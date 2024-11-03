import time
import warnings
import torch
import argparse

from tqdm import tqdm
from models.vqgan import VQGAN
from models.utils import adopt_weight
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
import config as cfg
from dataset import *
import os
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from models.shuffle_discriminator import ShuffleDiscriminator

warnings.filterwarnings("ignore")

def seed_everything(seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def train_epoch(model, triplet_disc, train_loader, optimizer_ae, optimizer_disc, optimizer_triplet, triplet_loss, use_cuda, epoch, opts, writer=None, local_rank=None):

    optimizer_idx = 0
    losses_ae = []
    losses_disc = []
    losses_triplet = []

    model.train()
    disc_factor = adopt_weight(epoch, threshold=opts.discriminator_iter_start)
    triplet_factor = adopt_weight(epoch, threshold=opts.triplet_iter_start)
    
    idx = 0
    loop = tqdm(train_loader, total=len(train_loader)) if dist.get_rank() == 0 else train_loader

    scaler = GradScaler()
    
    for data in loop:
        idx += 1
        if use_cuda:
            inputs_ancher = data[0].to(local_rank)
            inputs_pos = data[3].to(local_rank)
            inputs_neg = data[4].to(local_rank)
            inputs_ancher = inputs_ancher.permute(0,2,1,3,4)
            inputs_pos = inputs_pos.permute(0,2,1,3,4)
            inputs_neg = inputs_neg.permute(0,2,1,3,4)

        if optimizer_idx == 0:

            with autocast():
                recon_loss, commitment_loss_static, commitment_loss_dynamic, aeloss, perceptual_loss, gan_feat_loss = model(inputs_ancher, optimizer_idx)
                loss = recon_loss + commitment_loss_static + commitment_loss_dynamic + disc_factor * (aeloss + gan_feat_loss) + perceptual_loss
            
            losses_ae.append(loss.item())
            optimizer_ae.zero_grad()

            scaler.scale(loss).backward()
            scaler.step(optimizer_ae)
            scaler.update()
            optimizer_idx = 1
            
            if dist.get_rank() == 0:
                writer.add_scalar("train/recon_loss", recon_loss, epoch*len(train_loader)+idx)
                writer.add_scalar("train/commitment_loss_static", commitment_loss_static, epoch*len(train_loader)+idx)
                writer.add_scalar("train/commitment_loss_dynamic", commitment_loss_dynamic, epoch*len(train_loader)+idx)
                writer.add_scalar("train/aeloss", aeloss, epoch*len(train_loader)+idx)
                writer.add_scalar("train/perceptual_loss", perceptual_loss, epoch*len(train_loader)+idx)
                writer.add_scalar("train/gan_feat_loss", gan_feat_loss, epoch*len(train_loader)+idx)
                loop.set_description(f'Epoch [{epoch}/{opts.num_epochs}]')
                loop.set_postfix({'loss_recon': recon_loss.item()})

        if optimizer_idx == 1:
            if disc_factor > 0:

                with autocast():
                    discloss = model(inputs_ancher, optimizer_idx)
                    loss = disc_factor * discloss
                
                losses_disc.append(loss.item())
                optimizer_disc.zero_grad()

                scaler.scale(loss).backward()
                scaler.step(optimizer_disc)
                scaler.update()
                optimizer_idx = 0
                if dist.get_rank() == 0:
                    writer.add_scalar("train/discloss", discloss, epoch*len(train_loader)+idx)
                    loop.set_description(f'Epoch [{epoch}/{opts.num_epochs}]')
                    loop.set_postfix({'loss_disc': loss.item()})
            else:
                optimizer_idx = 2
        
        if optimizer_idx == 2:

            if triplet_factor > 0:
                ancher_static, _ = model(inputs_ancher, optimizer_idx)
                pos_static, _ = model(inputs_pos, optimizer_idx)
                neg_static, _ = model(inputs_neg, optimizer_idx)

                ancher_feature_static = triplet_disc(ancher_static)
                pos_feature_static = triplet_disc(pos_static)
                neg_feature_static = triplet_disc(neg_static)

                triplet_loss = triplet_loss(ancher_feature_static, pos_feature_static, neg_feature_static)
                losses_triplet.append(triplet_loss.item())

                optimizer_triplet.zero_grad()
                optimizer_disc.zero_grad()
                optimizer_ae.zero_grad()
                scaler.scale(triplet_loss).backward()
                scaler.step(optimizer_triplet)
                scaler.step(optimizer_disc)
                scaler.step(optimizer_ae)
                scaler.update()

                optimizer_idx = 0
                if dist.get_rank() == 0:
                    writer.add_scalar("train/triplet_loss", triplet_loss, epoch*len(train_loader)+idx)
                    loop.set_description(f'Epoch [{epoch}/{opts.num_epochs}]')
                    loop.set_postfix({'loss_triplet': triplet_loss.item()})
            else:
                optimizer_idx = 0

    if dist.get_rank() == 0:
        print('Training Epoch: %d, loss_ae: %.4f, loss_disc: %.4f, loss_triplet: %.4f' % (epoch, np.mean(losses_ae), np.mean(losses_disc), np.mean(losses_triplet)))


def val_epoch(model, val_loader, use_cuda, epoch, writer=None, local_rank=None):
    
    recon_losses =[]
    
    model.eval()

    idx = 0
    loop = tqdm(val_loader, total=len(val_loader)) if dist.get_rank() == 0 else val_loader
    for data in loop:
        idx += 1
        inputs = data[0]
        if len(inputs.shape) != 1:
            if use_cuda:
                inputs = inputs.to(local_rank)

            with torch.no_grad():
                outputs = inputs.permute(0,2,1,3,4)
                recon_loss, x_recon, vq_output, perceptual_loss = model(outputs)
                recon_losses.append(recon_loss.item())
                if dist.get_rank() == 0:
                    writer.add_scalar("val/recon_loss", recon_loss, epoch*len(val_loader)+idx)
                    writer.add_scalar("val/perceptual_loss", perceptual_loss, epoch*len(val_loader)+idx)
                    writer.add_scalar("val/perplexity", vq_output['perplexity'], epoch*len(val_loader)+idx)
                    writer.add_scalar("val/commitment_loss", vq_output['commitment_loss'], epoch*len(val_loader)+idx)
                if idx == 1:
                    tmp = []
                    for batch_idx in range(x_recon.shape[0]):
                        tmp.append(x_recon[batch_idx])
                    tmp = torch.stack(tmp).permute(0, 2, 1, 3, 4).detach()
                    tmp = (tmp + 1) / 2
                    if dist.get_rank() == 0:
                        writer.add_video(f"val/video", tmp, global_step=epoch)
        if dist.get_rank() == 0:
            loop.set_description(f'Epoch [{epoch}/{opts.num_epochs}]')
            loop.set_postfix(loss = recon_loss.item())

    if dist.get_rank() == 0:
        print(f'Validation Epoch: {epoch}, recon_loss: {np.mean(recon_losses)}')

    return np.mean(recon_losses)


def train_vqgan(opts, local_rank):
    use_cuda = True

    run_id = opts.run_id
    if dist.get_rank() == 0:
        print(f'run id============ {run_id} =============')
        writer = SummaryWriter(os.path.join(cfg.logs, str(run_id)))
        save_dir = os.path.join(cfg.saved_models_dir, run_id)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)


    #=================================prepare dataset============================================
    train_list = cfg.dataset_list_dir + f'{opts.src}/list_train.txt'
    val_list = cfg.dataset_list_dir + f'{opts.src}/list_val.txt'
    num_train = sum(1 for i in open(train_list))
    num_val = sum(1 for i in open(val_list))
    train_aug_num = opts.batch_size - num_train % opts.batch_size

    val_set = TSNDataSet("", val_list, num_dataload=num_val, num_segments=opts.num_frames,
                            new_length=1, modality='RGB',sample_every_n_frames = opts.sample_every_n_frames,
                            random_shift=False,
                            test_mode=True,
                            opts=opts,
                            data_name=opts.src
                            )
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
    val_loader = DataLoader(val_set, batch_size=opts.batch_size, shuffle=False, sampler=val_sampler,
                                                num_workers=opts.num_workers, pin_memory=True)

    train_set = TSNDataSet("", train_list, num_dataload=num_train+train_aug_num, num_segments=opts.num_frames,
                            new_length=1, modality='RGB',sample_every_n_frames = opts.sample_every_n_frames,
                            random_shift=True,
                            test_mode=False,
                            opts=opts,
                            data_name=opts.src
                            )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_loader = DataLoader(train_set, batch_size=opts.batch_size, shuffle=False, sampler=train_sampler,
                                                num_workers=opts.num_workers, pin_memory=True)
    if dist.get_rank() == 0:
        print(f'Train dataset length: {len(train_set)}')
        print(f'Train dataset steps per epoch: {len(train_set)/opts.batch_size}')
        print(f'Validation dataset length: {len(val_set)}')
        print(f'Validation dataset steps per epoch: {len(val_set)/opts.batch_size}')
    #================================end of dataset preparation=================================

    model = VQGAN(opts)
    triplet_disc = ShuffleDiscriminator(opts.embedding_dim, num_domains=2)
    model = model.to(local_rank)
    triplet_disc = triplet_disc.to(local_rank)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank,broadcast_buffers=False, find_unused_parameters=True)
    triplet_disc = DDP(triplet_disc, device_ids=[local_rank], output_device=local_rank,broadcast_buffers=False)

    optimizer_ae = torch.optim.Adam(list(model.module.encoder.parameters()) +
                                list(model.module.decoder.parameters()) +
                                list(model.module.pre_vq_conv.parameters()) +
                                list(model.module.post_vq_conv.parameters()) +
                                list(model.module.codebook.parameters()),
                                lr=opts.learning_rate, betas=(0.5, 0.9))
    optimizer_disc = torch.optim.Adam(list(model.module.image_discriminator.parameters()) +
                                list(model.module.video_discriminator.parameters()),
                                lr=opts.learning_rate, betas=(0.5, 0.9))
    optimizer_triplet = torch.optim.Adam(list(triplet_disc.parameters()),
                                lr=opts.learning_rate, betas=(0.5, 0.9))

    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2).to(local_rank)

    for epoch in range(0, opts.num_epochs):
        if local_rank == 0:
            print(f'Epoch {epoch} started')
            start=time.time()

        train_loader.sampler.set_epoch(epoch)
        if dist.get_rank() == 0:
            train_epoch(model, triplet_disc, train_loader, optimizer_ae, optimizer_disc, optimizer_triplet, triplet_loss, use_cuda, epoch, opts, writer = writer, local_rank = local_rank)
        else:
            train_epoch(model, triplet_disc, train_loader, optimizer_ae, optimizer_disc, optimizer_triplet, triplet_loss, use_cuda, epoch, opts, local_rank = local_rank)
        if epoch % opts.val_freq == 0:
            if dist.get_rank() == 0:
                val_epoch(model, val_loader, use_cuda, epoch, writer = writer, local_rank = local_rank)
            else:
                val_epoch(model, val_loader, use_cuda, epoch, local_rank=local_rank)
            if dist.get_rank() == 0: 
                save_file_path = os.path.join(save_dir, 'model.pth')
                states = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                }
                torch.save(states, save_file_path)
        
        if local_rank == 0:
            taken = time.time()-start
            print(f'Time taken for Epoch-{epoch} is {taken}')
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #=================RUN PARAMETERS=================
    parser.add_argument("--run_id", dest='run_id', type=str, required=False, default= "dummy_recon", help='run_id')
    parser.add_argument("--src", dest='src', type=str, required=False, default= "ucf", help='Source dataset')
    parser.add_argument("--tar", dest='tar', type=str, required=False, default= "hmdb", help='Target dataset')
    #=================TRAINING PARAMETERS=================
    parser.add_argument("--num_frames", dest='num_frames', type=int, required=False, default=16, help='Number of frames')
    parser.add_argument("--sample_every_n_frames", dest='sample_every_n_frames', type=int, required=False, default=2, help='Sample every n frames')
    parser.add_argument("--num_workers", dest='num_workers', type=int, required=False, default=10, help='Number of workers')
    parser.add_argument("--batch_size", dest='batch_size', type=int, required=False, default=32, help='Batch size')
    parser.add_argument("--v_batch_size", dest='v_batch_size', type=int, required=False, default=8, help='Validation batch size')
    parser.add_argument("--learning_rate", dest='learning_rate', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument("--num_epochs", dest='num_epochs', type=int, required=False, default=300, help='Number of epochs')
    parser.add_argument("--val_freq", dest='val_freq', type=int, required=False, default=10, help='Validation frequency')
    #=================DATA PARAMETERS=================
    parser.add_argument("--reso_h", dest='reso_h', type=int, required=False, default=128, help='Resolution height')
    parser.add_argument("--reso_w", dest='reso_w', type=int, required=False, default=128, help='Resolution width')
    parser.add_argument("--triple", dest='triple', type=int, required=False, default=1, help='Use triple sampling') #用于控制是否使用triplet loss
    #=================VQGAN PARAMETERS=================
    parser.add_argument('--embedding_dim_dynamic', type=int, default=256)
    parser.add_argument('--embedding_dim_static', type=int, default=256)
    parser.add_argument('--n_codes_dynamic', type=int, default=16384)
    parser.add_argument('--n_codes_static', type=int, default=16384)
    parser.add_argument('--n_hiddens', type=int, default=32)
    parser.add_argument('--downsample', nargs='+', type=int, default=(4, 8, 8))
    parser.add_argument('--image_channels', type=int, default=3)
    parser.add_argument('--disc_channels', type=int, default=64)
    parser.add_argument('--disc_layers', type=int, default=3)
    parser.add_argument('--discriminator_iter_start', type=int, default=10)
    parser.add_argument('--triplet_iter_start', type=int, default=20) # 用于控制什么时候开始使用triplet loss来分离动态和静态信息
    parser.add_argument('--disc_loss_type', type=str, default='hinge', choices=['hinge', 'vanilla'])
    parser.add_argument('--image_gan_weight', type=float, default=1.0)
    parser.add_argument('--video_gan_weight', type=float, default=1.0)
    parser.add_argument('--l1_weight', type=float, default=4.0)
    parser.add_argument('--gan_feat_weight', type=float, default=4.0)
    parser.add_argument('--perceptual_weight', type=float, default=4.0)
    parser.add_argument('--restart_thres', type=float, default=1.0)
    parser.add_argument('--no_random_restart', action='store_true')
    parser.add_argument('--norm_type', type=str, default='batch', choices=['batch', 'group'])
    parser.add_argument('--padding_type', type=str, default='replicate', choices=['replicate', 'constant', 'reflect', 'circular'])
    
    parser.add_argument("--local_rank", default=-1, type=int)
    
    seed_everything(1234)

    opts = parser.parse_args()
    opts.run_id = f'train_vqgan_{opts.src}_frames_{opts.num_frames}_every_{opts.sample_every_n_frames}_bs_{opts.batch_size}_lr_{opts.learning_rate}_amp_codes_{opts.n_codes}_dis_iter_{opts.discriminator_iter_start}'

##########################和DDP有关的参数################################
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
######################################################################
    print(f'local_rank: {local_rank}')
    train_vqgan(opts, local_rank)