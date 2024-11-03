import time
import warnings
import torch
import torch.nn as nn
import argparse

from tqdm import tqdm
from models.vqgan import VQGAN
from models.r3d_classifier import r2plus1d_18
from models.utils import adopt_weight
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
import config as cfg
from dataset import *
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore")

def seed_everything(seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def load_ft(opts):
    num_classes = opts.num_classes
    model = r2plus1d_18(pretrained = opts.kin_pretrained, progress = False)
    model.fc = nn.Linear(512, num_classes)

    if opts.self_pretrained_action:
        pretrained = torch.load(opts.self_pretrained_action)
        pretrained_kvpair = pretrained['action_model_state_dict']

        model_kvpair = model.state_dict()
        for layer_name, weights in pretrained_kvpair.items():
            layer_name = layer_name.replace('module.','')
            model_kvpair[layer_name] = weights   
        model.load_state_dict(model_kvpair, strict=True)
        print(f'action_model loaded successsfully!')
    
    # for param in model.parameters():
    #     param.requires_grad = False

    # model.fc = nn.Linear(512, num_classes)
    # for param in model.fc.parameters():
    #     param.requires_grad = True

    return model

def load_fa(opts):
    model = VQGAN(opts)
    if opts.self_pretrained_vqgan:
        pretrained = torch.load(opts.self_pretrained_vqgan)
        pretrained_kvpair = pretrained['model_state_dict']
        model_kvpair = model.state_dict()
        for layer_name, weights in pretrained_kvpair.items():
            layer_name = layer_name.replace('module.','')
            model_kvpair[layer_name] = weights   
        model.load_state_dict(model_kvpair, strict=True)
        print(f'vqgan loaded successsfully!')
    return model


def train_epoch(model, action_model, criterion_action, train_loader, optimizer_ae, optimizer_action, use_cuda, epoch, opts, writer=None, local_rank=None):

    losses_action = []

    model.train()
    action_model.train()

    # Initialize gradient scaler for FP16
    scaler = GradScaler()

    idx = 0
    loop = tqdm(train_loader, total=len(train_loader)) if dist.get_rank() == 0 else train_loader
    
    running_loss = 0.0
    total_iters = len(train_loader)

    for data in loop:
        idx += 1
        if use_cuda:
            inputs = data[0].to(local_rank)
            labels = data[1].to(local_rank)

        optimizer_action.zero_grad()
        optimizer_ae.zero_grad()

        # Use autocast for mixed precision training
        with autocast():
            _, _, _, inputs_recon = model(inputs.permute(0,2,1,3,4), log_image=True)
            outputs = action_model(inputs_recon)
            loss_action = criterion_action(outputs, labels)

        running_loss += loss_action.item()

        # Use scaler for backward pass and optimization
        scaler.scale(loss_action).backward()
        scaler.step(optimizer_action)
        scaler.step(optimizer_ae)
        scaler.update()

    # 在epoch结束时打印整体平均loss
    if dist.get_rank() == 0:
        epoch_avg_loss = running_loss / total_iters
        print(f'Training Epoch: {epoch}, Average Loss: {epoch_avg_loss:.4f}')


def val_epoch(model, action_model, val_loader, use_cuda, epoch, writer=None, local_rank=None):
    predictions, ground_truth = [], []
    model.eval()
    action_model.eval()
    
    loop = tqdm(val_loader, total=len(val_loader)) if dist.get_rank() == 0 else val_loader
    for data in loop:
        inputs = data[0]
        labels = data[1]
        if len(inputs.shape) != 1:
            if use_cuda:
                inputs = inputs.to(local_rank)
                labels = labels.to(local_rank)
            with torch.no_grad(), autocast():
                _, _, _, recon_x = model(inputs.permute(0,2,1,3,4), log_image=True)
                outputs = action_model(recon_x)
                probs = nn.functional.softmax(outputs, dim=1)
            predictions.append(probs)
            ground_truth.append(labels)

    predictions = torch.cat(predictions, dim=0)
    ground_truth = torch.cat(ground_truth, dim=0)
    
    _, pred_indices = torch.max(predictions, dim=1)
    correct_count = (pred_indices == ground_truth).sum()
    accuracy = correct_count.float() / ground_truth.size(0)
    
    print(f'Epoch {epoch} : Top1 {accuracy*100 :.3f}% ')
    writer.add_scalar(f'Validation Accuracy', accuracy.item(), epoch)

    return accuracy.item()


def train_vqgan(opts, local_rank):
    use_cuda = True

    run_id = opts.run_id
    if dist.get_rank() == 0:
        print(f'run id============ {run_id} =============')
        save_dir = os.path.join(cfg.save_dir_action, str(run_id))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        params_file = os.path.join(save_dir, 'hyper_params.txt')
        with open(params_file, 'w') as f:
            for arg in vars(opts):
                f.write(f'{arg}: {getattr(opts, arg)}\n')
        writer = SummaryWriter(save_dir)
        


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
                                                num_workers=opts.v_batch_size, pin_memory=True)

    train_set = TSNDataSet("", train_list, num_dataload=num_train+train_aug_num, num_segments=opts.num_frames,
                            new_length=1, modality='RGB',sample_every_n_frames = opts.sample_every_n_frames,
                            random_shift=True,
                            test_mode=False,
                            opts=opts,
                            data_name=opts.src
                            )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_loader = DataLoader(train_set, batch_size=opts.batch_size, shuffle=False, sampler=train_sampler,
                                                num_workers=opts.batch_size, pin_memory=True)
    if dist.get_rank() == 0:
        print(f'Train dataset length: {len(train_set)}')
        print(f'Train dataset steps per epoch: {len(train_set)/opts.batch_size}')
        print(f'Validation dataset length: {len(val_set)}')
        print(f'Validation dataset steps per epoch: {len(val_set)/opts.batch_size}')
    #================================end of dataset preparation=================================

    class_id_list = [int(line.strip().split(' ')[2]) for line in open(train_list)]
    _, class_data_counts = np.unique(np.array(class_id_list), return_counts=True)
    class_freq = (class_data_counts / class_data_counts.sum()).tolist()

    weight_class = torch.ones(opts.num_classes).cuda()
    if opts.weight_class:
        weight_class = 1 / torch.Tensor(class_freq).cuda()

    criterion_action = nn.CrossEntropyLoss(weight=weight_class)

    model = load_fa(opts)
    action_model = load_ft(opts)
    
    if use_cuda:
        model = model.to(local_rank)
        action_model = action_model.to(local_rank)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,broadcast_buffers=False, find_unused_parameters=True)
        action_model = DDP(action_model, device_ids=[local_rank], output_device=local_rank,broadcast_buffers=False)

    optimizer_ae = torch.optim.Adam(list(model.module.encoder.parameters()) +
                                list(model.module.decoder.parameters()) +
                                list(model.module.pre_vq_conv.parameters()) +
                                list(model.module.post_vq_conv.parameters()) +
                                list(model.module.codebook.parameters()),
                                lr=opts.learning_rate, betas=(0.5, 0.9))
    optimizer_action = torch.optim.SGD(action_model.parameters(), lr=opts.learning_rate_action)

    for epoch in range(0, opts.num_epochs):
        if local_rank == 0:
            print(f'Epoch {epoch} started')
            start=time.time()

        train_loader.sampler.set_epoch(epoch)
        if dist.get_rank() == 0:
            train_epoch(model, action_model, criterion_action, train_loader, optimizer_ae, optimizer_action, use_cuda, epoch, opts, writer = writer, local_rank = local_rank)
        else:
            train_epoch(model, action_model, criterion_action, train_loader, optimizer_ae, optimizer_action, use_cuda, epoch, opts, local_rank = local_rank)
        if epoch % opts.val_freq == 0:

            if dist.get_rank() == 0:

                acc = val_epoch(model, action_model, val_loader, use_cuda, epoch, writer = writer, local_rank = local_rank)

                save_file_path = os.path.join(save_dir, f'model_{acc:.4f}.pth')
                states = {
                    'epoch': epoch,
                    'action_model_state_dict': action_model.state_dict(),
                    'model_state_dict': model.state_dict(),
                    'acc': acc,
                }
                torch.save(states, save_file_path)

            dist.barrier()
        
        if local_rank == 0:
            taken = time.time()-start
            print(f'Time taken for Epoch-{epoch} is {taken}')
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #=================RUN PARAMETERS=================
    parser.add_argument("--run_id", dest='run_id', type=str, required=False, default= "dummy_recon", help='run_id')
    parser.add_argument("--src", dest='src', type=str, required=False, default= "ucf", help='Source dataset')
    parser.add_argument("--tar", dest='tar', type=str, required=False, default= "hmdb", help='Target dataset')
    parser.add_argument("--kin_pretrained", dest='kin_pretrained', type=int, required=False, default=0, help='Kin pretrained')
    parser.add_argument("--self_pretrained_vqgan", dest='self_pretrained_vqgan', type=str, required=False, default=None, help='Self pretrained')
    parser.add_argument("--self_pretrained_action", dest='self_pretrained_action', type=str, required=False, default=None, help='Self pretrained')
    parser.add_argument("--appendix", dest='appendix', type=str, required=False, default='', help='Appendix')
    #=================TRAINING PARAMETERS=================
    parser.add_argument("--num_frames", dest='num_frames', type=int, required=False, default=16, help='Number of frames')
    parser.add_argument("--sample_every_n_frames", dest='sample_every_n_frames', type=int, required=False, default=2, help='Sample every n frames')
    parser.add_argument("--num_workers", dest='num_workers', type=int, required=False, default=10, help='Number of workers')
    parser.add_argument("--batch_size", dest='batch_size', type=int, required=False, default=32, help='Batch size')
    parser.add_argument("--v_batch_size", dest='v_batch_size', type=int, required=False, default=8, help='Validation batch size')
    parser.add_argument("--learning_rate", dest='learning_rate', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument("--learning_rate_action", dest='learning_rate_action', type=float, required=False, default=1e-3, help='Learning rate for action model')
    parser.add_argument("--num_epochs", dest='num_epochs', type=int, required=False, default=300, help='Number of epochs')
    parser.add_argument("--val_freq", dest='val_freq', type=int, required=False, default=10, help='Validation frequency')
    #=================DATA PARAMETERS=================
    parser.add_argument("--reso_h", dest='reso_h', type=int, required=False, default=128, help='Resolution height')
    parser.add_argument("--reso_w", dest='reso_w', type=int, required=False, default=128, help='Resolution width')
    parser.add_argument("--triple", dest='triple', type=int, required=False, default=0, help='Use triple sampling')
    #=================VQGAN PARAMETERS=================
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--n_codes', type=int, default=16384)
    parser.add_argument('--n_hiddens', type=int, default=32)
    parser.add_argument('--downsample', nargs='+', type=int, default=(4, 8, 8))
    parser.add_argument('--image_channels', type=int, default=3)
    parser.add_argument('--disc_channels', type=int, default=64)
    parser.add_argument('--disc_layers', type=int, default=3)
    parser.add_argument('--discriminator_iter_start', type=int, default=10)
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
    #=================ACTION PARAMETERS=================
    parser.add_argument('--num_classes', type=int, default=12)
    parser.add_argument('--weight_class', action='store_true')
    
    parser.add_argument("--local_rank", default=-1, type=int)
    
    seed_everything(1234)

    opts = parser.parse_args()
    opts.run_id = f'action_vqgan_amp_{opts.src}_frames_{opts.num_frames}_every_{opts.sample_every_n_frames}_bs_{opts.batch_size}_lr_{opts.learning_rate}_lr_action_{opts.learning_rate_action}_codes_{opts.n_codes}_dis_iter_{opts.discriminator_iter_start}_{opts.appendix}'

##########################和DDP有关的参数################################
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
######################################################################
    print(f'local_rank: {local_rank}')
    train_vqgan(opts, local_rank)