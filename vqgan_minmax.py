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
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
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
    num_classes_action = opts.num_classes_action
    model = r2plus1d_18(pretrained = opts.kin_pretrained, progress = False)
    model.fc = nn.Linear(512, num_classes_action)

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

def load_fb(opts):
    num_classes_privacy = opts.num_classes_privacy
    model = r2plus1d_18(pretrained = opts.kin_pretrained, progress = False)
    model.fc = nn.Linear(512, num_classes_privacy)

    if opts.self_pretrained_privacy:
        pretrained = torch.load(opts.self_pretrained_privacy)
        pretrained_kvpair = pretrained['privacy_model_state_dict']

        model_kvpair = model.state_dict()
        for layer_name, weights in pretrained_kvpair.items():
            layer_name = layer_name.replace('module.','')
            model_kvpair[layer_name] = weights   
        model.load_state_dict(model_kvpair, strict=True)
        print(f'privacy_model loaded successsfully!')
    
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


def train_epoch(model, action_model, privacy_model, criterion_action, criterion_privacy, train_loader, optimizer_ae, optimizer_action, optimizer_privacy, use_cuda, epoch, opts, writer=None, local_rank=None):


    # Initialize gradient scaler for FP16
    scaler = GradScaler()

    loop = tqdm(train_loader, total=len(train_loader)) if dist.get_rank() == 0 else train_loader
    
    running_loss_action = 0.0
    running_loss_privacy = 0.0
    running_loss_anonymized = 0.0
    total_iters = len(train_loader)

    for data in loop:
        if use_cuda:
            inputs = data[0].to(local_rank)
            action_labels = data[1].to(local_rank)
            privacy_labels = data[2].to(local_rank)

        #==========================step 1=========================================
        model.train()
        action_model.eval()
        privacy_model.eval()

        optimizer_action.zero_grad()
        optimizer_privacy.zero_grad()
        optimizer_ae.zero_grad()

        with autocast():
            _, _, _, inputs_recon = model(inputs.permute(0,2,1,3,4), log_image=True)
            outputs_action = action_model(inputs_recon)
            outputs_privacy = privacy_model(inputs_recon)

            loss_action = criterion_action(outputs_action, action_labels)
            loss_privacy = criterion_privacy(outputs_privacy, privacy_labels)
            loss_anonymized = opts.action_weight * loss_action - opts.privacy_weight * loss_privacy

        running_loss_anonymized += loss_anonymized.item()

        scaler.scale(loss_anonymized).backward()
        scaler.step(optimizer_ae)
        scaler.update()
        
        #==========================step 2=========================================
        model.eval()
        action_model.train()
        privacy_model.train()

        optimizer_action.zero_grad()
        optimizer_privacy.zero_grad()

        with autocast():
            _, _, _, inputs_recon = model(inputs.permute(0,2,1,3,4), log_image=True)
            outputs_action = action_model(inputs_recon)
            outputs_privacy = privacy_model(inputs_recon)

            loss_action = criterion_action(outputs_action, action_labels)
            loss_privacy = criterion_privacy(outputs_privacy, privacy_labels)

        running_loss_action += loss_action.item()
        running_loss_privacy += loss_privacy.item()

        scaler.scale(loss_action).backward(retain_graph=True)
        scaler.scale(loss_privacy).backward()
        scaler.step(optimizer_action)
        scaler.step(optimizer_privacy)
        scaler.update()

        #==========================end=========================================
        if dist.get_rank() == 0:
            loop.set_description(f'Epoch [{epoch}/{opts.num_epochs}]')
            loop.set_postfix({
                'loss_anonymized': loss_anonymized.item(),
                'loss_action': loss_action.item(),
                'loss_privacy': loss_privacy.item()
            })

    if dist.get_rank() == 0:
        epoch_avg_loss_action = running_loss_action / total_iters
        epoch_avg_loss_privacy = running_loss_privacy / total_iters
        epoch_avg_loss_anonymized = running_loss_anonymized / total_iters
        print(f'Training Epoch: {epoch}, Average Loss: anonymized {epoch_avg_loss_anonymized:.4f}, action {epoch_avg_loss_action:.4f}, privacy {epoch_avg_loss_privacy:.4f}')


def val_epoch_action(model, action_model, val_loader, use_cuda, epoch, writer, local_rank):
    predictions, ground_truth = [], []
    model.eval()
    action_model.eval()
    
    loop = tqdm(val_loader, total=len(val_loader))
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
            loop.set_description(f'Epoch [{epoch}/{opts.num_epochs}]')

    predictions = torch.cat(predictions, dim=0)
    ground_truth = torch.cat(ground_truth, dim=0)
    
    _, pred_indices = torch.max(predictions, dim=1)
    correct_count = (pred_indices == ground_truth).sum()
    accuracy = correct_count.float() / ground_truth.size(0)
    
    print(f'Epoch {epoch} : Top1 {accuracy*100 :.3f}% ')
    writer.add_scalar(f'Validation Accuracy', accuracy.item(), epoch)

    return accuracy.item()

def val_epoch_privacy(model, privacy_model, val_loader, criterion, use_cuda, epoch, writer, local_rank):
    
    model.eval()
    privacy_model.eval()

    test_loss = 0.0
    predictions, ground_truth = [], []
    total_length = 0

    with torch.no_grad():
        loop = tqdm((val_loader), total = len(val_loader))
        for data in loop:
            inputs = data[0]
            privacy_label = data[2]
            if use_cuda:
                inputs = inputs.cuda(local_rank)
                privacy_label = privacy_label.cuda(local_rank)

            total_length = total_length + inputs.shape[0]

            _, _, _, inputs_recon = model(inputs.permute(0,2,1,3,4), log_image=True)
            outputs = privacy_model(inputs_recon)
            loss_privacy = criterion(outputs, privacy_label)
            predictions.extend(outputs.cpu().data.numpy())
            ground_truth.extend(privacy_label.cpu().data.numpy())

            test_loss += loss_privacy.item() * inputs.shape[0]
            loop.set_description(f'Epoch [{epoch}/{opts.num_epochs}]')

        test_loss = test_loss / total_length
        ground_truth = np.asarray(ground_truth)
        predictions = np.asarray(predictions)
        ap = average_precision_score(ground_truth, predictions, average=None)
        prec, recall, f1, _ = precision_recall_fscore_support(ground_truth, (np.array(predictions) > 0.5).astype(int))

        writer.add_scalar(f'Validation Loss :',np.mean(test_loss),epoch)
        writer.add_scalar(f'Validation cMAP :',np.mean(ap),epoch)
        writer.add_scalar(f'Validation f1 :',np.mean(f1),epoch)
        print(f"Epoch: {epoch}/{opts.num_epochs}, Testing Loss: {test_loss:.4f}, Testing cMAP: {np.mean(ap):.4f} Testing f1: {np.mean(f1):.4f}")

    return np.mean(ap), np.mean(f1)


def train_vqgan(opts, local_rank):
    use_cuda = True

    run_id = opts.run_id
    if dist.get_rank() == 0:
        print(f'run id============ {run_id} =============')
        save_dir = os.path.join(cfg.save_dir_minmax, str(run_id))
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
    # val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
    # val_loader = DataLoader(val_set, batch_size=opts.batch_size, shuffle=False, sampler=val_sampler,
    #                                             num_workers=opts.v_batch_size, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=opts.batch_size, shuffle=False,
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

    #=================================weight class action============================================
    class_id_list = [int(line.strip().split(' ')[2]) for line in open(train_list)]
    _, class_data_counts = np.unique(np.array(class_id_list), return_counts=True)
    class_freq = (class_data_counts / class_data_counts.sum()).tolist()

    weight_class_action = torch.ones(opts.num_classes_action)
    if opts.weight_class_action:
        weight_class_action = 1 / torch.Tensor(class_freq)
    #=================================weight class privacy============================================
    if opts.src == 'ucf':
        with open('/home/zhiwei/source/dataset/ucf101_privacy_label.pickle', 'rb') as f:
            privacy_data = pickle.load(f)
    elif opts.src == 'hmdb':
        with open('/home/zhiwei/source/dataset/hmdb51_privacy_label.pickle', 'rb') as f:
            privacy_data = pickle.load(f)
    for line in open(train_list):
        path = line.strip().split(' ')[0]
        privacy_label = privacy_data[path.split('/')[-1].split('.')[0]]

    attribute_counts = np.zeros(5)

    total_samples = 0
    for line in open(train_list):
        path = line.strip().split(' ')[0]
        privacy_label = privacy_data[path.split('/')[-1].split('.')[0]][0]
        attribute_counts += np.array(privacy_label)
        total_samples += 1

    attribute_frequencies = attribute_counts / total_samples
    weights = 1.0 / attribute_frequencies
    weights = weights / np.sum(weights)

    weight_class_privacy = torch.ones(opts.num_classes_privacy)

    if opts.weight_class_privacy :
        weight_class_privacy = torch.Tensor(weights)
    #=================================criterion============================================
    criterion_ft = nn.CrossEntropyLoss(weight=weight_class_action)
    criterion_fb = nn.BCEWithLogitsLoss(weight=weight_class_privacy)

    if use_cuda:
        criterion_action = criterion_ft.to(local_rank)
        criterion_privacy = criterion_fb.to(local_rank)
    #=================================cuda============================================

    model = load_fa(opts)
    action_model = load_ft(opts)
    privacy_model = load_fb(opts)
    
    if use_cuda:
        model = model.to(local_rank)
        action_model = action_model.to(local_rank)
        privacy_model = privacy_model.to(local_rank)

        model = DDP(model, device_ids=[local_rank], output_device=local_rank,broadcast_buffers=False, find_unused_parameters=True)
        action_model = DDP(action_model, device_ids=[local_rank], output_device=local_rank,broadcast_buffers=False)
        privacy_model = DDP(privacy_model, device_ids=[local_rank], output_device=local_rank,broadcast_buffers=False)

    optimizer_ae = torch.optim.Adam(list(model.module.encoder.parameters()) +
                                list(model.module.decoder.parameters()) +
                                list(model.module.pre_vq_conv.parameters()) +
                                list(model.module.post_vq_conv.parameters()) +
                                list(model.module.codebook.parameters()),
                                lr=opts.learning_rate, betas=(0.5, 0.9))
    optimizer_action = torch.optim.SGD(action_model.parameters(), lr=opts.learning_rate_action)
    optimizer_privacy = torch.optim.SGD(privacy_model.parameters(), lr=opts.learning_rate_privacy)

    for epoch in range(0, opts.num_epochs):
        if local_rank == 0:
            print(f'Epoch {epoch} started')
            start=time.time()

        train_loader.sampler.set_epoch(epoch)
        if dist.get_rank() == 0:
            train_epoch(model, action_model, privacy_model, criterion_action, criterion_privacy, train_loader, optimizer_ae, optimizer_action, optimizer_privacy, use_cuda, epoch, opts, writer = writer, local_rank = local_rank)
        else:
            train_epoch(model, action_model, privacy_model, criterion_action, criterion_privacy, train_loader, optimizer_ae, optimizer_action, optimizer_privacy, use_cuda, epoch, opts, local_rank = local_rank)
        
        if epoch % opts.val_freq == 0:

            if dist.get_rank() == 0:

                acc = val_epoch_action(model, action_model, val_loader, use_cuda, epoch, writer = writer, local_rank = local_rank)
                cMAP, f1 = val_epoch_privacy(model, privacy_model, val_loader, criterion_privacy, use_cuda, epoch, writer = writer, local_rank = local_rank)

                save_file_path = os.path.join(save_dir, f'model_{epoch}_{acc:.4f}_{cMAP:.4f}_{f1:.4f}.pth')
                states = {
                    'epoch': epoch,
                    'action_model_state_dict': action_model.state_dict(),
                    'privacy_model_state_dict': privacy_model.state_dict(),
                    'model_state_dict': model.state_dict(),
                    'acc': acc,
                    'cMAP': cMAP,
                    'f1': f1,
                }
                torch.save(states, save_file_path)

        dist.barrier()
        
        if local_rank == 0:
            taken = time.time()-start
            print(f'Time taken for Epoch-{epoch} is {taken}')
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #=================RUN PARAMETERS=================================
    parser.add_argument("--run_id", dest='run_id', type=str, required=False, default= "dummy_recon", help='run_id')
    parser.add_argument("--src", dest='src', type=str, required=False, default= "ucf", help='Source dataset')
    parser.add_argument("--tar", dest='tar', type=str, required=False, default= "hmdb", help='Target dataset')
    parser.add_argument("--kin_pretrained", dest='kin_pretrained', type=int, required=False, default=0, help='Kin pretrained')
    parser.add_argument("--self_pretrained_vqgan", dest='self_pretrained_vqgan', type=str, required=False, default=None, help='Self pretrained')
    parser.add_argument("--self_pretrained_action", dest='self_pretrained_action', type=str, required=False, default=None, help='Self pretrained')
    parser.add_argument("--self_pretrained_privacy", dest='self_pretrained_privacy', type=str, required=False, default=None, help='Self pretrained')
    #=================TRAINING PARAMETERS==========================
    parser.add_argument("--num_frames", dest='num_frames', type=int, required=False, default=16, help='Number of frames')
    parser.add_argument("--sample_every_n_frames", dest='sample_every_n_frames', type=int, required=False, default=2, help='Sample every n frames')
    parser.add_argument("--num_workers", dest='num_workers', type=int, required=False, default=10, help='Number of workers')
    parser.add_argument("--batch_size", dest='batch_size', type=int, required=False, default=32, help='Batch size')
    parser.add_argument("--v_batch_size", dest='v_batch_size', type=int, required=False, default=8, help='Validation batch size')
    parser.add_argument("--learning_rate_privacy", dest='learning_rate_privacy', type=float, required=False, default=1e-3, help='Learning rate for privacy model')
    parser.add_argument("--learning_rate_action", dest='learning_rate_action', type=float, required=False, default=1e-3, help='Learning rate for action model')
    parser.add_argument("--learning_rate", dest='learning_rate', type=float, required=False, default=1e-3, help='Learning rate for VQGAN')
    parser.add_argument("--num_epochs", dest='num_epochs', type=int, required=False, default=300, help='Number of epochs')
    parser.add_argument("--val_freq", dest='val_freq', type=int, required=False, default=10, help='Validation frequency')
    #=================DATA PARAMETERS==============================
    parser.add_argument("--reso_h", dest='reso_h', type=int, required=False, default=128, help='Resolution height')
    parser.add_argument("--reso_w", dest='reso_w', type=int, required=False, default=128, help='Resolution width')
    parser.add_argument("--triple", dest='triple', type=int, required=False, default=0, help='Use triple sampling')
    #=================VQGAN PARAMETERS=============================
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
    #=================MINMAX PARAMETERS==========================
    parser.add_argument('--privacy_weight', type=float, default=1.0)
    parser.add_argument('--action_weight', type=float, default=1.0)
    #=================ACTION PARAMETERS==========================
    parser.add_argument('--num_classes_action', type=int, default=12)
    parser.add_argument('--num_classes_privacy', type=int, default=5)
    parser.add_argument('--weight_class_action', action='store_true')
    parser.add_argument('--weight_class_privacy', action='store_true')
    #=================DDP PARAMETERS==========================
    parser.add_argument("--local_rank", default=-1, type=int)
    
    seed_everything(1234)

    opts = parser.parse_args()
    opts.run_id = f'minmax_real_vqgan_amp_{opts.src}_frames_{opts.num_frames}_every_{opts.sample_every_n_frames}_bs_{opts.batch_size}_lr_vqgan_{opts.learning_rate}_lr_privacy_{opts.learning_rate_privacy}_lr_action_{opts.learning_rate_action}_codes_{opts.n_codes}_dis_iter_{opts.discriminator_iter_start}_privacy_weight_{opts.privacy_weight}_action_weight_{opts.action_weight}'

##########################和DDP有关的参数################################
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
######################################################################
    print(f'local_rank: {local_rank}')
    train_vqgan(opts, local_rank)