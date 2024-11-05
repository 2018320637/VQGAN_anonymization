import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import *
import time, os, warnings
import numpy as np
from tqdm import tqdm
from itertools import cycle

from models.vqgan import VQGAN
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.cuda.amp import autocast, GradScaler

import config as cfg
from dataset import *
import argparse

from models.r3d_classifier import r2plus1d_18_domain

warnings.filterwarnings("ignore")

def fix_model(model):
    for name, param in model.named_parameters():
        param.requires_grad = False

def unfix_model(model):
    for name, param in model.named_parameters():
        param.requires_grad = True

def entropy_loss(predictions):
    """Compute the entropy loss for the target domain."""
    softmax_preds = nn.functional.softmax(predictions, dim=1)
    log_softmax_preds = nn.functional.log_softmax(predictions, dim=1)
    entropy = -torch.sum(softmax_preds * log_softmax_preds, dim=1)
    return torch.mean(entropy)

def generate_pseudo_labels(model, inputs, threshold=0.90):
    """Generate pseudo labels for the target domain."""
    model.eval()
    with torch.no_grad():
        outputs, _ = model(inputs)
        softmax_outputs = nn.functional.softmax(outputs, dim=1)
        max_probs, pseudo_labels = torch.max(softmax_outputs, dim=1)

    mask = max_probs > threshold
    pseudo_labels = pseudo_labels[mask]
    
    return pseudo_labels, mask

def seed_everything(seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def load_action(opts, type='action'):
    num_classes_action = opts.num_classes_action
    model = r2plus1d_18_domain(pretrained = opts.kin_pretrained, progress = False)
    model.fc = nn.Linear(512, num_classes_action)

    if opts.self_pretrained_action:
        pretrained = torch.load(opts.self_pretrained_action,map_location=f'cuda:0')
        pretrained_kvpair = pretrained['action_model_state_dict']

        model_kvpair = model.state_dict()
        for layer_name, weights in pretrained_kvpair.items():
            layer_name = layer_name.replace('module.','')
            model_kvpair[layer_name] = weights   
        model.load_state_dict(model_kvpair, strict=False)
        print(f'{type} model loaded successsfully!')

    return model

def load_fa(opts):
    model = VQGAN(opts)
    if opts.self_pretrained_fa:
        pretrained = torch.load(opts.self_pretrained_fa, map_location=f'cuda:0')
        pretrained_kvpair = pretrained['model_state_dict']
        model_kvpair = model.state_dict()
        for layer_name, weights in pretrained_kvpair.items():
            layer_name = layer_name.replace('module.','')
            model_kvpair[layer_name] = weights   
        model.load_state_dict(model_kvpair, strict=False)
        print(f'vqgan loaded successsfully!')
    return model

def train_epoch(fa_model, action_model, pseudo_model, train_loader_src, train_loader_tar, criterion, criterion_domain, optimizer, use_cuda, epoch, writer, opts, scaler):

    losses_action = []
    losses_entropy = []
    losses_pseudo = []
    losses_domain_src = []
    losses_domain_tar = []
    action_model.train()
    fa_model.eval()
    fix_model(fa_model)
    unfix_model(action_model)
    fix_model(pseudo_model)
    
    len_dataloader = max(len(train_loader_src), len(train_loader_tar))
    data_source_iter = cycle(train_loader_src)
    data_target_iter = cycle(train_loader_tar)
    for i in tqdm(range(len_dataloader), desc="Epoch: [{}]".format(epoch)):
        p = float(i + epoch * len_dataloader) / opts.num_epochs / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        data_src = next(data_source_iter)
        data_tar = next(data_target_iter)
        
        if use_cuda:
            inputs_src = data_src[0].cuda()
            labels_src = data_src[1].cuda(non_blocking=True)
            inputs_tar = data_tar[0].cuda()
            domain_labels_src = torch.zeros(len(inputs_src)).long().cuda()
            domain_labels_tar = torch.ones(len(inputs_tar)).long().cuda()
            
        optimizer.zero_grad()
        inputs_src = inputs_src.permute(0,2,1,3,4)
        inputs_tar = inputs_tar.permute(0,2,1,3,4)

        with autocast():
            outputs_fa_src = fa_model(inputs_src)
            outputs_fa_tar = fa_model(inputs_tar)
            outputs_src, domain_src = action_model(outputs_fa_src, alpha)
            outputs_tar, domain_tar = action_model(outputs_fa_tar, alpha)

            loss_action = criterion(outputs_src,labels_src)
            loss_entropy = entropy_loss(outputs_tar)
            loss_domain_src = criterion_domain(domain_src,domain_labels_src)
            loss_domain_tar = criterion_domain(domain_tar,domain_labels_tar)
            loss_pseudo = torch.tensor(0.0).cuda()
            pseudo_labels, mask = generate_pseudo_labels(pseudo_model, outputs_fa_tar, threshold=opts.pseude_threshold)
            if mask.sum() > 0:
                pseudo_outputs = outputs_tar[mask]
                pseudo_labels = pseudo_labels.cuda()

                loss_pseudo = criterion(pseudo_outputs, pseudo_labels)

            loss_action = opts.action_weight * loss_action
            loss_domain_src = opts.domain_weight * loss_domain_src
            loss_domain_tar = opts.domain_weight * loss_domain_tar
            loss_entropy = opts.entropy_weight * loss_entropy
            loss_pseudo = opts.pseudo_weight * loss_pseudo

            loss = loss_action + loss_domain_src + loss_domain_tar + loss_entropy + loss_pseudo
        
        losses_action.append(loss_action.item())
        losses_entropy.append(loss_entropy.item())
        losses_domain_src.append(loss_domain_src.item())
        losses_domain_tar.append(loss_domain_tar.item())
        losses_pseudo.append(loss_pseudo.item())
        writer.add_scalar(f'Train_loss_action {opts.src}', loss_action.item(), i)
        writer.add_scalar(f'Train_loss_entropy {opts.tar}', loss_entropy.item(), i)
        writer.add_scalar(f'Training_loss_domain_src {opts.src}', loss_domain_src.item(), i)
        writer.add_scalar(f'Training_loss_domain_tar {opts.src}', loss_domain_tar.item(), i)
        writer.add_scalar(f'Train_loss_pseudo {opts.tar}', loss_pseudo.item(), i)

        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    print('Training Epoch: %d, loss_action: %.4f' % (epoch, np.mean(losses_action)))
    writer.add_scalar(f'Training loss_action_epoch {opts.src}', np.mean(losses_action), epoch)
    writer.add_scalar(f'Training loss_entropy_epoch {opts.tar}', np.mean(losses_entropy), epoch)
    writer.add_scalar(f'Training loss_domain_src_epoch {opts.src}', np.mean(losses_domain_src), epoch)
    writer.add_scalar(f'Training loss_domain_tar_epoch {opts.src}', np.mean(losses_domain_tar), epoch)
    writer.add_scalar(f'Training loss_pseudo_epoch {opts.tar}', np.mean(losses_pseudo), epoch)

def val_epoch_src(fa_model, action_model, val_loader, criterion, criterion_domain, use_cuda, epoch, writer, val_type, val_dataset):
    val_losses =[]
    val_losses_domain = []
    predictions, ground_truth = [], []
    
    action_model.eval()
    fa_model.eval()

    loop = tqdm((val_loader), total = len(val_loader))
    for data in loop:
        inputs = data[0]
        labels = data[1]
        ground_truth.extend(labels)
        if len(inputs.shape) != 1:
            if use_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda(non_blocking=True)
                labels_domain = torch.zeros(len(inputs)).long().cuda()
            
            with torch.no_grad(), autocast():
                inputs = inputs.permute(0,2,1,3,4)
                outputs_fa = fa_model(inputs)
                outputs, domain = action_model(outputs_fa)
                loss = criterion(outputs,labels)
                loss_domain = criterion_domain(domain,labels_domain)
                val_losses.append(loss.item())
                val_losses_domain.append(loss_domain.item())

            predictions.extend(nn.functional.softmax(outputs, dim = 1).cpu().data.numpy())
    
    ground_truth = np.asarray(ground_truth)
    pred_array = np.flip(np.argsort(predictions,axis=1),axis=1) 
    c_pred = pred_array[:,0]
    
    correct_count = np.sum(c_pred==ground_truth)
    accuracy = float(correct_count)/len(c_pred)
    
    print(f'Epoch {epoch} : Top1 on {val_type} {val_dataset} is {accuracy*100 :.3f}% ')
    writer.add_scalar(f'Validation Loss {val_dataset}', np.mean(val_losses), epoch)
    writer.add_scalar(f'Validation Accuracy {val_dataset}', np.mean(accuracy), epoch)
    writer.add_scalar(f'Validation Loss Domain {val_dataset}', np.mean(val_losses_domain), epoch)
    return accuracy

def val_epoch_tar(fa_model, action_model, val_loader, criterion, criterion_domain, use_cuda, epoch, writer, val_type, val_dataset):
    val_losses =[]
    val_losses_domain = []
    val_entropy = []
    predictions, ground_truth = [], []
    
    action_model.eval()
    fa_model.eval()

    loop = tqdm((val_loader), total = len(val_loader))
    for data in loop:
        inputs = data[0]
        labels = data[1]
        ground_truth.extend(labels)
        if len(inputs.shape) != 1:
            if use_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda(non_blocking=True)
                labels_domain = torch.ones(len(inputs)).long().cuda()
            
            with torch.no_grad(), autocast():
                inputs = inputs.permute(0,2,1,3,4)
                outputs_fa = fa_model(inputs)
                outputs, domain = action_model(outputs_fa)
                loss = criterion(outputs,labels)
                loss_domain = criterion_domain(domain,labels_domain)
                val_losses.append(loss.item())
                val_losses_domain.append(loss_domain.item())
                val_entropy.append(entropy_loss(outputs).item())
            predictions.extend(nn.functional.softmax(outputs, dim = 1).cpu().data.numpy())
    
    ground_truth = np.asarray(ground_truth)
    pred_array = np.flip(np.argsort(predictions,axis=1),axis=1) 
    c_pred = pred_array[:,0]
    
    correct_count = np.sum(c_pred==ground_truth)
    accuracy = float(correct_count)/len(c_pred)
    
    print(f'Epoch {epoch} : Top1 accuracy on {val_dataset} is {accuracy*100 :.3f}% ')
    writer.add_scalar(f'Validation Loss {val_dataset}', np.mean(val_losses), epoch)
    writer.add_scalar(f'Validation Accuracy {val_dataset}', np.mean(accuracy), epoch)
    writer.add_scalar(f'Validation Loss Domain {val_dataset}', np.mean(val_losses_domain), epoch)
    writer.add_scalar(f'Validation Entropy {val_dataset}', np.mean(val_entropy), epoch)

    return accuracy

def train_action(opts):
    use_cuda = True
    scaler = GradScaler()

    run_id = opts.run_id
    print(f'run id============ {run_id} =============')
    save_dir = os.path.join(cfg.save_dir_action_eval, str(run_id))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    params_file = os.path.join(save_dir, 'hyper_params.txt')
    with open(params_file, 'w') as f:
        for arg in vars(opts):
            f.write(f'{arg}: {getattr(opts, arg)}\n')
    writer = SummaryWriter(save_dir)


    #=================================prepare dataset============================================
    train_list_src = cfg.dataset_list_dir + f'{opts.src}/list_train.txt'
    train_list_tar = cfg.dataset_list_dir + f'{opts.tar}/list_train.txt'
    val_list_src = cfg.dataset_list_dir + f'{opts.src}/list_val.txt'
    val_list_tar = cfg.dataset_list_dir + f'{opts.tar}/list_val.txt'
    num_train_src = sum(1 for i in open(train_list_src))
    num_train_tar = sum(1 for i in open(train_list_tar))
    num_val_src = sum(1 for i in open(val_list_src))
    num_val_tar = sum(1 for i in open(val_list_tar))
    train_aug_num_src = opts.batch_size - num_train_src % opts.batch_size
    train_aug_num_tar = opts.batch_size - num_train_tar % opts.batch_size

    val_set_src = TSNDataSet("", val_list_src, num_dataload=num_val_src, num_segments=opts.num_frames,
                            new_length=1, modality='RGB',
                            random_shift=False,
                            test_mode=True,
                            opts=opts,
                            data_name=opts.src,
                            )
    val_loader_src = DataLoader(val_set_src, batch_size=opts.batch_size, shuffle=False,
                                                num_workers=opts.batch_size, pin_memory=True)

    val_set_tar = TSNDataSet("", val_list_tar, num_dataload=num_val_tar, num_segments=opts.num_frames,
                            new_length=1, modality='RGB',
                            random_shift=False,
                            test_mode=True,
                            opts=opts,
                            data_name=opts.tar,
                            )
    val_loader_tar = DataLoader(val_set_tar, batch_size=opts.batch_size, shuffle=False,
                                                    num_workers=opts.batch_size, pin_memory=True)

    train_set_src = TSNDataSet("", train_list_src, num_dataload=num_train_src+train_aug_num_src, num_segments=opts.num_frames,
                            new_length=1, modality='RGB',
                            random_shift=True,
                            test_mode=False,
                            opts=opts,
                            data_name=opts.src,
                            )
    train_sampler_src = RandomSampler(train_set_src)
    train_loader_src = DataLoader(train_set_src, batch_size=opts.batch_size, shuffle=False,
                                                sampler=train_sampler_src, num_workers=opts.batch_size, pin_memory=True)
    
    train_set_tar = TSNDataSet("", train_list_tar, num_dataload=num_train_tar+train_aug_num_tar, num_segments=opts.num_frames,
                            new_length=1, modality='RGB',
                            random_shift=True,
                            test_mode=False,
                            opts=opts,
                            data_name=opts.tar,
                            )
    train_sampler_tar = RandomSampler(train_set_tar)
    train_loader_tar = DataLoader(train_set_tar, batch_size=opts.batch_size, shuffle=False,
                                                sampler=train_sampler_tar, num_workers=opts.batch_size, pin_memory=True)

    print(f'Train dataset length src: {len(train_set_src)}')
    print(f'Train dataset steps per epoch src: {len(train_set_src)/opts.batch_size}')
    print(f'Train dataset length tar: {len(train_set_tar)}')
    print(f'Train dataset steps per epoch tar: {len(train_set_tar)/opts.batch_size}')
    print(f'Validation dataset length src: {len(val_set_src)}')
    print(f'Validation dataset steps per epoch src: {len(val_set_src)/opts.batch_size}')
    print(f'Validation dataset length tar: {len(val_set_tar)}')
    print(f'Validation dataset steps per epoch tar: {len(val_set_tar)/opts.batch_size}')
    #================================end of dataset preparation=================================
    weight_class = torch.ones(opts.num_classes).cuda()
    if opts.weight_class:
        class_id_list = [int(line.strip().split(' ')[2]) for line in open(train_list_src)]
        _, class_data_counts = np.unique(np.array(class_id_list), return_counts=True)
        class_freq = (class_data_counts / class_data_counts.sum()).tolist()
        weight_class = 1 / torch.Tensor(class_freq).cuda()
    #================================end of weight class=========================================

    fa_model = load_fa(opts)
    fix_model(fa_model)
    action_model = load_action(opts, type='action')
    pseudo_model = load_action(opts, type='pseudo')
    unfix_model(action_model)
    criterion_ft = nn.CrossEntropyLoss(weight=weight_class)
    criterion_domain = nn.CrossEntropyLoss()

    if use_cuda:
        fa_model = fa_model.cuda()
        action_model = action_model.cuda()
        pseudo_model = pseudo_model.cuda()
        criterion_ft = criterion_ft.cuda()
        criterion_domain = criterion_domain.cuda()

    optimizer_ft = optim.SGD(action_model.parameters(), lr=opts.learning_rate)


    best_acc_src = .0
    best_acc_tar = .0
    for epoch in range(0, opts.num_epochs):
        print(f'Epoch {epoch} started')
        start=time.time()
        
        train_epoch(fa_model, action_model, pseudo_model, train_loader_src, train_loader_tar, criterion_ft, criterion_domain, optimizer_ft, use_cuda, epoch, writer, opts, scaler)

        if epoch % opts.val_freq == 0:

            acc_src = val_epoch_src(fa_model, action_model, val_loader_src, criterion_ft, criterion_domain, use_cuda, epoch, writer, 'src', opts.src)
            acc_tar = val_epoch_tar(fa_model, action_model, val_loader_tar, criterion_ft, criterion_domain, use_cuda, epoch, writer, 'tar', opts.tar)
            
            if acc_src > best_acc_src:
                best_acc_src = acc_src
                save_file_path = os.path.join(save_dir, 'model_{}2{}_{}_acc_{}.pth'.format(opts.src, opts.src, epoch, acc_src))
                states = {
                    'epoch': epoch,
                    'top1_acc': acc_src,
                    'action_model_state_dict': action_model.state_dict(),
                }
                torch.save(states, save_file_path)
            if acc_tar > best_acc_tar:
                best_acc_tar = acc_tar
                save_file_path = os.path.join(save_dir, 'model_{}2{}_{}_acc_{}.pth'.format(opts.src, opts.tar, epoch, acc_tar))
                states = {
                    'epoch': epoch,
                    'top1_acc': acc_tar,
                    'action_model_state_dict': action_model.state_dict(),
                }
                torch.save(states, save_file_path)
        
        taken = time.time()-start
        print(f'Time taken for Epoch-{epoch} is {taken}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #=================RUN PARAMETERS=================
    parser.add_argument("--run_id", dest='run_id', type=str, required=False, default= "dummy_recon", help='run_id')
    parser.add_argument("--src", dest='src', type=str, required=False, default= "ucf", help='src')
    parser.add_argument("--tar", dest='tar', type=str, required=False, default= "ucf", help='tar')
    #=================MODEL PARAMETERS=================
    parser.add_argument("--kin_pretrained", dest='kin_pretrained', type=int, required=False, default=1, help='kin_pretrained')
    parser.add_argument("--self_pretrained_action", dest='self_pretrained_action', type=int, required=False, default=False, help='self_pretrained_action')
    parser.add_argument("--self_pretrained_fa", dest='self_pretrained_fa', type=int, required=False, default=False, help='self_pretrained_fa')
    parser.add_argument("--self_pretrained_pseudo", dest='self_pretrained_pseudo', type=str, required=False, default=None, help='self_pretrained_pseudo')
    #=================TRAINING PARAMETERS=================
    parser.add_argument("--num_frames", dest='num_frames', type=int, required=False, default=16, help='Number of frames')
    parser.add_argument("--num_workers", dest='num_workers', type=int, required=False, default=10, help='Number of workers')
    parser.add_argument("--batch_size", dest='batch_size', type=int, required=False, default=32, help='Batch size')
    parser.add_argument("--learning_rate", dest='learning_rate', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument("--num_epochs", dest='num_epochs', type=int, required=False, default=300, help='Number of epochs')
    parser.add_argument("--weight_class", dest='weight_class', type=int, required=False, default=False, help='weight_class')
    parser.add_argument("--val_freq", dest='val_freq', type=int, required=False, default=10, help='Validation frequency')
    #=================DATA PARAMETERS=================
    parser.add_argument("--triple", dest='triple', type=int, required=False, default=0, help='triple')
    parser.add_argument("--reso_h", dest='reso_h', type=int, required=False, default=128, help='Resolution height')
    parser.add_argument("--reso_w", dest='reso_w', type=int, required=False, default=128, help='Resolution width')
    parser.add_argument("--num_classes_action", dest='num_classes_action', type=int, required=False, default=12, help='Number of action classes')
    #=================OPTIMIZER PARAMETERS=================
    parser.add_argument("--action_weight", dest='action_weight', type=float, required=False, default=1.0, help='Action weight')
    parser.add_argument("--pseudo_weight", dest='pseudo_weight', type=float, required=False, default=1.0, help='Pseudo weight')
    parser.add_argument("--entropy_weight", dest='entropy_weight', type=float, required=False, default=1.0, help='Entropy weight')
    parser.add_argument("--domain_weight", dest='domain_weight', type=float, required=False, default=1.0, help='Domain weight')
    parser.add_argument("--pseude_threshold", dest='pseude_threshold', type=float, required=False, default=0.90, help='Pseudo threshold')
    seed_everything(1)

    opts = parser.parse_args()
    opts.run_id = f'action_eval_da_{opts.src}2{opts.tar}_frames_{opts.num_frames}_bs_{opts.batch_size}_lr_{opts.learning_rate}_action_{opts.action_weight}_pseudo_{opts.pseudo_weight}_entropy_{opts.entropy_weight}_domain_{opts.domain_weight}_th_{opts.pseude_threshold}'

    train_action(opts)
