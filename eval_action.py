import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import *
import time, os, warnings
import numpy as np
from tqdm import tqdm

from models.vqgan import VQGAN
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.cuda.amp import autocast, GradScaler

import config as cfg
from dataset import *
import argparse

from models.r3d_classifier import r2plus1d_18

warnings.filterwarnings("ignore")

def fix_model(model):
    for name, param in model.named_parameters():
        param.requires_grad = False

def unfix_model(model):
    for name, param in model.named_parameters():
        param.requires_grad = True

def seed_everything(seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def load_action(opts):
    num_classes_action = opts.num_classes_action
    model = r2plus1d_18(pretrained = opts.kin_pretrained, progress = False)
    model.fc = nn.Linear(512, num_classes_action)

    if opts.self_pretrained_action:
        pretrained = torch.load(opts.self_pretrained_action,map_location=f'cuda:0')
        pretrained_kvpair = pretrained['action_model_state_dict']

        model_kvpair = model.state_dict()
        for layer_name, weights in pretrained_kvpair.items():
            layer_name = layer_name.replace('module.','')
            model_kvpair[layer_name] = weights   
        model.load_state_dict(model_kvpair, strict=False)
        print(f'action_model loaded successsfully!')

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

def train_epoch(fa_model, action_model, train_loader, criterion, optimizer, use_cuda, epoch, writer, opts, scaler):

    losses_action = []
    action_model.train()
    fa_model.eval()
    fix_model(fa_model)
    unfix_model(action_model)
    
    loop = tqdm((train_loader), total = len(train_loader))
    for data in loop:
        if use_cuda:
            inputs = data[0].cuda()
            labels = data[1].cuda(non_blocking=True)
            
        optimizer.zero_grad()

        inputs = inputs.permute(0,2,1,3,4)

        with autocast():
            outputs_fa = fa_model(inputs)
            outputs = action_model(outputs_fa)

        loss_action = criterion(outputs,labels) 

        losses_action.append(loss_action.item())
        scaler.scale(loss_action).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_description(f'Epoch [{epoch}/{opts.num_epochs}]')
        loop.set_postfix(loss = loss_action.item())
    del loss_action, inputs, outputs_fa, outputs, labels

    print('Training Epoch: %d, loss_action: %.4f' % (epoch, np.mean(losses_action)))
    writer.add_scalar(f'Training loss_action {opts.src}', np.mean(losses_action), epoch)
    writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

def val_epoch(fa_model, action_model, val_loader, criterion, use_cuda, epoch, writer, val_type, val_dataset):
    val_losses =[]
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
            
            with torch.no_grad(), autocast():
                inputs = inputs.permute(0,2,1,3,4)
                outputs_fa = fa_model(inputs)
                outputs = action_model(outputs_fa)

            loss = criterion(outputs,labels)
            val_losses.append(loss.item())

            predictions.extend(nn.functional.softmax(outputs, dim = 1).cpu().data.numpy())
    del inputs, outputs_fa, outputs, labels, loss
    
    ground_truth = np.asarray(ground_truth)
    pred_array = np.flip(np.argsort(predictions,axis=1),axis=1) 
    c_pred = pred_array[:,0]
    
    correct_count = np.sum(c_pred==ground_truth)
    accuracy = float(correct_count)/len(c_pred)
    
    print(f'Epoch {epoch} : Top1 on {val_type} {val_dataset} is {accuracy*100 :.3f}% ')
    writer.add_scalar(f'Validation Loss {val_dataset}', np.mean(val_losses), epoch)
    writer.add_scalar(f'Validation Accuracy {val_dataset}', np.mean(accuracy), epoch)

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
    train_list = cfg.dataset_list_dir + f'{opts.src}/list_train.txt'
    val_list_src = cfg.dataset_list_dir + f'{opts.src}/list_val.txt'
    val_list_tar = cfg.dataset_list_dir + f'{opts.tar}/list_val.txt'
    num_train = sum(1 for i in open(train_list))
    num_val_src = sum(1 for i in open(val_list_src))
    num_val_tar = sum(1 for i in open(val_list_tar))
    train_aug_num = opts.batch_size - num_train % opts.batch_size

    val_set_src = TSNDataSet("", val_list_src, num_dataload=num_val_src, num_segments=opts.num_frames,
                            new_length=1, modality='RGB',
                            random_shift=False,
                            test_mode=True,
                            opts=opts,
                            )
    val_loader_src = DataLoader(val_set_src, batch_size=opts.batch_size, shuffle=False,
                                                num_workers=opts.batch_size, pin_memory=True)

    val_set_tar = TSNDataSet("", val_list_tar, num_dataload=num_val_tar, num_segments=opts.num_frames,
                            new_length=1, modality='RGB',
                            random_shift=False,
                            test_mode=True,
                            opts=opts,
                            )
    val_loader_tar = DataLoader(val_set_tar, batch_size=opts.batch_size, shuffle=False,
                                                    num_workers=opts.batch_size, pin_memory=True)

    train_set = TSNDataSet("", train_list, num_dataload=num_train+train_aug_num, num_segments=opts.num_frames,
                            new_length=1, modality='RGB',
                            random_shift=True,
                            test_mode=False,
                            opts=opts,
                            )
    train_sampler = RandomSampler(train_set)
    train_loader = DataLoader(train_set, batch_size=opts.batch_size, shuffle=False,
                                                sampler=train_sampler, num_workers=opts.batch_size, pin_memory=True)

    print(f'Train dataset length: {len(train_set)}')
    print(f'Train dataset steps per epoch: {len(train_set)/opts.batch_size}')
    print(f'Validation dataset length src: {len(val_set_src)}')
    print(f'Validation dataset steps per epoch src: {len(val_set_src)/opts.batch_size}')
    print(f'Validation dataset length tar: {len(val_set_tar)}')
    print(f'Validation dataset steps per epoch tar: {len(val_set_tar)/opts.batch_size}')
    #================================end of dataset preparation=================================
    weight_class = torch.ones(opts.num_classes).cuda()
    if opts.weight_class:
        class_id_list = [int(line.strip().split(' ')[2]) for line in open(train_list)]
        _, class_data_counts = np.unique(np.array(class_id_list), return_counts=True)
        class_freq = (class_data_counts / class_data_counts.sum()).tolist()
        weight_class = 1 / torch.Tensor(class_freq).cuda()
    #================================end of weight class=========================================

    fa_model = load_fa(opts)
    fix_model(fa_model)
    action_model = load_action(opts)
    unfix_model(action_model)
    criterion_ft = nn.CrossEntropyLoss(weight=weight_class)

    if use_cuda:
        fa_model = fa_model.cuda()
        action_model = action_model.cuda()
        criterion_ft = criterion_ft.cuda()

    optimizer_ft = optim.SGD(action_model.parameters(), lr=opts.learning_rate)


    best_acc_src = .0
    best_acc_tar = .0
    for epoch in range(0, opts.num_epochs):
        print(f'Epoch {epoch} started')
        start=time.time()
        
        train_epoch(fa_model, action_model, train_loader, criterion_ft, optimizer_ft, use_cuda, epoch, writer, opts, scaler)

        if epoch % opts.val_freq == 0:

            acc_src = val_epoch(fa_model, action_model, val_loader_src, criterion_ft, use_cuda, epoch, writer, 'src', opts.src)
            acc_tar = val_epoch(fa_model, action_model, val_loader_tar, criterion_ft, use_cuda, epoch, writer, 'tar', opts.tar)
            
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
    
    seed_everything(1)

    opts = parser.parse_args()
    opts.run_id = f'action_eval_{opts.src}2{opts.tar}_frames_{opts.num_frames}_bs_{opts.batch_size}_lr_{opts.learning_rate}'

    train_action(opts)
