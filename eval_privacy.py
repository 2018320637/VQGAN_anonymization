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
import pickle

from sklearn.metrics import precision_recall_fscore_support, average_precision_score

from models.model_utils import build_model_privacy

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

def load_privacy(opts):
    
    model = build_model_privacy(
        architecture = opts.architecture,
        pretrained = opts.privacy_pretrained,
        num_classes = opts.num_classes_privacy,
        train_backbone = opts.train_backbone,
    )
    if opts.self_pretrained_privacy:
        weights = torch.load(opts.self_pretrained_privacy,map_location=f'cuda:0')
        model.load_state_dict(weights["privacy_model_state_dict"], strict=False)
        print("privacy model {} loaded successfully".format(opts.architecture))

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

def train_epoch(fa_model, privacy_model, train_loader, criterion, optimizer, use_cuda, epoch, writer, opts, scaler):

    losses_privacy = [] 
    privacy_model.train()
    fa_model.eval()
    fix_model(fa_model)
    unfix_model(privacy_model)
    
    loop = tqdm((train_loader), total = len(train_loader))
    for data in loop:
        if use_cuda:
            inputs = data[0].cuda()
            labels = data[2].cuda(non_blocking=True)
            labels = labels.unsqueeze(1).expand(-1, inputs.size(1), -1)
            labels = labels.reshape(-1, labels.size(2))

        optimizer.zero_grad()

        B, T, C, H, W = inputs.shape
        inputs = inputs.permute(0,2,1,3,4)

        with autocast():
            outputs_fa = fa_model(inputs)
            outputs_fa = outputs_fa.permute(0,2,1,3,4).reshape(B*T, C, H, W)
            outputs_privacy = privacy_model(outputs_fa)
            loss_privacy = criterion(outputs_privacy, labels)

        scaler.scale(loss_privacy).backward()
        scaler.step(optimizer)
        scaler.update()

        losses_privacy.append(loss_privacy.item())
        writer.add_scalar('Train loss privacy step', loss_privacy.item(), epoch)
        
        loop.set_description(f'Epoch [{epoch}/{opts.num_epochs}]')
        loop.set_postfix(loss = loss_privacy.item())
    del loss_privacy, inputs, outputs_fa, outputs_privacy, labels

    print('Training Epoch: %d, loss_privacy: %.4f' % (epoch, np.mean(losses_privacy)))
    writer.add_scalar(f'Training loss_privacy {opts.src}', np.mean(losses_privacy), epoch)

def val_epoch(fa_model, privacy_model, val_loader, criterion, use_cuda, epoch, writer, val_dataset, opts):
    
    fa_model.eval()
    privacy_model.eval()
    test_loss = 0.0
    predictions, ground_truth = [], []
    total_length = 0

    with torch.no_grad(), autocast():
        loop = tqdm((val_loader), total = len(val_loader))
        for data in loop:
            inputs = data[0]
            privacy_label = data[2]
            if use_cuda:
                inputs = inputs.cuda()
                privacy_label = privacy_label.cuda(non_blocking=True)
                privacy_label = privacy_label.unsqueeze(1).expand(-1, inputs.size(1), -1)
                privacy_label = privacy_label.reshape(-1, privacy_label.size(2))
            total_length = total_length+inputs.shape[0]

            B, T, C, H, W = inputs.shape
            inputs = inputs.permute(0,2,1,3,4)

            with autocast():
                outputs_fa = fa_model(inputs)
                outputs_fa = outputs_fa.permute(0,2,1,3,4).reshape(B*T, C, H, W)
                outputs_privacy = privacy_model(outputs_fa)
                loss = criterion(outputs_privacy, privacy_label)

            predictions.extend(outputs_privacy.cpu().data.numpy())
            ground_truth.extend(privacy_label.cpu().data.numpy())

            test_loss += loss.item() * inputs.size(0)
            loop.set_description(f'Epoch [{epoch}/{opts.num_epochs}]')

        test_loss = test_loss / total_length
        ground_truth = np.asarray(ground_truth)
        predictions = np.asarray(predictions)
        ap = average_precision_score(ground_truth, predictions, average=None)
        prec, recall, f1, _ = precision_recall_fscore_support(ground_truth, (np.array(predictions) > 0.5).astype(int))

        writer.add_scalar(f'Val Loss {val_dataset}',np.mean(test_loss),epoch)
        writer.add_scalar(f'Val cMAP {val_dataset}',np.mean(ap),epoch)
        writer.add_scalar(f'Val f1 {val_dataset}',np.mean(f1),epoch)
        print(f"Epoch: {epoch}/{opts.num_epochs}, Testing Loss: {test_loss:.4f}, Testing cMAP: {np.mean(ap):.4f} Testing f1: {np.mean(f1):.4f}")

    return np.mean(ap), np.mean(f1)

def train_privacy(opts):
    use_cuda = True
    scaler = GradScaler()

    run_id = opts.run_id
    print(f'run id============ {run_id} =============')
    save_dir = os.path.join(cfg.save_dir_privacy_da, str(run_id))
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
                            new_length=1, modality='RGB',
                            random_shift=False,
                            test_mode=True,
                            opts=opts,
                            data_name=opts.src
                            )
    val_loader = DataLoader(val_set, batch_size=opts.batch_size, shuffle=False,
                                                num_workers=opts.batch_size, pin_memory=True)

    train_set = TSNDataSet("", train_list, num_dataload=num_train+train_aug_num, num_segments=opts.num_frames,
                            new_length=1, modality='RGB',
                            random_shift=True,
                            test_mode=False,
                            opts=opts,
                            data_name=opts.src
                            )
    train_sampler = RandomSampler(train_set)
    train_loader = DataLoader(train_set, batch_size=opts.batch_size, shuffle=False,
                                                sampler=train_sampler, num_workers=opts.batch_size, pin_memory=True)

    print(f'Train dataset length: {len(train_set)}')
    print(f'Train dataset steps per epoch: {len(train_set)/opts.batch_size}')
    print(f'Validation dataset length: {len(val_set)}')
    print(f'Validation dataset steps per epoch: {len(val_set)/opts.batch_size}')
    #================================end of dataset preparation=================================
    weight_class = torch.ones(opts.num_classes_privacy).cuda()

    if opts.weight_class :

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
        weight_class = torch.Tensor(weights).cuda()

    #================================end of prepare dataset============================================


    fa_model = load_fa(opts)
    privacy_model = load_privacy(opts)
    criterion_privacy = nn.BCEWithLogitsLoss(weight=weight_class).cuda()
    fix_model(fa_model)
    unfix_model(privacy_model)

    if use_cuda:
        fa_model = fa_model.cuda()
        privacy_model = privacy_model.cuda()
        criterion_privacy = criterion_privacy.cuda()

    optimizer_privacy = optim.SGD(privacy_model.parameters(), lr=opts.learning_rate)
    

    best_cMAP = .0
    for epoch in range(0, opts.num_epochs):
        print(f'Epoch {epoch} started')
        start=time.time()
        
        train_epoch(fa_model, privacy_model, train_loader, criterion_privacy, optimizer_privacy, use_cuda, epoch, writer, opts, scaler)

        if epoch % opts.val_freq == 0:

            cMAP, f1 = val_epoch(fa_model, privacy_model, val_loader, criterion_privacy, use_cuda, epoch, writer, opts.src, opts)
            
            if cMAP > best_cMAP:
                best_cMAP = cMAP
                save_file_path = os.path.join(save_dir, 'model_{}_{}_cMAP_{}.pth'.format(opts.src, epoch, cMAP))
                states = {
                    'epoch': epoch,
                    'top1_acc': cMAP,
                    'privacy_model_state_dict': privacy_model.state_dict(),
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
    parser.add_argument("--privacy_pretrained", dest='privacy_pretrained', type=int, required=False, default=1, help='privacy_pretrained')
    parser.add_argument("--self_pretrained_privacy", dest='self_pretrained_privacy', type=str, required=False, help='self_pretrained')
    parser.add_argument("--self_pretrained_fa", dest='self_pretrained_fa', type=str, required=False, help='self_pretrained')
    parser.add_argument("--train_backbone", action='store_true')
    parser.add_argument("--architecture", dest='architecture', type=str, required=True, default='resnet50', help='architecture')
    #=================TRAINING PARAMETERS=================
    parser.add_argument("--num_frames", dest='num_frames', type=int, required=False, default=16, help='Number of frames')
    parser.add_argument("--sample_every_n_frames", dest='sample_every_n_frames', type=int, required=False, default=2, help='Sample every n frames')
    parser.add_argument("--num_workers", dest='num_workers', type=int, required=False, default=10, help='Number of workers')
    parser.add_argument("--batch_size", dest='batch_size', type=int, required=False, default=32, help='Batch size')
    parser.add_argument("--learning_rate", dest='learning_rate', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument("--num_epochs", dest='num_epochs', type=int, required=False, default=300, help='Number of epochs')
    parser.add_argument("-weight_class", dest='weight_class', action='store_true', help='weight_class')
    parser.add_argument("--val_freq", dest='val_freq', type=int, required=False, default=10, help='Validation frequency')
    #=================DATA PARAMETERS=================
    parser.add_argument("--triple", dest='triple', type=int, required=False, default=0, help='triple')
    parser.add_argument("--reso_h", dest='reso_h', type=int, required=False, default=128, help='Resolution height')
    parser.add_argument("--reso_w", dest='reso_w', type=int, required=False, default=128, help='Resolution width')
    parser.add_argument("--num_classes_privacy", dest='num_classes_privacy', type=int, required=False, default=12, help='Number of privacy classes')
    
    seed_everything(1)

    opts = parser.parse_args()
    opts.run_id = f'privacy_eval_{opts.src}2{opts.tar}_frames_{opts.num_frames}_bs_{opts.batch_size}_lr_{opts.learning_rate}_architecture_{opts.architecture}'

    train_privacy(opts)
