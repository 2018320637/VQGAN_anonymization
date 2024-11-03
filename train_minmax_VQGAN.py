import time
import warnings
import torch
import torch.nn as nn
import torchvision
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
from nt_xent_original import NTXentLoss
from models.unet_model import UNet
from models.model_utils import build_model_privacy
from torchvision.utils import save_image

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
        print(f'ft_model loaded successsfully!')

    return model

def load_fa(opts):
    fa_model = UNet(n_channels = 3, n_classes=3)
    saved = torch.load(opts.self_pretrained_fa)
    fa_model.load_state_dict(saved['fa_model_state_dict'], strict=True)
    print(f'fa_model loaded successsfully')
    return fa_model

def load_fb(opts, eval = False):
    
    model = build_model_privacy(
        architecture = opts.architecture,
        pretrained = opts.privacy_pretrained,
        num_classes = opts.num_classes_privacy,
        train_backbone = opts.train_backbone,
    )
    if eval:
        return model

    if opts.self_pretrained_privacy:
        weights = torch.load(opts.self_pretrained_privacy)
        model.load_state_dict(weights["privacy_model_state_dict"], strict=True)
        print("fb model {} loaded successfully".format(opts.architecture))

    return model

def train_epoch_minmax(model, action_model, privacy_model, criterion_action, criterion_privacy, train_loader, optimizer, optimizer_action, optimizer_privacy, epoch, opts, writer=None, local_rank=None):


    scaler = GradScaler()

    total_iters = len(train_loader)

    loop = tqdm(train_loader, total=total_iters) if dist.get_rank() == 0 else train_loader
    
    running_loss_action = 0.0
    running_loss_privacy = 0.0
    running_loss_anonymized = 0.0

    step = 0
    for data in loop:

        inputs_pos = data[0].to(local_rank)
        inputs_neg = data[1].to(local_rank)
        action_labels = data[2].to(local_rank)

        #==========================step 1=========================================
        if step % 2 == 0:
            model.train()
            action_model.eval()
            privacy_model.eval()

            optimizer_action.zero_grad()
            optimizer_privacy.zero_grad()
            optimizer.zero_grad()

            with autocast():
                ori_bs, ori_t, ori_c, ori_h, ori_w = inputs_pos.shape
                inputs_pos = inputs_pos.reshape(-1, ori_c, ori_h, ori_w)
                inputs_neg = inputs_neg.reshape(-1, ori_c, ori_h, ori_w)
                inputs_recon_pos = model(inputs_pos)
                inputs_recon_neg = model(inputs_neg)
                outputs_action_pos = action_model(inputs_recon_pos.reshape(ori_bs, ori_t, ori_c, ori_h, ori_w).permute(0,2,1,3,4))
                outputs_action_neg = action_model(inputs_recon_neg.reshape(ori_bs, ori_t, ori_c, ori_h, ori_w).permute(0,2,1,3,4))

                outputs_privacy_pos = privacy_model(inputs_recon_pos)
                outputs_privacy_neg = privacy_model(inputs_recon_neg)

                loss_action_pos = criterion_action(outputs_action_pos, action_labels)
                loss_action_neg = criterion_action(outputs_action_neg, action_labels)

                loss_action = loss_action_pos + loss_action_neg
                loss_privacy = criterion_privacy(outputs_privacy_pos, outputs_privacy_neg)
                loss_anonymized = opts.action_weight * loss_action - opts.privacy_weight * loss_privacy

            running_loss_anonymized += loss_anonymized.item()
            running_loss_action += loss_action.item()
            running_loss_privacy += loss_privacy.item()

            scaler.scale(loss_anonymized).backward()
            scaler.step(optimizer)
            scaler.update()

            step += 1

            if dist.get_rank() == 0:
                writer.add_scalar(f'Training loss_anonymized', loss_anonymized.item(), epoch * total_iters + step)
                writer.add_scalar(f'Training loss_action', loss_action.item(), epoch * total_iters + step)
                writer.add_scalar(f'Training loss_privacy', loss_privacy.item(), epoch * total_iters + step)
                loop.set_description(f'Epoch [{epoch}/{opts.num_epochs}]')
                loop.set_postfix({
                    'loss_anonymized': loss_anonymized.item(),
                    'loss_action': loss_action.item(),
                    'loss_privacy': loss_privacy.item()
                })

            continue
        
        #==========================step 2=========================================
        if step % 2 == 1:

            model.eval()
            action_model.eval()
            privacy_model.train()

            optimizer_action.zero_grad()
            optimizer_privacy.zero_grad()
            optimizer.zero_grad()

            with autocast():
                ori_bs, ori_t, ori_c, ori_h, ori_w = inputs_pos.shape
                inputs_pos = inputs_pos.reshape(-1, ori_c, ori_h, ori_w)
                inputs_neg = inputs_neg.reshape(-1, ori_c, ori_h, ori_w)
                inputs_recon_pos = model(inputs_pos)
                inputs_recon_neg = model(inputs_neg)
                outputs_action_pos = action_model(inputs_recon_pos)
                outputs_action_neg = action_model(inputs_recon_neg)

                outputs_privacy_pos = privacy_model(inputs_recon_pos)
                outputs_privacy_neg = privacy_model(inputs_recon_neg)

                loss_action_pos = criterion_action(outputs_action_pos, action_labels)
                loss_action_neg = criterion_action(outputs_action_neg, action_labels)

                loss_action = loss_action_pos + loss_action_neg
                loss_privacy = criterion_privacy(outputs_privacy_pos, outputs_privacy_neg)

                loss_anonymized = opts.action_weight * loss_action - opts.privacy_weight * loss_privacy

            running_loss_action += loss_action.item()
            running_loss_privacy += loss_privacy.item()
            running_loss_anonymized += loss_anonymized.item()

            scaler.scale(loss_action).backward(retain_graph=True)
            scaler.scale(loss_privacy).backward()
            scaler.step(optimizer_action)
            scaler.step(optimizer_privacy)
            scaler.update()

            step += 1

            if dist.get_rank() == 0:
                writer.add_scalar(f'Training loss_anonymized', loss_anonymized.item(), epoch * total_iters + step)
                writer.add_scalar(f'Training loss_action', loss_action.item(), epoch * total_iters + step)
                writer.add_scalar(f'Training loss_privacy', loss_privacy.item(), epoch * total_iters + step)
                loop.set_description(f'Epoch [{epoch}/{opts.num_epochs}]')
                loop.set_postfix({
                    'loss_anonymized': loss_anonymized.item(),
                    'loss_action': loss_action.item(),
                    'loss_privacy': loss_privacy.item()
                })

            continue

        #==========================end=========================================
    if dist.get_rank() == 0:
        epoch_avg_loss_action = running_loss_action / total_iters
        epoch_avg_loss_privacy = running_loss_privacy / total_iters
        epoch_avg_loss_anonymized = running_loss_anonymized / total_iters
        print(f'Training Epoch: {epoch}, Average Loss: anonymized {epoch_avg_loss_anonymized:.4f}, action {epoch_avg_loss_action:.4f}, privacy {epoch_avg_loss_privacy:.4f}')

    return model, action_model, privacy_model

def train_epoch_action(model, train_loader, criterion, optimizer, epoch, writer, opts):

    losses_action = [] 
    model.train()
    
    loop = tqdm((train_loader), total = len(train_loader))
    for data in loop:
        inputs = data[0].cuda()
        labels = data[1].cuda(non_blocking=True)
            
        optimizer.zero_grad()

        outputs = model((inputs).permute(0,2,1,3,4))

        loss_action = criterion(outputs,labels) 

        losses_action.append(loss_action.item())
        loss_action.backward()
        
        optimizer.step()
        loop.set_description(f'Epoch [{epoch}/{opts.num_epochs}]')
        loop.set_postfix(loss = loss_action.item())
    del loss_action, inputs, outputs, labels

    print('Training Epoch: %d, loss_action: %.4f' % (epoch, np.mean(losses_action)))
    writer.add_scalar(f'Training loss_action {opts.src}', np.mean(losses_action), epoch)
    writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

def train_epoch_privacy(model, train_loader, criterion, optimizer, epoch, writer, opts):

    losses_privacy = [] 
    model.train()
    scaler = GradScaler()
    
    loop = tqdm((train_loader), total = len(train_loader))
    for data in loop:
        inputs = data[0].cuda()
        labels = data[2].cuda(non_blocking=True)
        labels = labels.unsqueeze(1).expand(-1, inputs.size(1), -1)
        labels = labels.reshape(-1, labels.size(2))

        optimizer.zero_grad()

        B, T, C, H, W = inputs.shape
        inputs = inputs.reshape(-1, C, H, W)
        
        with autocast():
            outputs = model(inputs)
            loss_privacy = criterion(outputs, labels)

        scaler.scale(loss_privacy).backward()
        scaler.step(optimizer)
        scaler.update()

        losses_privacy.append(loss_privacy.item())
        writer.add_scalar('Train loss privacy step', loss_privacy.item(), epoch)
        
        loop.set_description(f'Epoch [{epoch}/{opts.num_epochs}]')
        loop.set_postfix(loss = loss_privacy.item())
    del loss_privacy, inputs, outputs, labels

    print('Training Epoch: %d, loss_privacy: %.4f' % (epoch, np.mean(losses_privacy)))
    writer.add_scalar(f'Training loss_privacy {opts.src}', np.mean(losses_privacy), epoch)

    return model

def val_epoch_action(model, action_model, val_loader, epoch, writer, local_rank):
    predictions, ground_truth = [], []
    model.eval()
    action_model.eval()
    
    loop = tqdm(val_loader, total=len(val_loader))
    for data in loop:
        inputs = data[0]
        labels = data[1]
        if len(inputs.shape) != 1:
            inputs = inputs.to(local_rank)
            labels = labels.to(local_rank)
            with torch.no_grad(), autocast():
                recon_x = model(inputs.permute(0,2,1,3,4))
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

def val_epoch_privacy(model, val_loader, criterion, epoch, writer, val_type, val_dataset, opts):
    model.eval()
    test_loss = 0.0
    predictions, ground_truth = [], []
    total_length = 0

    with torch.no_grad():
        loop = tqdm((val_loader), total = len(val_loader))
        for data in loop:
            inputs = data[0]
            privacy_label = data[2]

            inputs = inputs.cuda()
            privacy_label = privacy_label.cuda(non_blocking=True)
            privacy_label = privacy_label.unsqueeze(1).expand(-1, inputs.size(1), -1)
            privacy_label = privacy_label.reshape(-1, privacy_label.size(2))

            total_length = total_length+inputs.shape[0]

            B, T, C, H, W = inputs.shape
            inputs = inputs.reshape(-1, C, H, W)
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, privacy_label)

            predictions.extend(outputs.cpu().data.numpy())
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

def val_visualization(save_dir, epoch, validation_dataloader, model):

    for i, data in enumerate(validation_dataloader):
        if len(data[0].shape) == 1:
            continue
        
        inputs = data[0].cuda()
        ori_bs, ori_t, ori_c, ori_h, ori_w = inputs.shape
        inputs = inputs.reshape(-1, ori_c, ori_h, ori_w)
        image_full_name = save_dir + 'e'+ str(epoch) + '.png'
        with torch.no_grad():
            outputs = model(inputs).reshape(ori_bs, ori_t, ori_c, ori_h, ori_w)
            inputs = inputs.reshape(ori_bs, ori_t, ori_c, ori_h, ori_w)
            vis_image = torch.cat([inputs[:, 0], outputs[:, 0]], dim=0)
            save_image(vis_image, image_full_name, padding=5, nrow=int(inputs.shape[0]))
            break

def train_minmax(opts, local_rank):

    run_id = opts.run_id
    if dist.get_rank() == 0:
        print(f'run id============ {run_id} =============')
        save_dir = os.path.join(cfg.save_dir_minmax, str(run_id))
        save_dir_vis_src = os.path.join(save_dir, 'vis_src')
        save_dir_vis_tar = os.path.join(save_dir, 'vis_tar')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            os.makedirs(save_dir_vis_src)
            os.makedirs(save_dir_vis_tar)
            
            
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
                            new_length=1, modality='RGB',sample_every_n_frames = opts.sample_every_n_frames,
                            random_shift=False,
                            test_mode=True,
                            opts=opts,
                            data_name=opts.src
                            )
    val_set_tar = TSNDataSet("", val_list_tar, num_dataload=num_val_tar, num_segments=opts.num_frames,
                            new_length=1, modality='RGB',sample_every_n_frames = opts.sample_every_n_frames,
                            random_shift=False,
                            test_mode=True,
                            opts=opts,
                            data_name=opts.tar
                            )

    val_loader_src = DataLoader(val_set_src, batch_size=opts.v_batch_size, shuffle=False,
                                num_workers=opts.num_workers, pin_memory=True)
    val_loader_tar = DataLoader(val_set_tar, batch_size=opts.v_batch_size, shuffle=False,
                                num_workers=opts.num_workers, pin_memory=True)

    train_set_src = TSNDataSet("", train_list_src, num_dataload=num_train_src+train_aug_num_src, num_segments=opts.num_frames,
                            new_length=1, modality='RGB',sample_every_n_frames = opts.sample_every_n_frames,
                            random_shift=True,
                            test_mode=False,
                            opts=opts,
                            data_name=opts.src
                            )
    train_sampler_src = torch.utils.data.distributed.DistributedSampler(train_set_src)
    train_loader_src = DataLoader(train_set_src, batch_size=opts.batch_size, shuffle=False, sampler=train_sampler_src,
                                                num_workers=opts.batch_size, pin_memory=True)
    train_set_tar = TSNDataSet("", train_list_tar, num_dataload=num_train_tar+train_aug_num_tar, num_segments=opts.num_frames,
                            new_length=1, modality='RGB',sample_every_n_frames = opts.sample_every_n_frames,
                            random_shift=True,
                            test_mode=False,
                            opts=opts,
                            data_name=opts.tar
                            )
    train_sampler_tar = torch.utils.data.distributed.DistributedSampler(train_set_tar)
    train_loader_tar = DataLoader(train_set_tar, batch_size=opts.batch_size, shuffle=False, sampler=train_sampler_tar,
                                                num_workers=opts.batch_size, pin_memory=True)
    if dist.get_rank() == 0:
        print(f'Train dataset length src: {len(train_set_src)}')
        print(f'Train dataset steps per epoch src: {len(train_set_src)/opts.batch_size}')
        print(f'Train dataset length tar: {len(train_set_tar)}')
        print(f'Train dataset steps per epoch tar: {len(train_set_tar)/opts.batch_size}')
        print(f'Validation dataset length src: {len(val_set_src)}')
        print(f'Validation dataset steps per epoch src: {len(val_set_src)/opts.v_batch_size}')
        print(f'Validation dataset length tar: {len(val_set_tar)}')
        print(f'Validation dataset steps per epoch tar: {len(val_set_tar)/opts.v_batch_size}')
    #================================end of dataset preparation=================================

    #=================================weight class action============================================
    class_id_list = [int(line.strip().split(' ')[2]) for line in open(train_list_src)]
    _, class_data_counts = np.unique(np.array(class_id_list), return_counts=True)
    class_freq = (class_data_counts / class_data_counts.sum()).tolist()

    class_action_weight = torch.ones(opts.num_classes_action)
    if opts.weight_class_action:
        class_action_weight = 1 / torch.Tensor(class_freq)
    #=================================privacy loss weight============================================
    if opts.src == 'ucf':
        with open('/home/zhiwei/source/dataset/ucf101_privacy_label.pickle', 'rb') as f:
            privacy_data = pickle.load(f)
    elif opts.src == 'hmdb':
        with open('/home/zhiwei/source/dataset/hmdb51_privacy_label.pickle', 'rb') as f:
            privacy_data = pickle.load(f)
    for line in open(train_list_src):
        path = line.strip().split(' ')[0]
        privacy_label = privacy_data[path.split('/')[-1].split('.')[0]]

    attribute_counts = np.zeros(5)

    total_samples = 0
    for line in open(train_list_src):
        path = line.strip().split(' ')[0]
        privacy_label = privacy_data[path.split('/')[-1].split('.')[0]][0]
        attribute_counts += np.array(privacy_label)
        total_samples += 1

    attribute_frequencies = attribute_counts / total_samples
    weights = 1.0 / attribute_frequencies
    weights = weights / np.sum(weights)

    class_privacy_weight = torch.ones(opts.num_classes_privacy).cuda()

    if opts.weight_class_privacy:
        class_privacy_weight = torch.Tensor(weights).cuda()
    #=================================criterion============================================
    criterion_ft = nn.CrossEntropyLoss(weight=class_action_weight)
    criterion_fb = nn.BCEWithLogitsLoss(weight=class_privacy_weight)

    criterion_action = criterion_ft.to(local_rank)
    criterion_privacy = criterion_fb.to(local_rank)
    #=================================models construction============================================
    model = load_fa(opts)
    action_model = load_ft(opts)
    privacy_model = load_fb(opts)
    
    model = model.to(local_rank)
    action_model = action_model.to(local_rank)
    privacy_model = privacy_model.to(local_rank)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank,broadcast_buffers=False, find_unused_parameters=True)
    action_model = DDP(action_model, device_ids=[local_rank], output_device=local_rank,broadcast_buffers=False)
    privacy_model = DDP(privacy_model, device_ids=[local_rank], output_device=local_rank,broadcast_buffers=False)
    #=================================optimizer============================================
    optimizer = torch.optim.SGD(model.parameters(), lr=opts.learning_rate)
    optimizer_action = torch.optim.SGD(action_model.parameters(), lr=opts.learning_rate_action)
    optimizer_privacy = torch.optim.SGD(privacy_model.parameters(), lr=opts.learning_rate_privacy)
    #=================================training============================================
    for epoch in range(0, opts.num_epochs):
        if local_rank == 0:
            print(f'Epoch {epoch} started')
            start=time.time()

        train_loader_src.sampler.set_epoch(epoch)

        if dist.get_rank() == 0:
            model, action_model, privacy_model = train_epoch_minmax(model, action_model, privacy_model, criterion_action, criterion_privacy, train_loader_src, optimizer, optimizer_action, optimizer_privacy, epoch, opts, writer = writer, local_rank = local_rank)
        else:
            model, action_model, privacy_model = train_epoch_minmax(model, action_model, privacy_model, criterion_action, criterion_privacy, train_loader_src, optimizer, optimizer_action, optimizer_privacy, epoch, opts, local_rank = local_rank)
        
        if epoch % opts.val_freq == 0:
            #=================================validation============================================
            if dist.get_rank() == 0:
                val_visualization(save_dir_vis_src, epoch, val_loader_src, model)
                val_visualization(save_dir_vis_tar, epoch, val_loader_tar, model)

                acc_src = val_epoch_action(model, action_model, val_loader_src, epoch, writer = writer, local_rank = local_rank)
                acc_tar = val_epoch_action(model, action_model, val_loader_tar, epoch, writer = writer, local_rank = local_rank)

                cMAP_src, f1_src = val_epoch_privacy(privacy_model, val_loader_src, criterion_privacy, epoch, writer, 'src', opts.src, opts)
                cMAP_tar, f1_tar = val_epoch_privacy(privacy_model, val_loader_tar, criterion_privacy, epoch, writer, 'tar', opts.tar, opts)

                save_file_path = os.path.join(save_dir, f'model_{epoch}_{opts.src}_{acc_src:.4f}_{opts.tar}_{acc_tar:.4f}.pth')
                states = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'acc_src': acc_src,
                    'acc_tar': acc_tar,
                    'cMAP_src': cMAP_src,
                    'f1_src': f1_src,
                    'cMAP_tar': cMAP_tar,
                    'f1_tar': f1_tar,
                }
                torch.save(states, save_file_path)

        dist.barrier()
        
        if local_rank == 0:
            taken = time.time()-start
            print(f'Time taken for Epoch-{epoch} is {taken}')

    del action_model, privacy_model

    #=================================privacy eval============================================
    #=================================privacy loss weight=====================================
    if opts.tar == 'ucf':
        with open('/home/zhiwei/source/dataset/ucf101_privacy_label.pickle', 'rb') as f:
            privacy_data = pickle.load(f)
    elif opts.tar == 'hmdb':
        with open('/home/zhiwei/source/dataset/hmdb51_privacy_label.pickle', 'rb') as f:
            privacy_data = pickle.load(f)
    for line in open(train_list_tar):
        path = line.strip().split(' ')[0]
        privacy_label = privacy_data[path.split('/')[-1].split('.')[0]]

    attribute_counts = np.zeros(5)

    total_samples = 0
    for line in open(train_list_tar):
        path = line.strip().split(' ')[0]
        privacy_label = privacy_data[path.split('/')[-1].split('.')[0]][0]
        attribute_counts += np.array(privacy_label)
        total_samples += 1

    attribute_frequencies = attribute_counts / total_samples
    weights = 1.0 / attribute_frequencies
    weights = weights / np.sum(weights)

    class_privacy_weight = torch.ones(opts.num_classes_privacy).cuda()

    if opts.weight_class_privacy:
        class_privacy_weight = torch.Tensor(weights).cuda()

    privacy_model_eval = load_fb(opts,eval = True)


    privacy_model_eval = privacy_model_eval.to(local_rank)
    privacy_model = DDP(privacy_model, device_ids=[local_rank], output_device=local_rank,broadcast_buffers=False)

    criterion_privacy_eval = nn.BCEWithLogitsLoss(weight=class_privacy_weight).cuda()
    optimizer_privacy_eval = torch.optim.SGD(privacy_model_eval.parameters(), lr=opts.learning_rate_privacy)

    best_cMAP_src = .0
    for epoch in range(0, opts.num_epoch_eval):
        print(f'Epoch {epoch} started')
        start=time.time()
        
        privacy_model_eval = train_epoch_privacy(privacy_model_eval, train_loader_tar, criterion_privacy_eval, optimizer_privacy_eval, epoch, writer, opts)

        if epoch % opts.val_freq == 0:

            cMAP_src, f1_src = val_epoch_privacy(privacy_model_eval, val_loader_src, criterion_privacy_eval, epoch, writer, 'src', opts.src, opts)
            cMAP_tar, f1_tar = val_epoch_privacy(privacy_model_eval, val_loader_tar, criterion_privacy_eval, epoch, writer, 'tar', opts.tar, opts)

            
            if cMAP_src > best_cMAP_src:
                best_cMAP_src = cMAP_src
                save_file_path = os.path.join(save_dir, 'model_{}2{}_{}_cMAP_{}_f1_{}.pth'.format(opts.src, opts.src, epoch, cMAP_src, f1_src))
                states = {
                    'epoch': epoch,
                    'top1_acc': cMAP_src,
                    'privacy_model_state_dict': privacy_model_eval.state_dict(),
                }
                torch.save(states, save_file_path)
            
            if cMAP_tar > best_cMAP_tar:
                best_cMAP_tar = cMAP_tar
                save_file_path = os.path.join(save_dir, 'model_{}2{}_{}_cMAP_{}_f1_{}.pth'.format(opts.src, opts.tar, epoch, cMAP_tar, f1_tar))
                states = {
                    'epoch': epoch,
                    'top1_acc': cMAP_tar,
                    'privacy_model_state_dict': privacy_model_eval.state_dict(),
                }
                torch.save(states, save_file_path)

        
        taken = time.time()-start
        print(f'Time taken for Epoch-{epoch} is {taken}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #=================RUN PARAMETERS=================================
    parser.add_argument("--run_id", dest='run_id', type=str, required=False, default= "dummy_recon", help='run_id')
    parser.add_argument("--src", dest='src', type=str, required=False, default= "ucf", help='Source dataset')
    parser.add_argument("--tar", dest='tar', type=str, required=False, default= "hmdb", help='Target dataset')
    #=================MODEL PARAMETERS===============================
    parser.add_argument("--kin_pretrained", dest='kin_pretrained', type=int, required=False, default=1, help='Kin pretrained')
    parser.add_argument("--privacy_pretrained", dest='privacy_pretrained', type=int, required=False, default=1, help='privacy pretrained')
    parser.add_argument("--self_pretrained_action", dest='self_pretrained_action', type=str, required=False, default=None, help='Self pretrained')
    parser.add_argument("--self_pretrained_privacy", dest='self_pretrained_privacy', type=str, required=False, default=None, help='Self pretrained')
    parser.add_argument("--self_pretrained_fa", dest='self_pretrained_fa', type=int, required=False, default=None, help='self_pretrained_fa')
    parser.add_argument("--self_pretrained_ssl", dest='self_pretrained_ssl', type=str, required=False, default=None, help='self_pretrained_ssl')
    parser.add_argument("--train_backbone", dest='train_backbone', type=int, required=True, default=1, help='train_backbone')
    parser.add_argument("--architecture", dest='architecture', type=str, required=True, default='resnet50', help='architecture')
    #=================TRAINING PARAMETERS==========================
    parser.add_argument("--num_frames", dest='num_frames', type=int, required=False, default=16, help='Number of frames')
    parser.add_argument("--sample_every_n_frames", dest='sample_every_n_frames', type=int, required=False, default=2, help='Sample every n frames')
    parser.add_argument("--num_workers", dest='num_workers', type=int, required=False, default=10, help='Number of workers')
    parser.add_argument("--batch_size", dest='batch_size', type=int, required=False, default=32, help='Batch size')
    parser.add_argument("--v_batch_size", dest='v_batch_size', type=int, required=False, default=8, help='Validation batch size')
    parser.add_argument("--learning_rate_fb", dest='learning_rate_fb', type=float, required=False, default=1e-3, help='Learning rate for privacy model')
    parser.add_argument("--learning_rate_fa", dest='learning_rate_fa', type=float, required=False, default=1e-3, help='Learning rate for action model')
    parser.add_argument("--learning_rate_ft", dest='learning_rate_ft', type=float, required=False, default=1e-3, help='Learning rate for anonymization model')
    parser.add_argument("--learning_rate_privacy", dest='learning_rate_privacy', type=float, required=False, default=1e-3, help='Learning rate for privacy model')
    parser.add_argument("--num_epochs", dest='num_epochs', type=int, required=False, default=300, help='Number of epochs')
    parser.add_argument("--num_epoch_eval", dest='num_epoch_eval', type=int, required=False, default=100, help='Epoch eval')
    parser.add_argument("--val_freq", dest='val_freq', type=int, required=False, default=10, help='Validation frequency')
    #=================DATA PARAMETERS==============================
    parser.add_argument("--reso_h", dest='reso_h', type=int, required=False, default=128, help='Resolution height')
    parser.add_argument("--reso_w", dest='reso_w', type=int, required=False, default=128, help='Resolution width')
    parser.add_argument("--triple", dest='triple', type=int, required=False, default=0, help='Use triple sampling')
    #=================MINMAX PARAMETERS==========================
    parser.add_argument('--privacy_weight', type=float, default=1.0)
    parser.add_argument('--action_weight', type=float, default=1.0)
    #=================ACTION PARAMETERS==========================
    parser.add_argument('--num_classes_action', type=int, default=12)
    parser.add_argument('--num_classes_privacy', type=int, default=5)
    parser.add_argument('--weight_class_action', type=int, default=0)
    parser.add_argument('--weight_class_privacy', type=int, default=0)
    #=================DDP PARAMETERS==========================
    parser.add_argument("--local_rank", default=-1, type=int)
    
    seed_everything(1234)

    opts = parser.parse_args()
    opts.run_id = f'VITA_minmax_{opts.src}2{opts.tar}_frames_{opts.num_frames}_rate_{opts.sample_every_n_frames}_bs_{opts.batch_size}_lr_ft_{opts.learning_rate_ft}_lr_fb_{opts.learning_rate_fb}_lr_fa_{opts.learning_rate_fa}_privacy_weight_{opts.privacy_weight}_action_weight_{opts.action_weight}'

##########################和DDP有关的参数################################
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
######################################################################
    print(f'local_rank: {local_rank}')
    train_minmax(opts, local_rank)