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
from models.model_utils import build_model_privacy
from torchvision.utils import save_image
from models.shuffle_discriminator import ShuffleDiscriminator
from models.domain_discriminator import DomainDiscriminator
from itertools import cycle
import torchvision
from PIL import Image

warnings.filterwarnings("ignore")

def seed_everything(seed=1234):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 或者 ':16:8'

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def fix_model(model):
    for name, param in model.module.named_parameters():
        param.requires_grad = False

def unfix_model(model):
    for name, param in model.module.named_parameters():
        if 'perceptual_model' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

def load_ft(opts, local_rank):
    num_classes_action = opts.num_classes_action
    model = r2plus1d_18(pretrained = opts.kin_pretrained, progress = False)
    model.fc = nn.Linear(512, num_classes_action)

    if opts.self_pretrained_action:
        pretrained = torch.load(opts.self_pretrained_action, map_location=f'cuda:{local_rank}')
        pretrained_kvpair = pretrained['action_model_state_dict']

        model_kvpair = model.state_dict()
        for layer_name, weights in pretrained_kvpair.items():
            layer_name = layer_name.replace('module.','')
            model_kvpair[layer_name] = weights   
        model.load_state_dict(model_kvpair, strict=True)
        print(f'action model loaded successsfully!')

    return model

def load_fa(opts, local_rank):
    model = VQGAN(opts)
    if opts.self_pretrained_vqgan:
        pretrained = torch.load(opts.self_pretrained_vqgan, map_location=f'cuda:{local_rank}')
        pretrained_kvpair = pretrained['model_state_dict']
        model_kvpair = model.state_dict()
        for layer_name, weights in pretrained_kvpair.items():
            layer_name = layer_name.replace('module.','')
            model_kvpair[layer_name] = weights   
        model.load_state_dict(model_kvpair, strict=True)
        print(f'vqgan loaded successsfully!')
    return model

def load_fb(opts, local_rank):
    
    model = build_model_privacy(
        architecture = opts.architecture,
        pretrained = opts.privacy_pretrained,
        num_classes = opts.num_classes_privacy,
        train_backbone = opts.train_backbone,
    )

    if opts.self_pretrained_privacy:
        weights = torch.load(opts.self_pretrained_privacy, map_location=f'cuda:{local_rank}')
        model.load_state_dict(weights["privacy_model_state_dict"], strict=True)
        print("privacy model {} loaded successfully".format(opts.architecture))

    return model

def train_epoch_minmax(model, action_model, privacy_model, triplet_model, domain_classifier, criterion_action, criterion_privacy, criterion_triplet, criterion_domain, train_loader_src, train_loader_tar, optimizer_ae, optimizer_disc, optimizer_action, optimizer_privacy, optimizer_triplet, optimizer_domain, epoch, opts, writer=None, local_rank=None):
    
    losses_action = []
    losses_privacy = []
    losses_triplet = []
    losses_recon = []
    losses_anonymized = []
    losses_domain = []
    disc_factor = adopt_weight(epoch, threshold=opts.discriminator_iter_start)
    triplet_factor = adopt_weight(epoch, threshold=opts.triplet_iter_start)
    domain_factor = adopt_weight(epoch, threshold=opts.domain_iter_start)

    scaler = GradScaler()
    total_iters = len(train_loader_src)
    data_target_iter = cycle(train_loader_tar)
    loop = tqdm(train_loader_src, total=total_iters) if dist.get_rank() == 0 else train_loader_src

    model.train()
    unfix_model(model)

    step = 0
    for data in loop:

        inputs_ancher = data[0].to(local_rank)
        inputs_pos = data[3].to(local_rank)
        inputs_neg = data[4].to(local_rank)
        action_labels = data[1].to(local_rank)
        privacy_labels = data[2].to(local_rank)
        inputs_ancher = inputs_ancher.permute(0,2,1,3,4)
        inputs_pos = inputs_pos.permute(0,2,1,3,4)
        inputs_neg = inputs_neg.permute(0,2,1,3,4)

        data_tar = next(data_target_iter)
        inputs_anchor_tar = data_tar[0].to(local_rank)
        inputs_pos_tar = data_tar[3].to(local_rank)
        inputs_neg_tar = data_tar[4].to(local_rank)
        inputs_anchor_tar = inputs_anchor_tar.permute(0,2,1,3,4)
        inputs_pos_tar = inputs_pos_tar.permute(0,2,1,3,4)
        inputs_neg_tar = inputs_neg_tar.permute(0,2,1,3,4)

        #==========================step 1=============================================
        if step % 2 == 0:
            #==========================recon loss src=================================
            with autocast():
                recon_loss, commitment_loss_static, commitment_loss_dynamic, aeloss, perceptual_loss, gan_feat_loss = model(inputs_ancher, 0)
                loss = recon_loss + commitment_loss_static + commitment_loss_dynamic + disc_factor * (aeloss + gan_feat_loss) + perceptual_loss
            
            losses_recon.append(recon_loss.item())
            optimizer_ae.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer_ae)
            scaler.update()
                
            if dist.get_rank() == 0:
                writer.add_scalar("train/recon_loss", recon_loss, epoch*len(train_loader_src)+step)
                writer.add_scalar("train/commitment_loss_static", commitment_loss_static, epoch*len(train_loader_src)+step)
                writer.add_scalar("train/commitment_loss_dynamic", commitment_loss_dynamic, epoch*len(train_loader_src)+step)
                writer.add_scalar("train/aeloss", aeloss, epoch*len(train_loader_src)+step)
                writer.add_scalar("train/perceptual_loss", perceptual_loss, epoch*len(train_loader_src)+step)
                writer.add_scalar("train/gan_feat_loss", gan_feat_loss, epoch*len(train_loader_src)+step)
                loop.set_description(f'Epoch [{epoch}/{opts.num_epochs}]')
                loop.set_postfix({'loss_recon': recon_loss.item()})
            
            #==========================discriminator loss src===========================
            if disc_factor > 0:

                with autocast():
                    discloss = model(inputs_ancher, 1)
                    loss = disc_factor * discloss

                optimizer_disc.zero_grad()

                scaler.scale(loss).backward()
                scaler.step(optimizer_disc)
                scaler.update()

                if dist.get_rank() == 0:
                    writer.add_scalar("train/discloss", discloss, epoch*len(train_loader_src)+step)
                    loop.set_description(f'Epoch [{epoch}/{opts.num_epochs}]')
                    loop.set_postfix({'loss_disc': loss.item()})

            #==========================triplet loss src&tar disentangle static=================================
            if triplet_factor > 0:
                triplet_model.train()
                unfix_model(triplet_model)
                domain_classifier.train()
                unfix_model(domain_classifier)
                model.train()
                model.module.codebook_dynamic.eval()
                model.module.codebook_static.train()
                with autocast():
                    #==========================src data triplet=============================
                    ancher_static, _ = model(inputs_ancher, 2)
                    pos_static, _ = model(inputs_pos, 2)
                    neg_static, _ = model(inputs_neg, 2)

                    ancher_feature_static = triplet_model(ancher_static)
                    pos_feature_static = triplet_model(pos_static)
                    neg_feature_static = triplet_model(neg_static)
                    #==========================tar data triplet=============================
                    ancher_static_tar, _ = model(inputs_anchor_tar, 2)
                    pos_static_tar, _ = model(inputs_pos_tar, 2)
                    neg_static_tar, _ = model(inputs_neg_tar, 2)

                    ancher_feature_static_tar = triplet_model(ancher_static_tar)
                    pos_feature_static_tar = triplet_model(pos_static_tar)
                    neg_feature_static_tar = triplet_model(neg_static_tar)

                    triplet_loss_src = criterion_triplet(ancher_feature_static, pos_feature_static, neg_feature_static)
                    triplet_loss_tar = criterion_triplet(ancher_feature_static_tar, pos_feature_static_tar, neg_feature_static_tar)
                    triplet_loss = triplet_loss_src + triplet_loss_tar
                
                losses_triplet.append(triplet_loss.item())
                optimizer_triplet.zero_grad()
                optimizer_ae.zero_grad()
                scaler.scale(triplet_loss).backward()
                scaler.step(optimizer_triplet)
                scaler.step(optimizer_ae)
                scaler.update()

                if dist.get_rank() == 0:
                    writer.add_scalar("train/triplet_loss", triplet_loss, epoch*len(train_loader_src)+step)
                    loop.set_description(f'Epoch [{epoch}/{opts.num_epochs}]')
                    loop.set_postfix({'loss_triplet': triplet_loss.item()})
            #==========================domain loss src&tar align dynamic=======================================
            if domain_factor > 0:
                triplet_model.train()
                unfix_model(triplet_model)
                domain_classifier.train()
                unfix_model(domain_classifier)
                model.train()
                model.module.codebook_dynamic.train()
                model.module.codebook_static.eval()
                with autocast():
                    #==========================src data domain=============================
                    ancher_static, ancher_dynamic = model(inputs_ancher, 2)
                    pred_domain_src = domain_classifier(ancher_dynamic, domain_label_src)
                    #==========================tar data domain=============================
                    ancher_static_tar, ancher_dynamic_tar = model(inputs_anchor_tar, 2)
                    pred_domain_tar = domain_classifier(ancher_dynamic_tar, domain_label_tar)

                    domain_label_src = torch.zeros(inputs_ancher.shape[0]).long().to(local_rank)
                    domain_label_tar = torch.ones(inputs_anchor_tar.shape[0]).long().to(local_rank)

                    domain_loss_src = criterion_domain(pred_domain_src, domain_label_src)
                    domain_loss_tar = criterion_domain(pred_domain_tar, domain_label_tar)
                    domain_loss = domain_loss_src + domain_loss_tar
                
                losses_domain.append(domain_loss.item())
                optimizer_triplet.zero_grad()
                optimizer_ae.zero_grad()
                optimizer_domain.zero_grad()
                scaler.scale(domain_loss).backward()
                scaler.step(optimizer_ae)
                scaler.step(optimizer_domain)
                scaler.update()

                if dist.get_rank() == 0:
                    writer.add_scalar("train/domain_loss", domain_loss, epoch*len(train_loader_src)+step)
                    loop.set_description(f'Epoch [{epoch}/{opts.num_epochs}]')
                    loop.set_postfix({'loss_domain': domain_loss.item()})
            #==========================anonymization loss=================================     
            privacy_model.eval()
            fix_model(privacy_model)
            model.train()
            model.module.codebook_dynamic.eval()
            model.module.codebook_static.train()

            with autocast():
                _, recon_anchor, _, _, _ = model(inputs_ancher)
                output_action = action_model(recon_anchor)
                output_privacy = privacy_model(recon_anchor)
                loss_action = criterion_action(output_action, action_labels)
                loss_privacy = criterion_privacy(output_privacy, privacy_labels)
                loss_anonymized = opts.action_weight * loss_action - opts.privacy_weight * loss_privacy
                losses_action.append(loss_action.item())
                losses_privacy.append(loss_privacy.item())
                losses_anonymized.append(loss_anonymized.item())

                optimizer_ae.zero_grad()
                optimizer_action.zero_grad()
                optimizer_privacy.zero_grad()
                scaler.scale(loss_anonymized).backward()
                scaler.step(optimizer_ae)
                scaler.update()

            if dist.get_rank() == 0:
                writer.add_scalar(f'Training loss_fa', loss_anonymized.item(), epoch * total_iters + step)
                writer.add_scalar(f'Training loss_ft', loss_action.item(), epoch * total_iters + step)
                writer.add_scalar(f'Training loss_fb', loss_privacy.item(), epoch * total_iters + step)
                loop.set_description(f'Epoch [{epoch}/{opts.num_epochs}]')
                loop.set_postfix({
                    'loss_fa': loss_anonymized.item(),
                    'loss_ft': loss_action.item(),
                    'loss_fb': loss_privacy.item()
                })
            step += 1
            continue
        
        #==========================step 2=============================================
        if step % 2 == 1:

            model.eval()
            privacy_model.train()
            fix_model(model)
            unfix_model(privacy_model)


            with autocast():
                _, recon_anchor, _, _, _ = model(inputs_ancher)
                output_action = action_model(recon_anchor)
                output_privacy = privacy_model(recon_anchor)
                loss_action = criterion_action(output_action, action_labels)
                loss_privacy = criterion_privacy(output_privacy, privacy_labels)
                loss_anonymized = opts.action_weight * loss_action - opts.privacy_weight * loss_privacy
                loss = loss_action + loss_privacy
                losses_action.append(loss_action.item())
                losses_privacy.append(loss_privacy.item())
                losses_anonymized.append(loss_anonymized.item())

                optimizer_ae.zero_grad()
                optimizer_action.zero_grad()
                optimizer_privacy.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer_action)
                scaler.step(optimizer_privacy)
                scaler.update()

            if dist.get_rank() == 0:
                writer.add_scalar(f'Training loss_fa', loss_anonymized.item(), epoch * total_iters + step)
                writer.add_scalar(f'Training loss_ft', loss_action.item(), epoch * total_iters + step)
                writer.add_scalar(f'Training loss_fb', loss_privacy.item(), epoch * total_iters + step)
                loop.set_description(f'Epoch [{epoch}/{opts.num_epochs}]')
                loop.set_postfix({
                    'loss_fa': loss_anonymized.item(),
                    'loss_ft': loss_action.item(),
                    'loss_fb': loss_privacy.item()
                })
            step += 1
            continue

    #==========================end training epoch=========================================
    if dist.get_rank() == 0:
        epoch_avg_loss_action = np.mean(losses_action)
        epoch_avg_loss_privacy = np.mean(losses_privacy)
        epoch_avg_loss_anonymized = np.mean(losses_anonymized)
        print(f'Training Epoch: {epoch}, Average Loss: anonymized {epoch_avg_loss_anonymized:.4f}, action {epoch_avg_loss_action:.4f}, privacy {epoch_avg_loss_privacy:.4f}')

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
                _, recon_x, _, _, _ = model(inputs.permute(0,2,1,3,4))
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

def val_epoch_privacy(model, privacy_model, val_loader, criterion, epoch, writer, val_dataset, opts):
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

            inputs = inputs.cuda()
            privacy_label = privacy_label.cuda(non_blocking=True)
            privacy_label = privacy_label.unsqueeze(1).expand(-1, inputs.size(1), -1)
            privacy_label = privacy_label.reshape(-1, privacy_label.size(2))

            total_length = total_length+inputs.shape[0]
            
            with autocast():
                _, recon_x, _, _, _ = model(inputs.permute(0,2,1,3,4))
                recon_x = recon_x.permute(0,2,1,3,4)
                B, T, C, H, W = recon_x.shape
                recon_x = recon_x.reshape(-1, C, H, W)
                outputs = privacy_model(recon_x)
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

def val_visualization(save_dir, epoch, validation_dataloader, model, local_rank):
    """Visualize and save reconstruction results
    Args:
        save_dir (str): Directory to save visualization results
        epoch (int): Current epoch number
        validation_dataloader: Validation data loader
        model: VQGAN model
        local_rank: Device rank for distributed training
    """
    model.eval()
    for i, data in enumerate(validation_dataloader):
        if len(data[0].shape) == 1:
            continue
            
        inputs = data[0].to(local_rank)
        inputs = inputs.permute(0,2,1,3,4)  # [B,T,C,H,W] -> [B,C,T,H,W]
        
        with torch.no_grad(), autocast():
            _, x_recon, _, _, _ = model(inputs)
            
            inputs = inputs.permute(0,2,1,3,4)  # [B,C,T,H,W] -> [B,T,C,H,W]
            outputs = x_recon.permute(0,2,1,3,4)
            
            input_frames = inputs[:,0,:,:,:].squeeze(1)   # [B,C,H,W]
            output_frames = outputs[:,0,:,:,:].squeeze(1)  # [B,C,H,W]
            
            vis_frames = torch.cat([input_frames, output_frames], dim=0)  # [2B,C,H,W]
            vis_frames = torch.clamp(vis_frames, -0.5, 0.5)  # Clamp values
            
            grid = torchvision.utils.make_grid(vis_frames, nrow=inputs.shape[0], padding=2)
            
            # Convert from [-0.5,0.5] to [0,255]
            grid = grid + 0.5  # [-0.5,0.5] -> [0,1]
            grid = grid.transpose(0,1).transpose(1,2)  # [C,H,W] -> [H,W,C]
            grid = grid.cpu().numpy()
            grid = (grid * 255).astype(np.uint8)
            
            filename = f"recon_e{epoch:04d}_b{i:04d}.png"
            path = os.path.join(save_dir, filename)
            Image.fromarray(grid).save(path)
            
            break  # Only save first batch

def train_minmax(opts, local_rank):

    run_id = opts.run_id
    if dist.get_rank() == 0:
        print(f'run id============ {run_id} =============')
        save_dir = os.path.join(cfg.save_dir_vqgan_minmax, str(run_id))
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
    criterion_triplet = nn.TripletMarginLoss(margin=1.0, p=2)
    criterion_domain = nn.CrossEntropyLoss()

    criterion_action = criterion_ft.to(local_rank)
    criterion_privacy = criterion_fb.to(local_rank)
    criterion_triplet = criterion_triplet.to(local_rank)
    criterion_domain = criterion_domain.to(local_rank)
    #=================================models construction============================================
    model = load_fa(opts, local_rank)
    action_model = load_ft(opts, local_rank)
    privacy_model = load_fb(opts, local_rank)
    triplet_model = ShuffleDiscriminator(opts.embedding_dim_static, opts.embedding_dim_static)
    domain_classifier = DomainDiscriminator(opts.embedding_dim_static, 2)
    
    model = model.to(local_rank)
    action_model = action_model.to(local_rank)
    privacy_model = privacy_model.to(local_rank)
    triplet_model = triplet_model.to(local_rank)
    domain_classifier = domain_classifier.to(local_rank)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank,broadcast_buffers=False, find_unused_parameters=True)
    action_model = DDP(action_model, device_ids=[local_rank], output_device=local_rank,broadcast_buffers=False)
    privacy_model = DDP(privacy_model, device_ids=[local_rank], output_device=local_rank,broadcast_buffers=False)
    triplet_model = DDP(triplet_model, device_ids=[local_rank], output_device=local_rank,broadcast_buffers=False)
    domain_classifier = DDP(domain_classifier, device_ids=[local_rank], output_device=local_rank,broadcast_buffers=False)
    # fix action model all the time
    action_model.eval()
    fix_model(action_model)

    #=================================optimizer============================================
    optimizer_ae = torch.optim.Adam(list(model.module.encoder.parameters()) +
                            list(model.module.decoder.parameters()) +
                            list(model.module.pre_vq_conv_dynamic.parameters()) +
                            list(model.module.pre_vq_conv_static.parameters()) +
                            list(model.module.post_vq_conv.parameters()) +
                            list(model.module.codebook_dynamic.parameters()) +
                            list(model.module.codebook_static.parameters()),
                            lr=opts.learning_rate_fa, betas=(0.5, 0.9))
    
    optimizer_disc = torch.optim.Adam(list(model.module.image_discriminator.parameters()) +
                                list(model.module.video_discriminator.parameters()),
                                lr=opts.learning_rate_disc, betas=(0.5, 0.9))
    
    optimizer_action = torch.optim.Adam(action_model.parameters(), lr=opts.learning_rate_ft, betas=(0.5, 0.9))
    optimizer_privacy = torch.optim.Adam(privacy_model.parameters(), lr=opts.learning_rate_fb, betas=(0.5, 0.9))
    optimizer_triplet = torch.optim.Adam(triplet_model.parameters(), lr=opts.learning_rate_triplet, betas=(0.5, 0.9))
    optimizer_domain = torch.optim.Adam(domain_classifier.parameters(), lr=opts.learning_rate_domain, betas=(0.5, 0.9))
    #=================================training============================================
    for epoch in range(0, opts.num_epochs):
        if local_rank == 0:
            print(f'Epoch {epoch} started')
            start=time.time()

        train_loader_src.sampler.set_epoch(epoch)

        if dist.get_rank() == 0:
            train_epoch_minmax(model, action_model, privacy_model, triplet_model, domain_classifier, criterion_action, criterion_privacy, criterion_triplet, criterion_domain, train_loader_src, train_loader_tar, optimizer_ae, optimizer_disc, optimizer_action, optimizer_privacy, optimizer_triplet, optimizer_domain, epoch, opts, writer = writer, local_rank = local_rank)
        else:
            train_epoch_minmax(model, action_model, privacy_model, triplet_model, domain_classifier, criterion_action, criterion_privacy, criterion_triplet, criterion_domain, train_loader_src, train_loader_tar, optimizer_ae, optimizer_disc, optimizer_action, optimizer_privacy, optimizer_triplet, optimizer_domain, epoch, opts, local_rank = local_rank)
        
        if epoch % opts.val_freq == 0:
            #=================================validation============================================
            if dist.get_rank() == 0:
                val_visualization(save_dir_vis_src, epoch, val_loader_src, model, local_rank)
                val_visualization(save_dir_vis_tar, epoch, val_loader_tar, model, local_rank)

                acc_src = val_epoch_action(model, action_model, val_loader_src, epoch, writer = writer, local_rank = local_rank)
                acc_tar = val_epoch_action(model, action_model, val_loader_tar, epoch, writer = writer, local_rank = local_rank)

                cMAP_src, f1_src = val_epoch_privacy(model, privacy_model, val_loader_src, criterion_privacy, epoch, writer, opts.src, opts)
                cMAP_tar, f1_tar = val_epoch_privacy(model, privacy_model, val_loader_tar, criterion_privacy, epoch, writer, opts.tar, opts)

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
    parser.add_argument("--self_pretrained_vqgan", dest='self_pretrained_vqgan', type=int, required=False, default=None, help='self_pretrained_vqgan')
    parser.add_argument("--train_backbone", dest='train_backbone', action='store_true')
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
    parser.add_argument("--learning_rate_disc", dest='learning_rate_disc', type=float, required=False, default=1e-3, help='Learning rate for discriminator')
    parser.add_argument("--learning_rate_domain", dest='learning_rate_domain', type=float, required=False, default=1e-3, help='Learning rate for domain classifier')
    parser.add_argument("--learning_rate_triplet", dest='learning_rate_triplet', type=float, required=False, default=1e-3, help='Learning rate for triplet loss')
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
    #=================VQGAN PARAMETERS===========================
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
    parser.add_argument('--triplet_iter_start', type=int, default=20) # 用于控制什么时候开始使用triplet loss来分离源域和目标域的动态和静态信息
    parser.add_argument('--domain_iter_start', type=int, default=20) # 用于控制什么时候开始使用domain classifier来align源域和目标域的动态信息
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
    #=================DDP PARAMETERS==========================
    parser.add_argument("--local_rank", default=-1, type=int)
    
    seed_everything(1234)

    opts = parser.parse_args()
    opts.run_id = f'VQGAN_minmax_{opts.src}2{opts.tar}_frames_{opts.num_frames}_rate_{opts.sample_every_n_frames}_bs_{opts.batch_size}_lr_ft_{opts.learning_rate_ft}_lr_fb_{opts.learning_rate_fb}_lr_fa_{opts.learning_rate_fa}_privacy_weight_{opts.privacy_weight}_action_weight_{opts.action_weight}'

##########################和DDP有关的参数################################
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
######################################################################
    print(f'local_rank: {local_rank}')
    train_minmax(opts, local_rank)