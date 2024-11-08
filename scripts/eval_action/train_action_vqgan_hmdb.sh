#!/bin/bash
cd ..
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port 29502 vqgan_action_ddp_amp.py \
    --src 'hmdb' \
    --tar 'ucf' \
    --num_frames 16 \
    --sample_every_n_frames 2 \
    --num_workers 8 \
    --batch_size 8 \
    --v_batch_size 8 \
    --learning_rate 0.0002 \
    --learning_rate_action 0.002 \
    --num_epochs 500 \
    --val_freq 5 \
    --reso_h 128 \
    --reso_w 128 \
    --embedding_dim 256 \
    --n_codes 16384 \
    --n_hiddens 32 \
    --downsample 4 8 8 \
    --disc_channels 64 \
    --disc_layers 3 \
    --discriminator_iter_start 50 \
    --disc_loss_type hinge \
    --image_gan_weight 1.0 \
    --video_gan_weight 1.0 \
    --l1_weight 4.0 \
    --gan_feat_weight 4.0 \
    --perceptual_weight 4.0 \
    --weight_class \
    --restart_thres 1.0 \
    --no_random_restart \
    --norm_type batch \
    --padding_type replicate \
    --self_pretrained_vqgan '/home/zhiwei/source/SPAct/anonymization_training/saved_models/vqgan/train_vqgan_hmdb_frames_16_every_2_bs_16_lr_0.0001_amp_codes_16384_dis_iter_50/model.pth' \
    --self_pretrained_action '/home/zhiwei/source/SPAct/anonymization_training/saved_models/train_action_H2U_16_frames_bs_32/model_hmdb2hmdb_70_acc_0.9305555555555556.pth'