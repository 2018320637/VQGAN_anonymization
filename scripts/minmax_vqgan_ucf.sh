#!/bin/bash
cd ..
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port 29501 vqgan_minmax.py \
    --src 'ucf' \
    --tar 'hmdb' \
    --num_frames 16 \
    --sample_every_n_frames 2 \
    --num_workers 8 \
    --batch_size 8 \
    --v_batch_size 8 \
    --train_backbone \
    --architecture 'vit' \
    --learning_rate_fa 0.0001 \
    --learning_rate_fb 0.001 \
    --learning_rate_ft 0.001 \
    --learning_rate_disc 0.001 \
    --learning_rate_domain 0.001 \
    --learning_rate_triplet 0.001 \
    --num_epochs 500 \
    --val_freq 5 \
    --reso_h 128 \
    --reso_w 128 \
    --triple 1 \
    --embedding_dim_dynamic 128 \
    --embedding_dim_static 128 \
    --n_codes_dynamic 16384 \
    --n_codes_static 2048 \
    --n_hiddens 32 \
    --downsample 4 8 8 \
    --disc_channels 64 \
    --disc_layers 3 \
    --discriminator_iter_start 50 \
    --triplet_iter_start 30 \
    --domain_iter_start 30 \
    --disc_loss_type hinge \
    --image_gan_weight 1.0 \
    --video_gan_weight 1.0 \
    --l1_weight 4.0 \
    --gan_feat_weight 4.0 \
    --perceptual_weight 4.0 \
    --action_weight 1.0 \
    --privacy_weight 1.0 \
    --weight_class_action 1 \
    --weight_class_privacy 1 \
    --restart_thres 1.0 \
    --no_random_restart \
    --norm_type batch \
    --padding_type replicate
    # --self_pretrained_vqgan '/home/zhiwei/source/SPAct/anonymization_training/saved_models/vqgan/train_vqgan_ucf_frames_16_every_2_bs_8_lr_0.0001_amp_codes_16384_dis_iter_50/model.pth' \
    # --self_pretrained_action '/home/zhiwei/source/SPAct/anonymization_training/action_logs_vqgan/action_vqgan_amp_ucf_frames_16_every_2_bs_8_lr_0.0002_lr_action_0.002_codes_16384_dis_iter_50_/model_0.9510.pth' \
    # --self_pretrained_privacy '/home/zhiwei/source/SPAct/anonymization_training/privacy_logs_vqgan/privacy_vqgan_ucf_frames_16_every_2_bs_8_lr_0.0001_lr_privacy_0.001_codes_16384_dis_iter_50/model_0.6858_0.4815.pth'