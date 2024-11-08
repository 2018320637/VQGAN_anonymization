#!/bin/bash
cd ../..
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port 29501 vqgan_pretrain.py \
    --src 'hmdb' \
    --tar 'ucf' \
    --num_frames 16 \
    --sample_every_n_frames 2 \
    --num_workers 8 \
    --batch_size 8 \
    --v_batch_size 8 \
    --learning_rate 0.0001 \
    --num_epochs 300 \
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
    --disc_loss_type hinge \
    --image_gan_weight 1.0 \
    --video_gan_weight 1.0 \
    --l1_weight 4.0 \
    --gan_feat_weight 4.0 \
    --perceptual_weight 4.0 \
    --restart_thres 1.0 \
    --no_random_restart \
    --norm_type batch \
    --padding_type replicate