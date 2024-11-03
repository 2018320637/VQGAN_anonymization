#!/bin/bash
cd ..
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port 29502 vqgan_action_ddp_amp.py \
    --src 'ucf' \
    --tar 'hmdb' \
    --num_frames 16 \
    --sample_every_n_frames 2 \
    --num_workers 8 \
    --v_batch_size 8 \
    --reso_h 128 \
    --reso_w 128 \
    --embedding_dim 256 \
    --n_codes 16384 \
    --n_hiddens 32 \
    --downsample 4 8 8 \
    --disc_channels 64 \
    --disc_layers 3 \
    --no_random_restart \
    --self_pretrained_vqgan '/home/zhiwei/source/SPAct/anonymization_training/action_logs_vqgan/action_vqgan_amp_hmdb_frames_16_every_2_bs_8_lr_0.0002_lr_action_0.002_codes_16384_dis_iter_50_/model_0.8444.pth' \
    --self_pretrained_action '/home/zhiwei/source/SPAct/anonymization_training/action_logs_vqgan/action_vqgan_amp_hmdb_frames_16_every_2_bs_8_lr_0.0002_lr_action_0.002_codes_16384_dis_iter_50_/model_0.8444.pth'