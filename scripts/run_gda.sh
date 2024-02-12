#! /bin/bash
export CUDA_VISIBLE_DEVICES=0

type=context-based
bs=4
bl=3e-5
ul=3e-4
accum=4
epoch=20
num_timesteps=1000
sampling_timesteps=5

python -u  ./train_bio.py --data_dir ./dataset/gda \
  --max_height 35 \
  --channel_type $type \
  --bert_lr $bl \
  --transformer_type bert \
  --model_name_or_path ../../PLMs/scibert_scivocab_cased \
  --train_file train.data \
  --dev_file dev.data \
  --test_file test.data \
  --train_batch_size $bs \
  --test_batch_size $bs \
  --gradient_accumulation_steps $accum \
  --num_labels 1 \
  --learning_rate $ul \
  --max_grad_norm 1.0 \
  --warmup_ratio 0.06 \
  --num_train_epochs $epoch \
  --evaluation_steps 400 \
  --seed 66 \
  --num_class 2 \
  --num_timesteps $num_timesteps \
  --sampling_timesteps $sampling_timesteps \
  --save_path ./checkpoint/gda/train2_diffusion_scibert-lr${bl}_accum${accum}_unet-lr${ul}_bs${bs}_${num_timesteps}_${sampling_timesteps}_${epoch}.pt \
  --log_dir ./logs/gda/train2_diffusion_scibert-lr${bl}_accum${accum}_unet-lr${ul}_bs${bs}_${num_timesteps}_${sampling_timesteps}_${epoch}.log
