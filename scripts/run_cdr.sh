#! /bin/bash
export CUDA_VISIBLE_DEVICES=0


type=context-based
bs=4
bl=3e-5
ul=4e-4
accum=1

python -u  ./train_bio.py --data_dir ./dataset/cdr \
  --max_height 35 \
  --channel_type $type \
  --bert_lr $bl \
  --transformer_type bert \
  --model_name_or_path ../../PLMs/scibert_scivocab_cased \
  --train_file train_filter.data \
  --dev_file dev_filter.data \
  --test_file test_filter.data \
  --train_batch_size $bs \
  --test_batch_size $bs \
  --gradient_accumulation_steps $accum \
  --num_labels 1 \
  --learning_rate $ul \
  --max_grad_norm 1.0 \
  --warmup_ratio 0.06 \
  --num_train_epochs 30 \
  --seed 111 \
  --num_class 2 \
  --save_path ./checkpoint/cdr/train_scibert-lr${bl}_accum${accum}_unet-lr${ul}_bs${bs}.pt \
  --log_dir ./logs/cdr/train_scibert-lr${bl}_accum${accum}_unet-lr${ul}_bs${bs}.log


