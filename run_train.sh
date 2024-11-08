#!/bin/bash

python train.py \
    --dim 1024 \
    --hidden_dim 2048 \
    --max_seq_len 1024 \
    --num_layers 10 \
    --num_heads 8 \
    --gradient_accumulation_steps 6 \
    --warmup_steps 2000 \
    --lr 1e-4 \
    --max_grad_norm 1.0 \
    --label_smoothing 0.1 \
    --weight_decay 0.01 \
    --num_epoch 50 \
    --train_batch_size 8 \
    --eval_batch_size 64 \
    --random_seed 42 \
    --logging_steps 50 \
    --wandb_run_name "llama_run_02" \
    --data_path "chunks.txt.gz"