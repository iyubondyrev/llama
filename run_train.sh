#!/bin/bash

python train.py \
    --dim 64 \
    --hidden_dim 128 \
    --max_seq_len 1024 \
    --num_layers 2 \
    --num_heads 2 \
    --warmup_steps 500 \
    --lr 1e-4 \
    --max_grad_norm 1.0 \
    --label_smoothing 0.1 \
    --weight_decay 0.01 \
    --num_epoch 3 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --random_seed 42 \
    --logging_steps 100 \
    --wandb_run_name "llama_run_01" \
    --data_path "chunks.txt.gz"