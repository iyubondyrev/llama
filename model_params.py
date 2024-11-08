from dataclasses import dataclass
import torch


@dataclass
class LLAMAParams:
    dim: int
    vocab_size: int
    hidden_dim: int
    max_seq_len: int
    num_layers: int
    num_heads: int
    gradient_accumulation_steps: int
    warmup_steps: int
    lr: float
    max_grad_norm: float
    label_smoothing: float
    weight_decay: float
    num_epoch: int
    train_batch_size: int
    eval_batch_size: int
    random_seed: int
    logging_steps: int
    device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')