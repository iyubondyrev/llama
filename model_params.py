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
    device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')