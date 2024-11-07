import math
import torch
from torch import nn
from rope import RoPE


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int) -> None:
        super(SelfAttention, self).__init__()

        assert dim % num_heads == 0, "dim % num_heads != 0"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = self.dim // self.num_heads
        self.max_seq_len = max_seq_len

        self.W_q = nn.Linear(self.dim, self.dim, bias=False)
        self.W_k = nn.Linear(self.dim, self.dim, bias=False)
        self.W_v = nn.Linear(self.dim, self.dim, bias=False)
        self.W_o = nn.Linear(self.dim, self.dim, bias=False)
        self.rope = RoPE(dim=self.head_dim, max_seq_len=self.max_seq_len)
        self.register_buffer(
            'causal_mask', torch.tril(torch.ones(self.max_seq_len, self.max_seq_len)).bool()
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x [batch_size, seq_len, dim]

        seq_len = x.shape[1]

        q: torch.Tensor = self.W_q(x) # [batch_size, seq_len, dim]
        k: torch.Tensor = self.W_k(x) # [batch_size, seq_len, dim]
        v: torch.Tensor = self.W_v(x) # [batch_size, seq_len, dim]

        q = self.rope(q.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)) # [batch_size, num_heads, seq_len, head_dim]
        k = self.rope(k.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)) # [batch_size, num_heads, seq_len, head_dim]
        v = v.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2) # [batch_size, num_heads, seq_len, head_dim]

        scores = torch.matmul(q, k.transpose(-1, -2)) # [batch_size, num_heads, seq_len, seq_len]


        if mask is not None:
            mask = mask.to(scores.device)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        else:
            causal_mask = self.causal_mask[:seq_len, :seq_len]
            causal_mask.to(scores.device)
            scores = scores.masked_fill(~causal_mask, float('-inf'))

         
        scores = scores / math.sqrt(self.head_dim)

        scores = torch.softmax(scores, dim=-1)

        scores = torch.matmul(scores, v) # [batch_size, num_heads, seq_len, head_dim]

        res = self.W_o((scores.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1))) # [batch_size, seq_len, dim]

        return res


x = torch.Tensor([
    [[[1, 2, 3, 4]], [[5, 6, 7, 8]]],
    [[[1, 2, 3, 4]], [[5, 6, 7, 8]]]
])

attn = SelfAttention(dim=4, num_heads=2, max_seq_len=2)

attn(x)

