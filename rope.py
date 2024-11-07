from torch import nn
import torch
import torchtune

class RoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super(RoPE, self).__init__()
        self.max_seq_len = max_seq_len
        self.dim = dim

        theta = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.freqs = torch.outer(torch.arange(max_seq_len), theta.repeat_interleave(2)).float()
        # self.cos = torch.cos(self.freqs) # [max_seq_len, dim / 2]
        # self.sin = torch.sin(self.freqs) # [max_seq_len, dim / 2]
        self.register_buffer('cos', torch.cos(self.freqs), persistent=False)
        self.register_buffer('sin', torch.sin(self.freqs), persistent=False)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len, dim = x.shape[-2], x.shape[-1]
        assert seq_len <= self.max_seq_len, "Current RoPE implementation is for fixed length"
        assert dim == self.dim, "Dim for RoPE does not match dim of input x"

        even = torch.arange(0, dim, 2)
        odd = torch.arange(1, dim, 2)

        cos_x = x * self.cos[:seq_len, :]
        sin_x = x * self.sin[:seq_len, :]

        cos_x[..., odd - 1] -= sin_x[..., odd]
        cos_x[..., even + 1] += sin_x[..., even]

        return cos_x
    
class RoPE_alternative(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super(RoPE_alternative, self).__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        position = torch.arange(0, max_seq_len, dtype=torch.float)
        dim_index = torch.arange(0, dim // 2, dtype=torch.float)
        inv_freq = 1.0 / (10000 ** (dim_index / (dim // 2)))

        sinusoid_inp = torch.einsum('i,j->ij', position, inv_freq)
        self.register_buffer('cos_pos', torch.cos(sinusoid_inp))
        self.register_buffer('sin_pos', torch.sin(sinusoid_inp))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(-2)
        dim = x.size(-1)
        assert dim == self.dim, "Input dimension does not match RoPE dimension"

        cos_pos = self.cos_pos[:seq_len, :].unsqueeze(0).unsqueeze(0)
        sin_pos = self.sin_pos[:seq_len, :].unsqueeze(0).unsqueeze(0)

        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        x_rotated_even = x1 * cos_pos - x2 * sin_pos
        x_rotated_odd = x1 * sin_pos + x2 * cos_pos

        x_rotated = torch.stack((x_rotated_even, x_rotated_odd), dim=-1)
        x_rotated = x_rotated.flatten(-2)

        return x_rotated


# rope = RoPE(dim=4, max_seq_len=2)

# # x = torch.Tensor([
# #     [[[1, 2, 3, 4]], [[5, 6, 7, 8]]],
# #     [[[1, 2, 3, 4]], [[5, 6, 7, 8]]]
# # ])

# x = torch.Tensor(
#     [[[[1, 2, 3, 4], [5, 6, 7, 8]], [[1, 2, 3, 4], [5, 6, 7, 8]]], [[[1, 2, 3, 4], [5, 6, 7, 8]], [[1, 2, 3, 4], [5, 6, 7, 8]]]]
# )

# rope_2 = RoPE_2(dim=4, max_seq_len=2)
# print(rope_2(x))


# print(x.shape)

# res_1 = rope(x)

# print(res_1)

# test_rope = torchtune.modules.RotaryPositionalEmbeddings(dim=4, max_seq_len=2)

# print(test_rope(x))