import torch
from torch import nn
from torch.nn.functional import silu

class GLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super(GLU, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        swish = silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)
        return x
