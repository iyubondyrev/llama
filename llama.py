import torch
from torch import nn

from attention import SelfAttention
from feed_forward import GLU
from model_params import LLAMAParams
from rmsnorm import RMSNorm


class EncoderBlock(nn.Module):
    def __init__(self, num_heads: int, dim: int, max_seq_len: int, hidden_dim: int):
        super(EncoderBlock, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.attention = SelfAttention(dim=dim, num_heads=num_heads, max_seq_len=max_seq_len)
        self.head_dim = dim // num_heads
        self.ffn = GLU(dim=dim, hidden_dim=hidden_dim)
        self.attention_norm = RMSNorm(dim=dim)
        self.ffn_norm = RMSNorm(dim=dim)
    
    def forward(self, x: torch.Tensor):
        h = x + self.attention.forward(
            self.attention_norm(x)
        )
        out = h + self.ffn.forward(self.ffn_norm(h))
        return out


class LLAMA(nn.Module):
    def __init__(self, params: LLAMAParams):
        super().__init__()

        
        self.vocab_size = params.vocab_size
        self.num_layers = params.num_layers
        self.embeds = nn.Embedding(self.vocab_size, params.dim)
        self.criterion = nn.CrossEntropyLoss()

        self.layers = nn.ModuleList()
        for _ in range(params.num_layers):
            self.layers.append(EncoderBlock(num_heads=params.num_heads, dim=params.dim, max_seq_len=params.max_seq_len, hidden_dim=params.hidden_dim))

        self.norm = RMSNorm(params.dim)
        self.output = nn.Linear(params.dim, self.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None) -> dict[str, torch.Tensor]:
        h = self.embeds(input_ids)
        
        for layer in self.layers:
            h = layer(h)
        
        h = self.norm(h)
        logits = self.output(h)
        
        if labels is not None:
            loss = self.criterion(logits.view(-1, self.vocab_size), labels.view(-1))
            return {"loss": loss, "logits": logits}
        
        return {"logits": logits}


# params = LLAMAParams(
#     dim=64,
#     hidden_dim=128,
#     max_seq_len=128,
#     vocab_size=512,
#     num_layers=2,
#     num_heads=2,
# )

# llama = LLAMA(params)

# random_tensor = torch.randint(0, 512, (16, 15))
# print(llama(random_tensor).shape)
