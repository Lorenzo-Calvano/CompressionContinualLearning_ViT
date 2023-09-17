import torch
from torch import nn
import numpy as np
from MHA import scaled_dot_product



class MultiQueryAttention(nn.Module):
    def __init__(self, num_heads: int, input_dim: int, dropout: float = 0.0):
        super().__init__()
        assert input_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        #cretate q-k-v throught linearization, q is multi, k-v are singular (shared between the heads)
        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, self.head_dim)
        self.v_proj = nn.Linear(input_dim, self.head_dim)

        #last layer to return
        self.o_proj = nn.Linear(input_dim, input_dim)

    def forward(self, x : torch.Tensor):
        batch_size, seq_length, _ = x.shape

        #multi query
        q = self.q_proj(x)
        q = q.reshape(batch_size, seq_length, self.num_heads, self.head_dim)

        #shared key and values
        k = self.k_proj(x)
        v = self.v_proj(x)
        #reshape for later broadcasting
        k = k.reshape(batch_size, seq_length, 1, self.head_dim)
        v = v.reshape(batch_size, seq_length, 1, self.head_dim)

        values = scaled_dot_product(q,k,v)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, num_heads, head_Dim]

        #concatenation (compress num_heads and head_dim)
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)
        return o



