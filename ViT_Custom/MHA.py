import torch
from torch import nn
import numpy as np
import math

#to calculate the scaled dot product (attention mechanism)
def scaled_dot_product(q : torch.Tensor, k : torch.Tensor, v : torch.Tensor):
    #value of dimensionality
    d_k = q.size()[-1]

    #intermediate step of attention
    attent_logits = torch.matmul(q, k.transpose(-2, -1))
    attent_logits = attent_logits / math.sqrt(d_k)
    attention = torch.softmax(attent_logits, dim=-1)

    #final values 
    values = torch.matmul(attention, v)
    return values



class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, input_dim: int, dropout: float = 0.0):
        super().__init__()
        assert input_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

    
        #create q-k-v throught linearization
        self.q_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)

        #last layer to return
        self.o_proj = nn.Linear(input_dim, input_dim)

    def forward(self, x: torch.Tensor):
        batch_size, seq_length, _ = x.size()
        
        #compute q-k-v
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Separate Q, K, V from linear output
        q = q.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3) # [Batch, num_heads, SeqLen, head_Dim]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        # Determine value outputs
        values = scaled_dot_product(q, k, v)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, num_heads, head_Dim]

        #concatenation (compress num_heads and head_dim)
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        return o
