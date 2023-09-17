import torch
from torch import nn
from MHA import MultiHeadAttention
from MQA import MultiQueryAttention


class TransformerBlock(nn.Module):
    def __init__(self, num_heads: int, feedfor_dim : int, input_dim: int, dropout: float = 0.0):
        super().__init__()

        #variables to compute the block result
        self.attention = MultiHeadAttention(num_heads, input_dim)
        self.norm_1 = nn.LayerNorm(input_dim, eps=1e-6)

        self.feed_for = nn.Sequential(
            nn.Linear(input_dim, feedfor_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedfor_dim, input_dim),
            nn.Dropout(dropout),
        )
        self.norm_2 = nn.LayerNorm(input_dim, 1e-6)

        self.dropout = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor):
        #first step attention + norm & residual
        x = self.norm_1(x)
        x =x + self.attention(x)
        
        
        # second step feed forward + norm & residual
        x = self.norm_2(x) 
        x = x + self.feed_for(x)
        
        return x



class TransformerEncoder(nn.Module):
    def __init__(self, num_blocks : int, num_heads: int, feedfor_dim : int, input_dim : int, dropout : float= 0.0):
        super().__init__()
        #additional variables, such as dropout etc

        #define num_blocks transformer blocks
        self.layers = nn.ModuleList([TransformerBlock(num_heads, feedfor_dim, input_dim, dropout) for _ in range(num_blocks)])


    def forward(self, x: torch.Tensor):
        
        #compute the results from all layers sequentially
        for block in self.layers:
            x = block(x)

        return x
    

