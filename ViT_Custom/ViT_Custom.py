import torch
import torch.nn as nn
from Encoder import TransformerEncoder


#class to create patches from tensor x
class PatchEmbedding(nn.Module):
    def __init__(self, input_dim: int, dim_patches : int,  in_channels: int = 3, image_size: int = 224):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, input_dim, dim_patches, stride=dim_patches)

    def forward(self, x: torch.tensor):
        return self.conv(x)



class ViT(nn.Module):
    def __init__(self, num_classes : int = 10, input_dim: int = 768, 
                 dim_patches: int = 16, feedfor_dim: int = 3092, num_blocks: int = 12, 
                 num_heads = 12, image_size: int = 224, dropout: float = 0.0):
        super().__init__()

        #number of patches
        self.num_patches = (image_size // dim_patches) ** 2

        #variables for patches, pos embedding etc...
        self.patch_emb = PatchEmbedding(input_dim, dim_patches)
        self.pos_embedding = nn.Parameter(data = torch.randn(1, self.num_patches + 1, input_dim), requires_grad=True)
        self.cls_token_emb = nn.Parameter(torch.randn(1, 1, input_dim), requires_grad=True)

        #transformer encoder
        self.encoder = TransformerEncoder(num_blocks=num_blocks, num_heads=num_heads, input_dim=input_dim, feedfor_dim=feedfor_dim, dropout=dropout)

        #mlp head to classify
        self.head = nn.Linear(int(input_dim), num_classes)


    def forward(self, x: torch.Tensor):
        
        #patches embedding
        x = self.patch_emb(x)

        x = torch.flatten(x, start_dim=-2, end_dim=-1)
        x = x.transpose(1,2)
        #class token for classification
        x = torch.cat([self.cls_token_emb.repeat(x.size(0), 1, 1), x], dim=1)
        
        #positional embedding
        x = x + self.pos_embedding
        
        #tranformer encoder
        x = self.encoder(x)
        cls_token = x[:, 0]

        #classify
        return self.head(cls_token)