from torch import nn
from MQA import MultiQueryAttention
from ViT_Custom.ViT_Custom import ViT







#function that compresses an already trained model MHA into one with MQA
def compress_model(model, input_dim: int = 768, num_heads: int = 12):

    assert input_dim % num_heads == 0
    head_dim = input_dim // num_heads

    mean_pool = nn.AvgPool1d(kernel_size=num_heads, stride=num_heads)
    
    #mean pool of the projection matrixes of keys and values, into one that has only one head 
    for block in model.encoder.layers:
        
        q_proj = block.attention.q_proj

        #set weight and bias for new k-v shared
        k_proj_weight = mean_pool(block.attention.k_proj.weight).reshape(head_dim,input_dim)
        v_proj_weight = mean_pool(block.attention.v_proj.weight).reshape(head_dim, input_dim)
        
        tmp = block.attention.k_proj.bias
        tmp = tmp.reshape(head_dim, num_heads)
        k_proj_bias = mean_pool(tmp).squeeze(1)
        
        tmp = block.attention.v_proj.bias
        tmp = tmp.reshape(head_dim, num_heads)
        v_proj_bias = mean_pool(tmp).squeeze(1)

        #block multy query attention
        block.attention = MultiQueryAttention(num_heads, input_dim)

        #q-k-v initialization
        block.attention.q_proj = q_proj
        block.attention.v_proj = nn.Linear(input_dim, head_dim)
        block.attention.k_proj = nn.Linear(input_dim, head_dim)

        #k-v shared values added to linear layers
        block.attention.v_proj.bias = nn.Parameter(v_proj_bias)
        block.attention.v_proj.weight = nn.Parameter(v_proj_weight)
        block.attention.k_proj.bias = nn.Parameter(k_proj_bias)
        block.attention.k_proj.weight = nn.Parameter(k_proj_weight)


    #train to adapt the model to the new structure
    return model


