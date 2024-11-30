import torch
from torch import nn
from typing import Optional
from src.encoder import ViTPatchEmbeddings
from transformers import ViTConfig, ViTModel

class ReconDecEmbeddings(nn.Module):
    
    def __init__(self, config:ViTConfig, use_mask_token=False)-> None:

        super().__init__()

        self.mask_token = nn.Parameter(torch.randn(1, 1, config.hidden_size)) if use_mask_token else None
        
        # try:
        #     assert config.hidden_size == config.patch_size**2 * config.num_channels
        # except AssertionError:
        #     print(f"hidden_size: {config.hidden_size} != patch_size**2 * num_channels: {config.patch_size**2 * config.num_channels}")
            
        self.config = config
        
        self.image_init = nn.Parameter(torch.randn(1, config.num_channels, config.image_size[0], config.image_size[1]))
        
        self.patch_embeddings = ViTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        
        total_num_patches = num_patches + 1  # CLS token
        
        total_num_patches += config.num_global_tokens if config.num_global_tokens is not None else 0
        
        self.position_embeddings = nn.Parameter(torch.randn(1, total_num_patches, config.hidden_size))
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    
    def forward(self,
                global_tokens: torch.Tensor,
                bool_masked_pos: Optional[torch.BoolTensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                interpolate_pos_encoding: Optional[bool] = None,
                return_dict: Optional[bool] = None):
        
        # shape of global_tokens: [N, num_global_tokens, hidden_size]
        # print(global_tokens.shape)
        assert global_tokens.shape[-1] == self.config.hidden_size
        assert global_tokens.shape[-2] == self.config.num_global_tokens + 1
        
        # repeat image_init for N times
        
        N = global_tokens.shape[0]
        
        image_init = self.image_init.expand(N, -1, -1, -1)
        
        embeddings = self.patch_embeddings(image_init) # shape: [N, num_patches, hidden_size]
        
        # concatenate image_init with global_tokens
        embeddings = torch.cat([global_tokens, embeddings], dim=1)
        
        embeddings += self.position_embeddings
        
        embeddings = self.dropout(embeddings)
        
        return embeddings
        
        

class ReconDec(ViTModel):
    
    def __init__(self, config):
        super().__init__(config)
        
        self.config = config
        self.embeddings = ReconDecEmbeddings(config)