from transformers import (
    ViTConfig, 
    ViTModel
)

import torch
from torch import nn
# local imports
from src.encoder import *
from src.pretraining_decoder import *
import torch.nn.functional as F

# update the class namespace
import transformers 
transformers.models.vit.modeling_vit.ViTPatchEmbeddings = ViTPatchEmbeddings
transformers.models.vit.modeling_vit.ViTEmbeddings = ViTEmbeddings
transformers.models.vit.modeling_vit.ViTSelfAttention = ViTSelfAttention

class Autoencoder(nn.Module):
    def __init__(self, encoder_config: ViTConfig,  phase: str = 'pre-training'):
        super().__init__()
        
        assert phase in ['pre-training', 'ocr-training'], f"phase must be one of ['pre-training', 'ocr-training']"
        
        self.encoder_config = encoder_config

        self.encoder = ViTModel(encoder_config)
        
        self.decoder = ReconDec(encoder_config)
        expected_out = (encoder_config.patch_size**2) * encoder_config.num_channels

        self.projection_layer = nn.Linear(encoder_config.hidden_size, expected_out) if expected_out != encoder_config.hidden_size else nn.Identity()

    def forward(self, pixel_values, gt_labels = None):
        encoder_output = self.encoder(pixel_values)
        
        enc_tensors = encoder_output.last_hidden_state
        
        global_tokens = enc_tensors[:, :self.encoder_config.num_global_tokens+1]
        
        decoder_output = self.decoder(global_tokens).last_hidden_state
                
        recon_images = decoder_output[:, self.encoder_config.num_global_tokens+1:]

        recon_images = self.projection_layer(recon_images)
        # adding sigmoid activation
        # need to check
        recon_images = F.sigmoid(recon_images)

        recon_images = recon_images.view(-1, self.encoder_config.num_channels, *self.encoder_config.image_size)
      
        return recon_images