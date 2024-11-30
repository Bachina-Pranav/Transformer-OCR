from transformers import (
    VisionEncoderDecoderModel,
    TrOCRConfig,
    ViTConfig, 
    TrOCRForCausalLM
)

import torch 
import torch.nn as nn

# local imports
from src.encoder import *

# update the class namespace
import transformers
transformers.models.vit.modeling_vit.ViTPatchEmbeddings = ViTPatchEmbeddings
transformers.models.vit.modeling_vit.ViTEmbeddings = ViTEmbeddings
transformers.models.vit.modeling_vit.ViTSelfAttention = ViTSelfAttention

class PageOCR(nn.Module):
    def __init__(self, encoder_config: ViTConfig, decoder_config: TrOCRConfig):
        super().__init__()
        
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        
        self.pageocr = VisionEncoderDecoderModel(encoder = ViTModelCustom(encoder_config), decoder = TrOCRForCausalLM(decoder_config))
        
        
        self.pageocr.config.decoder_start_token_id = decoder_config.decoder_start_token_id
        self.pageocr.config.pad_token_id = decoder_config.pad_token_id
        self.pageocr.config.vocab_size = decoder_config.vocab_size
        self.pageocr.config.max_length = decoder_config.max_position_embeddings
        self.pageocr.config.eos_token_id = decoder_config.eos_token_id
        self.pageocr.eos_token_id = decoder_config.eos_token_id
        self.pageocr.config.early_stopping = True
        self.pageocr.config.no_repeat_ngram_size = 3
        self.pageocr.config.length_penalty = 2.0
        self.pageocr.config.num_beams = 4 


    
    def forward(self, pixel_values, gt_labels):
        return self.pageocr(pixel_values,labels= gt_labels)

