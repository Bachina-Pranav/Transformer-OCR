# Built in imports
import os
import collections.abc
import math
from typing import Dict, List, Optional, Set, Tuple, Union

# third party imports:
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from einops import rearrange


# torch imports
import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel
import pytorch_lightning as pl
from transformers.modeling_outputs import BaseModelOutputWithPooling


#monkey patching 
class ViTPatchEmbeddings(nn.Module):
    """
    **unchanged from github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py**


    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """
    
    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings
    
class ViTEmbeddings(nn.Module):

    """
    Construct the CLS token, global summary tokens and position and patch embeddings.
    """

    def __init__(self, config:ViTConfig, use_mask_token=False)-> None:

        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.randn(1, 1, config.hidden_size)) if use_mask_token else None
        self.patch_embeddings = ViTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        
        total_num_patches = num_patches + 1  # CLS token
        # handle global tokens
        if config.num_global_tokens > 0:
            assert num_patches % config.num_global_tokens == 0, "The number of global tokens should divide the number of patches." 
            self.global_tokens = nn.Parameter(torch.randn(1, config.num_global_tokens, config.hidden_size))
            total_num_patches += config.num_global_tokens
        
        self.position_embeddings = nn.Parameter(torch.randn(1, total_num_patches, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config
    

    #TODO: complete the interpolate function with appropriate adjustment of the global tokens

    def forward(
        self, 
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    )-> torch.Tensor:

        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask  = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1 - mask) + mask_tokens * mask 

        # add the [CLS] & [GLB0] to [GLB-k] tokens

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        global_tokens = self.global_tokens.expand(batch_size, -1, -1) if hasattr(self, "global_tokens") else None
        embeddings = torch.cat((cls_tokens, global_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings
    
class AttentionMask:

    def __init__(self, config: ViTConfig) -> None:
        self.config = config
        image_size, patch_size = config.image_size, config.patch_size
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        self.num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.num_global_tokens = config.num_global_tokens
        self.tot_num_tokens = self.num_patches + 1 + self.num_global_tokens
        self.image_size = config.image_size
        self.patch_h = config.patch_h
        self.patch_w = config.patch_w
        self.mask = self._generate_attention_mask()


        
    def cal_num_global_tokens(self,h,patch_h,w,patch_w):
        return w//patch_w * h//patch_h
    
    def is_possible(self,h,patch_h,w,patch_w):
        return w%patch_w == 0 and h%patch_h == 0
    
    def convert_2d_to_1d(self,x,y,W):
        return x*W + y
    
    def get_indices(self,start_x , start_y , end_x , end_y):
        indices = []
        for i in range(start_x , end_x):
            for j in range(start_y , end_y):
                indices.append((i,j))
        return indices
    
    def get_patch_indices(self,h,patch_h,w,patch_w):
        indices = []
        for i in range(0,h,patch_h):
            for j in range(0,w,patch_w):
                indices.append(self.get_indices(i,j,i+patch_h,j+patch_w))
        return indices
       
    def _generate_attention_mask(self):
        h = self.image_size[0] // 16
        w = self.image_size[1] // 16
        number_gt = self.num_global_tokens
        patch_h = self.patch_h
        patch_w = self.patch_w
        
        assert self.is_possible(h,patch_h,w,patch_w)
        assert number_gt == self.cal_num_global_tokens(h,patch_h,w,patch_w)
        assert self.tot_num_tokens == 1+ number_gt + (h * w) , f"{number_gt} {h} {w}"
        patch_indices = self.get_patch_indices(h,patch_h,w,patch_w)
        # Convert all of them to 1D
        patch_indices1d = []
        for patch in patch_indices:
            curr_path = []
            for i in patch:
                curr_path.append(self.convert_2d_to_1d(i[0],i[1],w))
            patch_indices1d.append(curr_path)
        mask = torch.zeros(1+number_gt+(h*w),1+number_gt+(h*w))
        mask[0,0] = 1
        
        mask[1:1+number_gt, 1:1+number_gt] = torch.eye(number_gt)
        for patch_index , patch in enumerate(patch_indices1d):
            for i in patch:
                    mask[1+patch_index,1+number_gt+i] = 1
                    mask[1+number_gt+i,1+patch_index] = 1
            for i in patch:
                for j in patch:
                    mask[1+number_gt+i,1+number_gt+j] = 1

                    
        if self.config.visualize_encoder_attention_mask:
            mask_save = mask.cpu().numpy()
            
            plt.imshow(mask_save, cmap='gray', vmin=0, vmax=1)
            plt.savefig("attention_mask.png")

        return mask
        # This code is for 2d patches now we are going to see how it will affect some of our things we did till now 
        
    # def _generate_attention_mask(self):
        
    #     mask = torch.zeros((self.tot_num_tokens, self.tot_num_tokens), dtype=torch.bool)

    #     # [CLS] token attends to everything, everything attends to [CLS] token
    #     mask[0] = 1
    #     mask[:, 0] = 1

    #     # global tokens attend to other global tokens 
    #     if self.num_global_tokens > 0:
    #         mask[1:1+self.num_global_tokens, 1:1+self.num_global_tokens] = 1

    #     num_tokens_per_global_token = self.num_patches // self.num_global_tokens

    #     # each global token attends to only a subset of patches
    #     # each patch attends to its specific global token only
    #     # each patch attends to a subset of patches

    #     for i in range(self.num_global_tokens):
    #         start = 1 + self.num_global_tokens + i * num_tokens_per_global_token
    #         end = 1 + self.num_global_tokens + (i + 1) * num_tokens_per_global_token
    #         # set patch-patch attention mask
    #         mask[start:end, start:end] = 1 
    #         # set patch-global token attention mask
    #         mask[start:end, 1+i] = 1
    #         mask[1+i, start:end] = 1
        
        
    #     if self.config.visualize_encoder_attention_mask:
            
    #         mask_save = mask.cpu().numpy()*255
            
    #         plt.imshow(mask_save,cmap='gray')
    #         plt.savefig("attention_mask.png")

    #     return mask
    
class ViTSelfAttention(pl.LightningModule):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        self.attn_mask = AttentionMask(config).mask

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        
        attn_mask = self.attn_mask.to(attention_scores.device)
        attn_mask = attn_mask.expand(hidden_states.shape[0], self.num_attention_heads, -1, -1)
        assert attention_scores.shape == attn_mask.shape

        attention_scores = attention_scores.masked_fill(attn_mask==0, float("-inf"))
        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
    
class ViTModelCustom(ViTModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple,BaseModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output[:,:self.config.num_global_tokens+1],
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

