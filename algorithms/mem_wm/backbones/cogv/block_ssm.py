# Copyright 2024 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# diffusers.models.transformers.cogvideox_transformer_3d.py
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
import math
import os
import pickle
from pathlib import Path

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_processor import AttentionProcessor
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from .embedding import CogVideoXPatchEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from mamba_ssm import Mamba2
from mamba_ssm.utils.generation import InferenceParams
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
import types
from einops import rearrange
from typing import List, Dict, Any, Optional, Tuple, Union
from omegaconf import DictConfig
from utils.distributed_utils import rank_zero_print
from utils.lightning_utils import FlexAttentionCallRecord, FlexAttentionFLOPsTracker


flex_attention = torch.compile(flex_attention, dynamic=False)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class FlexAttnProcessor(nn.Module):
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    Supports KV caching for autoregressive generation with chunk-wise processing.
    """

    def __init__(self, cache_dir: str = "./cache_dumps"):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        
        # KV cache will be registered as buffers when initialized
        self.cache_enabled = False
        self._cache_initialized = False
        
        # Local cache saving configuration
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.save_cache_locally = False
        self.cache_step_counter = 0
    
    def enable_kv_cache(self):
        """Enable KV caching for autoregressive generation."""
        self.cache_enabled = True
    
    def get_kv_cache(self):
        """Get the KV cache."""
        return self._cached_key, self._cached_value
    
    def disable_kv_cache(self):
        """Disable KV caching and clear existing cache."""
        self.cache_enabled = False
        self._clear_cache_buffers()
    
    def clear_kv_cache(self):
        """Clear the KV cache."""
        self._clear_cache_buffers()
    
    def enable_local_cache_saving(self, cache_dir: str = None):
        """Enable saving cache to local files before each update."""
        self.save_cache_locally = True
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def disable_local_cache_saving(self):
        """Disable saving cache to local files."""
        self.save_cache_locally = False
    
    def save_kv_cache_to_file(self, processor_id: str = "default"):
        """Save current KV cache to local file."""
        if not self.save_cache_locally or not self._cache_initialized:
            return
        
        try:
            cache_data = {
                'cached_key': self._cached_key.cpu() if hasattr(self, '_cached_key') else None,
                'cached_value': self._cached_value.cpu() if hasattr(self, '_cached_value') else None,
                'step': self.cache_step_counter,
                'shape_key': self._cached_key.shape if hasattr(self, '_cached_key') else None,
                'shape_value': self._cached_value.shape if hasattr(self, '_cached_value') else None,
                'dtype': str(self._cached_key.dtype) if hasattr(self, '_cached_key') else None,
            }
            
            # Create organized directory structure: cache_dir/kv_cache/step_X/
            step_dir = self.cache_dir / "kv_cache" / f"step_{self.cache_step_counter}"
            step_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"{processor_id}.pkl"
            filepath = step_dir / filename
            
            with open(filepath, 'wb') as f:
                pickle.dump(cache_data, f)
            
            rank_zero_print(f"KV cache saved to {filepath}")
                
        except Exception as e:
            rank_zero_print(f"Failed to save KV cache: {e}")
    
    def load_kv_cache_from_file(self, processor_id: str = "default", step: int = None):
        """Load KV cache from local file."""
        try:
            if step is None:
                step = self.cache_step_counter
            
            # Use organized directory structure: cache_dir/kv_cache/step_X/
            step_dir = self.cache_dir / "kv_cache" / f"step_{step}"
            filename = f"{processor_id}.pkl"
            filepath = step_dir / filename
            
            if not filepath.exists():
                rank_zero_print(f"KV cache file {filepath} not found")
                return False
            
            with open(filepath, 'rb') as f:
                cache_data = pickle.load(f)
            
            if cache_data['cached_key'] is not None and cache_data['cached_value'] is not None:
                # Move back to appropriate device
                device = next(self.parameters()).device if hasattr(self, 'parameters') else 'cuda'
                
                if not self._cache_initialized:
                    self._init_cache_buffers(
                        cache_data['shape_key'],
                        cache_data['shape_value'],
                        getattr(torch, cache_data['dtype'].split('.')[-1]),
                        device
                    )
                
                self._cached_key.copy_(cache_data['cached_key'].to(device))
                self._cached_value.copy_(cache_data['cached_value'].to(device))
                
                rank_zero_print(f"KV cache loaded from {filepath}")
                return True
            
        except Exception as e:
            rank_zero_print(f"Failed to load KV cache: {e}")
        
        return False
    
    def _clear_cache_buffers(self):
        """Clear registered cache buffers."""
        if self._cache_initialized:
            if hasattr(self, '_cached_key'):
                delattr(self, '_cached_key')
            if hasattr(self, '_cached_value'):
                delattr(self, '_cached_value')
            self._cache_initialized = False
    
    def _init_cache_buffers(self, key_shape, value_shape, dtype, device):
        """Initialize cache buffers with given shapes."""
        if not self._cache_initialized:
            # Register buffers for key and value cache
            self.register_buffer('_cached_key', torch.zeros(key_shape, dtype=dtype, device=device), persistent=False)
            self.register_buffer('_cached_value', torch.zeros(value_shape, dtype=dtype, device=device), persistent=False)
            self._cache_initialized = True

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        block_mask = None,
        token_chunk_size: int = 5,
        enable_caches: bool = True,
        do_update_cache: bool = True, # since we are executing diffusion step by step, we shall only update the cache for the last step
        is_loading_context: bool = False, # if we are loading context, we need to update the cache for the first step
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)
        """
        If we pass in text, then it will be just dummy tokens, so it should always be cached and don't need to be updated actually
        """
        if enable_caches:
            self.enable_kv_cache()

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states) 
        # we actually only need query from positions where tokens to be generated, and the chunk size is token_chunk_size
        # However, if we want to use block_mask making code provided by Ryan, then we need to pass the query from all positions

        if self.cache_enabled:
            if not do_update_cache and not is_loading_context:
                # Only calculate KV for chunk_to_generate part (skip context_chunk)
                key = attn.to_k(hidden_states[:, text_seq_length + token_chunk_size:])
                value = attn.to_v(hidden_states[:, text_seq_length + token_chunk_size:])
            else:
                # Calculate KV for full sequence (context loading or first diffusion step)
                key = attn.to_k(hidden_states)
                value = attn.to_v(hidden_states)
        else:
            # Cache disabled: always calculate KV for full sequence
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        # Handle KV caching for autoregressive generation
        if self.cache_enabled and self._cache_initialized and hasattr(self, '_cached_key'):
            # Extract previous chunk's KV from cache buffers
            cached_key = self._cached_key
            cached_value = self._cached_value
            
            # Concatenate cached KV with current KV
            # Only concatenate the image part (skip text tokens)
            
            # Concatenate along sequence dimension
            key = torch.cat([cached_key, key[:, :, - token_chunk_size:]], dim=2) 
            # the reason why we use token_chunk_size: is to make sure there is no overlapped keys/values between the cached and the current chunk
            value = torch.cat([cached_value, value[:, :, - token_chunk_size:]], dim=2)
            
        # Record Flex Attention FLOP estimation (before calling kernel)
        if FlexAttentionFLOPsTracker.is_active():
            density = 1.0
            if block_mask is not None and hasattr(block_mask, "sparsity"):
                try:
                    sparsity_val = float(block_mask.sparsity())
                    density = max(0.0, min(1.0, 1.0 - sparsity_val / 100.0))
                    rank_zero_print(f"Block sparsity: {sparsity_val:.2f}%")
                except Exception as e:
                    rank_zero_print(f"Warning: Failed to get sparsity from block_mask: {e}")
                    density = 1.0

            record = FlexAttentionCallRecord(
                batch_size=int(query.shape[0]),
                num_heads=int(query.shape[1]),
                q_len=int(query.shape[2]),
                kv_len=int(key.shape[2]),
                head_dim=int(head_dim),
                density=density,
            )
            FlexAttentionFLOPsTracker.record(record)

        # Temporarily disable FlopCounterMode to avoid conflicts with compiled flex_attention kernel
        from utils.lightning_utils import FlopCounterModeContext
        FlopCounterModeContext.pause()
        
        try:
            hidden_states = flex_attention(query.to(torch.bfloat16), key.to(torch.bfloat16), value.to(torch.bfloat16), block_mask=block_mask)
        finally:
            FlopCounterModeContext.resume()

        # Update KV cache with context_chunk KV for next iteration
        # Only update cache when explicitly requested (e.g., after getting clean frames)
        if self.cache_enabled and do_update_cache:
            # Note: KV cache saving is now handled by the CogVideoXBlock level to ensure consistency
            
            # Store the token_chunk_size last tokens from current key/value for next iteration
            current_img_seq_len = key.shape[2] - text_seq_length
            if current_img_seq_len >= token_chunk_size:
                # Extract the last token_chunk_size tokens worth of tokens
                start_of_tokens_to_cache = current_img_seq_len - token_chunk_size
                # say we have [chunk 1, chunk 2], we will cache chunk 2 here
                key_to_cache = key[:, :, start_of_tokens_to_cache:].detach()
                value_to_cache = value[:, :, start_of_tokens_to_cache:].detach()
            else:
                # If current chunk is smaller than token_chunk_size, store all of it
                key_to_cache = key.detach()
                value_to_cache = value.detach()
            
            # Initialize cache buffers if not already done
            if not self._cache_initialized:
                self._init_cache_buffers( # initialize with default values and shape of key_to_cache and value_to_cache
                    key_to_cache.shape, 
                    value_to_cache.shape, 
                    key_to_cache.dtype, 
                    key_to_cache.device
                )
            
            # Update cache buffers
            if hasattr(self, '_cached_key') and key_to_cache.shape == self._cached_key.shape:
                self._cached_key.copy_(key_to_cache)
                self._cached_value.copy_(value_to_cache)
            else:
                # If shape changed, reinitialize buffers
                self._clear_cache_buffers()
                self._init_cache_buffers(
                    key_to_cache.shape, 
                    value_to_cache.shape, 
                    key_to_cache.dtype, 
                    key_to_cache.device
                )
                self._cached_key.copy_(key_to_cache)
                self._cached_value.copy_(value_to_cache)
            
            # Note: step counter is now managed by CogVideoXBlock to ensure consistency

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        attn_dtype = attn.to_out[0].weight.dtype
        hidden_states = hidden_states.to(attn_dtype) # otherwise when we set precision to 32, the hidden_states here will be bfloat16

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states

class AgentPositionEmbedding(nn.Module):
    def __init__(
        self, 
        input_dim: int = 4, 
        hidden_dim: int = 128, 
        output_dim: int = 512, 
        dropout_rate: float = 0.0
    ) -> None:
        super().__init__()
        
        self.network = nn.Sequential(
            # First layer
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Second layer
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Initialize weights using Xavier/Glorot initialization
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights using Xavier/Glorot initialization."""
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class AdaLayerNorm(nn.Module):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`, *optional*): The size of the embeddings dictionary.
        output_dim (`int`, *optional*):
        norm_elementwise_affine (`bool`, defaults to `False):
        norm_eps (`bool`, defaults to `False`):
        chunk_dim (`int`, defaults to `0`):
    """

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: Optional[int] = None,
        output_dim: Optional[int] = None,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()

        self.chunk_dim = chunk_dim
        output_dim = output_dim or embedding_dim * 2

        if num_embeddings is not None:
            self.emb = nn.Embedding(num_embeddings, embedding_dim)
        else:
            self.emb = None

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self, x: torch.Tensor, timestep: Optional[torch.Tensor] = None, temb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.emb is not None:
            temb = self.emb(timestep)

        temb = self.linear(self.silu(temb))

        if self.chunk_dim == 1:
            # This is a bit weird why we have the order of "shift, scale" here and "scale, shift" in the
            # other if-branch. This branch is specific to CogVideoX for now.
            shift, scale = temb.chunk(2, dim=-1)
            shift = shift[:, :, None, :]
            scale = scale[:, :, None, :]
        else:
            scale, shift = temb.chunk(2, dim=0)

        x = self.norm(x) * (1 + scale) + shift
        return x

class CogVideoXLayerNormZero(nn.Module):
    def __init__(
        self,
        conditioning_dim: int,
        embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_dim, 3 * embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = self.linear(self.silu(temb)).chunk(3, dim=-1)
        hidden_states = self.norm(hidden_states) * (1 + scale)[:, :, None, :] + shift[:, :, None, :]
        # encoder_hidden_states = self.norm(encoder_hidden_states) * (1 + enc_scale)[:, None, :] + enc_shift[:, None, :]
        return hidden_states, encoder_hidden_states, gate[:, :, None, :]



@maybe_allow_in_graph
class CogVideoXBlock(nn.Module):
    r"""
    Transformer block used in [CogVideoX](https://github.com/THUDM/CogVideo) model.

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        time_embed_dim (`int`):
            The number of channels in timestep embedding.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to be used in feed-forward.
        attention_bias (`bool`, defaults to `False`):
            Whether or not to use bias in attention projection layers.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        ff_mult: int = 4,
        attention_out_bias: bool = True,
        layer_idx: int = None,
        sample_width: int = 8,
        sample_height: int = 8,
        b_h: int = 4,
        b_w: int = 4,
        expand: int = 1,
        headdim: int = 48,
        d_state: int = 128,
        share_child: bool = True,
        cache_dir: str = "./cache_dumps",
        mamba_cache_mode: str = "chunk",  # "chunk" or "step"
    ):
        super().__init__()

        self.chunk_size = int((b_h * b_w)**(0.5))

        self.layer_idx = layer_idx

        self.sample_width = sample_width
        self.sample_height = sample_height

        # cache related
        self.inference_params = None
        self.context_loading_completed = False
        
        # Mamba processing mode configuration
        self.mamba_cache_mode = mamba_cache_mode  # "chunk" or "step"
        
        # Local cache saving configuration
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.save_cache_locally = False
        self.cache_step_counter = 0
        #########


        # 1. Mamba + FF
        self.norm1_mamba = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.mamba = Mamba2(
                        # This module uses roughly 3 * expand * d_model^2 parameters
                        d_model=dim,  # Model dimension d_model
                        d_state=d_state,  # SSM state expansion factor, typically 64 or 128
                        d_conv=4,  # Local convolution width
                        expand=expand,  # Block expansion factor
                        layer_idx=self.layer_idx,
                    )

        # 2. Local Attention + FF
        self.norm1_attn = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.attn = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=FlexAttnProcessor(cache_dir=cache_dir)
        )

        self.norm2_attn = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)  

        self.ff_attn = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
    

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        block_mask = None,
        enable_caches: bool = False,
        max_seqlen: int = 4096,
        motion_block_size: int = 4,
        do_update_cache: bool = False,
        is_loading_context: bool = False,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)
        attention_kwargs = attention_kwargs or {}
        
        batch_size, seq_len, hidden_dim = hidden_states.shape


        orig_shape = hidden_states.shape
        _, num_frames, _ = temb.shape
        hidden_states = hidden_states.reshape(batch_size, num_frames, -1, hidden_dim) #(b, f, hw, d)


        ####### Below: Mamba Block #########
        # Norm
        norm_hidden_states, norm_encoder_hidden_states, gate_mamba = self.norm1_mamba(
            hidden_states, encoder_hidden_states, temb
        )
        
        # Mamba
        mamba_hidden_states = rearrange(norm_hidden_states, "b f (h w) d -> b f h w d", h = self.sample_height)
        mamba_hidden_states = mamba_hidden_states.unfold(2,self.chunk_size,self.chunk_size).unfold(3,self.chunk_size,self.chunk_size)

        if enable_caches:
            if not is_loading_context and self.context_loading_completed:
                mamba_hidden_states = rearrange(mamba_hidden_states[:,  - motion_block_size:, ...], "b f bh bw d ch cw -> (b bh bw) (f ch cw) d")
            else:
                mamba_hidden_states = rearrange(mamba_hidden_states, "b f bh bw d ch cw -> (b bh bw) (f ch cw) d")

            if self.inference_params is None:
                block_ssm_batch_size = mamba_hidden_states.shape[0]
                
                # Allocate cache tensors directly in inference_params
                ssm_states, conv_states = self.mamba.allocate_inference_cache(
                    batch_size=block_ssm_batch_size, 
                    max_seqlen=max_seqlen, 
                    dtype=hidden_states.dtype
                )
                
                # Create inference_params with proper InferenceParams class
                self.inference_params = InferenceParams(
                    max_seqlen=max_seqlen,
                    max_batch_size=block_ssm_batch_size
                )
                # Set the cache states directly
                self.inference_params.key_value_memory_dict[self.layer_idx] = (ssm_states, conv_states)
                self.inference_params.seqlen_offset = 0

                # Create backup of cache states for restoration after diffusion steps
                self._backup_ssm_cache()

            
            # Save SSM cache before updating (if enabled and cache exists)
            if do_update_cache and self.save_cache_locally and self.inference_params is not None:
                self.save_ssm_cache_to_file()
            
            # Process tokens based on selected mode
            if not is_loading_context and self.context_loading_completed:
                # For new tokens: choose processing mode
                if self.mamba_cache_mode == "chunk":
                    # this is currently problematic, see: https://github.com/state-spaces/mamba/issues/641
                    # Chunk Mode: Use parallel scan with existing cache states
                    # Keep seqlen_offset=0 to use normal forward path with cache as initial state
                    mamba_hidden_states = self.mamba(mamba_hidden_states, inference_params=self.inference_params)
                    
                elif self.mamba_cache_mode == "step":
                    # Step Mode: Process tokens one by one using Mamba2's step function
                    batch_spatial, seq_len, d_model = mamba_hidden_states.shape
                    
                    # Set correct seqlen_offset for step mode
                    if hasattr(self, '_context_length'):
                        self.inference_params.seqlen_offset = self._context_length
                    else:
                        # Fallback estimation
                        self.inference_params.seqlen_offset = max_seqlen - seq_len
                    
                    # Process each token step by step
                    token_outputs = []
                    for token_idx in range(seq_len):
                        single_token = mamba_hidden_states[:, token_idx:token_idx+1, :]  # Shape: (batch, 1, d_model)
                        
                        # Process single token using step mode
                        token_output = self.mamba(single_token, inference_params=self.inference_params)
                        token_outputs.append(token_output)
                        
                        # Update seqlen_offset for next token
                        self.inference_params.seqlen_offset += 1
                    
                    # Concatenate token outputs
                    mamba_hidden_states = torch.cat(token_outputs, dim=1)
                    
                else:
                    raise ValueError(f"Unknown mamba_cache_mode: {self.mamba_cache_mode}. Must be 'chunk' or 'step'")
                    
            else:
                # For context loading: always use parallel scan mode
                if is_loading_context:
                    # Reset seqlen_offset for fresh context processing
                    self.inference_params.seqlen_offset = 0
                    # Store context length for step mode
                    self._context_length = mamba_hidden_states.shape[1]
                
                # Process all tokens in batch (standard way)
                mamba_hidden_states = self.mamba(mamba_hidden_states, inference_params=self.inference_params)
            
            if do_update_cache: # if it's first diffusion
                # Update context length to include newly processed tokens
                if hasattr(self, '_context_length'):
                    # Add the length of newly processed tokens
                    current_seq_len = mamba_hidden_states.shape[1]
                    if not is_loading_context and self.context_loading_completed:
                        # For new tokens: add to existing context length
                        self._context_length += current_seq_len
                    else:
                        # For context loading: set to current sequence length
                        self._context_length = current_seq_len
                else:
                    raise ValueError("Context length not found")
                
                # Keep the updated cache and backup it for future use
                self._backup_ssm_cache()
            else:
                # Restore cache from backup to undo the in-place updates
                self._restore_ssm_cache()
        else:
            mamba_hidden_states = rearrange(mamba_hidden_states, "b f bh bw d ch cw -> (b bh bw) (f ch cw) d")
            mamba_hidden_states = self.mamba(mamba_hidden_states)
        mamba_hidden_states = rearrange(mamba_hidden_states, "(b bh bw) (f ch cw) d -> b f (bh ch) (bw cw) d", b = batch_size, bh = self.sample_height//self.chunk_size, bw = self.sample_width//self.chunk_size, ch = self.chunk_size, cw = self.chunk_size)
        mamba_hidden_states = rearrange(mamba_hidden_states, "b f h w d -> b f (h w) d")
        mamba_encoder_hidden_states = norm_encoder_hidden_states

        if enable_caches:
            if not is_loading_context and self.context_loading_completed:
                full = torch.zeros_like(hidden_states)  # (b, f, hw, d)
                full[:, -motion_block_size:] = mamba_hidden_states
                mamba_hidden_states = full

        # Gate
        hidden_states = hidden_states + gate_mamba * mamba_hidden_states
        encoder_hidden_states = encoder_hidden_states + mamba_encoder_hidden_states
        ####### Above: Mamba Block #########

        ####### Below: Local Attention Block #########
        # Save KV cache before attention computation (if enabled and cache exists)
        if enable_caches and do_update_cache and self.save_cache_locally and self._supports_kv_cache():
            if self.attn.processor._cache_initialized:
                self.attn.processor.save_kv_cache_to_file(f"layer_{self.layer_idx}")

            self.cache_step_counter += 1
            self.attn.processor.cache_step_counter = self.cache_step_counter

        # Norm
        norm_hidden_states, norm_encoder_hidden_states, gate_attn = self.norm1_attn(
            hidden_states, encoder_hidden_states, temb
        )

        # Attn
        norm_hidden_states = norm_hidden_states.reshape(orig_shape)
        attn_hidden_states, attn_encoder_hidden_states = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            block_mask = block_mask,
            token_chunk_size=motion_block_size * self.sample_width * self.sample_height,
            enable_caches=enable_caches,
            do_update_cache=do_update_cache,
            is_loading_context=is_loading_context,
            **attention_kwargs,
        )
        
        # Gate
        hidden_states = hidden_states + gate_attn * attn_hidden_states.reshape(batch_size, num_frames, -1, hidden_dim)
        encoder_hidden_states = encoder_hidden_states + attn_encoder_hidden_states

        # Norm
        norm_hidden_states, norm_encoder_hidden_states, gate_attnff = self.norm2_attn(
            hidden_states, encoder_hidden_states, temb
        )

        # FF
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states.reshape(orig_shape)], dim=1)
        ff_output = self.ff_attn(norm_hidden_states)

        hidden_states = hidden_states + gate_attnff * ff_output[:, text_seq_length:].reshape(batch_size, num_frames, -1, hidden_dim)
        encoder_hidden_states = encoder_hidden_states + ff_output[:, :text_seq_length]
        ####### Above: Local Attention Block #########

        hidden_states = hidden_states.reshape(orig_shape)

        return hidden_states, encoder_hidden_states
    
    def _backup_ssm_cache(self):
        """Create backup of current SSM cache states."""
        if self.inference_params is not None:
            # Get current cache tensors from inference_params
            ssm_cache, conv_cache = self.inference_params.key_value_memory_dict[self.layer_idx]
            
            # Create backup inference_params with cloned tensors
            if not hasattr(self, 'inference_params_backup') or self.inference_params_backup is None:
                # Create new backup tensors
                ssm_backup = torch.zeros_like(ssm_cache)
                conv_backup = torch.zeros_like(conv_cache)
                
                self.inference_params_backup = InferenceParams(
                    max_seqlen=self.inference_params.max_seqlen,
                    max_batch_size=self.inference_params.max_batch_size
                )
                self.inference_params_backup.key_value_memory_dict[self.layer_idx] = (ssm_backup, conv_backup)
                self.inference_params_backup.seqlen_offset = self.inference_params.seqlen_offset
            
            # Copy current cache to backup
            ssm_backup, conv_backup = self.inference_params_backup.key_value_memory_dict[self.layer_idx]
            ssm_backup.copy_(ssm_cache)
            conv_backup.copy_(conv_cache)
            
            # Also backup the seqlen_offset
            self.inference_params_backup.seqlen_offset = self.inference_params.seqlen_offset
        else:
            raise ValueError("No SSM cache to backup")

    def _restore_ssm_cache(self):
        """Restore SSM cache states from backup (undo in-place updates)."""
        if self.inference_params is not None and hasattr(self, 'inference_params_backup') and self.inference_params_backup is not None:
            # Get current cache tensors from inference_params
            ssm_cache, conv_cache = self.inference_params.key_value_memory_dict[self.layer_idx]
            
            # Get backup tensors from inference_params_backup
            ssm_backup, conv_backup = self.inference_params_backup.key_value_memory_dict[self.layer_idx]
            
            # Restore cache from backup
            ssm_cache.copy_(ssm_backup)
            conv_cache.copy_(conv_backup)
            
            # Also restore the seqlen_offset
            self.inference_params.seqlen_offset = self.inference_params_backup.seqlen_offset
        else:
            raise ValueError("No SSM cache to restore")

    def clear_ssm_cache(self):
        """Clear inference cache and free memory."""
        if self.inference_params is not None:
            # Clear the inference_params and backup - that's it!
            self.inference_params = None
            self.inference_params_backup = None
    
    def enable_kv_cache(self):
        """Enable KV caching for autoregressive generation."""
        if self._supports_kv_cache():
            self.attn.processor.enable_kv_cache()
    
    def disable_kv_cache(self):
        """Disable KV caching and clear existing cache."""
        if self._supports_kv_cache():
            self.attn.processor.disable_kv_cache()
    
    def clear_kv_cache(self):
        """Clear the KV cache."""
        if self._supports_kv_cache():
            self.attn.processor.clear_kv_cache()
    
    def _supports_kv_cache(self):
        """Check if the attention processor supports KV caching."""
        return (hasattr(self.attn, 'processor') and 
                isinstance(self.attn.processor, FlexAttnProcessor))

    def clear_cache(self):
        """Clear SSM and KV caches for all transformer blocks."""
        self.clear_ssm_cache()
        self.disable_kv_cache()
        self.context_loading_completed = False

    def reset_context_loading_flag(self):
        """Reset the context loading flag for all transformer blocks."""
        self.context_loading_completed = False
    
    def set_context_loading_flag(self):
        """Set the context loading flag for all transformer blocks."""
        self.context_loading_completed = True
    
    def set_mamba_cache_mode(self, mode: str):
        """
        Set Mamba processing mode.
        
        Args:
            mode (str): Processing mode, either "chunk" or "step"
                - "chunk": Use parallel scan with cache as initial state (faster, but problematic for ar inference)
                - "step": Use step-by-step processing (slower, but guaranteed compatibility)
        """
        if mode not in ["chunk", "step"]:
            raise ValueError(f"Invalid mamba_cache_mode: {mode}. Must be 'chunk' or 'step'")
        
        self.mamba_cache_mode = mode
        rank_zero_print(f"Layer {self.layer_idx}: Mamba mode set to '{mode}'")
    
    def get_mamba_cache_mode(self) -> str:
        """Get current Mamba processing mode."""
        return self.mamba_cache_mode
    
    def get_context_length(self) -> int:
        """Get current context length (total processed tokens)."""
        return getattr(self, '_context_length', 0)
    
    def reset_context_length(self):
        """Reset context length counter."""
        if hasattr(self, '_context_length'):
            delattr(self, '_context_length')
    
    def enable_local_cache_saving(self, cache_dir: str = None):
        """Enable saving cache to local files before each update."""
        self.save_cache_locally = True
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Also enable for attention processor and sync step counters
        if self._supports_kv_cache():
            self.attn.processor.enable_local_cache_saving(str(self.cache_dir))
            # Sync step counters to ensure consistency
            self.attn.processor.cache_step_counter = self.cache_step_counter
    
    def disable_local_cache_saving(self):
        """Disable saving cache to local files."""
        self.save_cache_locally = False
        
        # Also disable for attention processor
        if self._supports_kv_cache():
            self.attn.processor.disable_local_cache_saving()
    
    def save_ssm_cache_to_file(self):
        """Save current SSM cache to local file."""
        if not self.save_cache_locally or self.inference_params is None:
            return
        
        try:
            # Get cache tensors from inference_params
            ssm_cache, conv_cache = self.inference_params.key_value_memory_dict[self.layer_idx]
            
            cache_data = {
                'ssm_states': ssm_cache.cpu(),
                'conv_states': conv_cache.cpu(),
                'step': self.cache_step_counter,
                'layer_idx': self.layer_idx,
                'seqlen_offset': self.inference_params.seqlen_offset,
                'shape_ssm': ssm_cache.shape,
                'shape_conv': conv_cache.shape,
                'dtype': str(ssm_cache.dtype),
            }
            
            # Create organized directory structure: cache_dir/ssm_cache/step_X/
            step_dir = self.cache_dir / "ssm_cache" / f"step_{self.cache_step_counter}"
            step_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"layer_{self.layer_idx}.pkl"
            filepath = step_dir / filename
            
            with open(filepath, 'wb') as f:
                pickle.dump(cache_data, f)
            rank_zero_print(f"SSM cache saved to {filepath}")
                
        except Exception as e:
            rank_zero_print(f"Failed to save SSM cache: {e}")
    
    def load_ssm_cache_from_file(self, step: int = None):
        """Load SSM cache from local file."""
        try:
            if step is None:
                step = self.cache_step_counter
            
            # Use organized directory structure: cache_dir/ssm_cache/step_X/
            step_dir = self.cache_dir / "ssm_cache" / f"step_{step}"
            filename = f"layer_{self.layer_idx}.pkl"
            filepath = step_dir / filename
            
            if not filepath.exists():
                rank_zero_print(f"SSM cache file {filepath} not found")
                return False
            
            with open(filepath, 'rb') as f:
                cache_data = pickle.load(f)
            
            if self.inference_params is not None:
                # Get current device
                device = next(self.parameters()).device
                
                # Get cache tensors from inference_params
                ssm_cache, conv_cache = self.inference_params.key_value_memory_dict[self.layer_idx]
                
                # Restore cache states
                ssm_cache.copy_(cache_data['ssm_states'].to(device))
                conv_cache.copy_(cache_data['conv_states'].to(device))
                
                # Restore sequence offset
                self.inference_params.seqlen_offset = cache_data['seqlen_offset']
                
                rank_zero_print(f"SSM cache loaded from {filepath}")
                return True
            
        except Exception as e:
            rank_zero_print(f"Failed to load SSM cache: {e}")
        
        return False


class CogVideoHybridSSM(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    A Transformer model for video-like data in [CogVideoX](https://github.com/THUDM/CogVideo).

    Parameters:
        num_attention_heads (`int`, defaults to `30`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `64`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `16`):
            The number of channels in the output.
        flip_sin_to_cos (`bool`, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        time_embed_dim (`int`, defaults to `512`):
            Output dimension of timestep embeddings.
        ofs_embed_dim (`int`, defaults to `512`):
            Output dimension of "ofs" embeddings used in CogVideoX-5b-I2B in version 1.5
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        attention_bias (`bool`, defaults to `True`):
            Whether to use bias in the attention projection layers.
        sample_width (`int`, defaults to `90`):
            The width of the input latents.
        sample_height (`int`, defaults to `60`):
            The height of the input latents.
        forward_window_size (`int`, defaults to `49`):
            The number of frames in the input latents. Note that this parameter was incorrectly initialized to 49
            instead of 13 because CogVideoX processed 13 latent frames at once in its default and recommended settings,
            but cannot be changed to the correct value to ensure backwards compatibility. To create a transformer with
            K latent frames, the correct value to pass here would be: ((K - 1) * temporal_compression_ratio + 1).
        patch_size (`int`, defaults to `2`):
            The size of the patches to use in the patch embedding layer.
        temporal_compression_ratio (`int`, defaults to `4`):
            The compression ratio across the temporal dimension. See documentation for `forward_window_size`.
        max_text_seq_length (`int`, defaults to `226`):
            The maximum sequence length of the input text embeddings.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        timestep_activation_fn (`str`, defaults to `"silu"`):
            Activation function to use when generating the timestep embeddings.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use elementwise affine in normalization layers.
        norm_eps (`float`, defaults to `1e-5`):
            The epsilon value to use in normalization layers.
        spatial_interpolation_scale (`float`, defaults to `1.875`):
            Scaling factor to apply in 3D positional embeddings across spatial dimensions.
        temporal_interpolation_scale (`float`, defaults to `1.0`):
            Scaling factor to apply in 3D positional embeddings across temporal dimensions.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["CogVideoXBlock", "CogVideoXPatchEmbed"]

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        external_cond_dim: int = 25,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 32,
        sample_height: int = 16,
        forward_window_size: int = 13,
        patch_size: int = 2,
        patch_size_t: int = None,
        temporal_compression_ratio: int = 1,
        spatial_vae_scale_factor: int = 8,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = True, # this will disable the position embedding in the patch embedding
        use_learned_positional_embeddings: bool = False,
        ff_mult: int = 4,
        trainable_params: Optional[List[str]] = None,
        # Mamba Block Parameters
        b_h_list: List[int] = [1, 1, 1, 1, 2, 2, 4, 8],
        b_w_list: List[int] = [1, 1, 1, 1, 2, 2, 4, 8],
        expand: int = 1,
        headdim: int = 48,
        d_state: int = 128,
        share_child: bool = True,
        ofs_embed_dim: Optional[int] = None,
        mamba_cache_mode: str = "step",  # "chunk" or "step"
        **kwargs,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        if not use_rotary_positional_embeddings and use_learned_positional_embeddings:
            raise ValueError(
                "There are no CogVideoX checkpoints available with disable rotary embeddings and learned positional "
                "embeddings. If you're using a custom model and/or believe this should be supported, please open an "
                "issue at https://github.com/huggingface/diffusers/issues."
            )

        self.external_cond_dim = external_cond_dim

        # 1. Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            external_cond_dim=external_cond_dim,
            external_cond_length=0,
            bias=True,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=forward_window_size,
            temporal_compression_ratio=temporal_compression_ratio,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings, # True
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )
        self.embedding_dropout = nn.Dropout(dropout)

        # 2. Time embeddings and ofs embedding(Only CogVideoX1.5-5B I2V have)

        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)

        self.agent_pos_embedding = AgentPositionEmbedding(input_dim = external_cond_dim, output_dim = time_embed_dim)
        self.pos_dropout = nn.Dropout(0.2)

        #### Didn't use this
        self.action_embedding = nn.Embedding(
            num_embeddings=external_cond_dim, 
            embedding_dim=128,
        )
        self.action_map = nn.Linear(2*128, time_embed_dim)
        #########

        self.ofs_proj = None
        self.ofs_embedding = None
        if ofs_embed_dim: 
            self.ofs_proj = Timesteps(ofs_embed_dim, flip_sin_to_cos, freq_shift)
            self.ofs_embedding = TimestepEmbedding(
                ofs_embed_dim, ofs_embed_dim, timestep_activation_fn
            )  # same as time embeddings, for ofs
      
        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    layer_idx=layer_idx,
                    sample_width=sample_width // patch_size,
                    sample_height=sample_height // patch_size,
                    ff_mult = ff_mult,
                    b_h=b_h_list[layer_idx],
                    b_w=b_w_list[layer_idx],
                    expand = expand,
                    headdim = headdim,
                    d_state = d_state,
                    share_child = share_child,
                    cache_dir="./cache_dumps",
                    mamba_cache_mode=mamba_cache_mode,
                )
                for layer_idx in range(num_layers)
            ]
        )

        
        self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)

        # 4. Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )

        if patch_size_t is None:
            # For CogVideox 1.0
            output_dim = patch_size * patch_size * out_channels
        else:
            # For CogVideoX 1.5
            output_dim = patch_size * patch_size * patch_size_t * out_channels

        self.proj_out = nn.Linear(inner_dim, output_dim)

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def get_fsdp_modules(self):
        return [CogVideoXBlock]

    def prepare_diffusion_forcing(self, text_emb, x_seqlen, x, b, f, motion_block_size):
        # batch_size x (image_embed + text_embed + x) x C
        # image_emb_seq_length = image_emb.shape[-2]

        text_emb_seq_length = text_emb.shape[-2]

        # pad to multiple of 128
        padded_length = text_emb_seq_length + x.shape[-2] 

        starts = torch.zeros(b, padded_length, device=x.device, dtype=torch.long)
        ends = torch.zeros(b, padded_length, device=x.device, dtype=torch.long)
        pad_ends = torch.zeros(b, device=x.device, dtype=torch.long)

        for i in range(b):
            # caculate the left padding length for each sample
            current_text_emb_seq_length = text_emb_seq_length

            attention_block_size = x[i].shape[0] // f * motion_block_size

            image_start_list = torch.arange(
                current_text_emb_seq_length,
                padded_length,
                step=attention_block_size, device=x.device
            )
            image_end_list = image_start_list + attention_block_size

            # embedding bidirectional blocks
            starts[i, :current_text_emb_seq_length] = 0
            ends[i, :current_text_emb_seq_length] = current_text_emb_seq_length

            # image bidirectional blocks
            for start, end in zip(image_start_list, image_end_list):
                starts[i, start:end] = torch.max(end - (10//motion_block_size)*attention_block_size, torch.tensor(text_emb_seq_length, device=image_start_list.device))
                ends[i, start:end] = end
        

        target_length = math.ceil(padded_length / 128) * 128
        padding_needed = target_length - padded_length
        starts = torch.nn.functional.pad(starts, (0, padding_needed))
        ends = torch.nn.functional.pad(ends, (0, padding_needed))


        # text is fully bidirectional
        # image can see all the text
        def images_mask(b, h, q_idx, kv_idx):
            text_emb_mask = kv_idx < text_emb_seq_length
            bidirectional_mask = (kv_idx < ends[b, q_idx]) & (kv_idx >= starts[b, q_idx])
            return bidirectional_mask | text_emb_mask

        block_mask = create_block_mask(images_mask, B=b, H=None, Q_LEN=padded_length,
                                       KV_LEN=padded_length, _compile=False, device=x.device)

        return block_mask

    def get_caches(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        block: CogVideoXBlock = self.transformer_blocks[layer_idx]
        if block.inference_params is not None:
            ssm_states, conv_states = block.inference_params.key_value_memory_dict[layer_idx]
        else:
            ssm_states, conv_states = None, None
            
        if block._supports_kv_cache() and block.attn.processor._cache_initialized:
            k_cache, v_cache = block.attn.processor.get_kv_cache()
        else:
            k_cache, v_cache = None, None
            
        return {
            "ssm_states": ssm_states,
            "conv_states": conv_states,
            "k_cache": k_cache,
            "v_cache": v_cache
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        conditions: torch.Tensor,
        # Not used
        timestep_cond: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        #########
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = False,
        motion_block_size: int = 5,
        is_loading_context: bool = False,
        do_update_cache: bool = False,
        **kwargs,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        enable_caches = kwargs.get("enable_caches", False)

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep.reshape(-1)
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)

        emb = self.time_embedding(t_emb, None).reshape(batch_size, -1, self.time_embedding.linear_2.out_features)

        # Action embedding
        actions = conditions

        # pos_emb = self.action_embedding(actions)
        pos_emb = self.agent_pos_embedding(actions)
        # pos_emb = pos_emb.view(pos_emb.shape[0], pos_emb.shape[1]//2, 2, -1).permute(0, 1, 3, 2).reshape(pos_emb.shape[0], pos_emb.shape[1]//2, -1)
        # pos_emb = self.action_map(pos_emb)
        emb = emb + pos_emb

        # we don't use encoder_hidden_states in this model, as we don't have text input; we rely one AdaLN to inject information from conditions
        encoder_hidden_states = torch.zeros((batch_size, 0, self.external_cond_dim), device=hidden_states.device, dtype=hidden_states.dtype)

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        block_mask = self.prepare_diffusion_forcing(encoder_hidden_states, hidden_states.shape[1], hidden_states, batch_size, num_frames, motion_block_size)

        # 3. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    attention_kwargs,
                    block_mask,
                    enable_caches,
                    motion_block_size,
                    is_loading_context,
                    do_update_cache,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                    block_mask = block_mask,
                    motion_block_size = motion_block_size,
                    enable_caches = enable_caches,
                    is_loading_context = is_loading_context,
                    do_update_cache = do_update_cache,
                )

        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        # 4. Final block
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(batch_size, num_frames, -1, hidden_states.shape[-1])
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = hidden_states.reshape(orig_shape)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        p = self.config.patch_size
        p_t = self.config.patch_size_t

        if p_t is None:
            output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return output
        return Transformer2DModelOutput(sample=output)
        # return DenoiserOutput(x=output, token_keep_mask=None)

    
    def clear_cache(self):
        """Clear SSM and KV caches for all transformer blocks."""
        for block in self.transformer_blocks:
            block.clear_cache()

    def reset_context_loading_flag(self):
        """Reset the context loading flag for all transformer blocks."""
        for block in self.transformer_blocks:
            block.reset_context_loading_flag()

    def set_context_loading_flag(self):
        """Set the context loading flag for all transformer blocks."""
        for block in self.transformer_blocks:
            block.set_context_loading_flag()
    
    def set_mamba_cache_mode(self, mode: str):
        """
        Set Mamba processing mode for all transformer blocks.
        
        Args:
            mode (str): Processing mode, either "chunk" or "step"
                - "chunk": Use parallel scan with cache as initial state (faster, but problematic for ar inference)
                - "step": Use step-by-step processing (slower, but guaranteed compatibility)
        """
        if mode not in ["chunk", "step"]:
            raise ValueError(f"Invalid mamba_cache_mode: {mode}. Must be 'chunk' or 'step'")
        
        for block in self.transformer_blocks:
            block.set_mamba_cache_mode(mode)
        
        rank_zero_print(f"All transformer blocks: Mamba mode set to '{mode}'")
    
    def get_mamba_cache_mode(self) -> str:
        """Get current Mamba processing mode (from first block)."""
        if self.transformer_blocks:
            return self.transformer_blocks[0].get_mamba_cache_mode()
        return "chunk"  # default
    
    def get_context_lengths(self) -> Dict[int, int]:
        """Get context length for each transformer block."""
        context_lengths = {}
        for block in self.transformer_blocks:
            context_lengths[block.layer_idx] = block.get_context_length()
        return context_lengths
    
    def reset_context_lengths(self):
        """Reset context length counters for all transformer blocks."""
        for block in self.transformer_blocks:
            block.reset_context_length()
    
    def enable_local_cache_saving(self, cache_dir: str = "./cache_dumps"):
        """Enable saving cache to local files for all transformer blocks."""
        for block in self.transformer_blocks:
            block.enable_local_cache_saving(cache_dir)
    
    def disable_local_cache_saving(self):
        """Disable saving cache to local files for all transformer blocks."""
        for block in self.transformer_blocks:
            block.disable_local_cache_saving()
    
    def save_all_caches_to_file(self):
        """Save all transformer block caches to local files."""
        for block in self.transformer_blocks:
            block.save_ssm_cache_to_file()
            if block._supports_kv_cache():
                block.attn.processor.save_kv_cache_to_file(f"layer_{block.layer_idx}")
    
    def load_all_caches_from_file(self, step: int = None):
        """Load all transformer block caches from local files."""
        success_count = 0
        for block in self.transformer_blocks:
            if block.load_ssm_cache_from_file(step):
                success_count += 1
            if block._supports_kv_cache():
                if block.attn.processor.load_kv_cache_from_file(f"layer_{block.layer_idx}", step):
                    success_count += 1
        
        rank_zero_print(f"Successfully loaded {success_count} cache files")
        
        return success_count > 0
    
    def list_available_cache_steps(self, cache_base_dir: str = "./cache_dumps"):
        """List all available cache steps in the cache directory."""
        cache_path = Path(cache_base_dir)
        
        ssm_steps = set()
        kv_steps = set()
        
        # Check SSM cache steps
        ssm_cache_dir = cache_path / "ssm_cache"
        if ssm_cache_dir.exists():
            for step_dir in ssm_cache_dir.iterdir():
                if step_dir.is_dir() and step_dir.name.startswith("step_"):
                    try:
                        step_num = int(step_dir.name.split("_")[1])
                        ssm_steps.add(step_num)
                    except (IndexError, ValueError):
                        continue
        
        # Check KV cache steps
        kv_cache_dir = cache_path / "kv_cache"
        if kv_cache_dir.exists():
            for step_dir in kv_cache_dir.iterdir():
                if step_dir.is_dir() and step_dir.name.startswith("step_"):
                    try:
                        step_num = int(step_dir.name.split("_")[1])
                        kv_steps.add(step_num)
                    except (IndexError, ValueError):
                        continue
        
        # Return steps that have both SSM and KV caches
        available_steps = sorted(ssm_steps.intersection(kv_steps))
        
        rank_zero_print(f"Available cache steps: {available_steps}")
        rank_zero_print(f"SSM cache steps: {sorted(ssm_steps)}")
        rank_zero_print(f"KV cache steps: {sorted(kv_steps)}")
        
        return available_steps
    
    def cleanup_old_cache_steps(self, cache_base_dir: str = "./cache_dumps", keep_latest: int = 5):
        """Clean up old cache steps, keeping only the latest N steps."""
        import shutil
        
        available_steps = self.list_available_cache_steps(cache_base_dir)
        
        if len(available_steps) <= keep_latest:
            rank_zero_print(f"No cleanup needed. Current steps: {len(available_steps)}, keep_latest: {keep_latest}")
            return
        
        steps_to_remove = available_steps[:-keep_latest]
        cache_path = Path(cache_base_dir)
        
        removed_count = 0
        for step in steps_to_remove:
            # Remove SSM cache step directory
            ssm_step_dir = cache_path / "ssm_cache" / f"step_{step}"
            if ssm_step_dir.exists():
                shutil.rmtree(ssm_step_dir)
                removed_count += 1
            
            # Remove KV cache step directory
            kv_step_dir = cache_path / "kv_cache" / f"step_{step}"
            if kv_step_dir.exists():
                shutil.rmtree(kv_step_dir)
                removed_count += 1
        
        rank_zero_print(f"Cleaned up {len(steps_to_remove)} old cache steps, removed {removed_count} directories")
        rank_zero_print(f"Kept latest {keep_latest} steps: {available_steps[-keep_latest:]}")