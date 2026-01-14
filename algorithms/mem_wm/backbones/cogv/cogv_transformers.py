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

from typing import Any, Dict, Optional, Tuple, Union
from typing import List, Optional
import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from .attention_processor import AttentionProcessor, CogVideoXAttnProcessor2_0
from .embedding import CogVideoXPatchEmbed, CogVideoXFrameEmbed, TimestepEmbedding, Timesteps
from .normalization import AdaLayerNorm, CogVideoXLayerNormZero
from .utils import prepare_rotary_positional_embeddings
import lightning as L
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

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
        attention_out_bias: bool = True,
        layer_idx: int = None,
    ):
        super().__init__()

        self.layer_idx = layer_idx

        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)
        
        
        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0(),
        )

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.ff = FeedForward(
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
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        batch_size, fhw, dim = hidden_states.shape
        batch_size_t, f, dim2 = temb.shape

        external_cond_length = encoder_hidden_states.size(1)

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        # attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states, # (bs, fhw, dim)
            encoder_hidden_states=norm_encoder_hidden_states, # (bs, f, dim)
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
        )

        hidden_states = hidden_states + (gate_msa.unsqueeze(2) * attn_hidden_states.view(batch_size, f, -1, dim)).view(batch_size, fhw, dim)
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa[:,-1:] * attn_encoder_hidden_states

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + (gate_ff.unsqueeze(2) * ff_output[:, external_cond_length:].view(batch_size, f, -1, dim)).view(batch_size, fhw, dim)
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff[:,-1:] * ff_output[:, :external_cond_length]

        return hidden_states, encoder_hidden_states

class CogVideoXTransformer3DModel(L.LightningModule, ModelMixin, ConfigMixin, PeftAdapterMixin):
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
        external_cond_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        num_layers (`int`, defaults to `30`):
            The number of layers of Transformer blocks to use.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        attention_bias (`bool`, defaults to `True`):
            Whether or not to use bias in the attention projection layers.
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
        max_external_cond_length (`int`, defaults to `226`):
            The maximum sequence length of the input text embeddings.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        timestep_activation_fn (`str`, defaults to `"silu"`):
            Activation function to use when generating the timestep embeddings.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether or not to use elementwise affine in normalization layers.
        norm_eps (`float`, defaults to `1e-5`):
            The epsilon value to use in normalization layers.
        spatial_interpolation_scale (`float`, defaults to `1.875`):
            Scaling factor to apply in 3D positional embeddings across spatial dimensions.
        temporal_interpolation_scale (`float`, defaults to `1.0`):
            Scaling factor to apply in 3D positional embeddings across temporal dimensions.
    """

    _supports_gradient_checkpointing = True

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
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        trainable_params: Optional[List[str]] = None,
        use_depth: bool = False,
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
        self.use_rotary_positional_embeddings = use_rotary_positional_embeddings
        self.use_learned_positional_embeddings = use_learned_positional_embeddings
        self.RoPE_BASE_HEIGHT = sample_height * spatial_vae_scale_factor
        self.RoPE_BASE_WIDTH = sample_width * spatial_vae_scale_factor
        self.spatial_vae_scale_factor = spatial_vae_scale_factor
        self.patch_size_t = patch_size_t
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.forward_window_size = forward_window_size

        # 1. Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            external_cond_dim=external_cond_dim,
            bias=True,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=forward_window_size,
            temporal_compression_ratio=temporal_compression_ratio,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )
        self.embedding_dropout = nn.Dropout(dropout)

        # 2. Time embeddings
        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)

        
        block_class = CogVideoXBlock

        # 3. Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                block_class(
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
                )
                for layer_idx in range(num_layers)
            ]
        )

        self.use_depth = use_depth

        if self.use_depth:
            # if using depth, we treat depthmap as an RGB image
            self.depth_patch_embed = CogVideoXFrameEmbed(
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=inner_dim,
                external_cond_dim=external_cond_dim,
                bias=True,
                sample_width=sample_width,
                sample_height=sample_height,
                sample_frames=forward_window_size,
                temporal_compression_ratio=temporal_compression_ratio,
                spatial_interpolation_scale=spatial_interpolation_scale,
                temporal_interpolation_scale=temporal_interpolation_scale,
                use_positional_embeddings=not use_rotary_positional_embeddings,
                use_learned_positional_embeddings=use_learned_positional_embeddings,
            )
            self.depth_blending_linear = nn.Linear(inner_dim, inner_dim) # need to be initialized from zeros

        self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)

        # 4. Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=-1,
        )
        if self.use_depth:
            self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels * 2)
        else:
            self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels)

        self.gradient_checkpointing = False
        self.initialize_weights()

        if trainable_params is not None:
            self.freeze_parameters(trainable_params)

    def initialize_weights(self):
        # Initialize patch embedding
        if hasattr(self.patch_embed, "proj"):
            nn.init.normal_(self.patch_embed.proj.weight, std=0.02)
            if self.patch_embed.proj.bias is not None:
                nn.init.zeros_(self.patch_embed.proj.bias)
        
        if self.use_depth:
            nn.init.normal_(self.depth_patch_embed.proj.weight, std=0.02)
            if self.depth_patch_embed.proj.bias is not None:
                nn.init.zeros_(self.depth_patch_embed.proj.bias)
            
            if hasattr(self.depth_patch_embed, "position_embeddings") and self.depth_patch_embed.position_embeddings is not None:
                nn.init.normal_(self.depth_patch_embed.position_embeddings, std=0.02)
        
        # Initialize positional embeddings if they exist
        if hasattr(self.patch_embed, "position_embeddings") and self.patch_embed.position_embeddings is not None:
            nn.init.normal_(self.patch_embed.position_embeddings, std=0.02)
            
        # Initialize time embedding
        nn.init.normal_(self.time_embedding.linear_1.weight, std=0.02)
        nn.init.zeros_(self.time_embedding.linear_1.bias)
        nn.init.normal_(self.time_embedding.linear_2.weight, std=0.02)
        nn.init.zeros_(self.time_embedding.linear_2.bias)

        # Initialize transformer blocks
        self._initialize_blocks(self.transformer_blocks)
                
        # Initialize final normalization and output projection
        # AdaLayerNorm has a linear layer, not adaLN_modulation
        nn.init.normal_(self.norm_out.linear.weight, std=0.02)
        nn.init.zeros_(self.norm_out.linear.bias)
        nn.init.normal_(self.proj_out.weight, std=0.02)
        if self.proj_out.bias is not None:
            nn.init.zeros_(self.proj_out.bias)
        
        if self.use_depth:
            # initialize blending linear from zeros, do controlnet cold start
            nn.init.zeros_(self.depth_blending_linear.weight)
            nn.init.zeros_(self.depth_blending_linear.bias)
    
    def freeze_parameters(self, trainable_params: List[str]):

        for name, param in self.named_parameters():
            if any(x in name for x in trainable_params):
                # print(f"Training {name}")
                param.requires_grad = True
            else:
                # print(f"Freezing {name}")
                param.requires_grad = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def _initialize_blocks(self, blocks):
        for block in blocks:
            # Initialize attention weights
            if hasattr(block.attn1, "to_q"):
                nn.init.normal_(block.attn1.to_q.weight, std=0.02)
                nn.init.normal_(block.attn1.to_k.weight, std=0.02)
                nn.init.normal_(block.attn1.to_v.weight, std=0.02)
                nn.init.normal_(block.attn1.to_out[0].weight, std=0.02)
                if block.attn1.to_out[0].bias is not None:
                    nn.init.zeros_(block.attn1.to_out[0].bias)
            
            # Initialize feedforward weights - find linear layers directly
            for module in block.ff.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            
            # Initialize norm weights - CogVideoXLayerNormZero has a linear layer
            if hasattr(block.norm1, "linear"):
                nn.init.normal_(block.norm1.linear.weight, std=0.02)
                nn.init.zeros_(block.norm1.linear.bias)
            
            if hasattr(block.norm2, "linear"):
                nn.init.normal_(block.norm2.linear.weight, std=0.02)
                nn.init.zeros_(block.norm2.linear.bias)
    

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
        
    
    def apply_cfg_dropout(
        self,
        tensor: torch.Tensor, 
        cfg_drop_prob: float, 
        dropping_strategy: str = "batch_wise", 
        batch_size: Optional[int] = None, 
        num_frames: Optional[int] = None
    ) -> torch.Tensor:
        """
        Helper function to apply classifier-free guidance dropout to a tensor.
        
        Args:
            tensor: The tensor to apply dropout to.
            cfg_drop_prob: Probability of dropping (zeroing) values.
            dropping_strategy: Strategy for dropping - "batch_wise", "frame_wise", or "no-dropping".
            batch_size: Batch size (required if not inferrable from tensor).
            num_frames: Number of frames (required for frame_wise if not inferrable).
        
        Returns:
            Tensor with dropout applied according to the specified strategy.
        """
        if cfg_drop_prob <= 0 or dropping_strategy == "no-dropping":
            return tensor
        
        # Infer batch_size if not provided
        if batch_size is None:
            batch_size = tensor.shape[0]
        
        if dropping_strategy == "batch_wise":
            mask_shape = (batch_size,) + (1,) * (tensor.ndim - 1)
            mask = torch.bernoulli(torch.ones(mask_shape, device=tensor.device) * (1 - cfg_drop_prob))
            return tensor * mask.to(dtype=tensor.dtype)
        
        elif dropping_strategy == "frame_wise":
            if num_frames is None:
                # Try to infer num_frames from tensor shape
                if tensor.ndim >= 2:
                    # Assume second dimension is frames if tensor is at least 2D
                    num_frames = tensor.shape[1]
                else:
                    raise ValueError("num_frames must be provided for frame_wise dropping when not inferrable from tensor")
            
            mask_shape = (batch_size, num_frames) + (1,) * (tensor.ndim - 2)
            mask = torch.bernoulli(torch.ones(mask_shape, device=tensor.device) * (1 - cfg_drop_prob))
            return tensor * mask.to(dtype=tensor.dtype)
        
        else:
            raise ValueError(f"Invalid cfg_dropping_strategy: {dropping_strategy}")


    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],  # [B, ...]
        encoder_hidden_states: torch.Tensor,
        timestep_cond: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        bs = hidden_states.shape[0]
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

        batch_size, num_frames, channels, height, width = hidden_states.shape
        cfg_dropping_strategy = kwargs.get("cfg_dropping_strategy", "batch_wise") # "frame_wise"

        if image_rotary_emb is None: # image_rotary_emb should be None: https://huggingface.co/MatrixTeam/TheMatrix/blob/main/stage3/transformer/config.json
            image_rotary_emb = (
                prepare_rotary_positional_embeddings(
                    height=height * self.spatial_vae_scale_factor,
                    width=width * self.spatial_vae_scale_factor,
                    num_frames=num_frames,
                    vae_scale_factor_spatial=self.spatial_vae_scale_factor,
                    patch_size=self.patch_size,
                    patch_size_t=self.patch_size_t if hasattr(self, "patch_size_t") else None,
                    attention_head_dim=self.attention_head_dim,
                    device=self.device,
                    base_height=self.RoPE_BASE_HEIGHT,
                    base_width=self.RoPE_BASE_WIDTH,
                )
                if self.use_rotary_positional_embeddings
                else None
            )
        # 1. Time embedding
        if len(timestep.size()) == 1:
            timesteps = timestep.unsqueeze(1)
        else:
            timesteps = timestep
        B, F = timesteps.size()[:2]
        
        
        # Flatten timesteps for embedding, then reshape back to [B, F, embed_dim]
        timesteps = timesteps.flatten(0)  # [B*F]
        t_emb = self.time_proj(timesteps)  # [B*F, time_embed_dim]d

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)  # [B*F, embed_dim]
        emb = emb.view(B, F, -1)  # Reshape back to [B, F, embed_dim] - phrase structure preserved

        cfg_drop_prob = 0.1
        if self.training and cfg_drop_prob > 0:
            encoder_hidden_states = self.apply_cfg_dropout(tensor=encoder_hidden_states, cfg_drop_prob=cfg_drop_prob, dropping_strategy=cfg_dropping_strategy, batch_size=bs, num_frames=num_frames)

        external_cond_length = encoder_hidden_states.shape[1]

        # 2. Patch embedding
        if self.use_depth:
            hidden_states, depth_hidden_states = hidden_states.split([channels // 2, channels // 2], dim=2)
            depth_hidden_states = self.depth_patch_embed(depth_hidden_states)
            depth_hidden_states = self.embedding_dropout(depth_hidden_states)
        
        
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)
        encoder_hidden_states = hidden_states[:, :external_cond_length]
        hidden_states = hidden_states[:, external_cond_length:]

        if self.use_depth:
            hidden_states = hidden_states + self.depth_blending_linear(depth_hidden_states)

        bs, fhw, dim = hidden_states.shape
        
        
        
        # 3. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

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
                    attention_mask,
                    **ckpt_kwargs,
                )
            else:
                kwargs = {}

                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                    attention_mask=attention_mask,
                    **kwargs,
                )


        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, external_cond_length:]

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        # Note: we use `-1` instead of `channels`:
        #   - It is okay to `channels` use for CogVideoX-2b and CogVideoX-5b (number of input channels is equal to output channels)
        #   - However, for CogVideoX-5b-I2V also takes concatenated input image latents (number of input channels is twice the output channels)
        p = self.config.patch_size
        output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
        # after permute: shape is [b, f, c, h_p, p, w_p, p] -> flatten to [b, f, c, h_p * p, w_p * p]
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return output
        return Transformer2DModelOutput(sample=output)

