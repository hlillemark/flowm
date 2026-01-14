# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from __future__ import annotations
import inspect
import math
from typing import Callable, List, Optional, Tuple, Union
import os
from functools import partial
import math
from typing import Optional
import time
import torch
import torch.nn.functional as F
from torch import nn
import torch._dynamo
# torch._dynamo.disable()

from diffusers.utils import deprecate, logging
from diffusers.models.attention_processor import AttentionProcessor
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
from diffusers.models.attention_processor import Attention

class CogVideoXAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # batch_size, sequence_length, _ = (
        #     hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        # )
        batch_size, sequence_length, _ = hidden_states.shape

        if attention_mask is not None:
            # attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size) # this is incorrect when doing self-attention
            # attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
            # (bs, seq_len, seq_len) -> (bs, attn_heads, seq_len, seq_len)
            if attention_mask.dim() == 2:
                # (bs, seq) -> (bs, 1, 1, seq)
                attention_mask = attention_mask[:, None, None, :]
                attention_mask = attention_mask.expand(batch_size, attn.heads, sequence_length, sequence_length)
            elif attention_mask.dim() == 3:
                # (bs, seq, seq) -> (bs, 1, seq, seq)
                attention_mask = attention_mask.unsqueeze(1).expand(batch_size, attn.heads, sequence_length, sequence_length)
            elif attention_mask.dim() == 4:
                # already in (bs, heads, seq, seq) form
                pass
            else:
                raise ValueError(f"Unexpected attention_mask shape: {attention_mask.shape}")
            
        query = attn.to_q(hidden_states)
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

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


class FusedCogVideoXAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        qkv = attn.to_qkv(hidden_states)
        split_size = qkv.shape[-1] // 3
        query, key, value = torch.split(qkv, split_size, dim=-1)

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

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states



################################################################################
#       Not using, as this is slower than sdpa
################################################################################

class LocalFrameAttentionWithDiffuser(nn.Module):
    """Chunked frame-local attention via Diffuser's Attention processor."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        *,
        chunk_size: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        
        self.attn = Attention(query_dim=dim, heads=num_heads, dim_head=head_dim, dropout=dropout)
        self.processor = CogVideoXAttnProcessor2_0()
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F, N, D]
        B, F_, N, D = x.shape
        assert F_ % self.chunk_size == 0, "F must be divisible by chunk_size"
        C = F_ // self.chunk_size
        out_chunks = []

        for i in range(C):
            # compute window frames for this chunk
            start = i * self.chunk_size
            prev_start = max(0, (i - 1) * self.chunk_size)
            window = x[:, prev_start : start + self.chunk_size]  # [B, Wf, N, D]
            Wf = window.shape[1]  # either chunk_size or 2*chunk_size
            total_tokens = Wf * N  # total tokens in the window

            flat = window.reshape(B, total_tokens, D)  # [B, T, D]

            # build mask
            cutoff = max(0, Wf - self.chunk_size) * N
            q_indices = torch.arange(total_tokens, device=x.device)
            kv_indices = torch.arange(total_tokens, device=x.device)
            
            q_in_curr = q_indices[:, None] >= cutoff
            kv_in_curr = kv_indices[None, :] >= cutoff
            kv_in_prev = kv_indices[None, :] < cutoff
            q_in_prev = q_indices[:, None] < cutoff

            # For F.scaled_dot_product_attention, True means "don't attend"
            attn_mask_bool = ~((q_in_curr & (kv_in_curr | kv_in_prev)) | (q_in_prev & kv_in_prev))
            attn_mask = attn_mask_bool.unsqueeze(0).expand(B, -1, -1)

            encoder_hidden_states = torch.empty(B, 0, D, device=x.device, dtype=x.dtype)
            
            out, _ = self.processor(
                self.attn,
                hidden_states=flat,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attn_mask,
                image_rotary_emb=None
            )

            # select only current chunk outputs
            start_idx = cutoff
            end_idx = cutoff + self.chunk_size * N
            chunk = out[:, start_idx:end_idx]  # [B, chunk_size*N, D]
            out_chunks.append(chunk.reshape(B, self.chunk_size, N, D))

        # concatenate all chunk results along frame dim
        return torch.cat(out_chunks, dim=1)

################################################################################


########################################
#       Deprecated FlexAttention

# It's strangely slow; I tried this compiled version, but it's still slow.
########################################
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

def _compile_flex():
    """Return a compiled wrapper around `flex_attention`.

    We compile a tiny wrapper to keep the graph small and avoid re-compiling the
    whole module. This assumes fairly static shapes for (B, H, T, Hd).
    """
    def _flex_call(q, k, v, block_mask):
        return flex_attention(q, k, v, block_mask=block_mask)

    try:
        compiled = torch.compile(_flex_call)
    except Exception:
        print("Failed to compile FlexAttention")
        # Fallback to eager if compile is unavailable (e.g., CPU or old PyTorch).
        compiled = _flex_call
    return compiled


class FlexLocalFrameAttention(nn.Module):
    """Chunked frame‑local attention implemented with FlexAttention.

    Input/Output: [B, F, N, D]
    - F: number of frames
    - N: tokens per frame (e.g., spatial tokens)
    - D: model dim
    - Each chunk has `chunk_size` frames. For chunk *i*, queries from the current
      chunk attend to keys/values from (prev_chunk + current_chunk). Queries that
      belong to the prev_chunk only attend within the prev_chunk (causal across
      chunks).

    This mirrors the windowing described in (prev + curr) with selection of the
    current chunk outputs, matching the SDPA reference.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        *,
        chunk_size: int = 4,
        compile_flex: bool = True,
        block_size_override: Optional[int] = None,
        warmup_on_init: bool = False,
        warmup_device: Optional[torch.device] = None,
        warmup_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        inner = num_heads * head_dim

        self.qkv_proj = nn.Linear(dim, inner * 3, bias=False)
        self.out_proj = nn.Linear(inner, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Compiled flex call
        self._flex_call = _compile_flex() if compile_flex else (lambda q, k, v, m: flex_attention(q, k, v, block_mask=m))

        # BLOCK_SIZE for create_block_mask; default to N at runtime if None
        self.block_size_override = block_size_override

        if warmup_on_init:
            # Lazily warm up for the two window sizes: chunk and 2*chunk
            dev = warmup_device if warmup_device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dt = warmup_dtype if warmup_dtype is not None else torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self._warmup(dev, dt)

    # ---- mask logic -----------------------------------------------------
    @staticmethod
    def _mask_mod_fn(batch, head, q_idx, kv_idx, cutoff: int):
        # True => allow attention
        q_in_curr = q_idx >= cutoff
        kv_in_curr = kv_idx >= cutoff
        kv_in_prev = kv_idx < cutoff
        q_in_prev = q_idx < cutoff
        return (q_in_curr & (kv_in_curr | kv_in_prev)) | (q_in_prev & kv_in_prev)

    def _build_block_mask(self, B: int, H: int, Wf: int, N: int, device: torch.device):
        """Create BlockMask for a given window of Wf frames and N tokens per frame.
        cutoff is where the current chunk starts inside the window.
        """
        T = Wf * N
        # This is the TOKEN-level cutoff, used for slicing the output tensor.
        token_cutoff = max(0, Wf - self.chunk_size) * N
        # This is the FRAME-level cutoff, used for the mask logic.
        frame_cutoff = max(0, Wf - self.chunk_size)
        
        mask_mod = partial(self._mask_mod_fn, cutoff=frame_cutoff)
        block_sz = self.block_size_override if self.block_size_override is not None else N
        return (
            create_block_mask(
                mask_mod,
                B,
                H,
                T,
                T,
                device=device,
                BLOCK_SIZE=block_sz,
            ),
            token_cutoff, # Return the token-level cutoff for slicing
        )

    # ---- forward --------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, F, N, D] -> same shape"""
        B, F_, N, D = x.shape
        assert D == self.dim, f"Expected dim={self.dim}, got {D}"
        assert F_ % self.chunk_size == 0, "F must be divisible by chunk_size"
        H, Hd = self.num_heads, self.head_dim
        C = F_ // self.chunk_size

        out_chunks = []
        for i in range(C):
            start = i * self.chunk_size
            prev_start = max(0, (i - 1) * self.chunk_size)
            window = x[:, prev_start : start + self.chunk_size]  # [B, Wf, N, D]
            Wf = window.shape[1]
            T = Wf * N

            # QKV projection
            flat = window.reshape(B, T, D)
            qkv = self.qkv_proj(flat)
            qkv = qkv.view(B, T, 3, H, Hd).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, T, Hd]

            # Block mask
            block_mask, token_cutoff = self._build_block_mask(B, H, Wf, N, x.device)

            # FlexAttention
            attn_out = self._flex_call(q, k, v, block_mask)  # [B, H, T, Hd]

            # Merge heads
            out = attn_out.permute(0, 2, 1, 3).reshape(B, T, H * Hd)
            out = self.dropout(self.out_proj(out))  # [B, T, D]

            # only current chunk
            start_idx = token_cutoff
            end_idx = token_cutoff + self.chunk_size * N
            chunk = out[:, start_idx:end_idx].reshape(B, self.chunk_size, N, D)
            out_chunks.append(chunk)

        return torch.cat(out_chunks, dim=1)

    # ---- optional warmup ------------------------------------------------
    @torch.no_grad()
    def _warmup(self, device: torch.device, dtype: torch.dtype):
        """Compile warm‑up for two window sizes: chunk_size and 2*chunk_size.
        Creates small dummy inputs and runs once so later calls reuse kernels.
        """
        B = 1
        H = self.num_heads
        Hd = self.head_dim
        D = self.dim
        N = 16  # a typical per-frame token count; compile will re-trigger if actual N differs
        # If you want to avoid recompile, set this to your expected N.

        for Wf in {self.chunk_size, 2 * self.chunk_size}:
            T = Wf * N
            q = torch.randn(B, H, T, Hd, device=device, dtype=dtype)
            k = torch.randn(B, H, T, Hd, device=device, dtype=dtype)
            v = torch.randn(B, H, T, Hd, device=device, dtype=dtype)
            block_mask, _ = self._build_block_mask(B, H, Wf, N, device)
            # run once
            _ = self._flex_call(q, k, v, block_mask)

########################################
#       Deprecated FlexAttention
########################################


################################################################################
# START      Correct FlexAttention from Author
################################################################################

class FlexAttnProcessor:
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.

    with credits to [Ryan Lok Him Po](https://t.co/Fgk0ydyxNH) 
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        block_mask = None
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
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


        hidden_states = flex_attention(query.to(torch.bfloat16), key.to(torch.bfloat16), value.to(torch.bfloat16), block_mask=block_mask)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


################################################################################
# END       Correct FlexAttention from Author
################################################################################



class LocalFrameAttention(nn.Module):
    """
    Chunked frame-local attention via F.scaled_dot_product_attention.
    arxiv: https://www.arxiv.org/pdf/2505.20171, see Appendix S.2
    
    Frame-local attention with **chunk-causal** structure and **bidirectional** attention within each chunk.

    This module operates on video tokens shaped as ``[B, F, N, D]`` where ``F`` is the number of frames
    and ``N`` is the number of tokens per frame (e.g., a spatial grid). Frames are grouped into chunks of
    size ``chunk_size = k // 2`` (so a "window" spans ``k`` frames as **prev + curr = (k/2) + (k/2)``).

    **Attention pattern (per window):**
      * For chunk index ``i``, form a window consisting of the **previous chunk** (if ``i>0``) and the
        **current chunk**. Let the window length in frames be ``Wf ∈ {chunk_size, 2*chunk_size}`` and
        the window length in tokens be ``T = Wf * N``.
      * Define a token cutoff ``cutoff = max(0, Wf - chunk_size) * N`` that separates previous-chunk
        tokens ``[0 : cutoff)`` from current-chunk tokens ``[cutoff : T)``.
      * **Queries in the previous chunk** (indices ``< cutoff``) may attend **only to previous-chunk tokens**
        (bidirectional within the previous chunk, but **cannot** see the current chunk).
      * **Queries in the current chunk** (indices ``≥ cutoff``) may attend to **both** previous-chunk **and**
        current-chunk tokens (bidirectional within the current chunk and causal across chunks).
      * Only the outputs corresponding to the **current chunk** are kept; previous-chunk outputs inside the
        window are discarded. Concatenating all kept outputs over chunks restores the original shape.

    **Training note (prefix sampling; Section 4.2 & App. S.2 idea):**
      * For long videos, sample a random **prefix** (kept unnoised) with probability ``p=0.5``; its length must
        exceed **half** the training sequence length to encourage long-context learning. The remaining frames
        (the "denoising frames") are noised. When no prefix is sampled, all tokens are noised (equivalent to
        diffusion forcing).
      * With ``k=10`` and ``chunk_size=5``, the mask simulates denoising the current 5-frame chunk conditioned
        on the previous 5-frame context (bidirectional within each 5-frame chunk, causal between the two).

    Args:
        dim (int): Model embedding dimension ``D``.
        num_heads (int): Number of attention heads ``H``.
        head_dim (int): Per-head dimension ``Hd`` such that ``num_heads * head_dim == dim``.
        dropout (float, optional): Dropout after the output projection. Default: ``0.0``.
        chunk_size (int, optional): Frames per chunk (i.e., ``k/2``). The full window size is ``2*chunk_size``.
            Must divide ``F`` in the simple chunked implementation.

    Shapes:
        * Input:  ``x`` of shape ``[B, F, N, D]``.
        * Output: same shape as input, ``[B, F, N, D]``.

    Implementation notes:
        Need to update this when the implementation is fixed
        * **SDPA path**: uses a boolean mask where ``True`` means *disallow* (PyTorch convention).
          A one-shot implementation can build a block-lower-triangular band mask that permits
          (prev + curr) for current-chunk queries and only prev for previous-chunk queries, then call
          ``torch.nn.functional.scaled_dot_product_attention`` **once** over the full sequence.
        * **FlashAttention**: vanilla ``flash_attn_func`` does **not** accept arbitrary masks. To preserve the
          asymmetric rule in a Flash backend you must either (a) make **two calls** per window (split queries into
          prev and curr sets) and stitch, or (b) use a block-sparse FlashAttention kernel that encodes the
          block-band/lower-triangular layout. Passing ``window_size`` alone is insufficient to encode this asymmetry.
        * If strict autoregressive order **inside** a chunk is desired, add a per-chunk lower-triangular submask;
          by default this module is **bidirectional within a chunk**.

    Returns:
        Tensor: Attention outputs with the same shape as the input, ``[B, F, N, D]``.
    
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        *,
        chunk_size: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size # the size of the chunk for bidirectional / causal split
        inner = num_heads * head_dim
        self.qkv_proj = nn.Linear(dim, inner * 3, bias=False)
        self.out_proj = nn.Linear(inner, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, F, N, D]
        B, F_, N, D = x.shape
        if F_ % self.chunk_size != 0:
            raise ValueError(f"F must be divisible by chunk_size. Got F={F_}, chunk_size={self.chunk_size}.")
        C = F_ // self.chunk_size
        out_chunks = []

        for i in range(C):
            # compute window frames for this chunk
            start = i * self.chunk_size
            prev_start = max(0, (i - 1) * self.chunk_size)
            window = x[:, prev_start : start + self.chunk_size]  # [B, Wf, N, D]
            Wf = window.shape[1]  # either chunk_size for first iteration, or 2*chunk_size for all subsequent ones
            total_tokens = Wf * N  # total tokens in the window

            # project to QKV
            flat = window.reshape(B, total_tokens, D)  # [B, T, D]
            qkv = self.qkv_proj(flat)
            # reshape to (B, T, 3, H, Hd)
            qkv = qkv.view(B, total_tokens, 3, self.num_heads, self.head_dim)
            # permute to (3, B, H, T, Hd)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, H, total_tokens, Hd]

            
            cutoff = max(0, Wf - self.chunk_size) * N
            
            # build mask
            q_indices = torch.arange(total_tokens, device=x.device)
            kv_indices = torch.arange(total_tokens, device=x.device)
            
            # first dimension is q, second dimension is k
            q_in_curr = q_indices[:, None] >= cutoff
            kv_in_curr = kv_indices[None, :] >= cutoff
            q_in_prev = q_indices[:, None] < cutoff
            kv_in_prev = kv_indices[None, :] < cutoff

            # For F.scaled_dot_product_attention, True means "don't attend"
                # Ends up as this for the chunk size = 2
                # [0 0 1 1]
                # [0 0 1 1]
                # [0 0 0 0]
                # [0 0 0 0]
            attn_mask = ~((q_in_curr & (kv_in_curr | kv_in_prev)) | (q_in_prev & kv_in_prev))

            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
            )

            # merge heads
            out = attn_out.permute(0, 2, 1, 3).reshape(B, total_tokens, self.num_heads * self.head_dim)
            out = self.dropout(self.out_proj(out))  # [B, total_tokens, D]

            # select only current chunk outputs
            start_idx = cutoff
            end_idx = cutoff + self.chunk_size * N
            chunk = out[:, start_idx:end_idx]  # [B, chunk_size*N, D]
            out_chunks.append(chunk.reshape(B, self.chunk_size, N, D))

        # concatenate all chunk results along frame dim
        return torch.cat(out_chunks, dim=1)

########################################
#       Benchmarking
########################################


def benchmark_attentions(num_frames: int = 400):
    """Mock a N video and benchmark speed + peak GPU memory of vanilla vs. LocalFrameAttention."""
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    dtype = torch.float16 if device == "cuda" else torch.float32

    # ----- Synthetic video -----
    B, F, H, W, D = 1, num_frames, 16, 16, 1280  # → 256 tokens/frame
    N = H * W
    
    # the total tokens
    print(f"Total tokens: {B * F * N // 1000}k")

    x = torch.randn(B, F, N, D, device=device, dtype=dtype)

    num_heads = 20

    # ------------------------------------------------------------------
    # Vanilla full‑sequence MultiheadAttention (for baseline)
    # ------------------------------------------------------------------
    try:
        seq = x.reshape(B, F * N, D)  # [B, T, D]
        mha = nn.MultiheadAttention(D, num_heads, batch_first=True, device=device, dtype=dtype)

        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        with torch.no_grad():
            y_full, _ = mha(seq, seq, seq)
        if device == "cuda":
            torch.cuda.synchronize()
            vanilla_mem = torch.cuda.max_memory_allocated(device) / 1024**2
        else:
            vanilla_mem = 0.0
        vanilla_time = (time.perf_counter() - start) * 1000  # ms
    except Exception as e:
        print(f"Error in vanilla attention: {e}")
        import traceback
        traceback.print_exc()
        vanilla_time = float('inf')
        vanilla_mem = float('inf')

    # ------------------------------------------------------------------
    # Attention from diffuser
    # ------------------------------------------------------------------
    try:
        seq = x.reshape(B, F * N, D)
        attn = Attention(query_dim=D, heads=num_heads, dim_head=D // num_heads).to(device).to(dtype)
        processor = CogVideoXAttnProcessor2_0()

        encoder_hidden_states = torch.empty(B, 0, D, device=device, dtype=dtype)

        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        with torch.no_grad():
            y_diffuser, _ = processor(
                attn=attn,
                hidden_states=seq,
                encoder_hidden_states=encoder_hidden_states,
            )
        if device == "cuda":
            torch.cuda.synchronize()
            diffuser_mem = torch.cuda.max_memory_allocated(device) / 1024**2
        else:
            diffuser_mem = 0.0
        diffuser_time = (time.perf_counter() - start) * 1000  # ms
    except Exception as e:
        print(f"Error in diffuser attention: {e}")
        import traceback
        traceback.print_exc()
        diffuser_time = float('inf')
        diffuser_mem = float('inf')

    # ------------------------------------------------------------------
    # Attention from diffuser with Flash Attention
    # ------------------------------------------------------------------
    try:
        seq = x.reshape(B, F * N, D)
        attn = Attention(query_dim=D, heads=num_heads, dim_head=D // num_heads).to(device).to(dtype)
        processor = CogVideoXAttnProcessor2_0()

        encoder_hidden_states = torch.empty(B, 0, D, device=device, dtype=dtype)

        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            y_flash, _ = processor(
                attn=attn,
                hidden_states=seq,
                encoder_hidden_states=encoder_hidden_states,
            )
        if device == "cuda":
            torch.cuda.synchronize()
            flash_mem = torch.cuda.max_memory_allocated(device) / 1024**2
        else:
            flash_mem = 0.0
        flash_time = (time.perf_counter() - start) * 1000  # ms
    except Exception as e:
        print(f"Error in diffuser with flash attention: {e}")
        import traceback
        traceback.print_exc()
        flash_time = float('inf')
        flash_mem = float('inf')


    # ------------------------------------------------------------------
    # FlexLocalFrameAttention
    # ------------------------------------------------------------------
    try:
        lfa_flex = FlexLocalFrameAttention(D, num_heads=num_heads, head_dim=D // num_heads, chunk_size=4).to(device).to(dtype)
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        with torch.no_grad():
            y_local_flex = lfa_flex(x)
        if device == "cuda":
            torch.cuda.synchronize()
            flex_mem = torch.cuda.max_memory_allocated(device) / 1024**2
        else:
            flex_mem = 0.0
        flex_time = (time.perf_counter() - start) * 1000  # ms
    except Exception as e:
        print(f"Error in FlexLocalFrameAttention: {e}")
        import traceback
        traceback.print_exc()
        flex_time = float('inf')
        flex_mem = float('inf')

    # ------------------------------------------------------------------
    # LocalFrameAttention (SDPA)
    # ------------------------------------------------------------------
    try:
        lfa_sdpa = LocalFrameAttention(D, num_heads=num_heads, head_dim=D // num_heads, chunk_size=4).to(device).to(dtype)
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        with torch.no_grad():
            y_local_sdpa = lfa_sdpa(x)
        if device == "cuda":
            torch.cuda.synchronize()
            local_mem = torch.cuda.max_memory_allocated(device) / 1024**2
        else:
            local_mem = 0.0
        local_time = (time.perf_counter() - start) * 1000  # ms
    except Exception as e:
        print(f"Error in LocalFrameAttention (SDPA): {e}")
        local_time = float('inf')
        local_mem = float('inf')

    # ------------------------------------------------------------------
    # LocalFrameAttention (SDPA w/ Flash)
    # ------------------------------------------------------------------
    try:
        # Re-use the same module instance
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            y_local_flash = lfa_sdpa(x)
        if device == "cuda":
            torch.cuda.synchronize()
            local_flash_mem = torch.cuda.max_memory_allocated(device) / 1024**2
        else:
            local_flash_mem = 0.0
        local_flash_time = (time.perf_counter() - start) * 1000  # ms
    except Exception as e:
        print(f"Error in LocalFrameAttention (SDPA w/ Flash): {e}")
        import traceback
        traceback.print_exc()
        local_flash_time = float('inf')
        local_flash_mem = float('inf')

    # ------------------------------------------------------------------
    # LocalFrameAttention with Diffuser
    # ------------------------------------------------------------------
    try:
        lfa_diffuser = LocalFrameAttentionWithDiffuser(D, num_heads=num_heads, head_dim=D // num_heads, chunk_size=4).to(device).to(dtype)
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        with torch.no_grad():
            y_local_diffuser = lfa_diffuser(x)
        if device == "cuda":
            torch.cuda.synchronize()
            local_diffuser_mem = torch.cuda.max_memory_allocated(device) / 1024**2
        else:
            local_diffuser_mem = 0.0
        local_diffuser_time = (time.perf_counter() - start) * 1000  # ms
    except Exception as e:
        print(f"Error in LocalFrameAttention (Diffuser): {e}")
        import traceback
        traceback.print_exc()
        local_diffuser_time = float('inf')
        local_diffuser_mem = float('inf')

    # ------------------------------------------------------------------
    # 3. Report
    # ------------------------------------------------------------------
    print("Benchmark on {}, with total tokens: {}k".format("GPU" if device == "cuda" else "CPU", B * F * N // 1000))
    print(f"Vanilla attention     : {vanilla_time:.2f} ms, {vanilla_mem:.1f} MB peak")
    print(f"Diffuser attention    : {diffuser_time:.2f} ms, {diffuser_mem:.1f} MB peak")
    print(f"Flash Attention       : {flash_time:.2f} ms, {flash_mem:.1f} MB peak")
    print(f"FlexLocalFrameAttn    : {flex_time:.2f} ms, {flex_mem:.1f} MB peak")
    print(f"LocalFrameAttn (SDPA) : {local_time:.2f} ms, {local_mem:.1f} MB peak")
    print(f"LocalFrameAttn (SDPA w/ Flash) : {local_flash_time:.2f} ms, {local_flash_mem:.1f} MB peak")
    print(f"LocalFrameAttn (Diffuser) : {local_diffuser_time:.2f} ms, {local_diffuser_mem:.1f} MB peak")

    if local_time > 0 and vanilla_time != float('inf'):
        print(f"Speed‑up (Local vs Vanilla) : {vanilla_time / local_time:.2f}×")
    if flash_time > 0 and vanilla_time != float('inf'):
        print(f"Speed‑up (Flash vs Vanilla) : {vanilla_time / flash_time:.2f}×")

    if device == "cuda" and local_mem > 0 and vanilla_mem != float('inf'):
        print(f"Memory reduction (Local vs Vanilla) : {vanilla_mem / local_mem:.2f}×")
    if device == "cuda" and flash_mem > 0 and vanilla_mem != float('inf'):
        print(f"Memory reduction (Flash vs Vanilla) : {vanilla_mem / flash_mem:.2f}×")

    print("\n✓ Comparison finished.\n")



# -----------------------------------------------------------------------------
# Smoke‑test: 800‑frame video, 16×16 spatial grid, 8 channels → 256 tokens/frm
# -----------------------------------------------------------------------------
def test_local_frame_attention():
    torch.manual_seed(0)
    B, F, H, W, D = 1, 800, 16, 16, 128
    N = H * W
    x = torch.randn(B, F, N, D, device='cuda' if torch.cuda.is_available() else 'cpu')
    attn = LocalFrameAttention(D, num_heads=4, head_dim=D // 4, dropout=0.0, chunk_size=5).to(x.device)
    with torch.no_grad():
        y = attn(x)
    print('input :', x.shape)
    print('output:', y.shape)
    assert y.shape == x.shape
    print('✓ FlexAttention LocalFrameAttention works on 800-frame video!')


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    # test_local_frame_attention()
    benchmark_attentions(num_frames=1280)