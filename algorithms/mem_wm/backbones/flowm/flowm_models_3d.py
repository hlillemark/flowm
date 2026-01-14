import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange

from algorithms.mem_wm.backbones.embeddings.patch_embed import PatchEmbed  # type: ignore
from algorithms.mem_wm.backbones.flowm.flowm_utils import fov_mask_indices  # type: ignore
    

class SinCos2DPositionalEncoding(nn.Module):
    """Sine-cosine 2D positional embeddings.

    Produces a tensor of shape (H*W, D) given (H, W).
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        assert embed_dim % 4 == 0, "embed_dim should be divisible by 4 for 2D sin-cos"

    @staticmethod
    def _get_1d_sincos(pos: torch.Tensor, dim: int) -> torch.Tensor:
        device = pos.device
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / dim)
        )
        sinusoid_inp = pos[:, None] * div_term[None, :]
        emb = torch.zeros(pos.size(0), dim, device=device)
        emb[:, 0::2] = torch.sin(sinusoid_inp)
        emb[:, 1::2] = torch.cos(sinusoid_inp)
        return emb

    def forward(self, H: int, W: int, device: Optional[torch.device] = None) -> torch.Tensor:
        if device is None:
            device = torch.device("cpu")
        y = torch.arange(H, device=device, dtype=torch.float32)
        x = torch.arange(W, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        dim_half = self.embed_dim // 2
        emb_y = self._get_1d_sincos(yy.reshape(-1), dim_half)
        emb_x = self._get_1d_sincos(xx.reshape(-1), dim_half)
        return torch.cat([emb_y, emb_x], dim=1)  # (H*W, D)


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, drop: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=drop, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x, need_weights=False)
        return out


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim_q: int, dim_kv: int, num_heads: int, drop: float = 0.0):
        super().__init__()
        self.q_proj = nn.Linear(dim_q, dim_q)
        self.k_proj = nn.Linear(dim_kv, dim_q)
        self.v_proj = nn.Linear(dim_kv, dim_q)
        self.attn = nn.MultiheadAttention(embed_dim=dim_q, num_heads=num_heads, dropout=drop, batch_first=True)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q_proj = self.q_proj(q)
        k_proj = self.k_proj(kv)
        v_proj = self.v_proj(kv)
        out, _ = self.attn(q_proj, k_proj, v_proj, need_weights=False)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.attn = PreNorm(dim, MultiHeadSelfAttention(dim, num_heads, drop))
        self.ff = PreNorm(dim, FeedForward(dim, int(dim * mlp_ratio), drop))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.ff(x)
        return x


class DecoderBlock(nn.Module):
    """Transformer decoder block: self-attn -> cross-attn -> MLP.

    Matches the standard Transformer decoder ordering with PreNorm and residuals.
    """

    def __init__(self, dim_q: int, dim_kv: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm_kv = nn.LayerNorm(dim_kv)
        # Self-attention over query tokens
        self.self_attn = PreNorm(dim_q, MultiHeadSelfAttention(dim_q, num_heads, drop))
        # Cross-attention to kv tokens (memory)
        self.cross_attn = PreNorm(dim_q, MultiHeadCrossAttention(dim_q, dim_kv, num_heads, drop))
        # Feedforward
        self.ff = PreNorm(dim_q, FeedForward(dim_q, int(dim_q * mlp_ratio), drop))

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # q: (B, Nq, Dq), kv: (B, Nk, Dkv)
        q = q + self.self_attn(q)
        q = q + self.cross_attn(q, self.norm_kv(kv))
        q = q + self.ff(q)
        return q


class MapLatentProcessor(nn.Module):
    """Maintains and updates a world map latent using JOINT self-attention over map-view and image tokens.

    - Extract current-view subset using an FOV cone mask centered at agent position.
    - Concatenate map-view tokens (with map pos-enc) and image tokens (already pos-encoded by encoder).
    - Apply shared self-attention blocks over the concatenated sequence.
    - Take only the map-view token outputs and overwrite back into the map.
    - Support rolling the map latents according to actions (separate staticmethod).
    """

    def __init__(
        self,
        world_size: int = 16,
        embed_dim: int = 256,
        num_heads: int = 8,
        depth: int = 6,
        fov_deg: float = 60.0,
        in_chans: int = 3,
        img_size: Tuple[int, int] = (128, 128),
        patch_size: int = 16,
        v_range: int = 0,
        v_channel_identity_emb: bool = False,
    ):
        super().__init__()
        self.world_size = world_size * 2 + 1  
        self.embed_dim = embed_dim
        self.fov_deg = fov_deg
        self.img_size = img_size
        self.patch_size = patch_size
        self.v_range = v_range

        if v_range == 1:
            self.v_list = [(0,0), (1,0), (0,1), (-1,0), (0,-1)]
        elif v_range == 0:
            self.v_list = [(0,0)]
        else:
            raise NotImplementedError(f"v_range {v_range} not implemented")

        self.num_v = len(self.v_list)

        # Optional learnable embedding to identify velocity channels (v-index)
        self.v_channel_identity_emb = v_channel_identity_emb
        if v_channel_identity_emb:
            self.v_channel_id_embed = nn.Embedding(self.num_v, self.embed_dim)
        else:
            self.v_channel_id_embed = None

        # Joint self-attention stack over [map_view, image_tokens]
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=4.0) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Map positional encoding (sine-cos 2D)
        self.map_pos_enc = SinCos2DPositionalEncoding(embed_dim)

        # Internal image patchifier and position encoding
        self.patch = PatchEmbed(
            img_height=img_size[0], img_width=img_size[1], patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim, norm_layer=None, flatten=True,
        )
        self.img_pos_enc = SinCos2DPositionalEncoding(embed_dim)

        self.update_gate = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.Sigmoid()
        )

    def init_map(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # (B, V, H, W, D) latent map initialized to zeros
        return torch.zeros(batch_size, self.num_v, self.world_size, self.world_size, self.embed_dim, device=device)

    def compute_fov_mask(self, device: torch.device) -> torch.Tensor:
        # Agent at center looking +x
        mask, _ = fov_mask_indices(
            shape_HW=(self.world_size, self.world_size),
            agent_xy=(self.world_size // 2, self.world_size // 2),
            heading_deg=0.0,
            fov_deg=self.fov_deg,
            max_range=None,
            samples_per_edge=5,
            min_coverage=0.0,
            device=device,
        )
        return mask  # (H, W) bool

    def extract_view_tokens(self, h_map: torch.Tensor, fov_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h_map: (B,V,H,W,D)
        # FOV mask selection:
        B, V, H, W, D = h_map.shape
        idx = torch.nonzero(fov_mask.view(-1), as_tuple=False).squeeze(1)  # (Ncv,)
        h_flat = h_map.view(B, V, H * W, D)
        h_cv = torch.index_select(h_flat, dim=2, index=idx)  # (B, V, Ncv, D)
        return h_cv, idx


    def overwrite_view_tokens(self, h_map: torch.Tensor, h_cv_new: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        # h_map: (B,V,H,W,D), h_cv_new: (B,V,Ncv,D)
        B, V, H, W, D = h_map.shape
        # scatter by idx:
        h_flat = h_map.view(B, V, H * W, D)
        out = h_flat.clone()
        # h_cv_new: (B, V, Ncv, D), out: (B, V, H*W, D), idx: (Ncv,)
        # We want to scatter h_cv_new into out at positions idx along the H*W dimension (dim=2)
        out = out.scatter(dim=2, index=idx.view(1, 1, -1, 1).expand(B, V, -1, D), src=h_cv_new)
        return out.view(B, V,H, W, D)


    def roll_map(self, h_map: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Apply action to map latents to maintain egocentric alignment.

        action: [B] with codes {0: left, 1: right, 2: straight}
        h_map:  (B,V,H,W,D)
        - left/right: rotate map around agent center
        - straight: shift upward (agent moving forward)
        """
        B, V, H, W, D = h_map.shape
        out = h_map.clone()
        
        left_idx = (action == 0).nonzero(as_tuple=True)[0]
        right_idx = (action == 1).nonzero(as_tuple=True)[0]
        straight_idx = (action == 2).nonzero(as_tuple=True)[0]

        if left_idx.numel():
            x = out[left_idx]
            x = torch.rot90(x, k=-1, dims=(2, 3)) # (B, V, H, W, D) 
            # Need to permute the channels after turning
            x[:, 1:] = torch.roll(x[:, 1:], shifts=1, dims=1)
            out[left_idx] = x

        if right_idx.numel():
            x = out[right_idx]
            x = torch.rot90(x, k=1, dims=(2, 3)) # (B, V, H, W, D) 
            # Need to permute the channels after turning
            x[:, 1:] = torch.roll(x[:, 1:], shifts=-1, dims=1)
            out[right_idx] = x

        if straight_idx.numel():
            x = out[straight_idx] # (B, V, H, W, D) 
            x = torch.roll(x, shifts=-1, dims=3)  # shift W up 
            out[straight_idx] = x

        return out

    def shift_v_channels(self, h_map: torch.Tensor) -> torch.Tensor:
        # h_map: (B,V,H,W,D)
        # Shift each velocity channel by the corresponding velocity
        B, V, H, W, D = h_map.shape
        out = h_map.clone()
        for i, (vx, vy) in enumerate(self.v_list):
            out[:, i] = torch.roll(h_map[:, i], shifts=(vy, vx), dims=(1, 2))
        return out

    def forward(self, h_map: torch.Tensor, frame: torch.Tensor, fov_mask: torch.Tensor, action_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Extract current view tokens and add positional encodings
        B, V, H, W, D = h_map.shape
        
        # h_map: shape [bs, num_v, world_size, world_size, 256]
        # h_cv: shape [bs, num_v, Ncv, 256]
        h_cv, idx = self.extract_view_tokens(h_map, fov_mask) # (B, V, Ncv, D)
        Ncv = h_cv.size(2)
        pos_cv = self.map_pos_enc(H, W, device=h_map.device) # (H*W, D)
        # Check if this indexing looks correct for the position encoding
        pos_cv = torch.index_select(pos_cv, dim=0, index=idx)  # (Ncv, D)
        h_cv_plus = h_cv + pos_cv.unsqueeze(0).unsqueeze(0) # (B, V, Ncv, D)
        # Optionally add velocity-channel identity embedding before flatten
        if self.v_channel_id_embed is not None:
            v_ids = torch.arange(self.num_v, device=h_map.device)
            v_emb = self.v_channel_id_embed(v_ids).view(1, self.num_v, 1, self.embed_dim)  # (1, V, 1, D)
            h_cv_plus = h_cv_plus + v_emb
        h_cv_plus_vflat = h_cv_plus.view(B, V * Ncv, D)
        # Ends up as [bs, 158, 256]. seems correct.

        # Patchify image and add image positional encodings
        # for input frame of shape [bs, 3, 128, 128], and patch size 16, we end up with 
        # 8x8 patches for each direction, so 64 tokens in total for the image decoding portion
        img_tok = self.patch(frame)  # (B, Np, D)
        Htok = self.patch.grid_size[0]
        Wtok = self.patch.grid_size[1]
        pos_img = self.img_pos_enc(Htok, Wtok, device=frame.device)  # (Np, D)
        img_tok = img_tok + pos_img.unsqueeze(0)

        # Joint self-attention over concatenated tokens
        joint = torch.cat([h_cv_plus_vflat, img_tok], dim=1)  # (B, Ncv+Ni, D)
        if action_emb is not None:
            joint = torch.cat([joint, action_emb], dim=1)  # (B, Ncv+Ni+Da, D)

        for blk in self.blocks:
            joint = blk(joint)
        joint = self.norm(joint)

        Ncv_all_v = h_cv_plus_vflat.size(1)
        
        h_cv_new = joint[:, :Ncv_all_v]  # take only map-view token outputs

        h_cv_new = h_cv_new.view(B, V, Ncv, D)

        g = self.update_gate(torch.cat([h_cv, h_cv_new], dim=-1))

        h_cv_write = (1 - g) * h_cv + g * h_cv_new

        h_cv_write_v_channels = h_cv_write.view(B, V, Ncv, D)

        # Overwrite back into map
        h_map = self.overwrite_view_tokens(h_map, h_cv_write_v_channels, idx)

        return h_map


class ViTImageDecoder(nn.Module):
    """MAE-ViT-style image decoder with cross-attention to map-view tokens."""

    def __init__(
        self,
        out_chans: int = 3,
        img_size: Tuple[int, int] = (128, 128),
        patch_size: int = 16,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        drop: float = 0.0,
        mask_token_learnable: bool = True,   # set False to make it a fixed seed
        mask_scale: float = 1.0,             # tweak to downweight mask token if desired
    ):
        super().__init__()
        H, W = img_size
        assert H % patch_size == 0 and W % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.mask_scale = mask_scale

        self.num_patches = (H // patch_size) * (W // patch_size)

        # Single mask token (broadcast to all output patch positions)
        tok = torch.randn(1, 1, embed_dim) * 0.02
        if mask_token_learnable:
            self.mask_token = nn.Parameter(tok)
        else:
            self.register_buffer("mask_token", tok, persistent=False)

        # Absolute 2D sin-cos pos enc for the output grid
        self.pos_enc = SinCos2DPositionalEncoding(embed_dim)

        # Decoder stack (cross-attention expected inside DecoderBlock)
        self.blocks = nn.ModuleList([
            DecoderBlock(embed_dim, embed_dim, num_heads=num_heads, mlp_ratio=4.0, drop=drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Project tokens to patch pixels
        self.proj = nn.Linear(embed_dim, out_chans * patch_size * patch_size)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """
        patches: (B, Np, C*ps*ps), row-major order where Np = (H/ps)*(W/ps)
        returns: (B, C, H, W)
        """
        B, Np, Cp = patches.shape
        H, W = self.img_size
        ps = self.patch_size
        Ht, Wt = H // ps, W // ps

        assert Np == Ht * Wt, f"Np={Np} != {Ht*Wt}"
        assert Cp % (ps * ps) == 0, f"C*ps*ps must divide Cp={Cp}"
        C = Cp // (ps * ps)

        # (B, (Ht Wt), (C ps ps)) -> (B, C, Ht ps, Wt ps) == (B, C, H, W)
        img = rearrange(
            patches.contiguous(),
            "b (ht wt) (c p1 p2) -> b c (ht p1) (wt p2)",
            ht=Ht, wt=Wt, c=C, p1=ps, p2=ps,
        )
        return img

    def forward(self, map_view_tokens: torch.Tensor) -> torch.Tensor:
        # map_view_tokens: (B, N_ctx, D)  # your “memory” K/V from the map
        B = map_view_tokens.size(0)
        Ht = self.img_size[0] // self.patch_size
        Wt = self.img_size[1] // self.patch_size

        # MAE-style init: broadcast a single mask token + absolute position enc
        pos = self.pos_enc(Ht, Wt, device=map_view_tokens.device).unsqueeze(0)  # (1, Np, D)
        q = self.mask_scale * self.mask_token.expand(B, self.num_patches, -1) + pos  # (B, Np, D)

        # These map tokens have the position embedding from earlier. 
        # shape (bs, Ncv_all_v, 256)
        kv = map_view_tokens
        
        for blk in self.blocks:
            q = blk(q, kv)  # cross-attend: queries (output grid) attend to map-view tokens

        q = self.norm(q)
        patches = self.proj(q)              # (B, Np, C*ps*ps)
        img = self.unpatchify(patches)      # (B, C, H, W)
        return img



class ViTBlockworldModel(nn.Module):
    """
    Transformer-based Blockworld model with egocentric map latents.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (3, 128, 128),
        world_size: int = 16,
        embed_dim: int = 256,
        img_patch_size: int = 16,
        map_proc_encoder_depth: int = 6,
        dec_depth: int = 6,
        num_heads: int = 8,
        fov_deg: float = 60.0,
        v_range: int = 0, # v_range 0 means no velocity, 1 means +/- 1 velocity in x/y (5 channels total)
        v_channel_identity_emb: bool = False,
        v_channel_maxpool_decode: bool = False,
        v_channel_mixing_mode: str = 'none',  # one of {'none'|'2d_v_channel_flattened'|'3d'}
        no_self_motion_equiv: bool = False, # whether to turn off self motion equivariance
        external_cond_dim: int = 5, # number of external actions
    ):
        super().__init__()
        C, H, W = input_shape
        self.input_shape = input_shape
        self.world_size = world_size
        self.embed_dim = embed_dim
        self.img_size = (H, W)
        self.img_patch_size = img_patch_size
        self.external_cond_dim = external_cond_dim
        self.map_processor = MapLatentProcessor(
            world_size=world_size, embed_dim=embed_dim, num_heads=num_heads, depth=map_proc_encoder_depth, fov_deg=fov_deg,
            in_chans=C, img_size=self.img_size, patch_size=img_patch_size,
            v_range=v_range,
            v_channel_identity_emb=v_channel_identity_emb,
        )
        self.decoder = ViTImageDecoder(
            out_chans=C, img_size=self.img_size, patch_size=img_patch_size,
            embed_dim=embed_dim, depth=dec_depth, num_heads=num_heads,
        )

        # Decoder-side option: aggregation of velocity channels before decoder (boolean)
        self.v_channel_maxpool_decode = v_channel_maxpool_decode
        self.v_channel_mixing_mode = v_channel_mixing_mode
        self.no_self_motion_equiv = no_self_motion_equiv

        if self.v_channel_mixing_mode == 'none':
            self.v_channel_mixing = None
        elif self.v_channel_mixing_mode == '2d_v_channel_flattened':
            h_kernel_size = 3
            h_pad = h_kernel_size // 2
            in_out_ch = embed_dim * self.map_processor.num_v
            self.v_channel_mixing = nn.Conv2d(
                in_out_ch, in_out_ch,
                kernel_size=(h_kernel_size, h_kernel_size),
                padding=(h_pad, h_pad),
                bias=False,
                padding_mode="circular",
                # added for stability of gradients: depthwise conv
                groups=embed_dim
            )
            nn.init.dirac_(self.v_channel_mixing.weight)
        elif self.v_channel_mixing_mode == '3d':
            v_kernel_size = 3
            v_pad = v_kernel_size // 2
            h_kernel_size = 3
            h_pad = h_kernel_size // 2
            self.v_channel_mixing = nn.Conv3d(
                embed_dim, embed_dim, 
                kernel_size=(v_kernel_size, h_kernel_size, h_kernel_size), 
                padding=(v_pad, h_pad, h_pad), 
                bias=False,
                padding_mode="circular",
                # added for stability of gradients: depthwise conv
                groups=embed_dim
            )
            nn.init.dirac_(self.v_channel_mixing.weight) # initialize to identity
        else:
            raise RuntimeError(f"Invalid v_channel_mixing_mode: {v_channel_mixing_mode}")

        if no_self_motion_equiv:
            self.action_embedding = nn.Embedding(self.external_cond_dim, embed_dim) # action (one-hot) (bs, external_cond_dim) - > (bs, D)

        # Cache FOV mask (computed on the fly with the device of inputs)
        self.register_buffer("_cached_fov_mask", torch.tensor([], dtype=torch.bool), persistent=False)

    def _get_fov_mask(self, device: torch.device) -> torch.Tensor:
        if self._cached_fov_mask.numel() == 0 or self._cached_fov_mask.device != device:
            mask = self.map_processor.compute_fov_mask(device)
            self._cached_fov_mask = mask
        return self._cached_fov_mask
    def forward(
        self,
        input_seq: torch.Tensor,
        input_actions: torch.Tensor,
        target_actions: torch.Tensor,
        target_seq: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
        return_hidden: bool = False,
        return_latent_video: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Args:
            input_seq: (B, T_in, C, H, W)
            input_actions: (B, T_in) int actions {0,1,2}
            target_actions: (B, T_out)
            target_seq: (B, T_out, C, H, W) optional
        Returns:
            preds: (B, T_out, C, H, W)
        """
        B, T_in, C, H, W = input_seq.shape
        device = input_seq.device
        T_out = target_actions.size(1)
        
        # Init map
        h_map = self.map_processor.init_map(B, device)  # (B,V,Hm,Wm,D)
        fov_mask = self._get_fov_mask(device)          # (Hm,Wm)
    
        # -----------------------------
        # Ingest observed inputs (roll first, then update)
        # -----------------------------
        action_emb = None
        for t in range(T_in):

            # Apply internal flow to each velocity channel
            h_map = self.map_processor.shift_v_channels(h_map) # (B,V,Hm,Wm,D)

            if t > 0:
                # a_t maps (t-1) -> t, so align first
                # First action is thrown out
                if not self.no_self_motion_equiv:
                    h_map = self.map_processor.roll_map(h_map, input_actions[:, t]) # (B,V,Hm,Wm,D)
                else:
                    # Broadcasting issue: action_embedding(input_actions[:, t]) is (B, D) but h_map is (B,V,Hm,Wm,D)
                    # Need to expand action embedding to match h_map dimensions
                    action_emb = self.action_embedding((input_actions[:, t]).long())  # (B, D) -> (B, 1, D)
                    action_emb = action_emb.unsqueeze(1) # (B, 1, D)

            frame = input_seq[:, t]

            if self.v_channel_mixing is not None:
                # Mix v-channels with convolution over map
                if self.v_channel_mixing_mode == '2d_v_channel_flattened':
                    # Stack velocity channels into channel dim and apply 2D conv over space
                    Bm, Vm, Hm, Wm, Dm = h_map.shape
                    x = h_map.permute(0, 4, 1, 2, 3).contiguous()  # (B,D,V,H,W)
                    x = x.view(Bm, Dm * Vm, Hm, Wm)  # (B, D*V, H, W)
                    x = self.v_channel_mixing(x)  # (B, D*V, H, W)
                    x = x.view(Bm, Dm, Vm, Hm, Wm).permute(0, 2, 3, 4, 1).contiguous()  # (B,V,H,W,D)
                    h_map = x
                else:  # self.v_channel_mixing_mode == '3d'
                    h_map_perm = h_map.permute(0, 4, 1, 2, 3).contiguous() # (B,D,V,Hm,Wm)
                    h_map_perm = self.v_channel_mixing(h_map_perm) # (B,D,V,Hm,Wm)
                    h_map = h_map_perm.permute(0, 2, 3, 4, 1).contiguous() # (B,V,Hm,Wm,D)

            # Update map with new observation
            h_map = self.map_processor(h_map, frame, fov_mask, action_emb=action_emb) # (B,V,Hm,Wm,D)

        # -----------------------------
        # Predict future frames autoregressively
        # -----------------------------
        outputs = []
        latent_maps: list[torch.Tensor] = []

        for t in range(T_out):

            # Apply internal flow to each velocity channel
            h_map = self.map_processor.shift_v_channels(h_map) # (B,V,Hm,Wm,D)

            if not self.no_self_motion_equiv:
                # Align to pose at (current + 1) using a_{t} (self motion equivariance)
                h_map = self.map_processor.roll_map(h_map, target_actions[:, t]) # (B,V,Hm,Wm,D)
            else:
                # putting the action in as some sort of unstructured embedding condition instead.
                # Broadcasting issue: action_embedding(target_actions[:, t]) is (B, D) but h_map is (B,V,Hm,Wm,D)
                # Need to expand action embedding to match h_map dimensions
                action_emb = self.action_embedding((target_actions[:, t]).long())  # (B, D) -> (B, 1, D)
                action_emb = action_emb.unsqueeze(1) # (B, 1, D)

            if self.v_channel_mixing is not None:
                # Mix v-channels with convolution over map
                if self.v_channel_mixing_mode == '2d_v_channel_flattened':
                    Bm, Vm, Hm, Wm, Dm = h_map.shape
                    x = h_map.permute(0, 4, 1, 2, 3).contiguous()  # (B,D,V,H,W)
                    x = x.view(Bm, Dm * Vm, Hm, Wm)  # (B, D*V, H, W)
                    x = self.v_channel_mixing(x)  # (B, D*V, H, W)
                    x = x.view(Bm, Dm, Vm, Hm, Wm).permute(0, 2, 3, 4, 1).contiguous()  # (B,V,H,W,D)
                    h_map = x
                else:  # self.v_channel_mixing_mode == '3d'
                    h_map_perm = h_map.permute(0, 4, 1, 2, 3).contiguous() # (B,D,V,Hm,Wm)
                    h_map_perm = self.v_channel_mixing(h_map_perm) # (B,D,V,Hm,Wm)
                    h_map = h_map_perm.permute(0, 2, 3, 4, 1).contiguous() # (B,V,Hm,Wm,D)

            # Record latent map used for decoding (optional)
            if return_latent_video:
                # Also norm v channels
                latent_maps.append(torch.norm(torch.norm(h_map, dim=-1), dim=1))  # (B,Hm,Wm)

            # Decode from current view of the map
            h_cv, idx = self.map_processor.extract_view_tokens(h_map, fov_mask) # (B, V, Ncv, D)
            B, V, Ncv, D = h_cv.shape
            pos_cv = self.map_processor.map_pos_enc(
                self.map_processor.world_size, self.map_processor.world_size, device=device
            )
            pos_cv = torch.index_select(pos_cv, dim=0, index=idx) # (Ncv, D)
            h_cv_plus = h_cv + pos_cv.unsqueeze(0).unsqueeze(0) # (B, V, N_cv, D)
            # Mirror optional velocity-channel identity embedding prior to flattening or pooling
            if self.map_processor.v_channel_id_embed is not None:
                v_ids = torch.arange(self.map_processor.num_v, device=device)
                v_emb = self.map_processor.v_channel_id_embed(v_ids).view(1, self.map_processor.num_v, 1, self.map_processor.embed_dim)
                h_cv_plus = h_cv_plus + v_emb

            if self.v_channel_maxpool_decode:
                # Max pool across V dimension to aggregate velocity channels
                h_cv_for_dec, _ = torch.max(h_cv_plus, dim=1)  # (B, Ncv, D)
            else:
                # Flatten V and Ncv for decoder
                h_cv_for_dec = h_cv_plus.view(B, V * Ncv, D)
            out = self.decoder(h_cv_for_dec)
            outputs.append(out)

            # Choose observation for updating the map (teacher forcing)
            if self.training and target_seq is not None and torch.rand(1).item() < teacher_forcing_ratio:
                obs = target_seq[:, t]
            else:
                obs = out

            # Update map with the observation for the next step
            h_map = self.map_processor(h_map, obs, fov_mask, action_emb=action_emb)

        preds = torch.stack(outputs, dim=1)  # (B, T_out, C, H, W)

        if return_latent_video:
            latent_video = torch.stack(latent_maps, dim=1).unsqueeze(2)  # (B, T_out, 1, Hm, Wm)
            return preds, latent_video
        if return_hidden:
            return preds, h_map
        return preds
