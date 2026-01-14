"""
References:
    - VQGAN: https://github.com/CompVis/taming-transformers
    - MAE: https://github.com/facebookresearch/mae
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers.mlp import Mlp
from algorithms.mem_wm.backbones.embeddings import RotaryEmbedding, apply_rotary_emb, PatchEmbed
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from algorithms.vae.common.distribution import DiagonalGaussianDistribution
import torch
from utils.print_utils import cyan, red
from utils.distributed_utils import rank_zero_print


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        frame_height,
        frame_width,
        qkv_bias=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.frame_height = frame_height
        self.frame_width = frame_width

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        rotary_freqs = RotaryEmbedding(
            dim=head_dim // 4,
            freqs_for="pixel",
            max_freq=frame_height * frame_width,
        ).get_axial_freqs(frame_height, frame_width)
        self.register_buffer("rotary_freqs", rotary_freqs, persistent=False)

    def forward(self, x):
        B, N, C = x.shape
        assert N == self.frame_height * self.frame_width

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = rearrange(
            q,
            "b (H W) (h d) -> b h H W d",
            H=self.frame_height,
            W=self.frame_width,
            h=self.num_heads,
        )
        k = rearrange(
            k,
            "b (H W) (h d) -> b h H W d",
            H=self.frame_height,
            W=self.frame_width,
            h=self.num_heads,
        )
        v = rearrange(
            v,
            "b (H W) (h d) -> b h H W d",
            H=self.frame_height,
            W=self.frame_width,
            h=self.num_heads,
        )

        q = apply_rotary_emb(self.rotary_freqs, q)
        k = apply_rotary_emb(self.rotary_freqs, k)

        q = rearrange(q, "b h H W d -> b h (H W) d")
        k = rearrange(k, "b h H W d -> b h (H W) d")
        v = rearrange(v, "b h H W d -> b h (H W) d")

        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b h N d -> b N (h d)")

        x = self.proj(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        frame_height,
        frame_width,
        mlp_ratio=4.0,
        qkv_bias=False,
        attn_causal=False,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads,
            frame_height,
            frame_width,
            qkv_bias=qkv_bias,
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MAE_ViT(AutoencoderKL):
    def __init__(
        self,
        latent_dim,
        input_height=256,
        input_width=256,
        patch_size=16,
        enc_dim=768,
        enc_depth=6,
        enc_heads=12,
        dec_dim=768,
        dec_depth=6,
        dec_heads=12,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        use_variational=True,
        latent_mean=0,
        latent_std=1,
        **kwargs,
    ):
        super().__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.patch_size = patch_size
        self.seq_h = input_height // patch_size
        self.seq_w = input_width // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.patch_dim = 3 * patch_size**2

        self.latent_dim = latent_dim
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        
        # original code just multiplies when encoding, by scaling factor = 1 / std. if passing in latent_std then we will essentially do that
        # can also pass in a channel-wise version

        if isinstance(latent_std, int):
            if not isinstance(latent_mean, int):
                raise RuntimeError("latent mean and std should both be ints if one is int")
            latent_mean = torch.ones((1, 1, latent_dim)) * latent_mean
            latent_std = torch.ones((1, 1, latent_dim)) * latent_std
        else:
            if isinstance(latent_mean, int):
                raise RuntimeError("latent mean and std should both be ints if one is int")
            assert len(latent_mean) == self.latent_dim and len(latent_std) == self.latent_dim
            latent_mean = torch.tensor(latent_mean).reshape(1, 1, self.latent_dim)
            latent_std = torch.tensor(latent_std).reshape(1, 1, self.latent_dim)
        
        self.register_buffer("latent_mean", latent_mean)
        self.register_buffer("latent_std", latent_std)

        if (kwargs.get("depth_latent_mean", None) is not None) and (kwargs.get("depth_latent_std", None) is not None):
            depth_latent_mean = torch.tensor(kwargs.pop("depth_latent_mean")).reshape(1, 1, self.latent_dim)
            depth_latent_std = torch.tensor(kwargs.pop("depth_latent_std")).reshape(1, 1, self.latent_dim)
            self.register_buffer("depth_latent_mean", depth_latent_mean)
            self.register_buffer("depth_latent_std", depth_latent_std)

        # patch
        self.patch_embed = PatchEmbed(input_height, input_width, patch_size, 3, enc_dim)

        # encoder
        self.encoder = nn.ModuleList(
            [
                AttentionBlock(
                    enc_dim,
                    enc_heads,
                    self.seq_h,
                    self.seq_w,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(enc_depth)
            ]
        )
        self.enc_norm = norm_layer(enc_dim)

        # bottleneck
        self.use_variational = use_variational
        mult = 2 if self.use_variational else 1
        self.quant_conv = nn.Linear(enc_dim, mult * latent_dim)
        self.post_quant_conv = nn.Linear(latent_dim, dec_dim)

        # decoder
        self.decoder = nn.ModuleList(
            [
                AttentionBlock(
                    dec_dim,
                    dec_heads,
                    self.seq_h,
                    self.seq_w,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm = norm_layer(dec_dim)
        self.predictor = nn.Linear(dec_dim, self.patch_dim)  # decoder to patch

        # initialize this weight first
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        # patchify
        bsz, _, h, w = x.shape
        x = x.reshape(
            bsz,
            3,
            self.seq_h,
            self.patch_size,
            self.seq_w,
            self.patch_size,
        ).permute([0, 1, 3, 5, 2, 4])  # [b, c, h, p, w, p] --> [b, c, p, p, h, w]
        x = x.reshape(bsz, self.patch_dim, self.seq_h, self.seq_w)  # --> [b, cxpxp, h, w]
        x = x.permute([0, 2, 3, 1]).reshape(bsz, self.seq_len, self.patch_dim)  # --> [b, hxw, cxpxp]
        return x

    def unpatchify(self, x):
        bsz = x.shape[0]
        # unpatchify
        x = x.reshape(bsz, self.seq_h, self.seq_w, self.patch_dim).permute([0, 3, 1, 2])  # [b, h, w, cxpxp] --> [b, cxpxp, h, w]
        x = x.reshape(
            bsz,
            3,
            self.patch_size,
            self.patch_size,
            self.seq_h,
            self.seq_w,
        ).permute([0, 1, 4, 2, 5, 3])  # [b, c, p, p, h, w] --> [b, c, h, p, w, p]
        x = x.reshape(
            bsz,
            3,
            self.input_height,
            self.input_width,
        )  # [b, c, hxp, wxp]
        return x

    def encode(self, x):
        # patchify
        x = self.patch_embed(x)

        # encoder
        for blk in self.encoder:
            x = blk(x)
        x = self.enc_norm(x)

        # bottleneck
        moments = self.quant_conv(x)
        if not self.use_variational:
            moments = torch.cat((moments, torch.zeros_like(moments)), 2)
        posterior = DiagonalGaussianDistribution(moments, deterministic=(not self.use_variational), dim=2)
        return posterior

    def decode(self, z):
        # bottleneck
        z = self.post_quant_conv(z)

        # decoder
        for blk in self.decoder:
            z = blk(z)
        z = self.dec_norm(z)

        # predictor
        z = self.predictor(z)

        # unpatchify
        dec = self.unpatchify(z)
        return dec

    def autoencode(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if self.use_variational and sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior, z

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def forward(self, inputs, labels, split="train"):
        rec, post, latent = self.autoencode(inputs)
        return rec, post, latent

    def get_last_layer(self):
        return self.predictor.weight
    
    @torch.no_grad()
    def vae_encode(self, x: torch.Tensor, output_shape = "(b t) (h w) c", image_height = 256, image_width = 256, data_type = "rgb"):
        """
        Used for inference time
        vae encode
        input: b t c h w
        output: (b t) (h w) c
        """
        b, t, c, h, w = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")
        
        if x.max() >= 255:
            print(f"x.max()={x.max()}")
            x = x / 255.0

        x = self.encode(x * 2 - 1).mean
        
        if data_type == "rgb":
            x = (x - self.latent_mean) / self.latent_std
        elif data_type == "depth":
            x = (x - self.depth_latent_mean) / self.depth_latent_std
        else:
            raise ValueError(f"Data type {data_type} not supported")
        
        x = rearrange(x, "(b t) (h w) c -> b t c h w",  h=image_height // self.patch_size, w=image_width // self.patch_size, b = b, t = t)
        return x

    @torch.no_grad()
    def vae_decode(self, x: torch.Tensor, input_channels: int = 16, data_type = "rgb"):
        """
        vae decode
        input: b t c h w
        output: b c h w
        """
        bs = x.shape[0]
        x = rearrange(x, "b t c h w -> (b t) (h w) c")


        if data_type == "rgb":
            x = (x * self.latent_std) + self.latent_mean
        elif data_type == "depth":
            x = (x * self.depth_latent_std) + self.depth_latent_mean
        else:
            raise ValueError(f"Data type {data_type} not supported")

        x = (self.decode(x) + 1) / 2
        
        x = rearrange(x, "(b t) c h w -> b t c h w", b = bs)
        return x
    
    @classmethod
    def from_pretrained(cls, path, **kwargs) -> "MAE_ViT":
        if "latent_dim" in kwargs:
            latent_dim = kwargs.pop("latent_dim")
        else:
            latent_dim = 16
        
        model = cls(latent_dim=latent_dim, **kwargs)
        
        if path is not None:
            if path.endswith(".safetensors"):
                from safetensors.torch import load_model
                load_model(model, path) # type: ignore
            else:
                model.init_from_ckpt(path)
        else:
            raise ValueError("No pretrained vae path provided")

        return model

    # models trained from mae_vit_trainer state dict will have vae. prefix, we need to remove that to match the state dict here
    def init_from_ckpt(self, path, ignore_keys=list(), prefix_to_strip="vae."):
        sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        
        new_sd = {}
        for k, v in sd.items():
            if any(k.startswith(ik) for ik in ignore_keys):
                rank_zero_print(red(f"Deleting key {k} from state_dict."))
                continue
            # Strip the prefix if it exists
            if 'latent_mean' in k or 'latent_std' in k:
                continue
            if k.startswith(prefix_to_strip):
                new_k = k[len(prefix_to_strip):]
            else:
                new_k = k
            new_sd[new_k] = v

        result = self.load_state_dict(new_sd, strict=False)
        rank_zero_print(cyan(f"VAE initialized from {path}"))

