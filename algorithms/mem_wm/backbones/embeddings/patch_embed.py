import torch
import torch.nn as nn
from einops import rearrange
from timm.layers.helpers import to_2tuple
from typing import Optional, Callable


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_height:int =256,
        img_width:int =256,
        patch_size:int =16,
        in_chans:int =3,
        embed_dim:int =768,
        norm_layer:Optional[Callable] =None,
        flatten:bool =True,
    ):
        super().__init__()
        img_size = (img_height, img_width)
        patch_size: tuple[int, int] = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x, random_sample=False):
        B, C, H, W = x.shape
        if not (random_sample or (H == self.img_size[0] and W == self.img_size[1])):
            raise ValueError(f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = rearrange(x, "B C H W -> B (H W) C")
        else:
            x = rearrange(x, "B C H W -> B H W C")
        x = self.norm(x)
        return x
