import torch
from torch import Tensor
from typing import Literal
from algorithms.vae import VAE_CLS_DICT
from utils.torch_utils import freeze_model

class VAEMixin:


    def _load_vae(self) -> None:
        """
        Load the pretrained VAE model.
        """
        vae_cls = VAE_CLS_DICT[self.cfg.vae.cls]
        self.vae = vae_cls.from_pretrained(
            path=self.cfg.vae.pretrained_path,
            torch_dtype=(
                torch.float16 if self.cfg.vae.use_fp16 else torch.float32
            ),  # only for Diffuser's ImageVAE
            **self.cfg.vae.pretrained_kwargs,
        ).to(self.device)
        
        # dirty hack to load the model from the .pt ckpt
        # if self.cfg.vae.load_ckpt:
        #     self.vae.load_ckpt(self.cfg.vae.ckpt_path)
        freeze_model(self.vae)

        if self.cfg.vae.downsampling_factor[0] > 1: # temporal downsampling
            self.is_image_vae = False
        else:
            self.is_image_vae = True

    def _encode(self, frames: Tensor, data_type: Literal["rgb", "depth"] = "rgb") -> Tensor:
        """
        args:
            frames: (bs, t, c, h, w)
        """
        xs = self.vae.vae_encode(frames, output_shape = self.cfg.vae.pretrained_kwargs.output_shape, image_height = self.cfg.x_shape[1], image_width = self.cfg.x_shape[2], data_type = data_type)
        
        return xs

    def _decode(self, xs: Tensor, data_type: Literal["rgb", "depth"] = "rgb") -> Tensor:
        """
        Decode the latent codes to the original frames.
        """
        
        xs = self.vae.vae_decode(xs, input_channels = self.cfg.latent.num_channels, data_type = data_type)
        return xs
