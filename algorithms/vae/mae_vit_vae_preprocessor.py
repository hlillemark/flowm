from pathlib import Path
from typing import Literal, Union, Sequence
from omegaconf import DictConfig
import torch
import numpy as np
from torch import Tensor
from einops import rearrange
from lightning.pytorch.utilities.types import STEP_OUTPUT
from algorithms.common.base_pytorch_algo import BasePytorchAlgo
from utils.storage_utils import safe_torch_save
from utils.logging_utils import log_video
from .mae_vit import MAE_ViT
from utils.torch_utils import freeze_model
import torch.distributed as dist
from lightning.pytorch.utilities.rank_zero import rank_zero_info


class MAE_ViT_VAE_Preprocessor(BasePytorchAlgo):
    """
    An algorithm for preprocessing videos to latents using a pretrained MAE ViT VAE model.
    """

    def __init__(self, cfg: DictConfig):
        self.max_decode_length = cfg.logging.max_video_length
        self.log_every_n_batch = cfg.logging.every_n_batch
        self.cfg = cfg
        self.process_depth = getattr(cfg.depth, 'process_depth', False)
        self.is_latent_diffusion = False
        if self.process_depth:
            self.normalize_depth_with_log = getattr(cfg.depth, 'normalize_depth_with_log', False)
        super().__init__(cfg)
        
    def configure_model(self) -> None:
        """
        Load the pretrained VAE model.
        """
        self.vae = MAE_ViT.from_pretrained(
            path=self.cfg.vae.pretrained_path,
            torch_dtype=(
                torch.float16 if self.cfg.vae.use_fp16 else torch.float32
            ),
            **self.cfg.vae.pretrained_kwargs,
        ).to(self.device)
        
        freeze_model(self.vae)

        if self.process_depth and hasattr(self.cfg.depth, "depth_min") and hasattr(self.cfg.depth, "depth_max"):
            self.register_depth_min_max(self.cfg.depth.depth_min, self.cfg.depth.depth_max)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        raise NotImplementedError(
            "Training not implemented for VAEVideo. Only used for validation"
        )

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        raise NotImplementedError(
            "Testing not implemented for VAEVideo. Only used for validation"
        )

    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> STEP_OUTPUT:
        batch, latent_paths, latent_depth_paths = batch
        
        videos = batch['videos']
        latent_paths = [Path(path) for path in latent_paths]
        
        # Encode the video data into a latent space
        latents = self._encode(videos)
        
        if self.process_depth:
            depths = batch['depths']
            depths = self._normalize_depth_min_max(depths) # to make it in the range of [0, 1]
            latent_depth_paths = [Path(path) for path in latent_depth_paths]
            latent_depths = self._encode(depths, 'depth')

        # just to see the progress in wandb
        if batch_idx % 100 == 0:
            self.log("dummy", 0.0)

        # log gt vs reconstructed video to wandb
        if batch_idx % self.log_every_n_batch == 0 and self.logger:
            videos = videos.detach().cpu()[: self.max_decode_length]
            reconstructed_videos = self._decode(latents[: self.max_decode_length])
            reconstructed_videos = reconstructed_videos.detach().cpu()
            log_video(
                reconstructed_videos,
                videos,
                step=self.global_step,
                namespace="reconstruction_vis",
                logger=self.logger.experiment,
                captions=[
                    f"{p.parent.parent.name}/{p.parent.name}/{p.stem}"
                    for p in latent_paths
                ],
            )
        
        # save the latent to disk
        latents_to_save = (
            latents
            .detach()
            .cpu()
        )
        for latent, latent_path in zip(latents_to_save, latent_paths):
            # should clone latent to avoid having large file size
            safe_torch_save(latent.clone(), latent_path)
        
        if self.process_depth:
            latent_depths_to_save = (
                latent_depths
                .detach()
                .cpu()
            )
            for latent_depth, latent_depth_path in zip(latent_depths_to_save, latent_depth_paths):
                safe_torch_save(latent_depth.clone(), latent_depth_path)
            
        return None
    
    def on_validation_end(self) -> None:
        """
        Ensure all ranks finish validation and reach a barrier before anyone exits.
        Was running into an issue before where the first rank would exit too soon.
        """
        # Only bother if we're actually distributed
        if getattr(self.trainer, "world_size", 1) > 1:
            try:
                # Use Lightning's strategy-aware barrier when available (DDP/FSDP/TPU-safe)
                self.trainer.strategy.barrier()
            except Exception as e:
                # Fallback to vanilla torch barrier
                if dist.is_available() and dist.is_initialized():
                    try:
                        dist.barrier()
                    except Exception as e2:
                        # Don't crash teardown; just log on rank 0
                        rank_zero_info(f"Barrier at on_validation_end failed: {e!r} / {e2!r}")
        # (optional) log once
        rank_zero_info("Synchronized all ranks at on_validation_end.")

    def _encode(self, frames: Tensor, data_type: Literal["rgb", "depth"] = "rgb") -> Tensor:
        """
        args:
            frames: (bs, t, c, h, w)
        """
        image_height = frames.shape[3]
        image_width = frames.shape[4]
        xs = self.vae.vae_encode(frames, output_shape = self.cfg.vae.pretrained_kwargs.output_shape, image_height = image_height, image_width = image_width, data_type = data_type)
        
        return xs

    def _decode(self, xs: Tensor, data_type: Literal["rgb", "depth"] = "rgb") -> Tensor:
        """
        Decode the latent codes to the original frames.
        """
        
        xs = self.vae.vae_decode(xs, input_channels = self.cfg.vae.latent_dim, data_type = data_type)
        return xs

    def _normalize_depth_min_max(self, depth):
        shape = [1] * (depth.ndim - self.depth_min.ndim) + list(self.depth_min.shape)
        min = self.depth_min.reshape(shape)
        max = self.depth_max.reshape(shape)
        if self.normalize_depth_with_log:
            depth = torch.log(depth)
            min = torch.log(min)
            max = torch.log(max)
        return (depth - min) / (max - min)


    def register_depth_min_max(
        self,
        min: Union[str, float, Sequence],
        max: Union[str, float, Sequence],
        namespace: str = "depth",
    ):
        """
        Register min and max of depth as tensor buffer.

        Args:
            min: the min of depth.
            max: the max of depth.
            namespace: the namespace of the registered buffer.
        """
        for k, v in [("min", min), ("max", max)]:
            if isinstance(v, str):
                if v.endswith(".npy"):
                    v = torch.from_numpy(np.load(v))
                elif v.endswith(".pt"):
                    v = torch.load(v)
                else:
                    raise ValueError(f"Unsupported file type {v.split('.')[-1]}.")
            else:
                v = torch.tensor(v)
            self.register_buffer(f"{namespace}_{k}", v.float().to(self.device))
            self.register_buffer(f"{namespace}_{k}", v.float().to(self.device))
            self.register_buffer(f"{namespace}_{k}", v.float().to(self.device))
            self.register_buffer(f"{namespace}_{k}", v.float().to(self.device))
            self.register_buffer(f"{namespace}_{k}", v.float().to(self.device))
