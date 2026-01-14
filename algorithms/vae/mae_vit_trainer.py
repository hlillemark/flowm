"""
Adapted from CompVis/latent-diffusion
https://github.com/CompVis/stable-diffusion
"""

from typing import Tuple, Union, Dict, List
from functools import partial
from omegaconf import DictConfig
import torch
import lightning.pytorch as pl
from einops import rearrange
from torchmetrics.image import FrechetInceptionDistance
from utils.logging_utils import log_video
from utils.logging_utils import get_validation_metrics_for_videos
from .common.losses import LPIPSWithDiscriminator, warmup
from .mae_vit import MAE_ViT
from utils.distributed_utils import is_rank_zero
from lightning_utilities.core.apply_func import apply_to_collection
import numpy as np
from typing import Sequence

class MAEVITImageVAETrainer(pl.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.learning_rate = cfg.lr
        self.automatic_optimization = False
        lossconfig = cfg.lossconfig
        self.embed_dim = cfg.embed_dim
        self.warmup_steps = cfg.warmup_steps
        self.gradient_clip_val = cfg.gradient_clip_val
        self.loss = LPIPSWithDiscriminator(**lossconfig)
        self.fid_model = None
        self.finetune_depth = cfg.get("finetune_depth", False)
        if self.finetune_depth:
            self.register_depth_min_max(cfg.depth_min, cfg.depth_max, namespace="depth")
            self.normalize_depth_with_log = cfg.get("normalize_depth_with_log", False)
        
        pretrained_path = cfg.vae.pretrained_path
        if pretrained_path is not None:
            # load_model(self.vae, pretrained_path) 
            self.vae = MAE_ViT.from_pretrained(path=pretrained_path, **cfg.vae.pretrained_kwargs)
        else:
            self.vae = MAE_ViT(**cfg.vae.pretrained_kwargs)

    def gather_data(
        self, data: Union[torch.Tensor, Dict, List, Tuple], batch_dim: int = 0
    ):
        """
        Gather tensors or collections of tensors from all devices,
        and stack them along the batch dimension.
        Args:
            data: tensor or collection of tensors to gather
            batch_dim: the batch dimension of the original tensor
        """
        # if not ddp, skip gathering and return the original data
        if self.trainer.world_size == 1:
            return apply_to_collection(data, torch.Tensor, lambda x: x.to(self.device))

        # synchronize before gathering
        torch.distributed.barrier()
        gathered_data = self.all_gather(data)

        # (r ... b ...) -> (... (r b) ...)
        rearrange_fn = (
            lambda x: x.permute(
                list(range(1, batch_dim + 1))
                + [0]
                + list(range(batch_dim + 1, x.dim()))
            )
            .reshape(*x.shape[1 : batch_dim + 1], -1, *x.shape[batch_dim + 2 :])
            .contiguous()
        )

        return apply_to_collection(gathered_data, torch.Tensor, rearrange_fn)

    def init_from_ckpt(self, path, ignore_keys=list()):

        # from omegaconf.dictconfig import DictConfig
        # import torch.serialization
        # torch.serialization.add_safe_globals([DictConfig])
        sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored VAE from {path}")

    def on_save_checkpoint(self, checkpoint):
        """
        save cfgs together to enable easily loading the pretrained model
        """
        checkpoint["cfg"] = self.cfg
        return checkpoint

    def encode(self, x):
        posterior = self.vae.encode(x)
        return posterior 

    def decode(self, z):
        dec = self.vae.decode(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def _normalize_depth(self, depth):
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


    def on_after_batch_transfer(
        self, batch: tuple, dataloader_idx: int = 0
    ) -> torch.Tensor:
        if type(batch) == list:
            batch = batch[dataloader_idx]
        # Passed in as B, T, C, 360, 640

        videos = batch["videos"]
        videos = 2.0 * videos - 1.0  # normalize to [-1, 1]
        if self.finetune_depth:
            depths = batch["depths"] # (bs, T, 3, h, w), the depth has been repeated in channel dimension
            depths = self._normalize_depth(depths)
            depths = 2.0 * depths - 1.0  # normalize to [-1, 1]
            return {"depths": depths, "videos": videos}
        else:
            return {"videos": videos}

    def training_step(self, batch, batch_idx):
        # pylint: disable=unpacking-non-sequence
        opt_ae, opt_disc = self.optimizers()

        videos = batch["videos"]
        videos = rearrange(videos, "b t c h w -> (b t) c h w")

        if self.finetune_depth:
            depths = batch["depths"]
            depths = rearrange(depths, "b t c h w -> (b t) c h w")
        
            reconstructions_depth, posterior_depth = self(depths)
        
        reconstructions_video, posterior_video = self(videos)

        log_loss = partial(
            self.log,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        log_loss_dict = partial(
            self.log_dict,
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True
        )

        # Warm-up: at the beginning of training / after GAN loss starts being used
        # compute lr_scale
        should_warmup, lr_scale = False, 1.0
        if self.global_step < self.warmup_steps:
            should_warmup = True
            lr_scale = float(self.global_step + 1) / self.warmup_steps
        elif (
            self.global_step >= self.cfg.lossconfig.disc_start - 1
            and self.global_step < self.cfg.lossconfig.disc_start + self.warmup_steps
        ):
            should_warmup = True
            lr_scale = (
                float(self.global_step - self.cfg.lossconfig.disc_start + 1)
                / self.warmup_steps
            )
        lr_scale = min(1.0, lr_scale)

        # Optimize the autoencoder

        aeloss, log_dict_ae = self.loss(
            videos,
            reconstructions_video,
            posterior_video,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )
        if self.finetune_depth:
            aeloss_depth, log_dict_ae_depth = self.loss(
                depths,
                reconstructions_depth,
                posterior_depth,
                0,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train_depth",
            )
            aeloss = aeloss + self.cfg.lossconfig.depth_weight * aeloss_depth
            log_dict_ae = {**log_dict_ae, **log_dict_ae_depth}

        opt_ae.zero_grad()
        self.manual_backward(aeloss)
        self.clip_gradients(opt_ae, gradient_clip_val=self.gradient_clip_val)
        if should_warmup:
            opt_ae = warmup(opt_ae, self.learning_rate, lr_scale)
        opt_ae.step()

        # Optimize the discriminator
        discloss, log_dict_disc = self.loss(
            videos,
            reconstructions_video,
            posterior_video,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )
        if self.finetune_depth:
            discloss_depth, log_dict_disc_depth = self.loss(
                depths,
                reconstructions_depth,
                posterior_depth,
                1,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train_depth",
            )
            discloss = discloss + self.cfg.lossconfig.depth_weight * discloss_depth
            log_dict_disc = {**log_dict_disc, **log_dict_disc_depth}

        opt_disc.zero_grad()
        self.manual_backward(discloss)
        self.clip_gradients(opt_disc, gradient_clip_val=self.gradient_clip_val)
        if should_warmup:
            opt_disc = warmup(opt_disc, self.learning_rate, lr_scale)
        opt_disc.step() # BUG | NOTE : Important, we applied two steps here, so the self.globalstep will be updated twice! This will influence the max_steps in the trainer

        loss_dict = {
            "aeloss": aeloss,
            "discloss": discloss,
            **log_dict_ae,
            **log_dict_disc,
        }

        # log_loss_dict(log_dict_disc)
        if batch_idx % self.cfg.logging.loss_freq == 0:
            log_loss_dict(loss_dict)
            
        return loss_dict
        

    def on_validation_epoch_start(self):
        self.fid_model = FrechetInceptionDistance(feature=64).to(self.device)

    def on_validation_epoch_end(self):
        self.fid_model = None

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if type(batch) == list:
            batch = batch[dataloader_idx]
            
        videos = batch["videos"]
        batch_size = videos.size(0)
        if self.finetune_depth:
            depths = batch["depths"]
            depths = rearrange(depths, "b t c h w -> (b t) c h w")
            reconstructions_depth, posterior_depth = self(depths)
        else:
            depths = None
        videos = rearrange(videos, "b t c h w -> (b t) c h w")
        reconstructions_video, posterior_video = self(videos)
        
        aeloss, log_dict_ae = self.loss(
            videos,
            reconstructions_video,
            posterior_video,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
        )
        if self.finetune_depth:
            aeloss_depth, log_dict_ae_depth = self.loss(
                depths,
                reconstructions_depth,
                posterior_depth,
                0,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="val_depth",
            )
            aeloss = aeloss + self.cfg.lossconfig.depth_weight * aeloss_depth
            log_dict_ae = {**log_dict_ae, **log_dict_ae_depth}

        discloss, log_dict_disc = self.loss(
            videos,
            reconstructions_video,
            posterior_video,    
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
        )
        if self.finetune_depth:
            discloss_depth, log_dict_disc_depth = self.loss(
                depths,
                reconstructions_depth,
                posterior_depth,
                1,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="val_depth",
            )
            discloss = discloss + self.cfg.lossconfig.depth_weight * discloss_depth
            log_dict_disc = {**log_dict_disc, **log_dict_disc_depth}

        self.log_dict(log_dict_ae, sync_dist=True)
        self.log_dict(log_dict_disc, sync_dist=True)

        validation_metrics = get_validation_metrics_for_videos(
            *map(
                lambda x: rearrange(x, "(b t) c h w -> b t c h w", b=batch_size) # werid here
                .contiguous()
                .detach(),
                (videos, reconstructions_video),
            ),
            fid_model=self.fid_model,
        )

        if self.finetune_depth:
            validation_metrics_depth = get_validation_metrics_for_videos(
                *map(
                    lambda x: rearrange(x, "(b t) c h w -> b t c h w", b=batch_size) # werid here
                    .contiguous()
                    .detach(),
                    (depths, reconstructions_depth),
                ),
                fid_model=self.fid_model,
            )
            validation_metrics = {**validation_metrics, **validation_metrics_depth}

        self.log_dict(
            {f"val/{k}": v for k, v in validation_metrics.items()},
            prog_bar=True,
            sync_dist=True,
        )

        ## Until Now, the videos and depths are still in [-1, 1] range, we need to unnormalize them

        if batch_idx == 0:  # log visualizations
            videos, reconstructions_video = (
                self._rearrange_and_unnormalize(x, batch_size).detach().cpu()
                for x in (videos, reconstructions_video)
            )
            all_videos = {
                "recon": reconstructions_video,
                "gt_video": videos
            }
            if self.finetune_depth:
                
                all_videos["recon_depth"] = reconstructions_depth
                all_videos["gt_depth"] = depths

                depths, reconstructions_depth = (
                    self._rearrange_and_unnormalize(x, batch_size).detach().cpu()
                    for x in (depths, reconstructions_depth)
                )

                all_videos = {**all_videos, **{"recon_depth": reconstructions_depth, "gt_depth": depths}}

            all_videos = self.gather_data(all_videos)
            if is_rank_zero and self.logger is not None:
                log_video(
                    all_videos["recon"],
                    all_videos["gt_video"],
                    step=self.global_step,
                    namespace="vae_recon_vis",
                    logger=self.logger.experiment,
                    fps = 15
                )
                if self.finetune_depth:
                    log_video(
                        all_videos["recon_depth"],
                        all_videos["gt_depth"],
                        step=self.global_step,
                        namespace="vae_recon_vis_depth",
                        logger=self.logger.experiment,
                        fps = 15
                    )
    def _rearrange_and_unnormalize(
        self, batch: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        batch = rearrange(batch, "(b t) c h w -> b t c h w", b=batch_size)
        batch = 0.5 * batch + 0.5
        return batch

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(
            list(self.vae.parameters()),
            lr=lr,
            betas=(0.5, 0.9),
        )
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9)
        )
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.vae.predictor.weight

