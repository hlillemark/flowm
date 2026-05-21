from typing import Dict, Optional, Tuple

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig, open_dict
from torch import Tensor

from algorithms.common import BasePytorchAlgo, BaseWMMixin
from utils.distributed_utils import rank_zero_print
from utils.print_utils import cyan, red


class RSSMVideo(BaseWMMixin, BasePytorchAlgo):
    """
    Dreamer-style RSSM world model for Blockworld video prediction.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.x_shape = list(cfg.x_shape)
        self.frame_skip = cfg.frame_skip
        self.external_cond_dim = cfg.external_cond_dim * (
            cfg.frame_skip if cfg.external_cond_stack else 1
        )
        self.is_latent_diffusion = cfg.latent.enable
        self.use_preprocessed_latents = (
            cfg.latent.enable and cfg.latent.type.startswith("pre_")
        )
        if self.is_latent_diffusion:
            raise NotImplementedError("The first RSSM integration supports pixel-space training only.")

        self.temporal_downsampling_factor = 1
        self.backbone_cfg = cfg.backbone
        self.logging_cfg = cfg.logging
        self.tasks = [
            task
            for task in ["prediction", "interpolation", "reconstruction"]
            if getattr(cfg.tasks, task).enabled
        ]
        self.num_logged_videos = 0
        self.generator = None
        self.main_model_prefix = "main_model"

        super().__init__(cfg)

    def configure_model(self) -> None:
        backbone_cfg = self.backbone_cfg
        with open_dict(backbone_cfg):
            backbone_cfg.input_shape = list(self.cfg.x_shape)
            backbone_cfg.action_dim = self.cfg.external_cond_dim

        from algorithms.mem_wm.backbones import create_main_model

        self.main_model = create_main_model(backbone_cfg=backbone_cfg)

        if self.cfg.load_model_state:
            if not self.cfg.model_state_key:
                raise ValueError(
                    f"model_state_key is not set while loading model state from {self.cfg.load_model_state}"
                )
            model_dict = torch.load(
                self.cfg.load_model_state, map_location="cpu", weights_only=False
            )
            if self.cfg.model_state_key in model_dict:
                model_dict = model_dict[self.cfg.model_state_key]
            else:
                rank_zero_print(
                    red(
                        f"!!! No model state found in {self.cfg.load_model_state} using key {self.cfg.model_state_key} !!!"
                    )
                )

            model_dict = {
                k.replace(f"{self.main_model_prefix}.", ""): v
                for k, v in model_dict.items()
            }
            load_model = self.cfg.load_model_state_mode
            if load_model == "strict":
                self.main_model.load_state_dict(model_dict, strict=True)
            elif load_model == "partial":
                self.main_model.load_state_dict(model_dict, strict=False)
            else:
                raise ValueError(f"Invalid load_model_state_mode: {load_model}")

            rank_zero_print(
                cyan(
                    f"\n ==== Successfully loaded {len(model_dict)} parameters from {self.cfg.load_model_state}[{self.cfg.model_state_key}] ===="
                )
            )

        self._build_metrics()

    def _prepare_actions(self, conds: Tensor) -> Tensor:
        """
        The Blockworld shortcode sets `cond_alignment: t-1->t`, so `actions[:, t]`
        already denotes the control that led into frame `t`. Keep that convention
        explicit here and feed the RSSM action vectors, matching Dreamer's
        action-concatenation path.
        """
        if conds.dim() == 2:
            return torch.nn.functional.one_hot(
                conds.long(), num_classes=self.cfg.external_cond_dim
            ).float()
        if conds.dim() == 3 and conds.shape[-1] == 1:
            return torch.nn.functional.one_hot(
                conds.squeeze(-1).long(), num_classes=self.cfg.external_cond_dim
            ).float()
        return conds.float()

    def _prepare_reset(self, batch: Dict, device: torch.device) -> Tensor:
        if "reset" in batch:
            return batch["reset"].to(device=device, dtype=torch.bool)
        if "is_first" in batch:
            return batch["is_first"].to(device=device, dtype=torch.bool)
        if "resets" in batch:
            return batch["resets"].to(device=device, dtype=torch.bool)
        return torch.zeros_like(batch["nonterminal"], device=device, dtype=torch.bool)

    def training_step(
        self, batch, batch_idx, dataloader_idx: int = 0, namespace: str = "training"
    ) -> STEP_OUTPUT:
        xs, conds, reset, valid_mask, _, _ = batch
        actions = self._prepare_actions(conds)
        loss_dict = self.main_model.training_losses(
            xs, actions, reset=reset, valid_mask=valid_mask
        )

        output_dict = {
            "loss": loss_dict["loss"],
            "predicted_x": loss_dict["recon_video"],
            "original_x": xs,
        }

        if batch_idx % self.logging_cfg.loss_freq == 0:
            self.log_dict(
                {
                    f"{namespace}/loss": loss_dict["loss"],
                    f"{namespace}/recon_loss": loss_dict["recon_loss"],
                    f"{namespace}/dyn_loss": loss_dict["dyn_loss"],
                    f"{namespace}/rep_loss": loss_dict["rep_loss"],
                },
                on_step=namespace == "training",
                on_epoch=namespace != "training",
                sync_dist=True,
                prog_bar=True,
            )

        return output_dict

    def on_after_batch_transfer(
        self, batch: Dict, dataloader_idx: int
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
        if type(batch) == list:
            batch = batch[dataloader_idx]

        device = self.device
        videos = batch["videos"]
        if self.cfg.backbone.input_shape[0] == 1:
            videos = videos.mean(dim=2, keepdim=True)

        xs = videos.to(device)
        gt_videos = xs
        conds = batch["conds"].to(device)
        reset = self._prepare_reset(batch, device)
        valid_mask = batch["nonterminal"].to(device)
        
        return xs, conds, reset, valid_mask, gt_videos, batch["metadata"]

    def on_validation_epoch_start(self) -> None:
        self.num_logged_videos = [0] * len(self.trainer.val_dataloaders)
        if self.cfg.logging.deterministic is not None:
            self.generator = torch.Generator(device=self.device).manual_seed(
                self.cfg.logging.deterministic
            )

    @torch.no_grad()
    def validation_step(
        self, batch, batch_idx, dataloader_idx: int = 0, namespace: str = "validation"
    ) -> STEP_OUTPUT:
        xs, conds, reset, valid_mask, _, _ = batch
        actions = self._prepare_actions(conds)
        loss_dict = self.main_model.training_losses(
            xs, actions, reset=reset, valid_mask=valid_mask
        )
        self.log_dict(
            {
                f"{namespace}/loss": loss_dict["loss"],
                f"{namespace}/recon_loss": loss_dict["recon_loss"],
                f"{namespace}/dyn_loss": loss_dict["dyn_loss"],
                f"{namespace}/rep_loss": loss_dict["rep_loss"],
            },
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )

        all_videos, video_metadata = self._sample_all_videos(batch)

        if self.trainer.sanity_checking:
            rank_zero_print(cyan("--- videos successfully predicted for sanity checking ---"))
        elif not (
            self.trainer.sanity_checking and not self.logging_cfg.sanity_generation
        ):
            for key, value in all_videos.items():
                if value.shape[2] != 3:
                    all_videos[key] = value.repeat(1, 1, 3, 1, 1)
            self._update_metrics(all_videos, dataloader_idx=dataloader_idx)
            self._log_videos(
                all_videos,
                namespace,
                dataloader_idx=dataloader_idx,
                video_metadata=video_metadata,
            )
        return

    def _sample_all_videos(
        self, batch
    ) -> Tuple[Dict[str, Tensor], Dict]:
        xs, conds, reset, valid_mask, gt_videos, video_metadata = batch
        actions = self._prepare_actions(conds)

        context_video = xs[:, : self.n_context_frames]
        context_actions = actions[:, : self.n_context_frames]
        context_reset = reset[:, : self.n_context_frames]
        context_valid = valid_mask[:, : self.n_context_frames]
        future_actions = actions[:, self.n_context_frames :]
        future_reset = reset[:, self.n_context_frames :]
        future_valid = valid_mask[:, self.n_context_frames :]

        state, _, _ = self.main_model.observe(
            context_video,
            context_actions,
            reset=context_reset,
            valid_mask=context_valid,
        )
        _, preds = self.main_model.imagine(
            future_actions,
            state,
            reset=future_reset,
            valid_mask=future_valid,
        )
        
        preds = preds * future_valid[:, :, None, None, None].to(preds.dtype)

        all_videos: Dict[str, Tensor] = {"gt": gt_videos.detach(), "prediction": preds.detach()}
        all_videos["prediction"] = torch.cat(
            [all_videos["gt"][:, : self.n_context_frames], all_videos["prediction"]], dim=1
        )
        
        return all_videos, video_metadata
