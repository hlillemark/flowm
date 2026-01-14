from typing import Optional, Dict, Tuple
from omegaconf import DictConfig
import torch
from torch import Tensor
from lightning.pytorch.utilities.types import STEP_OUTPUT
from algorithms.common import BasePytorchAlgo, BaseWMMixin
import torch.nn as nn

from utils.print_utils import cyan, red
from utils.distributed_utils import rank_zero_print


class FloWMVideo(BaseWMMixin, BasePytorchAlgo):
    """
    An algorithm for training and evaluating
    World Models' memory ability on video datasets on different tasks
    """
    def __init__(self, cfg: DictConfig) -> None:
        
        # 1. Shape
        self.x_shape = list(cfg.x_shape)
        self.frame_skip = cfg.frame_skip
        self.external_cond_dim = cfg.external_cond_dim * (
            cfg.frame_skip if cfg.external_cond_stack else 1
        )

        # 2. Latent
        self.is_latent_diffusion = cfg.latent.enable
        self.use_preprocessed_latents = cfg.latent.enable and cfg.latent.type.startswith("pre_")
        self.temporal_downsampling_factor = cfg.latent.downsampling_factor[0]
        self.is_latent_video_vae = self.temporal_downsampling_factor > 1

        self.x_shape = [cfg.latent.num_channels] + [
            d // cfg.latent.downsampling_factor[1] for d in self.x_shape[1:]
        ]
        if self.is_latent_video_vae:
            self.check_video_vae_compatibility(cfg)

        self.backbone_cfg = cfg.backbone

        # 4. Logging
        self.logging_cfg = cfg.logging

        self.tasks = [
            task
            for task in ["prediction", "interpolation", "reconstruction"] # Flowm only supports prediction
            if getattr(cfg.tasks, task).enabled
        ]
        self.num_logged_videos = 0
        self.generator = None
        self.latent_size = None

        self.main_model_prefix = "main_model"
        
        super().__init__(cfg)


    # ---------------------------------------------------------------------
    # Model & Metrics building
    # ---------------------------------------------------------------------
    def configure_model(self) -> None:
        """
        Build the model
        """
        
        # Diffusion Model
        vae_patch_size = self.cfg.vae.pretrained_kwargs.patch_size
        _, H, W = map(int, list(self.cfg.x_shape))  
        self.latent_size = [W // vae_patch_size, H // vae_patch_size]
        
        backbone_cfg = self.backbone_cfg

        from algorithms.mem_wm.backbones import create_main_model

        self.main_model = create_main_model(
            backbone_cfg=backbone_cfg
        )

        if self.cfg.load_model_state:
            if not self.cfg.model_state_key:
                raise ValueError(f"model_state_key is not set in the config, while loading model state from {self.cfg.load_model_state}")
            model_dict = torch.load(self.cfg.load_model_state, map_location="cpu", weights_only=False)
            if self.cfg.model_state_key in model_dict:
                model_dict = model_dict[self.cfg.model_state_key]
            else:
                rank_zero_print(red(f"!!! No model state found in {self.cfg.load_model_state}, while loading from {self.cfg.model_state_key} !!!"))

            # remove "diffusion_model." prefix
            model_dict = {k.replace(f"{self.main_model_prefix}.", ""): v for k, v in model_dict.items()}
            load_model = self.cfg.load_model_state_mode
            if load_model == "strict":
                missing_keys, unexpected_keys = self.main_model.load_state_dict(model_dict, strict = True)
            elif load_model == "partial":
                missing_keys, unexpected_keys = self.main_model.load_state_dict(model_dict, strict = False)
            else:
                raise ValueError(f"Invalid load_model: {load_model}")
            
            rank_zero_print(cyan(f"\n ==== Successfully loaded {len(model_dict)} parameters from {self.cfg.load_model_state}[{self.cfg.model_state_key}], load model mode: {load_model} ===="))

        # VAE
        if self.is_latent_diffusion and not self.use_preprocessed_latents:
            raise NotImplementedError("VAE should be turned off for flowm experiments...")
            # self._load_vae()
            
        # Build metrics
        self._build_metrics()

    def training_step(self, batch, batch_idx, dataloader_idx = 0, namespace="training") -> STEP_OUTPUT:
        """
        Training step
        """
        criterion = nn.MSELoss()
        input_seq, input_actions, target_actions, target_seq, gt_videos, _ = batch

        preds = self.main_model(input_seq, input_actions, target_actions, target_seq=target_seq, teacher_forcing_ratio=self.cfg.teacher_forcing)

        loss = criterion(preds, target_seq)

        output_dict = {
            "loss": loss,
            "predicted_x": preds,
            "original_x": target_seq,
        }

        if batch_idx % self.logging_cfg.loss_freq == 0:
            metrics_dict = {
                f"{namespace}/loss": loss,
            }
            
            self.log_dict(
                metrics_dict,
                on_step=namespace == "training",
                on_epoch=namespace != "training",
                sync_dist=True,
                prog_bar=True
            )

        return output_dict


    def on_after_batch_transfer(
        self, batch: Dict, dataloader_idx: int
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Optional[Tensor]]:
        """
        Preprocess the batch before training/validation.

        Args:
            batch (Dict): The batch of data. Contains "videos" or "latents", (optional) "conditions", and "masks".
            dataloader_idx (int): The index of the dataloader.
        Returns:
            xs (Tensor, "B n_tokens *x_shape"): Tokens to be processed by the model.
            conditions (Optional[Tensor], "B n_tokens d"): External conditions for the tokens.
            masks (Tensor, "B n_tokens"): Masks for the tokens.
            gt_videos (Optional[Tensor], "B n_frames *x_shape"): Optional ground truth videos, used for validation in latent diffusion.
        """
        # 1. Tokenize the videos and optionally prepare the ground truth videos
        if type(batch) == list:
            batch = batch[dataloader_idx]
            
        device = self.device

        # Prepare tokens according to latent config
        if self.is_latent_diffusion:
            if self.use_preprocessed_latents:
                xs = batch["latents"]
            else:
                xs = self._encode(batch["videos"])  # encode on-the-fly
            xs = xs.to(device)

            # Ground-truth videos for metrics/logging
            if "videos" in batch:
                gt_videos = batch["videos"].to(device)
            else:
                if hasattr(self, "vae") and self.vae is not None:
                    gt_videos = self._decode(xs)
                else:
                    gt_videos = None
        else:
            videos = batch["videos"]
            if self.cfg.backbone.input_channels == 1:
                # convert rgb videos to grey_scale videos [B, T, 3, H, W] -> [B, T, 1, H, W]
                videos = videos.mean(dim=2, keepdim=True)
            xs = videos.to(device)
            gt_videos = xs

        conds = batch["conds"].to(device)

        # Split into context/target
        input_seq = xs[:, :self.n_context_frames]
        input_actions = conds[:, :self.n_context_frames]
        target_actions = conds[:, self.n_context_frames:]
        target_seq = xs[:, self.n_context_frames:]
        
        return input_seq, input_actions, target_actions, target_seq, gt_videos, batch["metadata"]

    
    # ---------------------------------------------------------------------
    # Validation & Test
    # ---------------------------------------------------------------------

    def on_validation_epoch_start(self) -> None:
        # If using preprocessed latents, we don't keep the vae in memory during training. need 
        # to reload it during validation. 
        if self.is_latent_diffusion and self.use_preprocessed_latents:
            self._load_vae()
        self.num_logged_videos = [0] * len(self.trainer.val_dataloaders)
        if self.cfg.logging.deterministic is not None:
            self.generator = torch.Generator(device=self.device).manual_seed(
                self.cfg.logging.deterministic
            )

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0, namespace="validation") -> STEP_OUTPUT:
        """Validation step"""
        # 1. If running validation while training a model, directly evaluate
        # the denoising performance to detect overfitting, etc.
        # Logs the "denoising_vis" visualization as well as "validation/loss" metric.

        all_videos, video_metadata = self._sample_all_videos(batch, batch_idx, namespace)

        if self.trainer.sanity_checking:
            rank_zero_print(cyan(f"--- videos successfully predicted for sanity checking ---"))

        # 2. Sample all videos (based on the specified tasks)
        # and log the generated videos and metrics
        elif not (
            self.trainer.sanity_checking and not self.logging_cfg.sanity_generation
        ):
            # check whether the video channel is 3, if not, need to repeat the video to 3 channels
            for k, v in all_videos.items():
                if v.shape[2] != 3:
                    all_videos[k] = v.repeat(1, 1, 3, 1, 1)
            self._update_metrics(all_videos, dataloader_idx=dataloader_idx)
            self._log_videos(all_videos, namespace, dataloader_idx=dataloader_idx, video_metadata=video_metadata)

        return

    # ---------------------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------------------


    def _sample_all_videos(
        self, batch, batch_idx, namespace="validation"
    ) -> Optional[Dict[str, Tensor]]:
        """
        Top functions for sampling videos, where we first decide which tasks to carry out,
        then call the corresponding sampling functions.
        This is also where we unnormalize the latents and decode to get videos as the return of the function.
        """
        input_seq, input_actions, target_actions, target_seq, gt_videos, video_metadata = batch
        input_actions = input_actions.to(self.dtype)
        target_actions = target_actions.to(self.dtype)
        all_videos: Dict[str, Tensor] = {"gt": gt_videos}
        preds = self.main_model(
            input_seq, input_actions, target_actions, return_latent_video=True
        )
        # all black baseline
        # preds = torch.zeros_like(target_seq)
        
        latent_video = None
        if isinstance(preds, tuple):
            preds, latent_video = preds

        all_videos["prediction"] = preds
        # latent video is magnitude of map (grayscale); replicate to 3ch for logging
        if latent_video is not None:
            # (B,T,1,H,W) -> (B,T,3,H,W)
            all_videos["latent_map_mag"] = latent_video.repeat(1, 1, 3, 1, 1)

        # remove None values
        all_videos = {k: v for k, v in all_videos.items() if v is not None}
        # rearrange/unnormalize/detach the videos
        all_videos = {k: v.detach() for k, v in all_videos.items()}
        # decode latents if using latents
        if self.is_latent_diffusion:
            all_videos = {
                k: self._decode(v) if k != "gt" else gt_videos
                for k, v in all_videos.items()
            }
            
        # concatenate the context frames of video predictions with the ground truth
        if "prediction" in all_videos:
            all_videos["prediction"] = torch.cat([
                all_videos["gt"][:, : self.n_context_frames],
                all_videos["prediction"],
            ], dim=1)
        if "latent_map_mag" in all_videos:
            # Use a blank placeholder for context to align with prediction concat
            B, T_pred, C, H, W = all_videos["latent_map_mag"].shape
            zeros_ctx = torch.zeros(B, self.n_context_frames, C, H, W, device=all_videos["latent_map_mag"].device, dtype=all_videos["latent_map_mag"].dtype)
            all_videos["latent_map_mag"] = torch.cat([
                zeros_ctx,
                all_videos["latent_map_mag"],
            ], dim=1)

        
        return all_videos, video_metadata
    
