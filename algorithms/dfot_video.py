from typing import Optional, Dict, Callable, Tuple
from functools import partial
from omegaconf import DictConfig, open_dict
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from lightning.pytorch.utilities.types import STEP_OUTPUT
from einops import rearrange, repeat, reduce
from tqdm import tqdm
import warnings
import sys
from algorithms.common import BasePytorchAlgo, BaseWMMixin

# Diffusion Models
from algorithms.mem_wm.diffusion import create_diffusion_algo
# VAE
from algorithms.mem_wm.diffusion.gaussian_diffusion import GaussianDiffusion
# Sampling
from .history_guidance import HistoryGuidance

from utils.print_utils import cyan, blue, bold_red, red
from utils.distributed_utils import rank_zero_print, is_rank_zero
from utils.torch_utils import bernoulli_tensor
from algorithms.mem_wm.diffusion.scheduler import DDIMScheduler, DDPMScheduler

class DFoTVideo(BaseWMMixin, BasePytorchAlgo):
    """
    An algorithm for training and evaluating
    World Models' memory ability on video datasets on different tasks
    """
    def __init__(self, cfg: DictConfig) -> None:
        
        # 1. Shape
        self.x_shape = list(cfg.x_shape)
        self.frame_skip = cfg.frame_skip
        self.chunk_size = cfg.chunk_size
        self.external_cond_dim = cfg.external_cond_dim * (
            cfg.frame_skip if cfg.external_cond_stack else 1
        )

        # 2. Latent
        self.is_latent_diffusion = cfg.latent.enable
        self.use_preprocessed_latents = cfg.latent.enable and cfg.latent.type.startswith("pre_")
        rank_zero_print(
            blue(
                f"latent information: \n - Using latent diffusion: {self.is_latent_diffusion}, \n - use preprocessed latents: {self.use_preprocessed_latents}"
            )
        )
        self.temporal_downsampling_factor = cfg.latent.downsampling_factor[0]
        self.is_latent_video_vae = self.temporal_downsampling_factor > 1

        self.x_shape = [cfg.latent.num_channels] + [
            d // cfg.latent.downsampling_factor[1] for d in self.x_shape[1:]
        ]
        if self.is_latent_video_vae:
            self.check_video_vae_compatibility(cfg)

        # 3. Diffusion
        self.use_causal_mask = cfg.diffusion.use_causal_mask
        self.clip_noise = cfg.diffusion.clip_noise
        
        self.diffusion_cfg = cfg.diffusion
        self.backbone_cfg = cfg.backbone
        self.timesteps = self.diffusion_cfg.timesteps

        self.attn_mask = None

        # 3.1 History Guidance
        self.is_full_sequence = (
            cfg.noise_level == "random_uniform" # default will be random independent noise
            and not cfg.fixed_context.enabled # no context
            and not cfg.variable_context.enabled # no variable context
        )

        # 4. Logging
        self.logging_cfg = cfg.logging
            
        self.tasks = [
            task
            for task in ["prediction", "interpolation", "reconstruction"]
            if getattr(cfg.tasks, task).enabled
        ]
        self.num_logged_videos = 0
        self.generator = None
        self.latent_size = None

        self.main_model_prefix = "diffusion_model"
        
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
        channel_num = backbone_cfg.in_channels

        # Update DictConfig with new keys
        with open_dict(backbone_cfg):
            backbone_cfg.in_channels = channel_num
            backbone_cfg.out_channels = channel_num
            backbone_cfg.sample_width = self.latent_size[0]
            backbone_cfg.sample_height = self.latent_size[1]

            
        self.diffusion_model = create_main_model(
            backbone_cfg=backbone_cfg
        )

        if self.cfg.load_model_state:
            model_dict = torch.load(self.cfg.load_model_state, map_location="cpu", weights_only=False)
            if self.cfg.model_state_key in model_dict:
                model_dict = model_dict[self.cfg.model_state_key]
            else:
                state_keys = list(model_dict.keys())
                rank_zero_print(red(f"\n !!! No model state found in {self.cfg.load_model_state}, while loading using \"{self.cfg.model_state_key}\" to load model state !!! \n"))
                rank_zero_print(red(f"Available keys: {state_keys}. You should use one of them to load the state for the model ! \n"))
                raise ValueError(f"No model state found in {self.cfg.load_model_state}, while loading using \"{self.cfg.model_state_key}\" to load model state")

            # remove "diffusion_model." prefix
            model_dict = {k.replace("diffusion_model.", ""): v for k, v in model_dict.items()}
            load_model = self.cfg.load_model_state_mode
            if load_model == "strict":
                missing_keys, unexpected_keys = self.diffusion_model.load_state_dict(model_dict, strict = True)
            elif load_model == "partial":
                missing_keys, unexpected_keys = self.diffusion_model.load_state_dict(model_dict, strict = False)
            elif load_model == "smart":
                from utils.ckpt_utils import smart_load_state_dict
                if hasattr(self.cfg, "handlers"):
                    handlers = self.cfg.handlers
                    if handlers == "depth_fine_tuning":
                        from utils.ckpt_utils import depth_fine_tuning_handlers
                        skipped_keys = smart_load_state_dict(self.diffusion_model, model_dict, custom_handlers = depth_fine_tuning_handlers, verbose = True)
                    else:
                        raise ValueError(f"Invalid handler: {handlers}")
            else:
                raise ValueError(f"Invalid load_model: {load_model}")
            
            rank_zero_print(cyan(f"\n ==== Successfully loaded {len(model_dict)} parameters from {self.cfg.load_model_state}[{self.cfg.model_state_key}], load model mode: {load_model} ===="))

    
        self.diffusion_algo: GaussianDiffusion = create_diffusion_algo(
                sampling_timesteps=self.sampling_timesteps,  
                noise_schedule = self.diffusion_cfg.noise_schedule,
                mean_type = self.diffusion_cfg.mean_type,
                var_type = self.diffusion_cfg.var_type,
                cfg = self.diffusion_cfg,
                logger = self.logger
            ).to(self.diffusion_model.dtype)  # default: 1000 steps, linear noise schedule 

        # VAE
        if self.is_latent_diffusion and not self.use_preprocessed_latents:
            self._load_vae()
        
        # Build metrics
        self._build_metrics()

    def training_step(self, batch, batch_idx, dataloader_idx = 0, namespace="training") -> STEP_OUTPUT:
        """
        Training step
        """
        
        diffusion_cfg = self.cfg.diffusion
        
        xs, conds, masks, gt_videos, metadata = batch
        
        t , masks, attn_mask = self._get_training_noise_levels(xs, masks, strategy = diffusion_cfg.strategy, strategy_kwargs = diffusion_cfg.strategy_kwargs)
        conds = conds.to(xs.dtype) # BUG: ugly hack to fix the dtype mismatch
        bs = xs.shape[0]
        
        if diffusion_cfg.strategy == "diffusion-forcing":
            cfg_dropping_strategy = "frame_wise"
        else:
            raise ValueError(f"Invalid strategy: {diffusion_cfg.strategy}")
        
        model_kwargs = dict(strategy=diffusion_cfg.strategy, 
                            noise_abx_max = diffusion_cfg.noise_abs_max, 
                            masks = masks,
                            cfg_dropping_strategy = cfg_dropping_strategy,
                            attention_mask = attn_mask,)
        
        loss_dict = self.diffusion_algo.training_loss(
            self.diffusion_model, x_start = xs, conditions = conds, masks = masks,
            t = t, model_kwargs = model_kwargs, )
        
        loss = loss_dict["loss"].mean()

        predicted_x_start = loss_dict["predicted_x_start"]
        original_x = loss_dict["original_x"]
        output_dict = {
            "loss": loss,
            "predicted_x_start": predicted_x_start,
            "original_x": original_x,
        }

        if "throughput" in loss_dict:
            output_dict["throughput"] = loss_dict["throughput"]


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

    def _get_training_noise_levels(
        self, xs: Tensor, masks: Tensor = None, strategy = "diffusion-forcing", strategy_kwargs: Dict = None
    ) -> Tuple[Tensor, Tensor]:
        match strategy:
            case "diffusion-forcing":
                match strategy_kwargs.type:
                    case "original":
                        return self._get_training_noise_levels_diffusion_forcing(xs, masks)
                    case "block-ssm":
                        return self._get_training_noise_levels_block_ssm(xs, masks)
                    case _:
                        raise ValueError(f"Invalid strategy: {strategy_kwargs.type}")
            case _:
                raise ValueError(f"Invalid strategy: {strategy}")

    def _get_training_noise_levels_diffusion_forcing(
        self, xs: Tensor, masks: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Generate random noise levels for training.

        Parameters:
            masks: from data loader / on_after_batch_transfer, one indicates not padding, so loss need to be calculated on that
        
        """
        batch_size, n_tokens, *_ = xs.shape

        # random function different for continuous and discrete diffusion
        rand_fn = partial(
            *(
                (torch.rand,)
                if self.cfg.diffusion.is_continuous
                else (torch.randint, 0, self.timesteps)
            ),
            device=xs.device,
            generator=self.generator,
        )

        # baseline training (SD: fixed_context, BD: variable_context)
        context_mask = None
        if self.cfg.variable_context.enabled:
            assert (
                not self.cfg.fixed_context.enabled
            ), "Cannot use both fixed and variable context"
            context_mask = bernoulli_tensor(
                (batch_size, n_tokens),
                self.cfg.variable_context.prob, # NOTE: variable length is randomly decided, but this looks weird, as the context don't need to be consecutive frames ?
                device=self.device,
                generator=self.generator,
            ).bool()
        elif self.cfg.fixed_context.enabled:
            context_indices = self.cfg.fixed_context.indices or list(
                range(self.n_context_tokens)
            )
            context_mask = torch.zeros(
                (batch_size, n_tokens), dtype=torch.bool, device=xs.device
            )
            context_mask[:, context_indices] = True

        match self.cfg.noise_level:
            case "random_independent":  # independent noise levels (Diffusion Forcing)
                noise_levels = rand_fn((batch_size, n_tokens))
            case "random_uniform":  # uniform noise levels (Typical Video Diffusion)
                noise_levels = rand_fn((batch_size, 1)).repeat(1, n_tokens)
        
        if self.cfg.varlen_context.enabled: # NOTE: this varlen_context requires the context to be consecutive frames
            # [Deprecated Branch, but leave here for now]
            raise ValueError(bold_red(f"varlen_context is not supported in diffusion forcing training"))

            n_context_tokens = torch.randint(
                max(self.cfg.varlen_context.min, 1),
                min(self.cfg.varlen_context.max, self.forward_window_size_in_tokens),
                (1,),
                device=xs.device,
                generator=self.generator,
            )
        else:
            n_context_tokens = self.n_context_tokens

        if self.cfg.uniform_future.enabled:  # simplified training (Appendix A.5)
            noise_levels[:, n_context_tokens :] = rand_fn((batch_size, 1)).repeat(
                1, n_tokens - n_context_tokens
            )

        # treat frames that are not available as "full noise"
        noise_levels = torch.where(
            reduce(masks.bool(), "b t ... -> b t", torch.any),
            noise_levels,
            torch.full_like(
                noise_levels,
                1 if self.cfg.diffusion.is_continuous else self.timesteps - 1,
            ),
        )

        if context_mask is not None:
            # binary dropout training to enable guidance
            dropout = (
                (
                    self.cfg.variable_context
                    if self.cfg.variable_context.enabled
                    else self.cfg.fixed_context
                ).dropout
                if self.trainer.training
                else 0.0
            ) # always zero
            if dropout != 0.0:
                raise ValueError("Dropout should always be 0.0 during training.")
            context_noise_levels = bernoulli_tensor(
                (batch_size, 1),
                dropout,
                device=xs.device,
                generator=self.generator,
            )
            if not self.cfg.diffusion.is_continuous:
                context_noise_levels = context_noise_levels.long() * (
                    self.timesteps - 1
                )
            noise_levels = torch.where(context_mask, context_noise_levels, noise_levels)

            # modify masks to exclude context frames from loss computation
            context_mask = rearrange(
                context_mask, "b t -> b t" + " 1" * len(masks.shape[2:])
            )
            masks = torch.where(context_mask, False, masks)

        return noise_levels, masks, None

    def _get_training_noise_levels_block_ssm(
        self, xs: Tensor, masks: Tensor = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Generate random noise levels for training.

        Parameters:
            masks: from data loader / on_after_batch_transfer, one indicates not padding, so loss need to be calculated on that

        """
        batch_size, n_tokens, *_ = xs.shape

        # random function different for continuous and discrete diffusion
        rand_fn = partial(
            *(
                (torch.rand,)
                if self.cfg.diffusion.is_continuous
                else (torch.randint, 0, self.timesteps)
            ),
            device=xs.device,
            generator=self.generator,
        )

        noise_levels = rand_fn((batch_size, n_tokens))
        
        if self.cfg.varlen_context.enabled: # NOTE: this varlen_context requires the context to be consecutive frames
            if not self.cfg.varlen_context.min or not self.cfg.varlen_context.max:
                raise ValueError(bold_red(f"varlen_context.min and varlen_context.max must be set when varlen_context is enabled"))
            
            if self.cfg.varlen_context.max != self.forward_window_size_in_tokens:
                if is_rank_zero:
                    warnings.warn(bold_red(f"varlen_context.max is set to {self.cfg.varlen_context.max}, but the size of forward window is {self.forward_window_size_in_tokens}. For Long Context SSM baseline training following that paper, the max context length should always be the forward window size \n"))
            
            max_context_tokens = self.cfg.varlen_context.max
            min_context_tokens = self.cfg.varlen_context.min
            if max_context_tokens > min_context_tokens:
                n_context_tokens = torch.randint(
                    max(min_context_tokens, 1),
                    max_context_tokens, 
                    (1,),
                    device=xs.device,
                    generator=self.generator,
                )
            elif max_context_tokens == min_context_tokens:
                n_context_tokens = torch.tensor(min_context_tokens, device=xs.device)
            else:
                raise ValueError(bold_red(f"varlen_context.max must be greater than or equal to varlen_context.min"))
        else:
            assert False
            
        use_clean_context_indices = torch.rand(batch_size) <= self.cfg.varlen_context.prob 
        noise_levels[use_clean_context_indices, :n_context_tokens] = 0 # clean frames
        context_mask = torch.ones_like(masks) # (bs, t), zero indicate the context frames, whose loss should be ignored
        context_mask[use_clean_context_indices, :n_context_tokens] = 0

        # treat frames that are not available as "full noise"
        noise_levels = torch.where(
            reduce(masks.bool(), "b t ... -> b t", torch.any),
            noise_levels,
            torch.full_like(
                noise_levels,
                1 if self.cfg.diffusion.is_continuous else self.timesteps - 1,
            ),
        )

        masks = masks & context_mask

        attn_masks = None # For ssm baseline, we use FrameLocalAttention; We handle attn masks there, so we leave it as None here

        return noise_levels, masks, attn_masks

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
            
        device = self.diffusion_model.device
        # device = self.device
        if self.is_latent_diffusion:
            if self.use_preprocessed_latents:
                xs = batch["latents"]
            else:
                xs = self._encode(batch["videos"])
            if "videos" in batch:
                gt_videos = batch["videos"].to(device)
            else:
                if hasattr(self, "vae") and self.vae is not None:
                    gt_videos = self._decode(xs) # this means we only have latents, so we have to decode them to get the ground truth videos
                else:
                    gt_videos = None
        else:
            xs = batch["videos"]

        xs = xs.to(device)
        # 2. Prepare external conditions
        conditions = batch.get("conds", None)
        conditions = conditions.to(device) if conditions is not None else None
        #
        # 3. Prepare the masks
        if "masks" in batch:
            assert (
                not self.is_latent_video_vae
            ), "Masks should not be provided from the dataset when using VideoVAE. " 
        else:
            masks = torch.ones(*xs.shape[:2]).bool().to(device)

        return xs, conditions, masks, gt_videos, batch["metadata"]
    

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
        
        if self.trainer.sanity_checking:
            if "reconstruction" in self.tasks:
                self._eval_denoising(batch, batch_idx, dataloader_idx=dataloader_idx, namespace=namespace)

        # 2. Sample all videos (based on the specified tasks)
        # and log the generated videos and metrics
        
        elif not (
            self.trainer.sanity_checking and not self.logging_cfg.sanity_generation
        ):
            if "reconstruction" in self.tasks:
                self._eval_denoising(batch, batch_idx, dataloader_idx=dataloader_idx, namespace=namespace)

            if self.cfg.validation.sample_during_training:
                all_videos, video_metadata = self._sample_all_videos(batch, batch_idx, namespace)
                self._update_metrics(all_videos, dataloader_idx=dataloader_idx)
                self._log_videos(all_videos, namespace, dataloader_idx=dataloader_idx, video_metadata=video_metadata)
        return

    @torch.no_grad()
    def _eval_denoising(self, batch, batch_idx, dataloader_idx, namespace="training") -> None:
        """Evaluate the denoising performance during training."""
        
        xs, conditions, masks, gt_videos, video_metadata = batch

        xs = xs[:, : self.forward_window_size_in_tokens]
        if conditions is not None:
            conditions = conditions[:, : self.forward_window_size_in_tokens]
        masks = masks[:, : self.forward_window_size_in_tokens]
        if gt_videos is not None:
            gt_videos = gt_videos[:, : self.forward_window_size_in_tokens]

        batch = (xs, conditions, masks, gt_videos, video_metadata)
        output = self.training_step(batch, batch_idx, dataloader_idx = dataloader_idx, namespace=namespace)

        gt_videos = gt_videos if self.is_latent_diffusion else output["original_x"]
        recons = output["predicted_x_start"]
        if self.is_latent_diffusion:
            recons = self._decode(recons)

        if recons.shape[1] < gt_videos.shape[1]:  # recons.ndim is 5
            recons = F.pad(
                recons,
                (0, 0, 0, 0, 0, 0, 0, gt_videos.shape[1] - recons.shape[1], 0, 0),
            )

        all_videos = {
            "gt": gt_videos,
            "reconstruction": recons,
        }
            
        self._log_videos(all_videos, namespace, dataloader_idx, video_metadata=video_metadata)
        
        if not self.trainer.sanity_checking:
            self._update_metrics(all_videos)
    

    # ---------------------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------------------

    # History Guidance
    def _history_guidance_inference(
        self, xs: Tensor, conditions: Optional[Tensor] = None, prediction_kwargs: Optional[Dict] = None
    ) -> Tensor:
        """
        Predict the videos with the given context, using sliding window rollouts if necessary.
        Optionally, if cfg.tasks.prediction.keyframe_density < 1, predict the keyframes first,
        then interpolate the missing intermediate frames.
        Input:
        - xs: (batch_size, n_tokens, *self.x_shape)
        - conditions: (batch_size, n_tokens, dim)
        - prediction_kwargs: Dict
        Output:
        - xs_pred: (batch_size, n_tokens, *self.x_shape)
        """
        xs_pred = xs.clone()

        history_guidance = HistoryGuidance.from_config(
            config=self.cfg.tasks.prediction.history_guidance,
            timesteps=self.timesteps,
        )

        density = self.cfg.tasks.prediction.history_guidance.keyframe_density or 1
        if density > 1:
            raise ValueError("tasks.prediction.keyframe_density must be <= 1")
        keyframe_indices = (
            torch.linspace(0, xs_pred.shape[1] - 1, round(density * xs_pred.shape[1]))
            .round()
            .long()
        )
        keyframe_indices = torch.cat(
            [torch.arange(self.n_context_tokens), keyframe_indices]
        ).unique()  # context frames are always keyframes
        key_conditions = (
            conditions[:, keyframe_indices] if conditions is not None else None
        )

        # 1. Predict the keyframes
        xs_pred_key, *_ = self._hg_predict_sequence(
            context=xs_pred[:, : self.n_context_tokens],
            length=len(keyframe_indices),
            # length=self.forward_window_size_in_tokens,
            conditions=key_conditions,
            history_guidance=history_guidance,
            reconstruction_guidance=self.cfg.diffusion.reconstruction_guidance,
            sliding_context_len=self.cfg.tasks.prediction.history_guidance.sliding_context_len # default is None
            or self.forward_window_size_in_tokens // 2,
            prediction_kwargs=prediction_kwargs,
        )
        
        xs_pred[:, keyframe_indices] = xs_pred_key.to(xs_pred.dtype)

        # 2. (Optional) Interpolate the intermediate frames
        if len(keyframe_indices) < xs_pred.shape[1]:
            context_mask = torch.zeros(xs_pred.shape[:2], device=self.device).bool()
            context_mask[:, keyframe_indices] = True
            xs_pred = self._interpolate_videos(
                context=xs_pred,
                context_mask=context_mask,
                conditions=conditions,
            )

        return xs_pred

    def _hg_predict_sequence(
        self,
        context: torch.Tensor,
        length: Optional[int] = None,
        conditions: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        reconstruction_guidance: float = 0.0,
        history_guidance: Optional[HistoryGuidance] = None,
        sliding_context_len: Optional[int] = None,
        return_all: bool = False,
        prediction_kwargs: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict a sequence given context tokens at the beginning, using sliding window if necessary.
        Args
        ----
        context: torch.Tensor, Shape (batch_size, init_context_len, *self.x_shape)
            Initial context tokens to condition on
        length: Optional[int]
            Desired number of tokens in sampled sequence.
            If None, fall back to to self.forward_window_size_in_tokens, and
            If bigger than self.forward_window_size_in_tokens, sliding window sampling will be used.
        conditions: Optional[torch.Tensor], Shape (batch_size, conditions_len, ...)
            Unprocessed external conditions for sampling, e.g. action or text, optional
        guidance_fn: Optional[Callable]
            Guidance function for sampling
        reconstruction_guidance: float
            Scale of reconstruction guidance (from Video Diffusion Models Ho. et al.)
        history_guidance: Optional[HistoryGuidance]
            History guidance object that handles compositional generation
        sliding_context_len: Optional[int]
            Max context length when using sliding window. -1 to use forward_window_size_in_tokens - 1. self.forward_window_size_in_tokens - sliding_context_len = sliding_stride
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,*32*(newly generated frame)
            Has no influence when length <= self.forward_window_size_in_tokens as no sliding window is needed.
        return_all: bool
            Whether to return all steps of the sampling process.

        Returns
        -------
        xs_pred: torch.Tensor, Shape (batch_size, length, *self.x_shape)
            Predicted sequence with both context and generated tokens
        record: Optional[torch.Tensor], Shape (num_steps, batch_size, length, *self.x_shape)
            Record of all steps of the sampling process
        """
        if length is None:
            length = self.forward_window_size_in_tokens
        else:
            if length < self.forward_window_size_in_tokens:
                raise ValueError(
                    f"length is expected to be >= forward_window_size_in_tokens, got {length}. Otherwise this will misalign with the training distribution"
                )
        if sliding_context_len is None:
            if self.forward_window_size_in_tokens < length:
                raise ValueError(
                    "when length > forward_window_size_in_tokens, sliding_context_len must be specified."
                )
            else:
                sliding_context_len = self.forward_window_size_in_tokens - 1
        if sliding_context_len == -1:
            sliding_context_len = self.forward_window_size_in_tokens - 1

        batch_size, context_len, *_ = context.shape

        if sliding_context_len < context_len:
            raise ValueError(
                "sliding_context_len is expected to be >= length of initial context,"
                f"got {sliding_context_len}. If you are trying to use max context, "
                "consider specifying sliding_context_len=-1."
            )

        # if the number of tokens we would like to generate does not exactly line up with the sliding window size,
        # we can add some padding, then cut it down later, so that we never pass the 'incorrect' number 
        # of frames into the forward function of the model, keeping it consistent throughout generation
        # case 1: context_length = 16, original_length = 128, forward_window_size_in_tokens = 128, sliding_context_len = 64. Then the length should be 128 -> one generation round
        # case 2: context_length = 16, original_length = 192, forward_window_size_in_tokens = 128, sliding_context_len = 64. Then the length should be 192 -> two generation rounds
        # case 3: context_length = 16, original_length = 162, forward_window_size_in_tokens = 128, sliding_context_len = 64. Then the length should be 192 -> two generation rounds, then cut it down to 162
        original_length = length
        stride = self.forward_window_size_in_tokens - sliding_context_len
        num_generation_rounds = 1 + max(0, (length - self.forward_window_size_in_tokens + stride - 1) // stride)
        # num_generation_rounds = 1 + (length - sliding_context_len - 1) // (self.forward_window_size_in_tokens - sliding_context_len)
        length = self.forward_window_size_in_tokens + stride * (num_generation_rounds - 1)
        if length > original_length:
            if conditions is not None:
                # Add zero padding to conditions to match the extended length
                padding_length = length - original_length
                rank_zero_print(cyan(
                    f"Zero-padding actions by {padding_length} frames to match inference block size; extra frames will be truncated after generation."
                ))
                padding = torch.zeros((batch_size, padding_length, *conditions.shape[2:]), dtype=conditions.dtype, device=conditions.device)
                conditions = torch.cat([conditions, padding], dim=1)

        chunk_size = self.chunk_size if self.use_causal_mask else self.forward_window_size_in_tokens # use_causal_mask default is False
        
        curr_token = context_len
        xs_pred = context
        x_shape = self.x_shape
        record = None
        disable = not sys.stdout.isatty()
        pbar = tqdm(
            total=self.sampling_timesteps * num_generation_rounds,
            initial=0,
            desc="Predicting with DFoT",
            leave=False,
            disable=disable
        )
 
        while curr_token < length:
            if record is not None:
                raise ValueError("return_all is not supported if using sliding window.")
            # actual context depends on whether it's during sliding window or not
            # corner case at the beginning
            c = min(sliding_context_len, curr_token) # c is the realtime context length
            # try biggest prediction chunk size
            h = min(length - curr_token, self.forward_window_size_in_tokens - c)
            # chunk_size caps how many future tokens are diffused at once to save compute for causal model
            h = min(h, chunk_size) if chunk_size > 0 else h
            l = c + h
            pad = torch.zeros((batch_size, h, *x_shape))
            # context is last c tokens out of the sequence of generated/gt tokens
            # pad to length that's required by _hs_sample_sequence
            context = torch.cat([xs_pred[:, -c:], pad.to(self.device)], 1)
            # calculate number of model generated tokens (not GT context tokens)
            generated_len = curr_token - max(curr_token - c, context_len) # context len is a fixed constant, for a given initial context length
            # make context mask
            context_mask = torch.ones((batch_size, c), dtype=torch.long)
            if generated_len > 0:
                context_mask[:, -generated_len:] = 2
            pad = torch.zeros((batch_size, h), dtype=torch.long)
            context_mask = torch.cat([context_mask, pad.long()], 1).to(context.device)

            # if use causal mask, then the window len is not guaranteed to be the forward_window_size_in_tokens
            window_len = l if self.use_causal_mask else self.forward_window_size_in_tokens
            cond_slice = None
            if conditions is not None:
                cond_slice = conditions[:, curr_token - c : curr_token - c + window_len]

            new_pred, record = self._hs_sample_sequence(
                batch_size,
                length=l,
                context=context,
                context_mask=context_mask,
                conditions=cond_slice,
                guidance_fn=guidance_fn,
                reconstruction_guidance=reconstruction_guidance,
                history_guidance=history_guidance,
                return_all=return_all,
                pbar=pbar,
            )
            xs_pred = torch.cat([xs_pred, new_pred[:, -h:]], 1)
            curr_token = xs_pred.shape[1]
        pbar.close()
        
        # Cut it back to original length of generation
        xs_pred = xs_pred[:, :original_length]
        
        return xs_pred, record

    def _generate_scheduling_matrix(
        self,
        window_size: int,
        padding: int = 0,
    ):
        match self.cfg.scheduling_matrix:
            case "full_sequence":
                scheduling_matrix = np.arange(self.sampling_timesteps - 1, -1, -1)[
                    :, None
                ].repeat(window_size, axis=1)
            case "autoregressive":
                assert False, "autoregressive scheduling matrix is not implemented yet."
                scheduling_matrix = self._generate_pyramid_scheduling_matrix(
                    window_size, self.sampling_timesteps
                )

        scheduling_matrix = torch.from_numpy(scheduling_matrix).long()

        scheduling_matrix = self.diffusion_algo.ddim_idx_to_noise_level(
            scheduling_matrix
        )

        # paded entries are labeled as pure noise
        scheduling_matrix = F.pad(
            scheduling_matrix, (0, padding, 0, 0), value=self.timesteps - 1
        )

        return scheduling_matrix

    def _extend_x_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Extend the tensor by adding dimensions at the end to match x_stacked_shape."""
        return rearrange(x, "... -> ..." + " 1" * len(self.x_shape))
    
    def _hs_sample_sequence(
        self,
        batch_size: int,
        length: Optional[int] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        conditions: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        reconstruction_guidance: float = 0.0,
        history_guidance: Optional[HistoryGuidance] = None,
        return_all: bool = False,
        pbar: Optional[tqdm] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        The unified sampling method, with length up to maximum token size.
        context of length can be provided along with a mask to achieve conditioning.

        Args
        ----
        batch_size: int
            Batch size of the sampling process
        length: Optional[int]
            Number of frames in sampled sequence
            If None, fall back to length of context, and then fall back to `self.forward_window_size_in_tokens`
        context: Optional[torch.Tensor], Shape (batch_size, length, *self.x_shape)
            Context tokens to condition on. Assumed to be same across batch.
            Tokens that are specified as context by `context_mask` will be used for conditioning,
            and the rest will be discarded.
        context_mask: Optional[torch.Tensor], Shape (batch_size, length)
            Mask for context
            0 = To be generated, 1 = Ground truth context, 2 = Generated context
            Some sampling logic may discriminate between ground truth and generated context.
        conditions: Optional[torch.Tensor], Shape (batch_size, length (causal) or self.forward_window_size_in_tokens (noncausal), ...)
            Unprocessed external conditions for sampling
        guidance_fn: Optional[Callable]
            Guidance function for sampling
        history_guidance: Optional[HistoryGuidance]
            History guidance object that handles compositional generation
        return_all: bool
            Whether to return all steps of the sampling process
        Returns
        -------
        xs_pred: torch.Tensor, Shape (batch_size, length, *self.x_shape)
            Complete sequence containing context and generated tokens
        record: Optional[torch.Tensor], Shape (num_steps, batch_size, length, *self.x_shape)
            All recorded intermediate results during the sampling process
        """
        x_shape = self.x_shape

        if length is None:
            length = self.forward_window_size_in_tokens if context is None else context.shape[1]
        if length > self.forward_window_size_in_tokens:
            raise ValueError(
                f"length is expected to <={self.forward_window_size_in_tokens}, got {length}."
            )

        if context is not None:
            if context_mask is None:
                raise ValueError("context_mask must be provided if context is given.")
            if context.shape[0] != batch_size:
                raise ValueError(
                    f"context batch size is expected to be {batch_size} but got {context.shape[0]}."
                )
            if context.shape[1] != length:
                raise ValueError(
                    f"context length is expected to be {length} but got {context.shape[1]}."
                )
            if tuple(context.shape[2:]) != tuple(x_shape):
                raise ValueError(
                    f"context shape not compatible with x_stacked_shape {x_shape}."
                )

        if context_mask is not None:
            if context is None:
                raise ValueError("context must be provided if context_mask is given. ")
            if context.shape[:2] != context_mask.shape:
                raise ValueError("context and context_mask must have the same shape.")

        if conditions is not None:
            if self.use_causal_mask and conditions.shape[1] != length:
                raise ValueError(
                    f"for causal models, conditions length is expected to be {length}, got {conditions.shape[1]}."
                )
            elif not self.use_causal_mask and conditions.shape[1] != self.forward_window_size_in_tokens:
                raise ValueError(
                    f"for noncausal models, conditions length is expected to be {self.forward_window_size_in_tokens}, got {conditions.shape[1]}."
                )

        window_size = length if self.use_causal_mask else self.forward_window_size_in_tokens
        padding = window_size - length
        # create initial xs_pred with noise
        xs_pred = torch.randn(
            (batch_size, window_size, *x_shape),
            device=self.device,
            generator=self.generator,
            dtype = context.dtype if context is not None else None,
        )
        xs_pred = torch.clamp(xs_pred, -self.clip_noise, self.clip_noise)

        if context is None:
            # create empty context and zero context mask
            context = torch.zeros_like(xs_pred)
            context_mask = torch.zeros_like(
                (batch_size, window_size), dtype=torch.long, device=self.device
            )
        elif padding > 0:
            # pad context and context mask to reach window_size
            context_pad = torch.zeros(
                (batch_size, padding, *x_shape), device=self.device
            )
            # NOTE: In context mask, -1 = padding, 0 = to be generated, 1 = GT context, 2 = generated context
            context_mask_pad = -torch.ones(
                (batch_size, padding), dtype=torch.long, device=self.device
            )
            context = torch.cat([context, context_pad], 1)
            context_mask = torch.cat([context_mask, context_mask_pad], 1)


        if history_guidance is None:
            # by default, use conditional sampling
            history_guidance = HistoryGuidance.conditional(
                timesteps=self.timesteps,
            )

        # replace xs_pred's context frames with context
        xs_pred = torch.where(self._extend_x_dim(context_mask) >= 1, context, xs_pred)

        # generate scheduling matrix
        scheduling_matrix = self._generate_scheduling_matrix(
            window_size - padding,
            padding,
        )
        scheduling_matrix = scheduling_matrix.to(self.device)
        scheduling_matrix = repeat(scheduling_matrix, "m t -> m b t", b=batch_size)
        # fill context tokens' noise levels as -1 in scheduling matrix
        if not self.is_full_sequence:
            scheduling_matrix = torch.where(
                context_mask[None] >= 1, -1, scheduling_matrix
            )

        # prune scheduling matrix to remove identical adjacent rows
        diff = scheduling_matrix[1:] - scheduling_matrix[:-1]
        skip = torch.argmax((~reduce(diff == 0, "m b t -> m", torch.all)).float())
        scheduling_matrix = scheduling_matrix[skip:]

        record = [] if return_all else None

        if pbar is None:
            pbar = tqdm(
                total=scheduling_matrix.shape[0] - 1,
                initial=0,
                desc="Sampling with DFoT",
                leave=False,
            )

        # Determine scheduler type based on respacing
        if "ddim" in str(self.diffusion_cfg.sampling_timesteps):
            scheduler = DDIMScheduler(self.diffusion_algo)
            num_inference_steps = int(self.diffusion_cfg.sampling_timesteps[len("ddim") :])
        elif "dpm++" in str(self.diffusion_cfg.sampling_timesteps):
            scheduler = DDPMScheduler(self.diffusion_algo)
            num_inference_steps = int(self.diffusion_cfg.sampling_timesteps[len("dpm++") :])
        elif self.diffusion_cfg.sampling_timesteps == "":
            scheduler = DDPMScheduler(self.diffusion_algo)
            num_inference_steps = 1000
        else:
            raise NotImplementedError(f"timesteps={self.diffusion_cfg.sampling_timesteps} is not implemented.")
        


        for m in range(scheduling_matrix.shape[0] - 1):
            from_noise_levels = scheduling_matrix[m]
            to_noise_levels = scheduling_matrix[m + 1]

            # update context mask by changing 0 -> 2 for fully generated tokens
            context_mask = torch.where(
                torch.logical_and(context_mask == 0, from_noise_levels == -1), #
                2,
                context_mask,
            )

            # create a backup with all context tokens unmodified
            xs_pred_prev = xs_pred.clone()
            if return_all:
                record.append(xs_pred.clone())

            conditions_mask = None
            with history_guidance(context_mask) as history_guidance_manager:
                nfe = history_guidance_manager.nfe
                pbar.set_postfix(NFE=nfe)
                xs_pred, from_noise_levels, to_noise_levels, conditions_mask = (
                    history_guidance_manager.prepare(
                        xs_pred,
                        from_noise_levels,
                        to_noise_levels,
                        replacement_fn=self.diffusion_algo.q_sample,
                        replacement_only=self.is_full_sequence,
                    )
                ) # NOTE: I think we can just apply this condition_mask to the conditions, and then pass it to the model, we don't need to further pass condition mask to sampling process

                if reconstruction_guidance > 0:

                    def composed_guidance_fn(
                        xk: torch.Tensor,
                        pred_x0: torch.Tensor,
                        alpha_cumprod: torch.Tensor,
                    ) -> torch.Tensor:
                        loss = (
                            F.mse_loss(pred_x0, context, reduction="none")
                            * alpha_cumprod.sqrt()
                        )
                        _context_mask = rearrange(
                            context_mask.bool(),
                            "b t -> b t" + " 1" * len(x_shape),
                        )
                        # scale inversely proportional to the number of context frames
                        loss = torch.sum(
                            loss
                            * _context_mask
                            / _context_mask.sum(dim=1, keepdim=True).clamp(min=1),
                        )
                        likelihood = -reconstruction_guidance * 0.5 * loss
                        return likelihood

                else:
                    composed_guidance_fn = guidance_fn

                # update xs_pred by DDIM or DDPM sampling
                model_kwargs = dict(
                    cfg_scale=5,
                    stabilization_level=1,
                    strategy="diffusion-forcing",
                )
                
                # So the thing that comes out of history guidance manager prepares the input 
                # to the diffusion model by batch-wising the different CFG inputs. So for some
                # variable number of history inputs, you can put them all in the same batch and 
                # pass them all to the model together. Then the .compose function takes them all 
                # and weights them properly according to some predefined strategy. 
                # The main challenge here is to just pass it into the step function, which previously
                # was sample_step in the dfot codebase. This should roughly take similar arguments, but might
                # have some slight differences in the arguments and strategy and such passed in. 
                
                repeated_conditions = self._process_conditions(
                        (
                            repeat(
                                conditions,
                                "b ... -> (b nfe) ...",
                                nfe=nfe,
                            ).clone()
                            if conditions is not None
                            else None
                        ),
                        from_noise_levels,
                    )
                result = self.diffusion_algo.sample_step(
                    x=xs_pred,
                    curr_noise_level=from_noise_levels,
                    next_noise_level=to_noise_levels,
                    conditions=repeated_conditions,
                    conditions_mask=conditions_mask,
                    guidance_fn=composed_guidance_fn,
                    model=self.diffusion_model.forward,
                    model_kwargs=model_kwargs,
                )
                xs_pred = result.x
                xs_pred = history_guidance_manager.compose(xs_pred)


            # only replace the tokens being generated (revert context tokens)
            xs_pred = torch.where(
                self._extend_x_dim(context_mask) == 0, xs_pred, xs_pred_prev
            )
            pbar.update(1)

        if return_all:
            record.append(xs_pred.clone())
            record = torch.stack(record)
        if padding > 0:
            xs_pred = xs_pred[:, :-padding]
            record = record[:, :, :-padding] if return_all else None

        return xs_pred, record

    # Block SSM
    def _block_ssm_inference(
        self, xs: Tensor, conditions: Optional[Tensor] = None, prediction_kwargs: Optional[Dict] = None
    ) -> Tensor:
        """
        Predict the videos with the given context, using block SSM.
        Input:
        - xs: (batch_size, n_tokens, *self.x_shape)
        - conditions: (batch_size, n_tokens, dim)
        - prediction_kwargs: Dict
            - chunk_size: int default should be 5
        Output:
        - xs_pred: (batch_size, n_tokens, *self.x_shape)
        """
        xs_pred = xs.clone()

        density = self.cfg.tasks.prediction.block_ssm.keyframe_density or 1
        if density > 1:
            raise ValueError("tasks.prediction.keyframe_density must be <= 1")
        keyframe_indices = (
            torch.linspace(0, xs_pred.shape[1] - 1, round(density * xs_pred.shape[1]))
            .round()
            .long()
        )
        keyframe_indices = torch.cat(
            [torch.arange(self.n_context_tokens), keyframe_indices]
        ).unique()  # context frames are always keyframes
        key_conditions = (
            conditions[:, keyframe_indices] if conditions is not None else None
        )

        self.diffusion_model.clear_cache()

        # 1. Predict the keyframes
        xs_pred_key, *_ = self._block_ssm_predict_sequence(
            context=xs_pred[:, : self.n_context_tokens],
            length=len(keyframe_indices),
            # length=self.forward_window_size_in_tokens,
            conditions=key_conditions,
            chunk_size=5,
            motion_block_size=5,
            prediction_kwargs=prediction_kwargs,
        )
        
        xs_pred[:, keyframe_indices] = xs_pred_key.to(xs_pred.dtype)
        # if is_rank_zero: # uncomment to visualize history guidance
        #     history_guidance.log(logger=self.logger)

        # 2. (Optional) Interpolate the intermediate frames
        if len(keyframe_indices) < xs_pred.shape[1]:
            context_mask = torch.zeros(xs_pred.shape[:2], device=self.device).bool()
            context_mask[:, keyframe_indices] = True
            xs_pred = self._interpolate_videos(
                context=xs_pred,
                context_mask=context_mask,
                conditions=conditions,
            )

        return xs_pred

    def _block_ssm_predict_sequence(
        self,
        context: torch.Tensor,
        length: Optional[int] = None,
        conditions: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        motion_block_size: Optional[int] = None,
        prediction_kwargs: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict the videos with the given context, using block SSM.
        """
        if length is None:
            length = self.forward_window_size_in_tokens # for ssm baseline, this usually means how much frames the model is trained on


        batch_size, context_len, *_ = context.shape

        if context_len > self.forward_window_size_in_tokens:
            raise ValueError(f"context_len is expected to be <= {self.forward_window_size_in_tokens}, got {context_len}.")
        
        curr_token = context_len
        xs_pred = context
        x_shape = self.x_shape
        record = None

        ### First load the context with the block ssm model


        context_loading_pbar = tqdm(
            total=1, # Usually context len should smaller than self.forward_window_size_in_tokens
            initial=0,
            desc="Loading context with Block SSM",
            leave=False,
        )

        context_condition_slice = conditions[:, :context_len]

        # all tokens in context are ground truth context
        context_mask = torch.ones((batch_size, context_len), dtype=torch.long, device=self.device)

        context_pred, context_pred_record = self._block_ssm_sample_sequence(
                batch_size,
                length=context_len,
                context=context,
                context_mask=context_mask,
                conditions=context_condition_slice,
                pbar=context_loading_pbar,
                is_loading_context=True,
            )

        self.diffusion_model.set_context_loading_flag()

        # KV caches are managed automatically through the do_update_cache parameter
        # No need to explicitly enable/disable here as it's handled in the model forward pass

        # All the caches (ssm states, kv caches) are binded with those blocks in the model, which will be maintained automatically
        # if you want to retrieve the caches of each layer, you can call `self.diffusion_model.get_caches(layer_idx)`

        #### Context loading is finished, now start generating target frames

        # Start generating target frames

        # the total sliding number should be [(length - context_len) + chunk_size + 1] // chunk_size
        # Each chunk goes through multiple diffusion steps
        num_chunks = (length - context_len + chunk_size - 1) // chunk_size
        generate_pbar = tqdm(
            total=num_chunks,
            initial=0,
            desc="Predicting with Block SSM",
            leave=False,
        )

        while curr_token < length:
            if record is not None:
                raise ValueError("return_all is not supported if using sliding window.")
            
            # Calculate how many tokens to generate in this step
            h = min(chunk_size, length - curr_token)  # tokens to generate this step
            
            # For block SSM, we use a sliding window of 2*chunk_size tokens:
            # - First chunk_size tokens are context (from previous generation)
            # - Second chunk_size tokens are to be generated
            c = min(chunk_size, curr_token)  # context length for this step
            l = c + h  # total window length
            
            # Extract the last c tokens as context for this generation step
            # This ensures sliding stride equals chunk_size (motion_block_size=5)
            if curr_token > chunk_size:
                # Use last chunk_size tokens as context
                context_tokens = xs_pred[:, -chunk_size:]
            else:
                # Use all generated tokens so far as context
                context_tokens = xs_pred[:, -curr_token:]
                
            # Pad to reach the desired window length
            pad = torch.zeros((batch_size, h, *x_shape), device=self.device)
            context = torch.cat([context_tokens, pad], dim=1)
            
            # Create context mask: 1 for GT context, 2 for generated context, 0 for to-be-generated
            context_mask = torch.zeros((batch_size, l), dtype=torch.long, device=self.device)
            
            # Mark context tokens
            if curr_token <= context_len:
                # Still within initial GT context
                context_mask[:, :c] = 1  # GT context
            else:
                # Past initial context, all context is generated
                context_mask[:, :c] = 2  # Generated context
            
            # Slice conditions to match the window
            cond_slice = None
            if conditions is not None:
                start_idx = max(0, curr_token - c)
                end_idx = start_idx + l
                cond_slice = conditions[:, start_idx:end_idx]

            new_pred, record = self._block_ssm_sample_sequence(
                batch_size,
                length=l,
                context=context,
                context_mask=context_mask,
                conditions=cond_slice,
                is_loading_context=False,
                motion_block_size=motion_block_size,
            )
            
            # Extract only the newly generated tokens (last h tokens)
            xs_pred = torch.cat([xs_pred, new_pred[:, -h:]], 1)
            curr_token = xs_pred.shape[1]
            generate_pbar.update(1)
        generate_pbar.close()
        
        # clear up cache
        self.diffusion_model.clear_cache()

        
        return xs_pred, record

    def _block_ssm_sample_sequence(
        self,
        batch_size: int,
        length: Optional[int] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        conditions: Optional[torch.Tensor] = None,
        return_all: bool = False,
        pbar: Optional[tqdm] = None,
        is_loading_context: bool = False,
        motion_block_size: int = 5,
        save_caches: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        The unified sampling method, with length up to maximum token size.
        context of length can be provided along with a mask to achieve conditioning.

        Args
        ----
        batch_size: int
            Batch size of the sampling process
        length: Optional[int]
            Number of frames in sampled sequence
            If None, fall back to length of context, and then fall back to `self.forward_window_size_in_tokens`
        context: Optional[torch.Tensor], Shape (batch_size, length, *self.x_shape)
            Context tokens to condition on. Assumed to be same across batch.
            Tokens that are specified as context by `context_mask` will be used for conditioning,
            and the rest will be discarded.
        context_mask: Optional[torch.Tensor], Shape (batch_size, length)
            Mask for context
            0 = To be generated, 1 = Ground truth context, 2 = Generated context
            Some sampling logic may discriminate between ground truth and generated context.
        conditions: Optional[torch.Tensor], Shape (batch_size, length (causal) or self.forward_window_size_in_tokens (noncausal), ...)
            Unprocessed external conditions for sampling
        guidance_fn: Optional[Callable]
            Guidance function for sampling
        history_guidance: Optional[HistoryGuidance]
            History guidance object that handles compositional generation
        return_all: bool
            Whether to return all steps of the sampling process
        Returns
        -------
        xs_pred: torch.Tensor, Shape (batch_size, length, *self.x_shape)
            Complete sequence containing context and generated tokens
        record: Optional[torch.Tensor], Shape (num_steps, batch_size, length, *self.x_shape)
            All recorded intermediate results during the sampling process
        """

        # Determine scheduler type based on respacing
        if "ddim" in str(self.diffusion_cfg.sampling_timesteps):
            scheduler = DDIMScheduler(self.diffusion_algo)
            num_inference_steps = int(self.diffusion_cfg.sampling_timesteps[len("ddim") :])
        elif "dpm++" in str(self.diffusion_cfg.sampling_timesteps):
            scheduler = DDPMScheduler(self.diffusion_algo)
            num_inference_steps = int(self.diffusion_cfg.sampling_timesteps[len("dpm++") :])
        elif self.diffusion_cfg.sampling_timesteps == "":
            scheduler = DDPMScheduler(self.diffusion_algo)
            num_inference_steps = 1000
        else:
            raise NotImplementedError(f"timesteps={self.diffusion_cfg.sampling_timesteps} is not implemented.")
        
        x_shape = self.x_shape

        if length is None:
            raise ValueError("length must be provided for block ssm.")
        if length > self.forward_window_size_in_tokens:
            raise ValueError(
                f"length is expected to <= {self.forward_window_size_in_tokens}, got {length}."
            )

        if context is not None:
            if context_mask is None:
                raise ValueError("context_mask must be provided if context is given.")
            if context.shape[0] != batch_size:
                raise ValueError(
                    f"context batch size is expected to be {batch_size} but got {context.shape[0]}."
                )
            if context.shape[1] != length:
                raise ValueError(
                    f"context length is expected to be {length} but got {context.shape[1]}."
                )
            if tuple(context.shape[2:]) != tuple(x_shape):
                raise ValueError(
                    f"context shape not compatible with x_stacked_shape {x_shape}."
                )

        if context_mask is not None:
            if context is None:
                raise ValueError("context must be provided if context_mask is given. ")
            if context.shape[:2] != context_mask.shape:
                raise ValueError("context and context_mask must have the same shape.")

        window_size = length
        padding = window_size - length # we actually don't need to pad here, but we keep it for consistency
        # create initial xs_pred with noise
        xs_pred = torch.randn(
            (batch_size, window_size, *x_shape),
            device=self.device,
            generator=self.generator,
            dtype = context.dtype if context is not None else None,
        )
        xs_pred = torch.clamp(xs_pred, -self.clip_noise, self.clip_noise)

        if context is None:
            raise ValueError("context must be provided for block ssm.")
        elif padding > 0:
            # pad context and context mask to reach window_size
            context_pad = torch.zeros(
                (batch_size, padding, *x_shape), device=self.device
            )
            # NOTE: In context mask, -1 = padding, 0 = to be generated, 1 = GT context, 2 = generated context
            context_mask_pad = -torch.ones(
                (batch_size, padding), dtype=torch.long, device=self.device
            )
            context = torch.cat([context, context_pad], 1)
            context_mask = torch.cat([context_mask, context_mask_pad], 1)


        # replace xs_pred's context frames with context
        xs_pred = torch.where(self._extend_x_dim(context_mask) >= 1, context, xs_pred)

        # generate scheduling matrix
        scheduling_matrix = self._generate_scheduling_matrix(
            window_size - padding,
            padding,
        )
        scheduling_matrix = scheduling_matrix.to(self.device)
        scheduling_matrix = repeat(scheduling_matrix, "m t -> m b t", b=batch_size)
        # fill context tokens' noise levels as -1 in scheduling matrix
        if not self.is_full_sequence:# Default value of self.is_full_sequence is False, so we will set context tokens' noise levels as -1
            scheduling_matrix = torch.where(
                context_mask[None] >= 1, -1, scheduling_matrix
            )
        
        # prune scheduling matrix to remove identical adjacent rows
        diff = scheduling_matrix[1:] - scheduling_matrix[:-1]
        skip = torch.argmax((~reduce(diff == 0, "m b t -> m", torch.all)).float())
        scheduling_matrix = scheduling_matrix[skip:]

        record = [] if return_all else None

        if pbar is None:
            pbar = tqdm(
                total=scheduling_matrix.shape[0] - 1,
                initial=0,
                desc="Sampling with Block SSM",
                leave=False,
            )

        # we add one more line to schedule matrix to indicate the cache update
        # scheduling_matrix (step, 2, num of frames)

        # Enable organized cache saving with step-based directory structure
        if save_caches:
            cache_base_dir = "./cache_dumps"
            self.diffusion_model.enable_local_cache_saving(cache_dir=cache_base_dir)


        for m in range(scheduling_matrix.shape[0] - 1):
            # if from_noise level and to_noise level are the same as -1, then this means we are loading the context,
            # we actually don't need to sample multiple steps, we can just run one step forward to get the caches (ssm states, kv caches)
            from_noise_levels = scheduling_matrix[m]
            to_noise_levels = scheduling_matrix[m + 1]

            # Determine if we should update caches
            # For context loading: always enable caches and update them
            # For generation: enable caches but only update on first diffusion step
            is_update_cache_diffusion_step = (m == scheduling_matrix.shape[0] - 2)
            
            # Update caches only when:
            # 1. Loading context (to build initial cache from clean context)
            # 2. First diffusion step (to update cache with clean context frames from previous generation)
            do_update_cache = is_loading_context or is_update_cache_diffusion_step

            # update context mask by changing 0 -> 2 for fully generated tokens
            context_mask = torch.where(
                torch.logical_and(context_mask == 0, from_noise_levels == -1), #
                2,
                context_mask,
            )

            # create a backup with all context tokens unmodified
            xs_pred_prev = xs_pred.clone()
            if return_all:
                record.append(xs_pred.clone())

            conditions_mask = None

            # mask first or mask last
            conditions = self._process_conditions(conditions)

            model_kwargs = dict(
                cfg_scale=5,
                stabilization_level=1,
                strategy="diffusion-forcing",
                motion_block_size=motion_block_size,
                enable_caches=True,  # Always enable caches when using block SSM
                do_update_cache=do_update_cache,  # Control when to actually update the caches
                is_loading_context=is_loading_context,
            )

            result = self.diffusion_algo.sample_step(
                x=xs_pred,
                curr_noise_level=from_noise_levels,
                next_noise_level=to_noise_levels,
                conditions=conditions,
                conditions_mask=conditions_mask, # this is not used, better to confirm again though
                guidance_fn=None,
                model=self.diffusion_model.forward,
                model_kwargs=model_kwargs,
            )

            if do_update_cache and save_caches:
                self.diffusion_model.save_all_caches_to_file()

            xs_pred = result.x

            # only replace the tokens being generated (revert context tokens)
            xs_pred = torch.where(
                self._extend_x_dim(context_mask) == 0, xs_pred, xs_pred_prev
            )

            pbar.update(1)
  
            if is_loading_context:
                break

        if return_all:
            record.append(xs_pred.clone())
            record = torch.stack(record)
        if padding > 0:
            xs_pred = xs_pred[:, :-padding]
            record = record[:, :, :-padding] if return_all else None

        return xs_pred, record

    def _predict_videos(
        self, xs: Tensor, 
        conditions: Optional[Tensor] = None,
        prediction_kwargs: Optional[Dict] = None,
    ) -> Tensor:
        """
        Predict the videos with the given context, using different algorithms.
        - history_guidance: use the history guidance algorithm to predict the videos.
            using sliding window rollouts if necessary.
            Optionally, if cfg.tasks.prediction.keyframe_density < 1, predict the keyframes first,
            then interpolate the missing intermediate frames.
        - Block SSM: use the Block SSM algorithm to predict the videos.
        """
        context_len = self.n_context_frames
        conditions = conditions.to(xs.dtype) # BUG: ugly hack to fix the dtype mismatch
        match self.cfg.tasks.prediction.sampling_strategy:
            case "history_guidance":
                res = self._history_guidance_inference(xs, conditions, prediction_kwargs=prediction_kwargs)
                return res
            case "block_ssm":
                res = self._block_ssm_inference(xs, conditions, prediction_kwargs=prediction_kwargs)
                return res
            case _:
                raise NotImplementedError(f"Sampling strategy {self.cfg.tasks.prediction.sampling_strategy} is not implemented.")

    def _sample_all_videos(
        self, batch, batch_idx, namespace="validation"
    ) -> Optional[Dict[str, Tensor]]:
        """
        Top functions for sampling videos, where we first decide which tasks to carry out,
        then call the corresponding sampling functions.
        This is also where we unnormalize the latents and decode to get videos as the return of the function.
        """
        xs, conditions, _, gt_videos, video_metadata = batch
        conditions = conditions.to(xs.dtype)
        all_videos: Dict[str, Tensor] = {"gt": xs}
        prediction_kwargs = {}


        for task in self.tasks:
            match task:
                case "prediction":
                    sample_fn = self._predict_videos
                case "interplotation":
                    sample_fn = self._interpolate_videos 
                case "reconstruction":
                    continue
                case _:
                    raise NotImplementedError

            all_videos[task] = sample_fn(xs, conditions=conditions, prediction_kwargs=prediction_kwargs)

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
            
        def cut_to_same_len(all_videos):
            min_len = min(v.shape[1] for v in all_videos.values())
            for k in all_videos.keys():
                all_videos[k] = all_videos[k][:, :min_len]
        
        cut_to_same_len(all_videos)
        
        return all_videos, video_metadata

