# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from typing import Literal, Optional
import os
from omegaconf import DictConfig, OmegaConf
from .gaussian_diffusion import GaussianDiffusion
from typing import Union
from logging import Logger


def create_diffusion_algo(
    sampling_timesteps: Union[int, str] = "ddim50",
    noise_schedule: Literal["linear", "cosine", "sigmoid", "squaredcos_cap_v2", "cosine_simple"] = "linear", 
    use_kl=False,
    mean_type: Literal["xstart", "epsilon", "velocity"] = "epsilon",
    var_type: Literal["fixed_large", "fixed_small", "learned_range"] = "fixed_large",
    rescale_learned_sigmas=False,
    diffusion_steps=1000,
    shift = 0.125,
    cfg: Optional[DictConfig] = None,
    logger: Optional[Logger] = None
) -> GaussianDiffusion:
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps, shift = shift)

    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas: # NOTE: should be paired with learned_range?
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE


    return GaussianDiffusion(
        betas=betas,
        sampling_timesteps=sampling_timesteps,
        model_mean_type=gd.ModelMeanType[mean_type],
        model_var_type=gd.ModelVarType[var_type],
        loss_type=loss_type, # Default to MSE
        cfg = cfg,
        logger = logger
    )
