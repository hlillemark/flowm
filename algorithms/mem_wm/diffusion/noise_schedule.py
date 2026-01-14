import torch as th
import numpy as np
import math
from typing import Literal

###### Functions Tools ########

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas

###### Different Beta Schedule ######


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = th.linspace(0, timesteps, steps, dtype=th.float64) / timesteps
    v_start = th.tensor(start / tau).sigmoid()
    v_end = th.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return th.clip(betas, 0, 0.999)


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """
    This is the deprecated API for creating beta schedules.
    See get_named_beta_schedule() for the new library of schedules.
    """
    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "warmup10":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == "warmup50":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def cosine_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    from https://github.com/kwsong0113/diffusion-forcing-transformer
    """
    steps = timesteps + 1
    t = th.linspace(0, timesteps, steps, dtype=th.float64) / timesteps
    alphas_cumprod = th.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    return alphas_cumprod[1:]

def shift_beta_schedule(alphas_cumprod: th.Tensor, shift: float):
    """
    scale alphas_cumprod so that SNR is multiplied by shift ** 2
    from https://github.com/kwsong0113/diffusion-forcing-transformer
    
    """
    snr_scale = shift**2

    return (snr_scale * alphas_cumprod) / (
        snr_scale * alphas_cumprod + 1 - alphas_cumprod
    )
    
    
def cosine_simple_diffusion_schedule(
    timesteps,
    logsnr_min=-15.0,
    logsnr_max=15.0,
    shifted: float = 1.0,
    interpolated: bool = False,
):
    """
    cosine schedule with different parameterization
    following Simple Diffusion - https://arxiv.org/abs/2301.11093
    Supports "shifted cosine schedule" and "interpolated cosine schedule"

    Args:
        timesteps: number of timesteps
        logsnr_min: minimum log SNR
        logsnr_max: maximum log SNR
        shifted: shift the schedule by a factor. Should be base_resolution / current_resolution
        interpolated: interpolate between the original and the shifted schedule, requires shifted != 1.0
    """
    t_min = th.atan(th.exp(-0.5 * th.tensor(logsnr_max, dtype=th.float64)))
    t_max = th.atan(th.exp(-0.5 * th.tensor(logsnr_min, dtype=th.float64)))
    t = th.linspace(0, 1, timesteps, dtype=th.float64)
    logsnr = -2 * th.log(th.tan(t_min + t * (t_max - t_min)))
    if shifted != 1.0:
        shifted_logsnr = logsnr + 2 * th.log(
            th.tensor(shifted, dtype=th.float64)
        )
        if interpolated:
            logsnr = t * logsnr + (1 - t) * shifted_logsnr
        else:
            logsnr = shifted_logsnr

    alphas_cumprod = 1 / (1 + th.exp(-logsnr))
    return alphas_cumprod

###### Get Named Beta Schedule ######

def get_named_beta_schedule(
        schedule_name: Literal['linear', 'sigmoid', 'squaredcos_cap_v2', 'cosine', 'cosine_simple'], 
        num_diffusion_timesteps,
        clip_min=1e-9, 
        shift = 1.0):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        return get_beta_schedule(
            "linear",
            beta_start=scale * 0.0001,
            beta_end=scale * 0.02,
            num_diffusion_timesteps=num_diffusion_timesteps,
        )
    elif schedule_name == "squaredcos_cap_v2":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == "sigmoid":
        return sigmoid_beta_schedule(num_diffusion_timesteps, start=-3, end=3, tau=1, clamp_min=1e-5)
    elif schedule_name == "cosine":
        alphas_cumprod = cosine_schedule(num_diffusion_timesteps)
        alphas_cumprod = shift_beta_schedule(alphas_cumprod, shift)
        alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
        alphas = th.cat([alphas_cumprod[0:1], alphas])
        betas = 1 - alphas
        return th.clip(betas, clip_min, 1.0)
    elif schedule_name == "cosine_simple":
        alphas_cumprod = cosine_simple_diffusion_schedule(
            num_diffusion_timesteps,
            logsnr_min=-15.0,
            logsnr_max=15.0,
            shifted=shift,
            interpolated=False,
        )
        alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
        alphas = th.cat([alphas_cumprod[0:1], alphas])
        betas = 1 - alphas
        return th.clip(betas, clip_min, 1.0)
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")
