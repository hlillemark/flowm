import torch
from algorithms.mem_wm.diffusion.diffusion_utils import _extract_into_tensor
import inspect
from typing import Union, Optional, List
from omegaconf import DictConfig
from .interface import StepOutput

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class DiffusionScheduler:
    """Base class for diffusion schedulers that handle the step-by-step sampling logic."""
    
    def __init__(self, diffusion, **kwargs):
        self.diffusion = diffusion
    
    def step(self, model, x, t, conditions, **kwargs) -> StepOutput:
        """Perform a single sampling step."""
        raise NotImplementedError("Subclasses must implement step method")


class DDIMScheduler(DiffusionScheduler):
    """DDIM sampling scheduler."""
    
    def step(self, model, x, t, prev_t, conditions, clip_denoised=True, denoised_fn=None, 
             cond_fn=None, model_kwargs=None, eta=0.0, **kwargs) -> StepOutput:
        """
        Sample x_{t-1} from the model using DDIM.
        """
        x_t = x
        
        # FIXME: for swin dpm, we don't need that wrap mapper; I think we can just remove this wrapper logics for all inference code
        out = self.diffusion.p_mean_variance(
            model,
            x_t,
            t,
            conditions=conditions,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            **model_kwargs
        )
        if cond_fn is not None:
            out = self.diffusion.condition_score(cond_fn, out, x_t, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self.diffusion._predict_eps_from_xstart(x_t, t, out["pred_xstart"])

        if eps.shape[0] == 1:
            t = t[:1]

        alpha_bar = _extract_into_tensor(self.diffusion.alphas_cumprod, t, x_t.shape)
        alpha_bar_prev = _extract_into_tensor(self.diffusion.alphas_cumprod, prev_t, x_t.shape)
            
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = torch.randn_like(x_t)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (t != 0).float()
        while nonzero_mask.dim() < noise.dim():
            nonzero_mask = nonzero_mask.unsqueeze(-1)
        sample = mean_pred + nonzero_mask * sigma * noise
        
        return StepOutput(x=sample, pred_xstart=out["pred_xstart"])

class DDPMScheduler(DiffusionScheduler):
    """DDPM sampling scheduler."""
    
    def step(self, model, x, t, conditions, clip_denoised=True, denoised_fn=None, 
             cond_fn=None, model_kwargs=None, **kwargs) -> StepOutput:
        """
        Sample x_{t-1} from the model at the given timestep.
        """
        x_t = x
        
        out = self.diffusion.p_mean_variance(
            model,
            x_t,
            t,
            conditions=conditions,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            **model_kwargs,
        )
        
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float()
        while nonzero_mask.dim() < noise.dim():
            nonzero_mask = nonzero_mask.unsqueeze(-1)
        
        if cond_fn is not None:
            out["mean"] = self.diffusion.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs)
            
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        
        return StepOutput(x=sample, pred_xstart=out["pred_xstart"])

class EulerScheduler(DiffusionScheduler):
    """
    Simple Euler-based sampler (first-order).
    You can adapt it to "ancestral" form by adding noise at each step.
    """

    def __init__(self, diffusion, eta=0.0):
        """
        diffusion: an object that has:
           - p_mean_variance(...)
           - _predict_eps_from_xstart(...)
           - condition_score(...), etc.
        eta: controlling extra noise, like in DDIM (0 = deterministic)
        """
        self.diffusion = diffusion
        self.eta = eta

    def step(
        self,
        model,
        x,
        t,
        conditions=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        if model_kwargs is None:
            model_kwargs = {}

        # Evaluate model => get mean & pred_xstart
        out = self.diffusion.p_mean_variance(
            model,
            x,
            t,
            conditions=conditions,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            **model_kwargs
        )
        # Optionally apply classifier guidance or other condition
        if cond_fn is not None:
            out = self.diffusion.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Predict the noise eps from the model output
        eps = self.diffusion._predict_eps_from_xstart(x, t, out["pred_xstart"])

        # For a simple Euler step: x_{t-1} = x_t - dt * f(...), 
        # but in diffusion we typically do a reparameterized update.

        alpha_bar = self.diffusion.alphas_cumprod[t]
        alpha_bar_prev = self.diffusion.alphas_cumprod_prev[t]

        alpha_bar = alpha_bar.view(-1, *([1]*(x.ndim-1)))
        alpha_bar_prev = alpha_bar_prev.view(-1, *([1]*(x.ndim-1)))

        # standard re-derive pred x_0:
        pred_x0 = (x - torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha_bar)
        if clip_denoised:
            pred_x0 = pred_x0.clamp(-1, 1)

        # optional "ancestral" noise
        sigma = (
            self.eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        noise = torch.randn_like(x)

        mean_pred = (
            torch.sqrt(alpha_bar_prev) * pred_x0
            + torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps
        )
        nonzero_mask = (t != 0).float().view(-1, *([1] * (x.ndim - 1)))
        x_next = mean_pred + nonzero_mask * sigma * noise

        return StepOutput(x=x_next, pred_xstart=pred_x0)

class HeunScheduler(DiffusionScheduler):
    """
    2-step explicit method. For each integer step t->t-1, it calls the model twice.
    Similar to a "predict & correct" approach.
    """

    def __init__(self, diffusion, eta=0.0):
        self.diffusion = diffusion
        self.eta = eta

    def _euler_update(self, x_t, t, eps, clip_denoised=True):
        """
        Single Euler-like update to go from t to t-1 given eps.
        This is basically the same single-step formula as in a DDIM approach.
        """
        alpha_bar = self.diffusion.alphas_cumprod[t].view(-1, *([1]*(x_t.ndim-1)))
        alpha_bar_prev = self.diffusion.alphas_cumprod_prev[t].view(-1, *([1]*(x_t.ndim-1)))

        pred_x0 = (x_t - torch.sqrt(1 - alpha_bar)*eps)/torch.sqrt(alpha_bar)
        if clip_denoised:
            pred_x0 = pred_x0.clamp(-1,1)

        sigma = (
            self.eta
            * torch.sqrt((1 - alpha_bar_prev)/(1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar/alpha_bar_prev)
        )
        noise = torch.randn_like(x_t)

        mean_pred = (
            torch.sqrt(alpha_bar_prev) * pred_x0
            + torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps
        )
        nonzero_mask = (t != 0).float().view(-1, *([1]*(x_t.ndim-1)))
        x_next = mean_pred + nonzero_mask * sigma*noise
        return x_next, pred_x0

    def step(
        self,
        model,
        x,
        t,
        conditions=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None
    ):
        if model_kwargs is None:
            model_kwargs = {}

        # 1) Evaluate model => eps_t
        out1 = self.diffusion.p_mean_variance(
            model,
            x,
            t,
            conditions=conditions,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            **model_kwargs
        )
        if cond_fn is not None:
            out1 = self.diffusion.condition_score(cond_fn, out1, x, t, model_kwargs=model_kwargs)
        eps_t = self.diffusion._predict_eps_from_xstart(x, t, out1["pred_xstart"])

        # 2) Euler step => intermediate x_euler
        x_euler, _ = self._euler_update(x, t, eps_t, clip_denoised=clip_denoised)

        # 3) Evaluate model again => eps_{t-1} at x_euler
        t_next = torch.maximum(t-1, torch.zeros_like(t))
        out2 = self.diffusion.p_mean_variance(
            model,
            x_euler,
            t_next,
            conditions=conditions,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            **model_kwargs
        )
        if cond_fn is not None:
            out2 = self.diffusion.condition_score(cond_fn, out2, x_euler, t_next, model_kwargs=model_kwargs)
        eps_euler = self.diffusion._predict_eps_from_xstart(x_euler, t_next, out2["pred_xstart"])

        # 4) Heun correction: average the two derivatives
        # delta_1 = (x_euler - x)
        # delta_2 is the update you'd get if you used eps_euler at x
        x_temp2, _ = self._euler_update(x, t, eps_euler, clip_denoised=clip_denoised)
        delta_1 = x_euler - x
        delta_2 = x_temp2 - x
        x_final = x + 0.5*(delta_1 + delta_2)

        # pred_xstart from second call is typically better
        return StepOutput(x=x_final, pred_xstart=out2["pred_xstart"])

class DPM2Scheduler(DiffusionScheduler):

    """
    A second-order DPM solver approach in discrete timesteps.
    Inspired by certain 'DPM2' variants in KDiffusion.
    """

    def __init__(self, diffusion, eta=0.0):
        self.diffusion = diffusion
        self.eta = eta

    def _single_step_dpm(self, x_t, t, eps, clip_denoised=True):
        """
        Single DPM or DDIM-like step from t->t-1, given the noise eps.
        """
        alpha_bar = self.diffusion.alphas_cumprod[t].view(-1, *([1]*(x_t.ndim-1)))
        alpha_bar_prev = self.diffusion.alphas_cumprod_prev[t].view(-1, *([1]*(x_t.ndim-1)))

        # Re-derive x_0
        pred_x0 = (x_t - torch.sqrt(1 - alpha_bar)*eps)/torch.sqrt(alpha_bar)
        if clip_denoised:
            pred_x0 = pred_x0.clamp(-1,1)

        sigma = (
            self.eta
            * torch.sqrt((1 - alpha_bar_prev)/(1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar/alpha_bar_prev)
        )
        noise = torch.randn_like(x_t)

        mean_pred = (
            torch.sqrt(alpha_bar_prev)*pred_x0
            + torch.sqrt(1 - alpha_bar_prev - sigma**2)*eps
        )
        nonzero_mask = (t!=0).float().view(-1,*([1]*(x_t.ndim-1)))
        x_next = mean_pred + nonzero_mask*sigma*noise
        return x_next, pred_x0

    def step(self, model, x, t, conditions=None, clip_denoised=True,
             denoised_fn=None, cond_fn=None, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        # Evaluate model => eps_t
        out1 = self.diffusion.p_mean_variance(
            model,
            x,
            t,
            conditions=conditions,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            **model_kwargs
        )
        if cond_fn is not None:
            out1 = self.diffusion.condition_score(cond_fn, out1, x, t, model_kwargs=model_kwargs)
        eps_t = self.diffusion._predict_eps_from_xstart(x, t, out1["pred_xstart"])

        # Euler half-step
        x_half, _ = self._single_step_dpm(x, t, eps_t, clip_denoised=clip_denoised)

        # Evaluate model => eps_half
        t_next = torch.maximum(t-1, torch.zeros_like(t))
        out2 = self.diffusion.p_mean_variance(
            model,
            x_half,
            t_next,
            conditions=conditions,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            **model_kwargs
        )
        if cond_fn is not None:
            out2 = self.diffusion.condition_score(cond_fn, out2, x_half, t_next, model_kwargs=model_kwargs)
        eps_half = self.diffusion._predict_eps_from_xstart(x_half, t_next, out2["pred_xstart"])

        # Full step from x using eps_half
        x_full, _ = self._single_step_dpm(x, t, eps_half, clip_denoised=clip_denoised)

        # Combine (Heun or midpoint style)
        x_final = 0.5*(x_half + x_full)

        return StepOutput(x=x_final, pred_xstart=out2["pred_xstart"])

class DPMpp2SScheduler(DiffusionScheduler):
    """
    Discrete DPM++ 2S-like approach. Another 2-stage method often seen in KDiffusion.
    
    The difference from 2M is in how it handles the predicted x0 vs. the predicted eps,
    and how it constructs the half-step. Typically '2S' uses x0-based formula for each sub-step
    rather than an eps-based re-derivation. 
    """

    def __init__(self, diffusion, eta=0.0):
        self.eta = eta
        super().__init__(diffusion)

    def _single_step_update_x0(self, x_t, t, pred_x0, clip_denoised=True):
        """
        A single step update from a known x0 prediction.
        Instead of focusing on eps, we use x0 directly:
           x_{t-1} = sqrt(alpha_{t-1})*x0 + ...
        """
        shape = x_t.shape
        alpha_bar_t = self.diffusion.alphas_cumprod[t].view(-1,*([1]*(x_t.ndim-1)))
        alpha_bar_prev_t = self.diffusion.alphas_cumprod_prev[t].view(-1,*([1]*(x_t.ndim-1)))

        if clip_denoised:
            pred_x0 = pred_x0.clamp(-1,1)

        eps = (x_t - torch.sqrt(alpha_bar_t)*pred_x0)/torch.sqrt(1 - alpha_bar_t)

        sigma = (
            self.eta
            * torch.sqrt((1 - alpha_bar_prev_t)/(1 - alpha_bar_t))
            * torch.sqrt(1 - alpha_bar_t/alpha_bar_prev_t)
        )
        noise = torch.randn_like(x_t)

        mean_pred = (
            torch.sqrt(alpha_bar_prev_t)*pred_x0
            + torch.sqrt(1 - alpha_bar_prev_t - sigma**2)*eps
        )

        nonzero_mask = (t!=0).float().view(-1,*([1]*(x_t.ndim-1)))
        x_next = mean_pred + nonzero_mask*sigma*noise
        return x_next

    def step(self, model, x, t, conditions=None, clip_denoised=True,
             denoised_fn=None, cond_fn=None, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        # 1) Evaluate => x0_t
        out = self.diffusion.p_mean_variance(
            model, x, t,
            conditions=conditions,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            **model_kwargs
        )
        if cond_fn is not None:
            out = self.diffusion.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)
        pred_x0_t = out["pred_xstart"]

        # 2) Euler sub-step using x0_t
        x_euler = self._single_step_update_x0(x, t, pred_x0_t, clip_denoised=clip_denoised)

        # 3) Evaluate => x0_{t-1} at x_euler
        t_next = torch.maximum(t-1, torch.zeros_like(t))
        out2 = self.diffusion.p_mean_variance(
            model, x_euler, t_next,
            conditions=conditions,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            **model_kwargs
        )
        if cond_fn is not None:
            out2 = self.diffusion.condition_score(cond_fn, out2, x_euler, t_next, model_kwargs=model_kwargs)
        pred_x0_euler = out2["pred_xstart"]

        # 4) Final step from the original x using the new x0
        x_temp = self._single_step_update_x0(x, t, pred_x0_euler, clip_denoised=clip_denoised)

        # Combine for a second-order correction
        x_final = 0.5*(x_euler + x_temp)

        return StepOutput(x=x_final, pred_xstart=out2["pred_xstart"])


def expand_timesteps_with_group(
    timesteps: torch.Tensor,  # [T], sequence of inference timesteps in descending order
    num_frames: int,  # F, where F = F'+1
    num_noise_groups: int = 4,  # G
    pad_cond_timesteps: bool = True,
    pad_cond_timesteps_value: int = 0,
    pad_prev_timesteps: bool = False,
    pad_prev_value: int = 1000,
) -> torch.Tensor:  # [T//G , F'+1] if pad_cond_timesteps, where F'=W*G else [T//G, F'].

    if pad_cond_timesteps:    
        num_frames_nocond = num_frames - 1
    else:
        num_frames_nocond = num_frames

    assert (
        num_frames_nocond % num_noise_groups == 0
    ), "num_frames (without conditional frame) should be divisible by num_noise_groups"
    assert (
        timesteps.shape[0] % num_noise_groups == 0
    ), "total inference step number should be divisible by num_noise_groups"

    window_size = num_frames_nocond // num_noise_groups
    num_inner_group_steps = timesteps.shape[0] // num_noise_groups
    timesteps_expand = (
        timesteps.view(-1, num_inner_group_steps)
        .repeat_interleave(window_size, dim=0)
        .transpose(0, 1)
    )

    if pad_prev_timesteps:
        prev_timesteps = torch.cat(
            timesteps_expand[0, window_size:],
            torch.zeros(
                [1, window_size], dtype=timesteps.dtype, device=timesteps.device
            ).fill_(pad_prev_value),
            dim=1,
        )
        timesteps_expand = torch.cat(
            [timesteps_expand, prev_timesteps],
            dim=0,
        )

    if pad_cond_timesteps:
        timesteps_expand = torch.cat(
            [
                torch.zeros_like(timesteps_expand[:, :1]).fill_(pad_cond_timesteps_value),
                timesteps_expand,
            ],
            dim=1,
        )

    return timesteps_expand.flip(0)

"""
timesteps_expand:
tensor([[  0,   0,   0,   0, 250, 250, 250, 500, 500, 500, 750, 750, 750],
        [  0,  25,  25,  25, 275, 275, 275, 525, 525, 525, 775, 775, 775],
        [  0,  50,  50,  50, 300, 300, 300, 550, 550, 550, 800, 800, 800],
        [  0,  75,  75,  75, 325, 325, 325, 575, 575, 575, 825, 825, 825],
        [  0, 100, 100, 100, 350, 350, 350, 600, 600, 600, 850, 850, 850],
        [  0, 125, 125, 125, 375, 375, 375, 625, 625, 625, 875, 875, 875],
        [  0, 150, 150, 150, 400, 400, 400, 650, 650, 650, 900, 900, 900],
        [  0, 175, 175, 175, 425, 425, 425, 675, 675, 675, 925, 925, 925],
        [  0, 200, 200, 200, 450, 450, 450, 700, 700, 700, 950, 950, 950],
        [  0, 225, 225, 225, 475, 475, 475, 725, 725, 725, 975, 975, 975]])
"""