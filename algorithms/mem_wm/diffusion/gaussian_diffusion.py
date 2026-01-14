# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

from einops import rearrange
import torch as th
import torch.nn as nn
from .diffusion_utils import discretized_gaussian_log_likelihood, normal_kl, _extract_into_tensor
from utils.print_utils import bold_red
from typing import Literal, Optional, Callable
import torch.nn.functional as F
from logging import getLogger
from .scheduler import DDIMScheduler, DDPMScheduler
from .interface import ModelMeanType, ModelVarType, LossType, StepOutput
from .noise_schedule import get_named_beta_schedule
logger = getLogger(__name__)

from collections import namedtuple
ModelPrediction = namedtuple(
    "ModelPrediction", ["pred_noise", "pred_x_start", "model_out"]
)


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def masked_mean_flat(tensor, mask=None, dims_to_exclude=None):
    """
    Take the mean over all dimensions not specified in dims_to_exclude with an optional mask.
        only calculate the mean where the mask is True
    Args:
        tensor: Tensor of any shape (bs, ...)
        mask: Optional mask of shape (bs, t) for tensors of shape (bs, t, c, h, w)
            or (bs,) for tensors of shape (bs, c, h, w). 
        dims_to_exclude: Optional list of dimensions to exclude from mean calculation.
            Default is [0], which excludes only the batch dimension.
    
    Returns:
        Mean tensor with shape determined by the excluded dimensions,
        averaging only over unmasked elements
    """
    if dims_to_exclude is None:
        dims_to_exclude = [0]  # Default: exclude only batch dimension
    
    # Determine which dimensions to average over
    dims_to_avg = [d for d in range(tensor.dim()) if d not in dims_to_exclude]
    
    if mask is None:
        return tensor.mean(dim=dims_to_avg)
    
    # For tensors with mask
    # Expand mask to match tensor dimensions for proper broadcasting
    expanded_mask = mask
    
    # Determine how the mask should be expanded based on the tensor and dims_to_exclude
    for dim in range(mask.dim(), tensor.dim()):
        if dim not in dims_to_exclude:
            expanded_mask = expanded_mask.unsqueeze(-1)
    
    # Apply mask
    masked_tensor = tensor * expanded_mask
    
    # Count valid elements per kept dimension
    mean = masked_tensor.mean(dim=dims_to_avg)
    
    return mean


class GaussianDiffusion(nn.Module):
    """
    Utilities for training and sampling diffusion models.
    Original ported from this codebase:
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42
    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    """

    def __init__(
        self,
        *,
        betas,
        sampling_timesteps,
        model_mean_type,
        model_var_type,
        loss_type,
        cfg,
        logger, 
        **kwargs,
    ):
        super().__init__()
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type

        self.num_timesteps = int(betas.shape[0])
        self.clip_noise = cfg.clip_noise
        self.loss_weighting = cfg.loss_weighting
        self.sampling_timesteps = sampling_timesteps
        self.register_buffer("betas", betas)
        
        alphas = 1.0 - self.betas
        self.register_buffer("alphas_cumprod", th.cumprod(alphas, dim=0))
        self.register_buffer("alphas_cumprod_prev", 
                            th.cat([th.tensor([1.0], device=alphas.device), 
                                self.alphas_cumprod[:-1]]))
        self.register_buffer("alphas_cumprod_next", 
                            th.cat([self.alphas_cumprod[1:], 
                                th.tensor([0.0], device=alphas.device)]))
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", th.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", th.sqrt(1.0 - self.alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", th.log(1.0 - self.alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", th.sqrt(1.0 / self.alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", th.sqrt(1.0 / self.alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        # calculations for posterior q(x_{t-1} | x_t, x_0)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        posterior_log_variance_clipped = th.log(
            th.cat([self.posterior_variance[:1], self.posterior_variance[1:]])
        ) if len(self.posterior_variance) > 1 else th.tensor([])

        self.register_buffer("posterior_log_variance_clipped", posterior_log_variance_clipped)

        self.register_buffer("posterior_mean_coef1", 
                            self.betas * th.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.register_buffer("posterior_mean_coef2", 
                            (1.0 - self.alphas_cumprod_prev) * th.sqrt(alphas) / (1.0 - self.alphas_cumprod))

        self.use_causal_mask = cfg.use_causal_mask
        
        # snr: signal noise ratio
        snr = self.alphas_cumprod / (1 - self.alphas_cumprod)
        self.register_buffer("snr", snr)
        if self.loss_weighting.strategy in {"min_snr", "fused_min_snr"}:
            clipped_snr = snr.clone()
            clipped_snr = th.clamp(clipped_snr, min=None, max=self.loss_weighting.snr_clip)
            # self.clipped_snr = clipped_snr.float()
            self.register_buffer("clipped_snr", clipped_snr)
        elif self.loss_weighting.strategy == "sigmoid":
            # logsnr = th.log(snr).float()
            logsnr = th.log(snr) 
            self.register_buffer("logsnr", logsnr)

        self.logger = logger
            
    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None) -> th.Tensor:
        r"""
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        
        Math: x_t = \sqrt{\bar{\alpha_t}} x_0 + \sqrt{1-\bar{\alpha_t}} \epsilon
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        if x_start.ndim in [4, 5]:
            # 4: images: (bs, c, h, w)
            # 5: video: (bs, f, c, h, w)
            return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
            )
        else:
            raise ValueError(f"x_start.ndim must be 4 or 5, but got {x_start.ndim}")

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t, conditions, clip_denoised=True, denoised_fn=None, model_kwargs=None, **kwargs):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B = x.shape[0]
        C = x.shape[-3]
        # assert t.shape == (B,)
        strategy = kwargs.get("strategy", "next-frame")
        model_output = model(x, t, conditions, strategy = strategy, **model_kwargs) # fixed for diffusion-forcing, using x_t_cur
        if model_output.shape[0] != B:
            model_output = model_output[:B, ...]

        if isinstance(model_output, tuple):
            model_output, extra = model_output
        else:
            extra = None
        
        if model_output.shape[0] == 1:
            x_t = x[:1]
            t = t[:1]
        else:
            x_t = x

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            """
            https://arxiv.org/abs/2102.09672
            """
            assert False, "Not implemented"
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    th.cat([self.posterior_variance[1].unsqueeze(0), self.betas[1:]], dim=0),
                    th.log(th.cat([self.posterior_variance[1].unsqueeze(0), self.betas[1:]], dim=0)),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, model_output.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, model_output.shape)

        def process_xstart(x):
            if denoised_fn is not None: # NOTE: denoised_fn is None when sampling
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = model_output
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_xstart = self._predict_xstart_from_eps(x_t = x_t, t = t, eps = model_output)
        elif self.model_mean_type == ModelMeanType.VELOCITY:
            pred_xstart = self._predict_xstart_from_v(x_t = x_t, t = t, v = model_output)
        else:
            raise NotImplementedError(f"{self.model_mean_type} is not supported yet")
        
        pred_xstart = process_xstart(pred_xstart)
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x_t, t=t)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x_t.shape
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "extra": extra,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        r"""
        Predict the clean image x_0 from a noisy image x_t and predicted noise.
        
        This function implements the deterministic reverse process formula that 
        computes the predicted clean image (x_0) given a noisy image at timestep t 
        and the predicted noise.
        
        Args:
            x_t: A tensor of shape [N x C x ...] or [N x T x C x ...] representing 
                the noisy image at timestep t.
            t: A 1-D tensor of timesteps (one per batch element or per frame).
            eps: A tensor of the same shape as x_t containing the predicted noise.
        
        Returns:
            A tensor of the same shape as x_t containing the predicted clean image x_0.
        
        Math:
            x_0 = \frac{x_t - \sqrt{1-\bar{\alpha_t}} \epsilon}{\sqrt{\bar{\alpha_t}}}
            
        where:
            - \bar{\alpha_t} is the cumulative product of (1 - \beta_i) for i=1...t
            - \beta_i is the noise schedule
        """
        assert x_t.shape == eps.shape

        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _predict_xstart_from_v(self, x_t, t, v):
        r"""
        Predict the clean image x_0 from a noisy image x_t and velocity.
        
        This function implements the deterministic reverse process formula that 
        computes the predicted clean image (x_0) given a noisy image at timestep t 
        and the velocity.
        
        Math: x_0 = \sqrt{\bar{\alpha_t}} x_t - \sqrt{1-\bar{\alpha_t}} v
        """
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )
        
    def _predict_v(self, x_start, t, noise):
        # Math: v = \bar{\alpha} \epsilon - \sqrt{1-\bar{\alpha}} x_0
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def _predict_eps_from_v(self, x_t, t, v):
        """
        Get predicted epsilon from velocity
        """
        alpha_cum_t = _extract_into_tensor(self.alphas_cumprod, t, v.shape)
        beta_cum_t = _extract_into_tensor(1 - self.alphas_cumprod, t, v.shape)

        epsilon = (alpha_cum_t**0.5) * v + (beta_cum_t**0.5) * x_t
        return epsilon
    
    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.
        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, t, **model_kwargs)
        new_mean = p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.
        See condition_mean() for details on cond_fn.
        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, t, **model_kwargs)

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(x_start=out["pred_xstart"], x_t=x, t=t)
        return out

    def sample(
        self,
        model,
        x,
        conditions,
        scheduler_type="ddim",
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=True,
        eta=0.0,
        record_each_step=False,
        context_mask=None,
        *args,
        **kwargs,
    ):
        """
        Generate samples from the model using the specified scheduler.
        
        Args:
            scheduler_type: Either "ddim" or "ddpm" to specify sampling algorithm
            x: Starting noise tensor
            conditions: Model conditions
            (other args same as previous implementations)
            
        Returns:
            A tuple of (final_sample, predicted_x0, all_samples_by_step)
        """
        if device is None:
            device = next(model.parameters()).device
            
        if model_kwargs is None:
            model_kwargs = {}
            
        # Initialize the appropriate scheduler
        if scheduler_type.lower() == "ddim":
            scheduler = DDIMScheduler(self, **kwargs)
            step_kwargs = {"eta": eta}
        elif scheduler_type.lower() == "ddpm":
            scheduler = DDPMScheduler(self, **kwargs)
            step_kwargs = {}
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
            
        bs = x.shape[0]
        indices = list(range(self.sampling_timesteps))[::-1]
        noise_levels = self.ddim_idx_to_noise_level(th.tensor(indices, device=device))
        
        all_samples_by_step = []
        if record_each_step:
            all_samples_by_step.append(x)
            
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            noise_levels = tqdm(noise_levels, desc="Sampling")
            
        if 'memory_content_img_latent' in model_kwargs:
            mem = model_kwargs['memory_content_img_latent']
        else:
            logger.warning("Memory content image latent is not provided, this may make the generated video low quality")
            mem = None
            
        stabilization_level = 0
        for i in range(len(noise_levels) - 1):
            curr_noise_level = noise_levels[i]
            next_noise_level = th.clamp(noise_levels[i + 1], min=0) # make sure the next noise level is not negative
            if model_kwargs is not None and "stabilization_level" in model_kwargs:
                stabilization_level = model_kwargs["stabilization_level"]
                hist_len = x.shape[1] - 1  # in oasis inference, we only have one frame to predict
                current_t = F.pad(th.Tensor([curr_noise_level]), (hist_len, 0), value=stabilization_level)
                current_t = current_t.to(th.long).to(device)
                prev_t = F.pad(th.Tensor([next_noise_level]), (hist_len, 0), value=stabilization_level)
                prev_t = prev_t.to(th.long).to(device)
                # If context_mask provided, set noise level to 0 for context frames
                if context_mask is not None:
                    current_t = th.where(context_mask, th.zeros_like(current_t), current_t)
                    prev_t = th.where(context_mask, th.zeros_like(prev_t), prev_t)
                elif stabilization_level > 0:
                    logger.warning(bold_red("Context mask is not provided with non-zero stabilization level, this may affect video quality"))
            else:
                # uniform noise level
                current_t = th.tensor([curr_noise_level] * x.shape[1], device=device)
                prev_t = th.tensor([next_noise_level] * x.shape[1], device=device)
            current_t = current_t.repeat(2, 1)
            prev_t = prev_t.repeat(2, 1)
                
            with th.no_grad():
                out = scheduler.step(
                    model=model,
                    x=x,
                    t=current_t,
                    prev_t=prev_t,
                    conditions=conditions,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    **step_kwargs
                )
                
                # Update x to the new sample
                x = out.x
                
                if record_each_step:
                    all_samples_by_step.append(x)

                # if mem is not None:
                #     x[:, :mem.shape[1]] = mem
                    
                # # Apply memory conditioning if needed
                if stabilization_level > 0:
                    if mem is not None:
                        x[:, :mem.shape[1]] = mem
                    else:
                        logger.warning(bold_red("Memory content image latent is not provided with non-zero stabilization level"))
                        
        pred_xstart = out.pred_xstart
        all_samples = th.stack(all_samples_by_step, dim=1) if all_samples_by_step else None
        
        return x, pred_xstart, all_samples

    def model_predictions(self, x, k, conditions=None, conditions_mask=None, model= None, model_kwargs=None):
        model_output = model(x, k, conditions, **model_kwargs)

        if self.model_mean_type == ModelMeanType.EPSILON:
            pred_noise = th.clamp(model_output, -self.clip_noise, self.clip_noise)
            x_start = self._predict_xstart_from_eps(x, k, pred_noise)

        elif self.model_mean_type == ModelMeanType.START_X:
            x_start = model_output
            pred_noise = self._predict_eps_from_xstart(x, k, x_start)

        elif self.model_mean_type == ModelMeanType.VELOCITY:
            v = model_output
            x_start = self._predict_xstart_from_v(x, k, v)
            pred_noise = self._predict_eps_from_v(x, k, v)

        model_pred = ModelPrediction(pred_noise, x_start, model_output)

        return model_pred
    
    def ddim_idx_to_noise_level(self, indices: th.Tensor):
        shape = indices.shape
        real_steps = th.linspace(-1, self.num_timesteps - 1, self.sampling_timesteps + 1)
        real_steps = real_steps.long().to(indices.device)
        k = real_steps[indices.flatten()]
        return k.view(shape)

    def sample_step(
        self,
        x: th.Tensor,
        curr_noise_level: th.Tensor,
        next_noise_level: th.Tensor,
        conditions: Optional[th.Tensor],
        conditions_mask: Optional[th.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        scheduler_type="ddim",
        model = None,
        model_kwargs = None,
        **kwargs,
    ):
        if conditions.dtype != x.dtype:
            x = x.to(conditions.dtype) # x dtype is likely to be affected as we need to apply noise to x, while conditions are just loaded from dataloader
        if scheduler_type == "ddim":
            return self.ddim_sample_step(
                x=x,
                curr_noise_level=curr_noise_level,
                next_noise_level=next_noise_level,
                conditions=conditions,
                conditions_mask=conditions_mask,
                guidance_fn=guidance_fn,
                model=model,
                model_kwargs=model_kwargs,
            )

        # FIXME: temporary code for checking ddpm sampling
        assert th.all(
            (curr_noise_level - 1 == next_noise_level)
            | ((curr_noise_level == -1) & (next_noise_level == -1))
        ), "Wrong noise level given for ddpm sampling."

        assert (
            self.sampling_timesteps == self.num_timesteps
        ), "sampling_timesteps should be equal to timesteps for ddpm sampling."

        return self.ddpm_sample_step(
            x=x,
            curr_noise_level=curr_noise_level,
            conditions=conditions,
            conditions_mask=conditions_mask,
            guidance_fn=guidance_fn,
            model=model,
            model_kwargs=model_kwargs,
        )
    
    def ddpm_sample_step(
        self,
        x: th.Tensor,
        curr_noise_level: th.Tensor,
        conditions: Optional[th.Tensor],
        conditions_mask: Optional[th.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        model = None,
        model_kwargs = None,
        **kwargs,
    ):
        if guidance_fn is not None:
            raise NotImplementedError("guidance_fn is not yet implmented for ddpm.")

        clipped_curr_noise_level = th.clamp(curr_noise_level, min=0)

        model_mean, _, model_log_variance = self.p_mean_variance(
            model=model,
            x=x,
            t=clipped_curr_noise_level,
            conditions=conditions,
            conditions_mask=conditions_mask,
            model_kwargs=model_kwargs,
        )

        noise = th.where(
            self.add_shape_channels(clipped_curr_noise_level > 0),
            th.randn_like(x),
            0,
        )
        noise = th.clamp(noise, -self.clip_noise, self.clip_noise)
        x_pred = model_mean + th.exp(0.5 * model_log_variance) * noise

        # only update frames where the noise level decreases
        output = StepOutput(
            x = th.where(self.add_shape_channels(curr_noise_level == -1), x, x_pred),
            pred_xstart = model_mean
        )
        return output

    def ddim_sample_step(
        self,
        x: th.Tensor,
        curr_noise_level: th.Tensor,
        next_noise_level: th.Tensor,
        conditions: Optional[th.Tensor],
        conditions_mask: Optional[th.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        ddim_sampling_eta: float = 0.0,
        model = None,
        model_kwargs = None,
    ):

        clipped_curr_noise_level = th.clamp(curr_noise_level, min=0)

        alpha = self.alphas_cumprod[clipped_curr_noise_level]
        alpha_next = th.where(
            next_noise_level < 0,
            th.ones_like(next_noise_level),
            self.alphas_cumprod[next_noise_level],
        )
        sigma = th.where(
            next_noise_level < 0,
            th.zeros_like(next_noise_level),
            ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt(),
        )
        c = (1 - alpha_next - sigma**2).sqrt()

        alpha = self.add_shape_channels(alpha, x.shape)
        alpha_next = self.add_shape_channels(alpha_next, x.shape)
        c = self.add_shape_channels(c, x.shape)
        sigma = self.add_shape_channels(sigma, x.shape)

        if guidance_fn is not None:
            with th.enable_grad():
                x = x.detach().requires_grad_()

                model_pred = self.model_predictions(
                    x=x,
                    k=clipped_curr_noise_level,
                    conditions=conditions,
                    conditions_mask=conditions_mask,
                    model=model,
                    model_kwargs=model_kwargs,
                )

                guidance_loss = guidance_fn(
                    xk=x, pred_x0=model_pred.pred_x_start, alpha_cumprod=alpha
                )

                grad = -th.autograd.grad(
                    guidance_loss,
                    x,
                )[0]
                grad = th.nan_to_num(grad, nan=0.0)

                pred_noise = model_pred.pred_noise + (1 - alpha).sqrt() * grad
                x_start = th.where(
                    alpha > 0,  # to avoid NaN from zero terminal SNR
                    self.predict_start_from_noise(
                        x, clipped_curr_noise_level, pred_noise
                    ),
                    model_pred.pred_x_start,
                )

        else:
            model_pred = self.model_predictions(
                x=x,
                k=clipped_curr_noise_level,
                conditions=conditions,
                conditions_mask=conditions_mask,
                model=model,
                model_kwargs=model_kwargs,
            )
            x_start = model_pred.pred_x_start
            pred_noise = model_pred.pred_noise

        noise = th.randn_like(x)
        noise = th.clamp(noise, -self.clip_noise, self.clip_noise)

        x_pred = x_start * alpha_next.sqrt() + pred_noise * c + sigma * noise

        # only update frames where the noise level decreases
        mask = curr_noise_level == next_noise_level
        x_pred = th.where(
            self.add_shape_channels(mask, x.shape),
            x,
            x_pred,
        )
        output = StepOutput(
            x = x_pred,
            pred_xstart = x_start
        )

        return output
    
    def _vb_terms_bpd(
            self, model, x_start, x_t, t, conditions, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.
        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, conditions, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )

        masks = model_kwargs.get("masks", None)
        
        dims_to_exclude = [i for i in range(0, x_start.dim() - 3)]
        kl = masked_mean_flat(kl, masks, dims_to_exclude) / th.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = masked_mean_flat(decoder_nll, masks, dims_to_exclude) / th.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        if output.dim() == 2:
            # (bs, t) -> (bs) 
            output = masked_mean_flat(output)        
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def compute_loss_weights(
        self,
        k: th.Tensor,
        strategy: Literal["min_snr", "fused_min_snr", "uniform", "sigmoid"],
    ) -> th.Tensor:
        """
        Compute the loss weights for the given timesteps.
        Used for training, reweighting the loss for different timesteps.
        :param k: the timesteps to compute the loss weights for.
        :param strategy: the strategy to use for computing the loss weights.
        :return: a tensor of shape [N, **] containing the loss weights for each timestep.
        """
        if strategy == "uniform":
            return th.ones_like(k)
        
        self.snr = self.snr.to(k.device)
        snr = self.snr[k]
        epsilon_weighting = None
        match strategy:
            case "sigmoid":
                self.logsnr = self.logsnr.to(k.device)
                logsnr = self.logsnr[k]
                # sigmoid reweighting proposed by https://arxiv.org/abs/2303.00848
                # and adopted by https://arxiv.org/abs/2410.19324
                epsilon_weighting = th.sigmoid(
                    self.loss_weighting.sigmoid_bias - logsnr
                )
            case "snr+1":
                assert False, "Not used anymore"
                assert self.model_mean_type == ModelMeanType.VELOCITY, "snr+1 is used for Matrix, and only support velocity model"
                weightning =  1 / (1 - self.alphas_cumprod[k]).clamp(min=1e-8)
                return weightning
            case "min_snr":
                # min-SNR reweighting proposed by https://arxiv.org/abs/2303.09556
                self.clipped_snr = self.clipped_snr.to(k.device)
                clipped_snr = self.clipped_snr[k]
                epsilon_weighting = clipped_snr / snr.clamp(min=1e-8)  # avoid NaN
            case "fused_min_snr":
                # fused min-SNR reweighting proposed by Diffusion Forcing v1
                # with an additional support for bi-directional Fused min-SNR for non-causal models
                snr_clip, cum_snr_decay = (
                    self.loss_weighting.snr_clip,
                    self.loss_weighting.cum_snr_decay,
                )
                clipped_snr = self.clipped_snr[k]
                normalized_clipped_snr = clipped_snr / snr_clip
                normalized_snr = snr / snr_clip

                def compute_cum_snr(reverse: bool = False):
                    new_normalized_clipped_snr = (
                        normalized_clipped_snr.flip(1)
                        if reverse
                        else normalized_clipped_snr
                    )
                    cum_snr = th.zeros_like(new_normalized_clipped_snr)
                    for t in range(0, k.shape[1]):
                        if t == 0:
                            cum_snr[:, t] = new_normalized_clipped_snr[:, t]
                        else:
                            cum_snr[:, t] = (
                                cum_snr_decay * cum_snr[:, t - 1]
                                + (1 - cum_snr_decay) * new_normalized_clipped_snr[:, t]
                            )
                    cum_snr = F.pad(cum_snr[:, :-1], (1, 0, 0, 0), value=0.0)
                    return cum_snr.flip(1) if reverse else cum_snr

                if self.use_causal_mask:
                    cum_snr = compute_cum_snr()
                else:
                    # bi-directional cum_snr when not using causal mask
                    cum_snr = compute_cum_snr(reverse=True) + compute_cum_snr()
                    cum_snr *= 0.5
                clipped_fused_snr = 1 - (1 - cum_snr * cum_snr_decay) * (
                    1 - normalized_clipped_snr
                )
                fused_snr = 1 - (1 - cum_snr * cum_snr_decay) * (1 - normalized_snr)
                clipped_snr = clipped_fused_snr * snr_clip
                snr = fused_snr * snr_clip
                epsilon_weighting = clipped_snr / snr.clamp(min=1e-8)  # avoid NaN
            case _:
                raise ValueError(f"unknown loss weighting strategy {strategy}")

        match self.model_mean_type:
            case ModelMeanType.EPSILON:
                return epsilon_weighting
            case ModelMeanType.START_X:
                return epsilon_weighting * snr
            case ModelMeanType.VELOCITY:
                return epsilon_weighting * snr / (snr + 1)
            case _:
                raise ValueError(f"unknown objective {self.model_mean_type}")

    def _prior_bpd(self, x_start, model_kwargs=None):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        # 
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return masked_mean_flat(kl_prior, model_kwargs.get("masks", None)) / th.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        # 
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        masks = model_kwargs.get("masks", None)
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(masked_mean_flat((out["pred_xstart"] - x_start) ** 2, masks))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(masked_mean_flat((eps - noise) ** 2, masks))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start, model_kwargs)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }

    ### Training losses ###
    def training_loss(self, model, x_start, t, masks, conditions, model_kwargs, noise=None, noise_abs_max = None, ):
        """
        Compute training losses for a single timestep.
        x_start: (bs, f, c, h, w)
        conds: (bs, f, d)
        t: (bs, f)
        """

        model_kwargs['masks'] = masks

        debug_depth = model_kwargs.get('debug_depth', False) # for depth fine-tuning

        if noise is None:
            noise = th.randn_like(x_start)
            if noise_abs_max is not None:
                noise = th.clamp(noise, -noise_abs_max, noise_abs_max)
        x_t = self.q_sample(x_start, t, noise=noise)
        x_t = x_t.to(x_start.dtype)
        learn_sigma = self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]
        terms = {}

        if learn_sigma:
            B = x_t.shape[0]
            C = x_t.shape[-3] 
        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            assert False, "Not supported"
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, t.to(x_t.dtype), conditions, **model_kwargs)
            additional_loss = None
            if isinstance(model_output, tuple):
                model_output, additional_output = model_output
                additional_loss = additional_output
                    
            model_output, model_var_values = th.split(model_output, C, dim=-3) if learn_sigma else (model_output, None)
            
            
            # Learn the variance using the variational bound, but don't let
            # it affect our mean prediction.
            if learn_sigma:
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=2)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out, **kwargs: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    conditions = conditions,
                    clip_denoised=False,
                    model_kwargs=model_kwargs
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0
                

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
                ModelMeanType.VELOCITY: self._predict_v(x_start, t, noise), 
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape

            terms["mse"] = masked_mean_flat((target - model_output) ** 2, masks, dims_to_exclude=[0, 1]) 
            
            if debug_depth:
                target_rgb, target_depth = target.chunk(2, dim=2) # channel dimension
                model_output_rgb, model_output_depth = model_output.chunk(2, dim=2)
                terms["rgb_loss"] = masked_mean_flat((target_rgb - model_output_rgb) ** 2, masks, dims_to_exclude=[0, 1])
                terms["depth_loss"] = masked_mean_flat((target_depth - model_output_depth) ** 2, masks, dims_to_exclude=[0, 1])
                # terms["mse"] = 5 * terms["rgb_loss"] + 1 * terms["depth_loss"]

            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
            
            if additional_loss is not None:
                terms["loss"] += additional_loss
            
            if th.isnan(terms["loss"]).any():
                raise RuntimeError(bold_red("loss is nan"))
            
        else:
            raise NotImplementedError(self.loss_type)


        match self.model_mean_type:
            case ModelMeanType.EPSILON:
                predicted_x_start = self._predict_xstart_from_eps(x_t, t, model_output)
            case ModelMeanType.VELOCITY:
                predicted_x_start = self._predict_xstart_from_v(x_t, t, model_output)
            case _:
                raise NotImplementedError(self.model_mean_type)
            
        terms['original_x'] = x_start # (bs, t, c, h, w)
        # terms['noised_x'] = x_t
        terms['predicted_x_start'] = predicted_x_start

        loss_weight = self.compute_loss_weights(t, self.loss_weighting.strategy)
        # loss_weight shape [8, 16]
        # loss shape [8, 16]
        # loss_weight = self.add_shape_channels(loss_weight, x_start.shape)
        terms["loss"] = terms["loss"] * loss_weight

        if debug_depth:
            terms["rgb_loss"] = terms["rgb_loss"] * loss_weight
            terms["depth_loss"] = terms["depth_loss"] * loss_weight
        
        return terms

    def add_shape_channels(self, x, target_shape):
        """
        Reshape tensor x to have the right dimensions for broadcasting with a tensor of shape target_shape.
        For example, if x has shape [B] and target_shape is [C, H, W], this will return a tensor of shape [B, 1, 1, 1].
        If x has shape [B, T] and target_shape is [B, T, C, H, W], this will return a tensor of shape [B, T, 1, 1, 1].
        """
        if isinstance(target_shape, int):
            # If target_shape is just a single int, treat as a 1D shape
            target_shape = [target_shape]
        
        # If target_shape is a tensor, get its shape
        if hasattr(target_shape, 'shape'):
            target_shape = target_shape.shape
        
        # Determine how many dimensions to add
        dims_to_add = len(target_shape) - x.dim()
        if dims_to_add <= 0:
            # x already has enough dimensions
            return x
        
        # Add the necessary 1 dimensions
        return x.view(*x.shape, *([1] * dims_to_add))

