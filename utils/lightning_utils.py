"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple, Optional, cast
import contextlib
import copy
import os
import threading
import math
import torch
from lightning import Trainer, LightningModule
from lightning.pytorch import Callback
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.trainer.states import TrainerFn
from utils.print_utils import cyan
from utils.distributed_utils import rank_zero_print
import time

try:
    from torch.utils.flop_counter import FlopCounterMode  # PyTorch ≥ 2.2
except Exception as e:
    raise ImportError("Requires torch.utils.flop_counter.FlopCounterMode (PyTorch >= 2.2)") from e


@dataclass
class FlexAttentionCallRecord:
    batch_size: int
    num_heads: int
    q_len: int
    kv_len: int
    head_dim: int
    density: float
    
    # reference: https://gist.github.com/Chillee/2e270fc5413dbbce58c779f8c4eac66c

    def _base_flops(self) -> float:
        return float(self.batch_size * self.num_heads * self.q_len * self.kv_len * self.head_dim * self.density)

    def forward_flops(self) -> float:
        # empirical FlexAttention benchmark: ~4 * (B * H * Q * K * head_dim)
        return 4.0 * self._base_flops()

    def backward_flops(self) -> float:
        # Empirical heuristic from FlexAttention benchmarks (~10x base cost)
        return 10.0 * self._base_flops()


class FlexAttentionFLOPsTracker:
    _lock = threading.Lock()
    _records: List[FlexAttentionCallRecord] = []
    _active: bool = False

    @classmethod
    def begin_capture(cls) -> None:
        with cls._lock:
            cls._active = True
            cls._records.clear()

    @classmethod
    def is_active(cls) -> bool:
        return cls._active

    @classmethod
    def record(cls, record: FlexAttentionCallRecord) -> None:
        if not cls._active:
            return
        with cls._lock:
            if cls._active:
                cls._records.append(record)

    @classmethod
    def collect(cls) -> Tuple[float, float, List[FlexAttentionCallRecord]]:
        with cls._lock:
            records = list(cls._records)
            cls._records.clear()
            cls._active = False

        fwd = float(sum(r.forward_flops() for r in records))
        bwd = float(sum(r.backward_flops() for r in records))
        return fwd, bwd, records


@dataclass
class ModuleSpecState:
    name: str
    attr_path: str
    hook_type: str = "module"
    required: bool = True
    module: Optional[torch.nn.Module] = None
    pre_handle: Optional[Any] = None
    post_handle: Optional[Any] = None
    bw_pre_handle: Optional[Any] = None
    bw_post_handle: Optional[Any] = None
    capturing: bool = False
    recorded: bool = False
    bw_capturing: bool = False
    bw_recorded: bool = False
    fw_start_ev: Any = None
    fw_end_ev: Any = None
    fw_t0: Optional[float] = None
    forward_ms: Optional[float] = None
    pre_forward_flops: Optional[float] = None
    forward_flops: Optional[float] = None
    forward_call_count: int = 0  # Track number of forward calls
    bw_start_ev: Any = None
    bw_end_ev: Any = None
    bw_t0: Optional[float] = None
    backward_ms: Optional[float] = None
    pre_backward_flops: Optional[float] = None
    backward_flops: Optional[float] = None
    backward_call_count: int = 0  # Track number of backward calls
    fw_event_pairs: List[Tuple[Any, Any]] = field(default_factory=list)
    bw_event_pairs: List[Tuple[Any, Any]] = field(default_factory=list)
    _fw_tmp_pair: Optional[Tuple[Any, Any]] = None
    _bw_tmp_pair: Optional[Tuple[Any, Any]] = None

    def cleanup(self) -> None:
        try:
            if self.pre_handle is not None:
                self.pre_handle.remove()
        finally:
            self.pre_handle = None
        try:
            if self.post_handle is not None:
                self.post_handle.remove()
        finally:
            self.post_handle = None
        try:
            if self.bw_pre_handle is not None:
                self.bw_pre_handle.remove()
        finally:
            self.bw_pre_handle = None
        try:
            if self.bw_post_handle is not None:
                self.bw_post_handle.remove()
        finally:
            self.bw_post_handle = None


class FlopCounterModeContext:
    """Global context to store and manage the active FlopCounterMode instance."""
    _lock = threading.Lock()
    _instance: Any = None  # For FlopCounterMode instance
    _saved_counts: Any = None  # Saved flop_counts for pause/resume

    @classmethod
    def set_instance(cls, instance: Any) -> None:
        """Store the active FlopCounterMode instance."""
        with cls._lock:
            cls._instance = instance
            cls._saved_counts = None

    @classmethod
    def get_instance(cls) -> Any:
        """Retrieve the active FlopCounterMode instance."""
        with cls._lock:
            return cls._instance

    @classmethod
    def pause(cls) -> None:
        """Temporarily pause FlopCounterMode while preserving counts."""
        with cls._lock:
            if cls._instance is not None:
                # Save current flop_counts
                cls._saved_counts = copy.deepcopy(cls._instance.flop_counts)
                # Exit the context
                cls._instance.__exit__(None, None, None)

    @classmethod
    def resume(cls) -> None:
        """Resume FlopCounterMode and restore previously saved counts."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.__enter__()
                if cls._saved_counts is not None:
                    # Merge saved counts with current (which may be empty or have new counts)
                    for key in cls._saved_counts:
                        if key not in cls._instance.flop_counts:
                            cls._instance.flop_counts[key] = copy.deepcopy(cls._saved_counts[key])
                        else:
                            for op, count in cls._saved_counts[key].items():
                                cls._instance.flop_counts[key][op] += count
                    cls._saved_counts = None

    @classmethod
    def clear_instance(cls) -> None:
        """Clear the stored instance."""
        with cls._lock:
            cls._instance = None
            cls._saved_counts = None




class EMA(Callback):
    """
    Implements Exponential Moving Averaging (EMA).

    When training a model, this callback will maintain moving averages of the trained parameters.
    When evaluating, we use the moving averages copy of the trained parameters.
    When saving, we save an additional set of parameters with the prefix `ema`.

    Args:
        decay: The exponential decay used when calculating the moving average. Has to be between 0-1.
        validate_original_weights: Validate the original weights, as apposed to the EMA weights.
        every_n_steps: Apply EMA every N steps.
        cpu_offload: Offload weights to CPU.

    Adapted from:
    - https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/callbacks/ema.py
    - https://github.com/BioinfoMachineLearning/bio-diffusion/blob/e4bad15139815e562a27fb94dab0c31907522bc5/src/utils/__init__.py
    """

    def __init__(
        self,
        enable: bool = True,
        decay: float = 0.0,
        validate_original_weights: bool = False,
        every_n_steps: int = 1,
        cpu_offload: bool = False,
        optimizer_indices: Tuple[int] | None = None,
    ):
        self.enable = enable
        if not (0 <= decay <= 1):
            raise MisconfigurationException("EMA decay value must be between 0 and 1")
        self.decay = decay
        self.validate_original_weights = validate_original_weights
        self.every_n_steps = every_n_steps
        self.cpu_offload = cpu_offload
        self.optimizer_indices = optimizer_indices
        if not self.enable:
            return

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.enable:
            return
        device = pl_module.device if not self.cpu_offload else torch.device("cpu")
        trainer.optimizers = [
            (
                EMAOptimizer(
                    optim,
                    device=device,
                    decay=self.decay,
                    every_n_steps=self.every_n_steps,
                    current_step=trainer.global_step,
                )
                if not isinstance(optim, EMAOptimizer)
                and (self.optimizer_indices is None or i in self.optimizer_indices)
                else optim
            )
            for i, optim in enumerate(trainer.optimizers)
        ]

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if (
            self.enable
            and stage != TrainerFn.FITTING
            and not self.validate_original_weights
        ):
            # if not fitting, there's no optimizer so we need to manually put ema weights to checkpoint['state_dict']
            # this should be handled at lightning module level
            if not hasattr(pl_module, "should_validate_ema_weights"):
                rank_zero_print(
                    cyan("WARNING: this pl_module is incompatible with EMA callback"),
                    "You will be validating with the original weights, not the EMA weights.",
                )
            pl_module.should_validate_ema_weights = True

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def _should_validate_ema_weights(self, trainer: Trainer) -> bool:
        return not self.validate_original_weights and self._ema_initialized(trainer)

    def _ema_initialized(self, trainer: Trainer) -> bool:
        return any(
            isinstance(optimizer, EMAOptimizer) for optimizer in trainer.optimizers
        )

    def swap_model_weights(self, trainer: Trainer, saving_ema_model: bool = False):
        for optimizer in trainer.optimizers:
            if isinstance(optimizer, EMAOptimizer):
                optimizer.switch_main_parameter_weights(saving_ema_model)

    @contextlib.contextmanager
    def save_ema_model(self, trainer: Trainer):
        """
        Saves an EMA copy of the model + EMA optimizer states for resume.
        """
        self.swap_model_weights(trainer, saving_ema_model=True)
        try:
            yield
        finally:
            self.swap_model_weights(trainer, saving_ema_model=False)

    @contextlib.contextmanager
    def save_original_optimizer_state(self, trainer: Trainer):
        for optimizer in trainer.optimizers:
            if isinstance(optimizer, EMAOptimizer):
                optimizer.save_original_optimizer_state = True
        try:
            yield
        finally:
            for optimizer in trainer.optimizers:
                optimizer.save_original_optimizer_state = False

    def on_load_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        """
        Enable loading EMA-enabled optimizer states to EMA-disabled optimizer states.
        """
        if not self.enable:
            optimizer_states = checkpoint["optimizer_states"]
            new_optimizer_states = []
            for optimizer_state in optimizer_states:
                new_optimizer_states.append(
                    optimizer_state["opt"]
                    if "opt" in optimizer_state
                    else optimizer_state
                )
            checkpoint["optimizer_states"] = new_optimizer_states


@torch.no_grad()
def ema_update(ema_model_tuple, current_model_tuple, decay):
    torch._foreach_mul_(ema_model_tuple, decay)
    torch._foreach_add_(
        ema_model_tuple,
        current_model_tuple,
        alpha=(1.0 - decay),
    )


def run_ema_update_cpu(
    ema_model_tuple, current_model_tuple, decay, pre_sync_stream=None
):
    if pre_sync_stream is not None:
        pre_sync_stream.synchronize()

    ema_update(ema_model_tuple, current_model_tuple, decay)


class EMAOptimizer(torch.optim.Optimizer):
    r"""
    EMAOptimizer is a wrapper for torch.optim.Optimizer that computes
    Exponential Moving Average of parameters registered in the optimizer.

    EMA parameters are automatically updated after every step of the optimizer
    with the following formula:

        ema_weight = decay * ema_weight + (1 - decay) * training_weight

    To access EMA parameters, use ``swap_ema_weights()`` context manager to
    perform a temporary in-place swap of regular parameters with EMA
    parameters.

    Notes:
        - EMAOptimizer is not compatible with APEX AMP O2.

    Args:
        optimizer (torch.optim.Optimizer): optimizer to wrap
        device (torch.device): device for EMA parameters
        decay (float): decay factor

    Returns:
        returns an instance of torch.optim.Optimizer that computes EMA of
        parameters

    Example:
        model = Model().to(device)
        opt = torch.optim.Adam(model.parameters())

        opt = EMAOptimizer(opt, device, 0.9999)

        for epoch in range(epochs):
            training_loop(model, opt)

            regular_eval_accuracy = evaluate(model)

            with opt.swap_ema_weights():
                ema_eval_accuracy = evaluate(model)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        decay: float = 0.9999,
        every_n_steps: int = 1,
        current_step: int = 0,
    ):
        self.optimizer = optimizer
        self.decay = decay
        self.device = device
        self.current_step = current_step
        self.every_n_steps = every_n_steps
        self.save_original_optimizer_state = False

        self.first_iteration = True
        self.rebuild_ema_params = True
        self.stream = None
        self.thread = None

        self.ema_params = ()
        self.in_saving_ema_model_context = False

    def all_parameters(self) -> Iterable[torch.Tensor]:
        return (param for group in self.param_groups for param in group["params"])

    def step(self, closure=None, grad_scaler=None, **kwargs):
        self.join()

        if self.first_iteration:
            if any(p.is_cuda for p in self.all_parameters()):
                self.stream = torch.cuda.Stream()

            self.first_iteration = False

        if self.rebuild_ema_params:
            opt_params = list(self.all_parameters())

            self.ema_params += tuple(
                copy.deepcopy(param.data.detach()).to(self.device)
                for param in opt_params[len(self.ema_params) :]
            )
            self.rebuild_ema_params = False

        if (
            getattr(self.optimizer, "_step_supports_amp_scaling", False)
            and grad_scaler is not None
        ):
            loss = self.optimizer.step(closure=closure, grad_scaler=grad_scaler)
        else:
            loss = self.optimizer.step(closure)

        if self._should_update_at_step():
            self.update()
        self.current_step += 1
        return loss

    def _should_update_at_step(self) -> bool:
        return self.current_step % self.every_n_steps == 0

    @torch.no_grad()
    def update(self):
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self.stream):
            current_model_state = tuple(
                param.data.to(self.device, non_blocking=True)
                for param in self.all_parameters()
            )

            if self.device.type == "cuda":
                ema_update(self.ema_params, current_model_state, self.decay)

        if self.device.type == "cpu":
            self.thread = threading.Thread(
                target=run_ema_update_cpu,
                args=(
                    self.ema_params,
                    current_model_state,
                    self.decay,
                    self.stream,
                ),
            )
            self.thread.start()

    def swap_tensors(self, tensor1, tensor2):
        tmp = torch.empty_like(tensor1)
        tmp.copy_(tensor1)
        tensor1.copy_(tensor2)
        tensor2.copy_(tmp)

    def switch_main_parameter_weights(self, saving_ema_model: bool = False):
        self.join()
        self.in_saving_ema_model_context = saving_ema_model
        for param, ema_param in zip(self.all_parameters(), self.ema_params):
            self.swap_tensors(param.data, ema_param)

    @contextlib.contextmanager
    def swap_ema_weights(self, enabled: bool = True):
        r"""
        A context manager to in-place swap regular parameters with EMA
        parameters.
        It swaps back to the original regular parameters on context manager
        exit.

        Args:
            enabled (bool): whether the swap should be performed
        """

        if enabled:
            self.switch_main_parameter_weights()
        try:
            yield
        finally:
            if enabled:
                self.switch_main_parameter_weights()

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

    def join(self):
        if self.stream is not None:
            self.stream.synchronize()

        if self.thread is not None:
            self.thread.join()

    def state_dict(self):
        self.join()

        if self.save_original_optimizer_state:
            return self.optimizer.state_dict()

        # if we are in the context of saving an EMA model, the EMA weights are in the modules' actual weights
        ema_params = (
            self.ema_params
            if not self.in_saving_ema_model_context
            else list(self.all_parameters())
        )
        state_dict = {
            "opt": self.optimizer.state_dict(),
            "ema": ema_params,
            "current_step": self.current_step,
            "decay": self.decay,
            "every_n_steps": self.every_n_steps,
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.join()

        if "opt" in state_dict:
            self.optimizer.load_state_dict(state_dict["opt"])
            self.ema_params = tuple(
                param.to(self.device) for param in copy.deepcopy(state_dict["ema"])
            )
            self.current_step = state_dict["current_step"]
            self.decay = state_dict["decay"]
            self.every_n_steps = state_dict["every_n_steps"]
        else:  # loading non-EMA state dict
            self.optimizer.load_state_dict(state_dict)
            self.ema_params = tuple(
                copy.deepcopy(param.data.detach()).to(self.device)
                for param in self.all_parameters()
            )
        self.rebuild_ema_params = False

    def add_param_group(self, param_group):
        self.optimizer.add_param_group(param_group)
        self.rebuild_ema_params = True

class OneShotFwBwFLOPsDispatch(Callback):
    """
    Forward window: target_module.forward_pre_hook -> forward_post_hook
      - Excludes dataloader / batch transfer / preprocessing / loss compute.
    Backward window: on_before_backward -> on_after_backward
      - Excludes optimizer.step() / zero_grad().
    FLOPs are counted with FlopCounterMode in the same windows.
    Runs once (rank-0) after optional warmup_steps.

    Notes:
      - We reset CUDA peak stats ONLY once at forward-begin, so the peak read
        at backward-end is the *true whole-step* training peak.
      - We log both per-batch and per-sample FLOPs for fair cross-BS comparison.
    """

    def __init__(
        self,
        target_attr: str = "main_model",
        log_wandb_table: bool = True,
        warmup_steps: int = 63,
        eval_steps: int = 16,  # Number of steps to collect metrics over
        core_module_attr: Optional[str] = None,  # e.g., "transformer_blocks" to skip preprocessing
        module_specs: Optional[List[Dict[str, Any]]] = None,
        sync_each_hook: bool = True,  # when False, defer cuda synchronize/elapsed to end-of-step
    ):
        super().__init__()
        self.target_attr = target_attr
        self.core_module_attr = core_module_attr
        self.log_wandb_table = log_wandb_table
        self.warmup_steps = int(warmup_steps)
        self.eval_steps = int(eval_steps)
        self.module_specs_cfg = module_specs or []
        self.sync_each_hook = bool(sync_each_hook)

        self._done = False
        self._use_cuda = torch.cuda.is_available()
        self._current_eval_step = 0  # Track which eval step we're on

        # forward timing
        self._fw_start_ev = None
        self._fw_end_ev = None
        self._fw_t0 = None
        self._forward_ms = None

        # backward timing
        self._bw_start_ev = None
        self._bw_end_ev = None
        self._bw_t0 = None
        self._backward_ms = None

        # FLOP counters
        self._fc_fwd = None
        self._fc_bwd = None
        self._fwd_flops = None
        self._bwd_flops = None

        # hooks / state
        self._target_mod = None
        self._h_pre = None
        self._h_post = None
        self._capturing_fw = False
        self._trainer = None

        # memory peaks (bytes)
        self._fw_peak_alloc = None
        self._fw_peak_reserved = None
        self._train_peak_alloc = None  # whole step
        self._train_peak_reserved = None

        # misc
        self._dev = None
        self._batch_size = None  # captured from inputs during forward
        self._module_spec_states: List[ModuleSpecState] = []
        
        # Storage for multiple eval steps
        self._collected_metrics: List[Dict[str, float]] = []

    # -------- helpers
    def _get_target_module(self, pl_module):
        candidates: List[str] = []
        if self.target_attr:
            candidates.append(self.target_attr)

        prefix = getattr(pl_module, "main_model_prefix", None)
        if isinstance(prefix, str):
            candidates.append(prefix)

        for attr in candidates:
            if hasattr(pl_module, attr):
                target = getattr(pl_module, attr)
                
                # If core_module_attr is specified, drill down further to skip preprocessing
                if self.core_module_attr and hasattr(target, self.core_module_attr):
                    core = getattr(target, self.core_module_attr)
                    rank_zero_print(
                        cyan(
                            f"[OneShotFwBwFLOPsDispatch] Hooking core module '{attr}.{self.core_module_attr}' "
                            "to exclude preprocessing overhead from timing/FLOPs."
                        )
                    )
                    return core
                
                return target

        rank_zero_print(
            cyan(
                "[OneShotFwBwFLOPsDispatch] Falling back to LightningModule; forward timing will include training_step overhead."
            )
        )
        return pl_module  # fallback: time LightningModule.forward

    def _infer_device(self, pl_module):
        dev = getattr(self._target_mod, "device", None)
        if dev is None:
            dev = getattr(pl_module, "device", None)
        if isinstance(dev, str):
            dev = torch.device(dev)
        if dev is None and self._use_cuda:
            dev = torch.device("cuda")
        return dev

    def _resolve_attr_path(self, root: Any, path: str) -> Optional[Any]:
        current = root
        for part in path.split("."):
            if part == "":
                continue
            if not hasattr(current, part):
                return None
            current = getattr(current, part)
        return current

    def _normalize_module_spec_cfg(self, spec_cfg: Any) -> Optional[Dict[str, Any]]:
        if isinstance(spec_cfg, str):
            return {
                "name": spec_cfg.split(".")[-1],
                "attr_path": spec_cfg,
                "hook_type": "auto",
                "required": True,
                "enabled": True,
            }

        if spec_cfg is None:
            return None

        name = spec_cfg.get("name")
        attr_path = spec_cfg.get("attr") or spec_cfg.get("path") or spec_cfg.get("attr_path")
        if attr_path is None:
            return None
        if not name:
            name = attr_path.split(".")[-1]

        hook_type = str(spec_cfg.get("hook_type", "auto")).lower()
        required = bool(spec_cfg.get("required", True))
        enabled = bool(spec_cfg.get("enabled", True))

        return {
            "name": name,
            "attr_path": attr_path,
            "hook_type": hook_type,
            "required": required,
            "enabled": enabled,
        }

    def _setup_module_spec_hooks(self, pl_module, trainer):
        if not self.module_specs_cfg:
            return

        for raw_spec in self.module_specs_cfg:
            spec = self._normalize_module_spec_cfg(raw_spec)
            if spec is None or not spec.get("enabled", True):
                continue

            attr_path = spec["attr_path"]
            module = self._resolve_attr_path(pl_module, attr_path)
            if module is None:
                if spec["required"]:
                    rank_zero_print(
                        cyan(
                            f"[OneShotFwBwFLOPsDispatch] Module spec '{spec['name']}' (path '{attr_path}') not found; skipping."
                        )
                    )
                continue
            if not isinstance(module, torch.nn.Module):
                rank_zero_print(
                    cyan(
                        f"[OneShotFwBwFLOPsDispatch] Module spec '{spec['name']}' path '{attr_path}' is not nn.Module; skipping."
                    )
                )
                continue

            hook_type = spec["hook_type"]
            if hook_type == "auto":
                if isinstance(module, (torch.nn.ModuleList, torch.nn.Sequential)):
                    hook_type = "sequence"
                else:
                    hook_type = "module"

            state = ModuleSpecState(
                name=spec["name"],
                attr_path=attr_path,
                hook_type=hook_type,
                required=spec["required"],
                module=module,
            )
            self._attach_module_spec_hooks(state, trainer)
            self._module_spec_states.append(state)

    def _attach_module_spec_hooks(self, state: ModuleSpecState, trainer):
        if state.module is None:
            return

        def _should_capture_forward() -> bool:
            return (
                not self._done
                and trainer.global_step >= self.warmup_steps
                and self._current_eval_step < self.eval_steps
            )

        def _should_capture_backward() -> bool:
            return (
                not self._done
                and trainer.global_step >= self.warmup_steps
                and self._current_eval_step < self.eval_steps
                and self._fc_bwd is not None
            )

        def _fw_pre(_, __):
            if not _should_capture_forward():
                return
            # Don't skip if already recorded - we want to accumulate across multiple calls
            state.capturing = True
            state.forward_call_count += 1

            if self._use_cuda:
                with torch.cuda.device(self._dev):
                    stream = torch.cuda.current_stream(self._dev)
                    if self.sync_each_hook:
                        if state.fw_start_ev is None:
                            state.fw_start_ev = torch.cuda.Event(enable_timing=True)
                        if state.fw_end_ev is None:
                            state.fw_end_ev = torch.cuda.Event(enable_timing=True)
                        state.fw_start_ev.record(stream)
                    else:
                        start_ev = torch.cuda.Event(enable_timing=True)
                        end_ev = torch.cuda.Event(enable_timing=True)
                        start_ev.record(stream)
                        state._fw_tmp_pair = (start_ev, end_ev)
            else:
                state.fw_t0 = time.perf_counter()

            if self._fc_fwd is not None:
                try:
                    state.pre_forward_flops = float(self._fc_fwd.get_total_flops())
                except Exception:
                    state.pre_forward_flops = None
            else:
                state.pre_forward_flops = None

        def _fw_post(_, __, ___):
            if not state.capturing:
                return

            # Accumulate timing across multiple calls
            if self._use_cuda:
                with torch.cuda.device(self._dev):
                    stream = torch.cuda.current_stream(self._dev)
                    if self.sync_each_hook and state.fw_start_ev is not None and state.fw_end_ev is not None:
                        state.fw_end_ev.record(stream)
                        torch.cuda.synchronize()
                        elapsed = state.fw_start_ev.elapsed_time(state.fw_end_ev)
                        if state.forward_ms is None:
                            state.forward_ms = elapsed
                        else:
                            state.forward_ms += elapsed
                    elif not self.sync_each_hook and state._fw_tmp_pair is not None:
                        start_ev, end_ev = state._fw_tmp_pair
                        end_ev.record(stream)
                        state.fw_event_pairs.append((start_ev, end_ev))
                        state._fw_tmp_pair = None
            elif not self._use_cuda:
                t1 = time.perf_counter()
                elapsed = (t1 - (state.fw_t0 or t1)) * 1000.0
                if state.forward_ms is None:
                    state.forward_ms = elapsed
                else:
                    state.forward_ms += elapsed

            # Accumulate FLOPs across multiple calls
            if self._fc_fwd is not None and state.pre_forward_flops is not None:
                try:
                    total = float(self._fc_fwd.get_total_flops())
                    delta = total - state.pre_forward_flops
                    # Only accumulate if we got a valid positive delta
                    if delta >= 0 and not math.isnan(delta) and not math.isinf(delta):
                        if state.forward_flops is None:
                            state.forward_flops = delta
                        else:
                            state.forward_flops += delta
                except Exception:
                    pass

            state.recorded = True
            state.capturing = False

        def _bw_pre(_, __):
            if not _should_capture_backward():
                return
            # Don't skip if already recorded - we want to accumulate across multiple calls
            state.bw_capturing = True
            state.backward_call_count += 1

            if self._use_cuda:
                with torch.cuda.device(self._dev):
                    stream = torch.cuda.current_stream(self._dev)
                    if self.sync_each_hook:
                        if state.bw_start_ev is None:
                            state.bw_start_ev = torch.cuda.Event(enable_timing=True)
                        if state.bw_end_ev is None:
                            state.bw_end_ev = torch.cuda.Event(enable_timing=True)
                        state.bw_start_ev.record(stream)
                    else:
                        start_ev = torch.cuda.Event(enable_timing=True)
                        end_ev = torch.cuda.Event(enable_timing=True)
                        start_ev.record(stream)
                        state._bw_tmp_pair = (start_ev, end_ev)
            else:
                state.bw_t0 = time.perf_counter()

            if self._fc_bwd is not None:
                try:
                    state.pre_backward_flops = float(self._fc_bwd.get_total_flops())
                except Exception:
                    state.pre_backward_flops = None
            else:
                state.pre_backward_flops = None

        def _bw_post(_, __, ___):
            if not state.bw_capturing:
                return

            # Accumulate timing across multiple calls
            if self._use_cuda:
                with torch.cuda.device(self._dev):
                    stream = torch.cuda.current_stream(self._dev)
                    if self.sync_each_hook and state.bw_start_ev is not None and state.bw_end_ev is not None:
                        state.bw_end_ev.record(stream)
                        torch.cuda.synchronize()
                        elapsed = state.bw_start_ev.elapsed_time(state.bw_end_ev)
                        if state.backward_ms is None:
                            state.backward_ms = elapsed
                        else:
                            state.backward_ms += elapsed
                    elif not self.sync_each_hook and state._bw_tmp_pair is not None:
                        start_ev, end_ev = state._bw_tmp_pair
                        end_ev.record(stream)
                        state.bw_event_pairs.append((start_ev, end_ev))
                        state._bw_tmp_pair = None
            elif not self._use_cuda:
                t1 = time.perf_counter()
                elapsed = (t1 - (state.bw_t0 or t1)) * 1000.0
                if state.backward_ms is None:
                    state.backward_ms = elapsed
                else:
                    state.backward_ms += elapsed

            # Accumulate FLOPs across multiple calls
            if self._fc_bwd is not None and state.pre_backward_flops is not None:
                try:
                    total = float(self._fc_bwd.get_total_flops())
                    delta = total - state.pre_backward_flops
                    # Only accumulate if we got a valid positive delta
                    if delta >= 0 and not math.isnan(delta) and not math.isinf(delta):
                        if state.backward_flops is None:
                            state.backward_flops = delta
                        else:
                            state.backward_flops += delta
                except Exception:
                    pass

            state.bw_recorded = True
            state.bw_capturing = False

        attach_sequence = False
        sequence_modules: List[torch.nn.Module] = []
        if isinstance(state.module, torch.nn.Sequential):
            sequence_modules = list(state.module.children())
            attach_sequence = True
        elif isinstance(state.module, torch.nn.ModuleList):
            sequence_modules = list(state.module)
            attach_sequence = True

        if attach_sequence:
            if len(sequence_modules) == 0:
                rank_zero_print(
                    cyan(
                        f"[OneShotFwBwFLOPsDispatch] ModuleList '{state.attr_path}' is empty; module spec '{state.name}' skipped."
                    )
                )
                return
            first_block = sequence_modules[0]
            last_block = sequence_modules[-1]
            state.pre_handle = first_block.register_forward_pre_hook(_fw_pre, with_kwargs=False)
            state.post_handle = last_block.register_forward_hook(_fw_post, with_kwargs=False)
            state.bw_pre_handle = last_block.register_full_backward_pre_hook(_bw_pre)
            state.bw_post_handle = first_block.register_full_backward_hook(_bw_post)
            rank_zero_print(
                cyan(
                    f"[OneShotFwBwFLOPsDispatch] Hooked module spec '{state.name}' on sequence '{state.attr_path}' (first & last blocks)."
                )
            )
        else:
            state.pre_handle = state.module.register_forward_pre_hook(_fw_pre, with_kwargs=False)
            state.post_handle = state.module.register_forward_hook(_fw_post, with_kwargs=False)
            state.bw_pre_handle = state.module.register_full_backward_pre_hook(_bw_pre)
            state.bw_post_handle = state.module.register_full_backward_hook(_bw_post)
            rank_zero_print(
                cyan(
                    f"[OneShotFwBwFLOPsDispatch] Hooked module spec '{state.name}' on '{state.attr_path}'."
                )
            )

    def _teardown_module_spec_hooks(self):
        for state in self._module_spec_states:
            state.cleanup()
        self._module_spec_states = []

    def _log_module_spec_metrics(self, pl_module):
        if not self._module_spec_states:
            return

        rank_zero_print(
            cyan(
                "[OneShotFwBwFLOPsDispatch] Note: Per-module metrics are CUMULATIVE across all calls during the training step. "
                "For autoregressive models, modules may be called many times. "
                "FLOPs may show 'nan' if FlopCounterMode was paused/resumed."
            )
        )

        summary_lines: List[str] = []
        for state in self._module_spec_states:
            if not state.recorded:
                if state.required:
                    rank_zero_print(
                        cyan(
                            f"[OneShotFwBwFLOPsDispatch] Module spec '{state.name}' did not run before measurement; consider adjusting warmup_steps."
                        )
                    )
                continue

            forward_ms = state.forward_ms if state.forward_ms is not None else float("nan")
            forward_gflops = (
                (state.forward_flops / 1e9)
                if state.forward_flops is not None
                else float("nan")
            )
            fw_calls = state.forward_call_count

            if not state.bw_recorded and state.required:
                rank_zero_print(
                    cyan(
                        f"[OneShotFwBwFLOPsDispatch] Module spec '{state.name}' did not produce backward stats; ensure it participates in loss graph."
                    )
                )

            backward_ms = state.backward_ms if state.backward_ms is not None else float("nan")
            backward_gflops = (
                (state.backward_flops / 1e9)
                if state.backward_flops is not None
                else float("nan")
            )
            bw_calls = state.backward_call_count

            summary_lines.append(
                f"        - {state.name}: fw_calls={fw_calls}, forward_ms={forward_ms:.3f}, forward_GFLOPs={forward_gflops:.3f}, "
                f"bw_calls={bw_calls}, backward_ms={backward_ms:.3f}, backward_GFLOPs={backward_gflops:.3f}"
            )

            pl_module.log(
                f"flops/module/{state.name}_forward_ms_once",
                float(forward_ms),
                on_step=True,
                on_epoch=False,
            )
            pl_module.log(
                f"flops/module/{state.name}_forward_GFLOPs_once",
                float(forward_gflops),
                on_step=True,
                on_epoch=False,
            )
            pl_module.log(
                f"flops/module/{state.name}_backward_ms_once",
                float(backward_ms),
                on_step=True,
                on_epoch=False,
            )
            pl_module.log(
                f"flops/module/{state.name}_backward_GFLOPs_once",
                float(backward_gflops),
                on_step=True,
                on_epoch=False,
            )

        if summary_lines:
            msg = "\n" + "        Module-spec breakdown:\n" + "\n".join(summary_lines) + "\n"
            rank_zero_print(msg)

    def _start_fw_timer(self):
        if self._use_cuda:
            with torch.cuda.device(self._dev):
                if self.sync_each_hook:
                    torch.cuda.synchronize()
                fw_start = torch.cuda.Event(enable_timing=True)
                fw_end = torch.cuda.Event(enable_timing=True)
                self._fw_start_ev = fw_start
                self._fw_end_ev = fw_end
                stream = torch.cuda.current_stream(self._dev)
                fw_start.record(stream)
        else:
            self._fw_t0 = time.perf_counter()

    def _stop_fw_timer(self):
        if self._use_cuda:
            with torch.cuda.device(self._dev):
                if self._fw_end_ev is not None and self._fw_start_ev is not None:
                    stream = torch.cuda.current_stream(self._dev)
                    self._fw_end_ev.record(stream)
                    if self.sync_each_hook:
                        torch.cuda.synchronize()
                        self._forward_ms = self._fw_start_ev.elapsed_time(self._fw_end_ev)
        else:
            t1 = time.perf_counter()
            self._forward_ms = (t1 - (self._fw_t0 or t1)) * 1000.0

    def _finalize_deferred_events(self):
        """If we're deferring synchronization, flush once and compute elapsed times."""
        if self.sync_each_hook or not self._use_cuda:
            return
        with torch.cuda.device(self._dev):
            torch.cuda.synchronize()
        if self._fw_start_ev is not None and self._fw_end_ev is not None and self._forward_ms is None:
            self._forward_ms = self._fw_start_ev.elapsed_time(self._fw_end_ev)
        if self._bw_start_ev is not None and self._bw_end_ev is not None and self._backward_ms is None:
            self._backward_ms = self._bw_start_ev.elapsed_time(self._bw_end_ev)

        for state in self._module_spec_states:
            if state.fw_event_pairs:
                total = sum(s.elapsed_time(e) for s, e in state.fw_event_pairs)
                state.forward_ms = (state.forward_ms or 0.0) + total if state.forward_ms is not None else total
                state.fw_event_pairs.clear()
            if state.bw_event_pairs:
                total = sum(s.elapsed_time(e) for s, e in state.bw_event_pairs)
                state.backward_ms = (state.backward_ms or 0.0) + total if state.backward_ms is not None else total
                state.bw_event_pairs.clear()

    # -------- Lightning lifecycle
    def on_train_start(self, trainer, pl_module):
        self._trainer = trainer

        # only rank-0 for measurement and register hook
        if not getattr(trainer, "is_global_zero", True):
            self._done = True
            return

        rank_zero_print(
            cyan(
                f"[OneShotFwBwFLOPsDispatch] on_train_start: warmup_steps={self.warmup_steps}, "
                f"eval_steps={self.eval_steps}, "
                f"target_attr={self.target_attr}, core_module_attr={self.core_module_attr}, "
                f"module_specs count={len(self.module_specs_cfg)}"
            )
        )

        self._target_mod = self._get_target_module(pl_module)
        self._dev = self._infer_device(pl_module)

        def _pre_hook(mod, inputs):
            if self._done:
                return
            if trainer.global_step < self.warmup_steps:
                rank_zero_print(
                    cyan(
                        f"[OneShotFwBwFLOPsDispatch] _pre_hook skipped: global_step={trainer.global_step} < warmup={self.warmup_steps}"
                    )
                )
                return
            if self._current_eval_step >= self.eval_steps:
                return
            if self._capturing_fw:
                return
            # Skip hook for FlexAttnProcessor to avoid interfering with compiled kernels
            from algorithms.mem_wm.backbones.cogv.block_ssm_official_impl import FlexAttnProcessor
            if isinstance(mod, FlexAttnProcessor):
                return
            
            rank_zero_print(
                cyan(f"[OneShotFwBwFLOPsDispatch] _pre_hook triggered at global_step={trainer.global_step}")
            )
            self._capturing_fw = True
            FlexAttentionFLOPsTracker.begin_capture()
            bs = 0
            dataloaders = getattr(self._trainer, "train_dataloader", None)
            if dataloaders is not None:
                for dataloader in dataloaders:
                    bs = max(bs, dataloader.batch_size)

            # if isinstance(inputs, (tuple, list)) and len(inputs) > 0 and torch.is_tensor(inputs[0]):
            #     bs = inputs[0].shape[0]
            # elif torch.is_tensor(inputs):
            #     bs = inputs.shape[0]
            self._batch_size = int(bs) if bs is not None else None

            if self._use_cuda:
                with torch.cuda.device(self._dev):
                    torch.cuda.reset_peak_memory_stats()

            # start forward timing
            self._start_fw_timer()

            # start forward FLOP counter
            self._fc_fwd = FlopCounterMode(display=False)
            self._fc_fwd.__enter__()
            
            # Store the FlopCounterMode instance so FlexAttnProcessor can temporarily disable it
            FlopCounterModeContext.set_instance(self._fc_fwd)

        def _post_hook(mod, inputs, output):
            if not self._capturing_fw or self._done:
                return

            # stop forward timer FIRST
            self._stop_fw_timer()

            # collect forward FLOPs
            if self._fc_fwd is not None:
                self._fc_fwd.__exit__(None, None, None)
                self._fwd_flops = float(self._fc_fwd.get_total_flops())
                self._fc_fwd = None
                FlopCounterModeContext.clear_instance()

            # collect forward-only peak (same counter; still valid now)
            if self._use_cuda:
                with torch.cuda.device(self._dev):
                    torch.cuda.synchronize()
                    self._fw_peak_alloc = torch.cuda.max_memory_allocated()
                    self._fw_peak_reserved = torch.cuda.max_memory_reserved()

            self._capturing_fw = False

        # attach hooks
        # If target is a ModuleList/Sequential, hook first and last submodule only
        if isinstance(self._target_mod, (torch.nn.ModuleList, torch.nn.Sequential)):
            if len(self._target_mod) > 0:
                first_block = self._target_mod[0]
                last_block = self._target_mod[-1]
                self._h_pre = first_block.register_forward_pre_hook(_pre_hook, with_kwargs=False)
                self._h_post = last_block.register_forward_hook(_post_hook, with_kwargs=False)
                rank_zero_print(
                    cyan(
                        f"[OneShotFwBwFLOPsDispatch] Hooked first and last block of {len(self._target_mod)}-block sequence."
                    )
                )
            else:
                rank_zero_print(cyan("[OneShotFwBwFLOPsDispatch] WARNING: ModuleList is empty, no hooks attached."))
                self._done = True
        else:
            self._h_pre = self._target_mod.register_forward_pre_hook(_pre_hook, with_kwargs=False)
            self._h_post = self._target_mod.register_forward_hook(_post_hook, with_kwargs=False)

        self._setup_module_spec_hooks(pl_module, trainer)

    def on_before_backward(self, trainer, pl_module, loss):
        if self._done or not getattr(trainer, "is_global_zero", True):
            return
        if trainer.global_step < self.warmup_steps:
            rank_zero_print(
                cyan(
                    f"[OneShotFwBwFLOPsDispatch] on_before_backward skipped: global_step={trainer.global_step} < warmup={self.warmup_steps}"
                )
            )
            return
        if self._current_eval_step >= self.eval_steps:
            return
        
        rank_zero_print(
            cyan(f"[OneShotFwBwFLOPsDispatch] on_before_backward at global_step={trainer.global_step}, eval_step={self._current_eval_step + 1}/{self.eval_steps}")
        )

        if self._use_cuda:
            with torch.cuda.device(self._dev):
                bw_start = torch.cuda.Event(enable_timing=True)
                bw_end = torch.cuda.Event(enable_timing=True)
                self._bw_start_ev = bw_start
                self._bw_end_ev = bw_end
                stream = torch.cuda.current_stream(self._dev)
                bw_start.record(stream)
        else:
            self._bw_t0 = time.perf_counter()

        # start backward FLOP counter
        self._fc_bwd = FlopCounterMode(display=False)
        self._fc_bwd.__enter__()

    def on_after_backward(self, trainer, pl_module):
        if self._done or not getattr(trainer, "is_global_zero", True):
            return
        if trainer.global_step < self.warmup_steps:
            rank_zero_print(
                cyan(
                    f"[OneShotFwBwFLOPsDispatch] on_after_backward skipped: global_step={trainer.global_step} < warmup={self.warmup_steps}"
                )
            )
            return
        if self._current_eval_step >= self.eval_steps:
            return
        
        rank_zero_print(
            cyan(f"[OneShotFwBwFLOPsDispatch] on_after_backward at global_step={trainer.global_step}, eval_step={self._current_eval_step + 1}/{self.eval_steps}")
        )

        # stop backward timing FIRST
        if self._use_cuda:
            with torch.cuda.device(self._dev):
                if self._bw_end_ev is not None and self._bw_start_ev is not None:
                    stream = torch.cuda.current_stream(self._dev)
                    self._bw_end_ev.record(stream)
                    if self.sync_each_hook:
                        torch.cuda.synchronize()
                        self._backward_ms = self._bw_start_ev.elapsed_time(self._bw_end_ev)
        else:
            t2 = time.perf_counter()
            self._backward_ms = (t2 - (self._bw_t0 or t2)) * 1000.0

        # If we're deferring sync, flush once here
        self._finalize_deferred_events()

        # collect backward FLOPs
        if self._fc_bwd is not None:
            self._fc_bwd.__exit__(None, None, None)
            self._bwd_flops = float(self._fc_bwd.get_total_flops())
            self._fc_bwd = None
            
            # Sanity check: backward FLOPs should be 2-3x forward FLOPs typically
            if self._fwd_flops is not None and self._fwd_flops > 0:
                ratio = self._bwd_flops / self._fwd_flops
                if ratio > 5.0 or ratio < 0.5:
                    rank_zero_print(
                        cyan(
                            f"[OneShotFwBwFLOPsDispatch] WARNING: Backward/Forward FLOPs ratio is {ratio:.2f}, "
                            "which is unusual (typical range: 1.5-3.0). FlopCounterMode may be inaccurate for "
                            "this model. Trust timing measurements instead."
                        )
                    )

        # whole-step memory peaks (since we didn't reset before backward)
        if self._use_cuda:
            with torch.cuda.device(self._dev):
                torch.cuda.synchronize()
                self._train_peak_alloc = torch.cuda.max_memory_allocated()
                self._train_peak_reserved = torch.cuda.max_memory_reserved()

        flex_forward_add, flex_backward_add, flex_records = FlexAttentionFLOPsTracker.collect()
        flex_forward_gflops = float("nan")
        flex_backward_gflops = float("nan")
        flex_density = float("nan")
        flex_calls = 0

        if flex_records:
            if self._fwd_flops is None:
                self._fwd_flops = 0.0
            if self._bwd_flops is None:
                self._bwd_flops = 0.0

            self._fwd_flops += flex_forward_add
            self._bwd_flops += flex_backward_add

            flex_calls = len(flex_records)
            flex_forward_gflops = flex_forward_add / 1e9
            flex_backward_gflops = flex_backward_add / 1e9
            flex_density = sum(record.density for record in flex_records) / flex_calls

        # ---- Collect metrics for this eval step
        forward_ms = float(self._forward_ms) if self._forward_ms is not None else float("nan")
        backward_ms = float(self._backward_ms) if self._backward_ms is not None else float("nan")

        ratio = float("nan")
        if math.isfinite(forward_ms) and math.isfinite(backward_ms) and forward_ms != 0.0:
            ratio = backward_ms / forward_ms

        # GB (1e9) and GiB (1024**3)
        def _to_gb(x):  return (x / 1e9) if x is not None else float("nan")
        def _to_gib(x): return (x / (1024 ** 3)) if x is not None else float("nan")

        fw_alloc_gb = _to_gb(self._fw_peak_alloc)
        fw_res_gb   = _to_gb(self._fw_peak_reserved)
        tr_alloc_gb = _to_gb(self._train_peak_alloc)
        tr_res_gb   = _to_gb(self._train_peak_reserved)

        fw_alloc_gib = _to_gib(self._fw_peak_alloc)
        fw_res_gib   = _to_gib(self._fw_peak_reserved)
        tr_alloc_gib = _to_gib(self._train_peak_alloc)
        tr_res_gib   = _to_gib(self._train_peak_reserved)

        fg = (self._fwd_flops / 1e9) if self._fwd_flops is not None else float("nan")
        bg = (self._bwd_flops / 1e9) if self._bwd_flops is not None else float("nan")
        bs = self._batch_size or float("nan")
        fg_ps = (fg / bs) if (isinstance(bs, int) and bs > 0 and isinstance(fg, float)) else float("nan")
        bg_ps = (bg / bs) if (isinstance(bs, int) and bs > 0 and isinstance(bg, float)) else float("nan")

        # Store metrics for this eval step
        step_metrics = {
            "forward_ms": forward_ms,
            "backward_ms": backward_ms,
            "bw_fw_ratio": ratio,
            "forward_GFLOPs": fg,
            "backward_GFLOPs": bg,
            "forward_per_sample_GFLOPs": fg_ps,
            "backward_per_sample_GFLOPs": bg_ps,
            "forward_peak_alloc_GB": fw_alloc_gb,
            "forward_peak_reserved_GB": fw_res_gb,
            "training_total_peak_alloc_GB": tr_alloc_gb,
            "training_total_peak_reserved_GB": tr_res_gb,
            "forward_peak_alloc_GiB": fw_alloc_gib,
            "forward_peak_reserved_GiB": fw_res_gib,
            "training_total_peak_alloc_GiB": tr_alloc_gib,
            "training_total_peak_reserved_GiB": tr_res_gib,
            "flex_forward_est_GFLOPs": flex_forward_gflops,
            "flex_backward_est_GFLOPs": flex_backward_gflops,
            "flex_avg_density": flex_density,
            "flex_call_count": float(flex_calls),
        }
        self._collected_metrics.append(step_metrics)
        
        # Increment eval step counter
        self._current_eval_step += 1
        
        rank_zero_print(
            cyan(
                f"[OneShotFwBwFLOPsDispatch] Collected eval step {self._current_eval_step}/{self.eval_steps}"
            )
        )

        # If we haven't collected all eval steps yet, return early
        if self._current_eval_step < self.eval_steps:
            return

        # ---- All eval steps completed, compute statistics
        rank_zero_print(cyan("[OneShotFwBwFLOPsDispatch] All eval steps completed, computing statistics..."))
        
        # Compute mean and std for each metric
        def _compute_stats(metric_name: str) -> Tuple[float, float]:
            values = [m[metric_name] for m in self._collected_metrics if math.isfinite(m[metric_name])]
            if not values:
                return float("nan"), float("nan")
            mean_val = sum(values) / len(values)
            if len(values) > 1:
                variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
                std_val = math.sqrt(variance)
            else:
                std_val = 0.0
            return mean_val, std_val

        stats = {}
        for metric_name in self._collected_metrics[0].keys():
            mean_val, std_val = _compute_stats(metric_name)
            stats[f"{metric_name}_mean"] = mean_val
            stats[f"{metric_name}_std"] = std_val

        # Log mean and std
        for key, value in stats.items():
            pl_module.log(f"flops/{key}", float(value), on_step=True, on_epoch=False)

        # Print summary with mean ± std
        summary_msg = (
            f"\n        ========== Statistics over {self.eval_steps} eval steps ==========\n"
            f"        forward_ms: {stats['forward_ms_mean']:.6f} ± {stats['forward_ms_std']:.6f}\n"
            f"        backward_ms: {stats['backward_ms_mean']:.6f} ± {stats['backward_ms_std']:.6f}\n"
            f"        bw/fw: {stats['bw_fw_ratio_mean']:.6f} ± {stats['bw_fw_ratio_std']:.6f}\n"
            f"        forward_GFLOPs: {stats['forward_GFLOPs_mean']:.6f} ± {stats['forward_GFLOPs_std']:.6f}\n"
            f"        forward_per_sample_GFLOPs: {stats['forward_per_sample_GFLOPs_mean']:.6f} ± {stats['forward_per_sample_GFLOPs_std']:.6f}\n"
            f"        backward_GFLOPs: {stats['backward_GFLOPs_mean']:.6f} ± {stats['backward_GFLOPs_std']:.6f}\n"
            f"        backward_per_sample_GFLOPs: {stats['backward_per_sample_GFLOPs_mean']:.6f} ± {stats['backward_per_sample_GFLOPs_std']:.6f}\n"
            f"        forward_peak_alloc: {stats['forward_peak_alloc_GB_mean']:.6f} ± {stats['forward_peak_alloc_GB_std']:.6f} GB "
            f"({stats['forward_peak_alloc_GiB_mean']:.6f} ± {stats['forward_peak_alloc_GiB_std']:.6f} GiB)\n"
            f"        forward_peak_reserved: {stats['forward_peak_reserved_GB_mean']:.6f} ± {stats['forward_peak_reserved_GB_std']:.6f} GB "
            f"({stats['forward_peak_reserved_GiB_mean']:.6f} ± {stats['forward_peak_reserved_GiB_std']:.6f} GiB)\n"
            f"        training_total_peak_alloc: {stats['training_total_peak_alloc_GB_mean']:.6f} ± {stats['training_total_peak_alloc_GB_std']:.6f} GB "
            f"({stats['training_total_peak_alloc_GiB_mean']:.6f} ± {stats['training_total_peak_alloc_GiB_std']:.6f} GiB)\n"
            f"        training_total_peak_reserved: {stats['training_total_peak_reserved_GB_mean']:.6f} ± {stats['training_total_peak_reserved_GB_std']:.6f} GB "
            f"({stats['training_total_peak_reserved_GiB_mean']:.6f} ± {stats['training_total_peak_reserved_GiB_std']:.6f} GiB)\n"
        )

        if not math.isnan(stats['flex_forward_est_GFLOPs_mean']):
            summary_msg += (
                f"        [Note: forward_GFLOPs and backward_GFLOPs above INCLUDE Flex Attention]\n"
                f"        flex_forward_est_GFLOPs: {stats['flex_forward_est_GFLOPs_mean']:.6f} ± {stats['flex_forward_est_GFLOPs_std']:.6f}\n"
                f"        flex_backward_est_GFLOPs: {stats['flex_backward_est_GFLOPs_mean']:.6f} ± {stats['flex_backward_est_GFLOPs_std']:.6f}\n"
                f"        flex_avg_density: {stats['flex_avg_density_mean']:.6f} ± {stats['flex_avg_density_std']:.6f}\n"
                f"        flex_call_count: {stats['flex_call_count_mean']:.6f} ± {stats['flex_call_count_std']:.6f}\n"
            )

        rank_zero_print(cyan("[OneShotFwBwFLOPsDispatch] FLOP summary (mean ± std):"))
        rank_zero_print(summary_msg)

        if self.log_wandb_table and getattr(trainer, "is_global_zero", True):
            try:
                import wandb
                tbl = wandb.Table(columns=[
                    "metric", "mean", "std",
                ])
                for metric_name in ["forward_ms", "backward_ms", "bw_fw_ratio",
                                   "forward_GFLOPs", "backward_GFLOPs",
                                   "forward_per_sample_GFLOPs", "backward_per_sample_GFLOPs",
                                   "forward_peak_alloc_GB", "forward_peak_reserved_GB",
                                   "training_total_peak_alloc_GB", "training_total_peak_reserved_GB",
                                   "flex_forward_est_GFLOPs", "flex_backward_est_GFLOPs",
                                   "flex_avg_density", "flex_call_count"]:
                    tbl.add_data(
                        metric_name,
                        float(stats[f"{metric_name}_mean"]),
                        float(stats[f"{metric_name}_std"]),
                    )
                wandb.log({"profile/oneshot_precise_stats": tbl})
            except Exception:
                pass

        rank_zero_print(cyan("[OneShotFwBwFLOPsDispatch] Logging module-spec metrics..."))
        self._log_module_spec_metrics(pl_module)
        self._teardown_module_spec_hooks()

        # mark done & remove hooks
        rank_zero_print(cyan("[OneShotFwBwFLOPsDispatch] Measurement complete, marking done."))
        self._done = True
        try:
            if self._h_pre is not None:
                self._h_pre.remove()
            if self._h_post is not None:
                self._h_post.remove()
        except Exception:
            pass
