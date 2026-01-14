from typing import Any, Tuple
import torch
from torch import Tensor
from torchmetrics.metric import Metric


class PerFrameMSE(Metric):
    r"""
    Framewise mean squared error over C,H,W with batch & time preserved.

    Inputs:
        preds:  Tensor of shape [B, T, C, H, W]
        target: Tensor of shape [B, T, C, H, W]

    State across updates:
        - Accumulates per-(b,t) sum of squared errors and counts, then
          concatenates sequences across updates in arrival order.

    Output (compute):
        Tensor of shape [N_seq, T], where N_seq is the total sequences
        accumulated across updates. Each value is MSE over C,H,W for that frame.

    Notes:
        - All updates must have the same T; an error is raised otherwise.
        - If you want RMSE instead, take `.sqrt()` of the result.
    """
    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    # Flat buffers that store per-(b,t) accumulations in arrival order
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # sum of squared errors per (b,t), 1D; length grows as updates are called
        self.add_state("sum_sq", default=torch.empty(0), dist_reduce_fx="sum")
        # counts (number of pixels * channels) per (b,t), 1D; same length as sum_sq
        self.add_state("count", default=torch.empty(0, dtype=torch.long), dist_reduce_fx="sum")
        # time dimension (must be consistent across updates). -1 means "unset"
        self.add_state("T", default=torch.tensor(-1, dtype=torch.long), dist_reduce_fx="max")

    @staticmethod
    def _validate_inputs(preds: Tensor, target: Tensor) -> Tuple[int, int, int, int, int]:
        if preds.ndim != 5 or target.ndim != 5:
            raise ValueError(f"Expected preds/target with 5 dims [B,T,C,H,W], got {preds.shape} and {target.shape}")
        if preds.shape != target.shape:
            raise ValueError(f"Preds/target must have identical shape, got {preds.shape} vs {target.shape}")
        B, T, C, H, W = preds.shape
        if B == 0 or T == 0:
            raise ValueError("Batch size B and time T must be > 0.")
        return B, T, C, H, W

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        B, T, C, H, W = self._validate_inputs(preds, target)

        # Enforce consistent T across updates
        if int(self.T.item()) == -1:
            self.T = torch.tensor(T, dtype=torch.long, device=preds.device)
        elif int(self.T.item()) != T:
            raise ValueError(f"All updates must use the same time length T. "
                             f"Seen T={int(self.T.item())}, got T={T}.")

        # Compute per-(b,t) SSE and counts over C,H,W
        # diff: [B, T, C, H, W] -> [B*T, C*H*W]
        diff = (preds - target).reshape(B * T, C * H * W)
        sse_bt = (diff ** 2).sum(dim=1)                            # [B*T]
        n_bt = torch.full((B * T,), C * H * W, dtype=torch.long, device=preds.device)  # [B*T]

        # Append to running state (concatenate along 0)
        if self.sum_sq.numel() == 0:
            self.sum_sq = sse_bt.detach()
            self.count = n_bt
        else:
            self.sum_sq = torch.cat([self.sum_sq, sse_bt.detach()], dim=0)
            self.count = torch.cat([self.count, n_bt], dim=0)

    def compute(self) -> Tensor:
        if self.sum_sq.numel() == 0:
            # No data seen; return empty [0, T] (T may be -1 if truly empty)
            T = int(self.T.item()) if int(self.T.item()) > 0 else 0
            return self.sum_sq.new_zeros((0, T))
        T = int(self.T.item())
        # Per-(b,t) MSE = SSE / count
        mse_flat = self.sum_sq / self.count.clamp_min(1).to(self.sum_sq.dtype)  # [N_seq*T]
        # Reshape back to [N_seq, T]
        if mse_flat.numel() % T != 0:
            # This should not happen if update enforced a fixed T
            raise RuntimeError(f"Inconsistent internal state: total frames {mse_flat.numel()} not divisible by T={T}.")
        N_seq = mse_flat.numel() // T
        return mse_flat.view(N_seq, T)
