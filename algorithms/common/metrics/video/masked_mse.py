from typing import Any, Optional
import torch
from torch import Tensor
from torchmetrics import Metric
from .shared_registry import SharedVideoMetricModelRegistry
from .types import VideoMetricModelType


class MaskedMeanSquaredError(Metric):
    """
    Computes the Mean Squared Error (MSE) between two tensors, with an option to mask certain positions.

    Args:
        preds (Tensor): The predicted values.
        target (Tensor): The target values.
        mask (Optional[Tensor]): A mask to ignore certain values. If None, all values are considered.
    """

    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    plot_lower_bound: float = 0.0

    sum_squared_error: Tensor
    total: Tensor

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> None:
        """
        Update the metric with predictions and targets.
        Args:
            preds (Tensor): The predicted values.
            target (Tensor): The target values.
            mask (Optional[Tensor]): A mask to ignore certain positions. If None, all values are considered.
        """
        if mask is None:
            mask = torch.ones_like(target)
        else:
            assert mask.shape == target.shape, "Mask shape must match target shape."
            mask = mask.float()
        squared_error = (preds - target) ** 2
        masked_squared_error = squared_error * mask
        
        self.sum_squared_error += masked_squared_error.sum()
        self.total += mask.sum()

    def compute(self) -> Tensor:
        return self.sum_squared_error / self.total if self.total > 0 else torch.tensor(0.0)