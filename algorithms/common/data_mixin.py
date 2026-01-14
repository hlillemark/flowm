from typing import Optional
from torch import Tensor
import torch


class DataMixin:

    @torch.no_grad()
    def _process_conditions(
        self,
        conditions: Optional[Tensor],
        noise_levels: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        """
        Post-process the conditions before feeding them to the model.
        For example, conditions that should be computed relatively (e.g. relative poses)
        should be processed here instead of the dataset.

        Args:
            conditions (Optional[Tensor], "B T ..."): The external conditions for the video.
            noise_levels (Optional[Tensor], "B T"): Current noise levels for each token during sampling
        """

        if conditions is None:
            return conditions
        match self.cfg.external_cond_processing:
            case "mask_first": # NOTE: because first condition is meaningless, fr_{i} + condition x_{i+1} -> fr_{i+1}, we don't have fr_{-1}
                mask = torch.ones_like(conditions)
                mask[:, :1, : self.external_cond_dim] = 0
                return conditions * mask
            case "mask_last":
                mask = torch.ones_like(conditions)
                mask[:, -1, : self.external_cond_dim] = 0
                return conditions * mask
            case _:
                raise NotImplementedError(
                    f"External condition processing {self.cfg.external_cond_processing} is not implemented."
                )

