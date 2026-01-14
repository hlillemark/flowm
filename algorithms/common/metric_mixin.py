from typing import Literal, Optional
from typing import Dict
from torch import Tensor
from algorithms.common.metrics.video import VideoMetric, SharedVideoMetricModelRegistry
from utils.distributed_utils import rank_zero_print
from utils.print_utils import cyan, bold_red
from utils.torch_utils import freeze_model
import torch

class MetricMixin:

    def _build_metrics(self) -> None:
        """
        Build the metrics.
        """
        
        # Metrics 
        if len(self.tasks) == 0:
            return
        registry = SharedVideoMetricModelRegistry()
        metric_types = self.logging_cfg.metrics
        if self.cfg.training.strategy == "fsdp":
            rank_zero_print(bold_red("Using FSDP, metrics like FVD, IS, REAL IS is not compatible, because they use torch_script."))
            metric_types.remove("fvd") if "fvd" in metric_types else None
            metric_types.remove("is") if "is" in metric_types else None
            metric_types.remove("real_is") if "real_is" in metric_types else None

        for task in self.tasks:
            match task:
                case "prediction":
                    self.metrics_prediction = VideoMetric(
                        registry,
                        metric_types,
                        resolution = (self.cfg.x_shape[1], self.cfg.x_shape[2]), # (h, w)
                        split_batch_size=self.logging_cfg.metrics_batch_size,
                    )
                    freeze_model(self.metrics_prediction)
                case "interpolation":
                    assert (
                        not self.use_causal_mask
                        and not self.is_full_sequence
                        and self.forward_window_size_in_tokens > 2
                    ), "To execute interpolation, the model must be non-causal, not full sequence, and be able to process more than 2 tokens."
                    self.metrics_interpolation = VideoMetric(
                        registry,
                        metric_types,
                        resolution = (self.cfg.x_shape[1], self.cfg.x_shape[2]), # (h, w)
                        split_batch_size=self.logging_cfg.metrics_batch_size,
                    )
                    freeze_model(self.metrics_interpolation)
                case "reconstruction":
                    if not self.training:
                        self.tasks.remove("reconstruction")
                        rank_zero_print(cyan("Reconstruction is not supported during training, removing it from the tasks."))
                    else:
                        self.metrics_reconstruction = VideoMetric(
                            registry,
                            metric_types,
                            resolution = (self.cfg.x_shape[1], self.cfg.x_shape[2]), # (h, w)
                            split_batch_size=self.logging_cfg.metrics_batch_size,
                        )
                        freeze_model(self.metrics_reconstruction)
    

    def _metrics(
        self,
        task: Literal["prediction", "interpolation", "reconstruction"],
    ) -> Optional[VideoMetric]:
        """
        Get the appropriate metrics object for the given task.
        """
        return getattr(self, f"metrics_{task}", None)


    def _update_metrics(self, all_videos: Dict[str, Tensor], dataloader_idx: int = 0, verbose: bool = False) -> None:
        """Update metrics for the specific dataloader during validation/test step."""
        if (
            self.logging_cfg.n_metrics_frames is not None
        ):  # only consider the first n_metrics_frames for evaluation
            all_videos = {
                k: v[:, : self.logging_cfg.n_metrics_frames] for k, v in all_videos.items()
            }

        gt_videos = all_videos["gt"]
        for task in self.tasks:
            if task in all_videos:
                metric = self._metrics(task)
                videos = all_videos[task]
            else:
                if verbose:
                    rank_zero_print(cyan(f"{task} is not carried in this run, check whether this is expected behavior!"))
                continue
            # context_mask = torch.zeros(self.n_frames).bool().to(self.device)
            context_mask = torch.zeros(videos.shape[1]).bool().to(self.device)
            match task:
                case "prediction":
                    context_mask[: self.n_context_frames] = True
                case "interpolation":
                    context_mask[[0, -1]] = True
                case "reconstruction":
                    context_mask = context_mask
            if self.logging_cfg.n_metrics_frames is not None:
                context_mask = context_mask[: self.logging_cfg.n_metrics_frames]
            metric(videos, gt_videos, context_mask=context_mask)
  