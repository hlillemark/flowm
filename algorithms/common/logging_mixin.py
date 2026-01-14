from typing import Dict, List, Literal, Any, Optional
from torch import Tensor
from utils.distributed_utils import is_rank_zero, rank_zero_print, get_rank
from utils.print_utils import cyan
from utils.logging_utils import log_video, log_video_as_images
import torch

class LoggingMixin:

    def _log_videos(self, all_videos: Dict[str, Tensor], namespace: str, dataloader_idx: int = 0, video_metadata: Optional[Dict[str, Tensor]] = None) -> None:
        """Log videos during validation/test step."""
        rank = get_rank()

        if self.trainer.sanity_checking and (not self.logging_cfg.sanity_generation):
            return

        if self.num_logged_videos[dataloader_idx] >= self.logging_cfg.max_num_videos:
            return

        should_log_to_logger = bool(self.logger) and is_rank_zero
        should_save_raw = self.logging_cfg.raw_dir is not None

        # Nothing to log on this rank
        if not should_log_to_logger and not should_save_raw:
            return

        batch_size, n_frames = all_videos["gt"].shape[:2]

        num_videos_to_log = min(
            self.logging_cfg.max_num_videos - self.num_logged_videos[dataloader_idx],
            batch_size,
        )
        cut_videos = lambda x: x[:num_videos_to_log]
        
        # Also handle auxiliary videos if present
        extra_video_keys = []
        if "latent_map_mag" in all_videos:
            extra_video_keys.append("latent_map_mag")

        for task in list(self.tasks) + extra_video_keys:
            if task == "interpolation":
                context_frames = torch.tensor(
                    [0, n_frames - 1], device=self.device, dtype=torch.long
                )
            elif task == "prediction":
                context_frames = self.n_context_frames
            elif task == "reconstruction":
                context_frames = 0 # reconstruction in diffusion won't have context frames
            elif task == "latent_map_mag":
                # already aligned with prediction concat; no special context overlay in visualization
                context_frames = 0
            else:
                raise ValueError(f"Invalid task: {task}")
                
            if task not in all_videos:
                # rank_zero_print(cyan(f"{task} is not carried in this run, check whether this is expected behavior!"))
                continue

            raw_dir = self.logging_cfg.raw_dir if task == "prediction" else None # Only log raw if doing prediction, not reconstruction
            log_images_this_task = (
                self.logging_cfg.log_video_as_images.enable
                and (should_log_to_logger or raw_dir is not None)
            )

            # Skip if this rank has nothing to do for this task
            if not should_log_to_logger and raw_dir is None and not log_images_this_task:
                continue

            log_video(
                cut_videos(all_videos[task].clone()), # to avoid modifying the original tensor
                cut_videos(all_videos["gt"].clone()) if task != "latent_map_mag" else None,
                step=None if namespace == "test" else self.global_step,
                namespace=f"{task}_vis_{namespace}_loader_id={dataloader_idx}",
                logger=self.logger.experiment if should_log_to_logger else None,
                indent=self.num_logged_videos[dataloader_idx],
                raw_dir=raw_dir,
                context_frames=context_frames,
                captions=f"{task} | gt" if task != "latent_map_mag" else f"{task}",
                video_metadata=video_metadata,
                rank=rank,
                log_to_logger=should_log_to_logger,
            )

            if log_images_this_task:
                video_format = self.logging_cfg.log_video_as_images.format # png, pdf
                indices = list(self.logging_cfg.log_video_as_images.indices)
                if indices is None:
                    raise ValueError("indices is required when log_video_as_images is enabled")
                log_video_as_images(
                    cut_videos(all_videos[task].clone()), 
                    cut_videos(all_videos["gt"].clone()) if task != "latent_map_mag" else None,
                    step = None if namespace == "test" else self.global_step,
                    namespace=f"{task}_vis_as_images_{namespace}_loader_id={dataloader_idx}",
                    logger=self.logger.experiment if should_log_to_logger else None,
                    indent=self.num_logged_videos[dataloader_idx],
                    indices=indices, 
                    format=video_format,
                    raw_dir=raw_dir,
                    video_metadata=video_metadata,
                    rank=rank,
                    log_to_logger=should_log_to_logger,
                )

        self.num_logged_videos[dataloader_idx] += batch_size
    

    def _log_videos_for_validation_only(self, all_videos: Dict[str, Tensor], namespace: str, dataloader_idx: int = 0) -> None:
        """
        We want to use multiple devices to run extremely long videos, so it's not good practice to use gather_data here, in this case. Therefore, we want each device to log their own videos.
        """
        rank = get_rank()
        should_log_to_logger = bool(self.logger) and is_rank_zero

        too_large_to_gather = False
        try:
            gathered_videos = self.gather_data(all_videos)
            all_videos = gathered_videos
        except:
            print("gather_data failed")
            too_large_to_gather = True
            torch.cuda.empty_cache()

        batch_size, n_frames = all_videos["gt"].shape[:2]

        if self.num_logged_videos[dataloader_idx] >= self.logging_cfg.max_num_videos:
            return

        for task in self.tasks:
            if task == "prediction":
                context_frames = self.n_context_frames
            else:
                context_frames = torch.tensor(
                    [0, n_frames - 1], device=self.device, dtype=torch.long
                )
            if task not in all_videos:
                # rank_zero_print(cyan(f"{task} is not carried in this run, check whether this is expected behavior!"))
                continue
            
            num_videos_to_log = min(
                self.logging_cfg.max_num_videos - self.num_logged_videos[dataloader_idx],
                batch_size,
            )

            verbose_namespace = f"{task}_vis_{namespace}_loader_id={dataloader_idx}"
            cut_videos = lambda x: x[:num_videos_to_log]

            raw_dir = self.logging_cfg.raw_dir if not too_large_to_gather else f"{self.logger.save_dir}/{task}_vis_{namespace}_loader_id={dataloader_idx}/device_id={self.trainer.local_rank}"
            if too_large_to_gather:
                log_video(
                    cut_videos(all_videos[task].clone()),
                    cut_videos(all_videos["gt"].clone()),
                    step=None if namespace == "test" else self.global_step,
                    namespace=verbose_namespace,
                    logger=self.logger.experiment if should_log_to_logger else None,
                    indent=self.num_logged_videos[dataloader_idx],
                    raw_dir=raw_dir,
                    context_frames=context_frames,
                    captions=f"{task} | gt",
                    rank=rank,
                    log_to_logger=should_log_to_logger,
                )
            else:
                if should_log_to_logger:
                    log_video(
                        cut_videos(all_videos[task].clone()),
                        cut_videos(all_videos["gt"].clone()),
                        step=None if namespace == "test" else self.global_step,
                        namespace=verbose_namespace,
                        logger=self.logger.experiment,
                        indent=self.num_logged_videos[dataloader_idx],
                        raw_dir=raw_dir,
                        context_frames=context_frames,
                        captions=f"{task} | gt",
                        rank=rank,
                        log_to_logger=should_log_to_logger,
                    )


    def _temp_save_videos(self, latents: Tensor, path: str):
        """
        Temp save the videos to the given path.
        """
        videos = self._decode(latents)
        from utils.video_utils import write_video_to_file
        write_video_to_file(videos[0], path)
