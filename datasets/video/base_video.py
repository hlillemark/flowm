from typing import Literal, List, Dict, Any, Callable, Tuple, Optional
from abc import ABC, abstractmethod
import random
import bisect
from pathlib import Path
import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torchvision.datasets.video_utils import _collate_fn, _VideoTimestampsDataset
from tqdm import tqdm
from einops import rearrange
from utils.distributed_utils import rank_zero_print
from utils.print_utils import cyan, red
from decord import cpu, VideoReader

SPLIT = Literal["training", "validation", "test"]
RANDOME_SEED = 42

class BaseVideoDataset(torch.utils.data.Dataset, ABC):
    """
    Common base class for video dataset.
    Methods here are shared between simple and advanced video datasets

    Folder structure of each dataset:
    - {save_dir} (specified in config, e.g., data/phys101)
        - /{split}
            - data files (e.g. 000001.mp4, 000001.pt)
        - /metadata
            - {split}.pt
    - {save_dir}_latent_{latent_resolution} (same structure as save_dir)
    """

    _ALL_SPLITS = ["training", "validation", "test"]
    metadata: Dict[str, Any]

    def __init__(
        self,
        cfg: DictConfig,
        split: SPLIT = "training",
    ):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.resolution = cfg.resolution
        spatial_downsampling = cfg.downsampling_factor[1]
        self.latent_resolution = [cfg.resolution[0] // spatial_downsampling, cfg.resolution[1] // spatial_downsampling]
        self.save_dir = Path(cfg.save_dir)
        self.latent_dir = self.save_dir.with_name(
            f"{self.save_dir.name}_latent_{'x'.join(map(str, self.latent_resolution))}{'_' + cfg.latent.suffix if cfg.latent.suffix else ''}"
        )
        self.split_dir = self.save_dir / split
        self.metadata_dir = self.save_dir / "metadata"
        
        self.cond_loading_style = cfg.cond_loading_style

        rank_zero_print( "\n\n" + 10*"==" + f"SPLIT={self.split}" + 10*"==")

        # Download dataset if not exists
        if self._should_download():
            self.download_dataset()
        if not self.metadata_dir.exists():
            self.metadata_dir.mkdir(exist_ok=True, parents=True)
        
        for split in self._ALL_SPLITS:
            self.build_metadata(split)

        try:
            self.metadata = self.load_metadata()
            self.augment_dataset()
            self.subsample_dataset()
            self.transform = self.build_transform()
        except Exception as e:
            rank_zero_print(red(f"{e}"))
            self.metadata = []

    def subsample_dataset(self) -> None:
        """
        Subsample the dataset
        """
        if hasattr(self.cfg, f"num_{self.split}_videos"):
            num_videos = getattr(self.cfg, f"num_{self.split}_videos")
            if num_videos is None:
                return None
            self.num_videos = num_videos
            rank_zero_print(f"Subsampling {self.split} dataset to {self.num_videos} videos")
            random.seed(RANDOME_SEED)
            self.metadata = random.sample(self.metadata, self.num_videos)
        
        return None

    def _should_download(self) -> bool:
        """
        Check if the dataset should be downloaded
        """
        return not (self.save_dir / self.split).exists()

    @abstractmethod
    def download_dataset(self) -> None:
        """
        Download dataset from the internet and build it in save_dir
        """
        raise NotImplementedError

    def build_metadata(self, split: SPLIT) -> None:
        """
        Build metadata for the dataset and save it in metadata_dir
        This may vary depending on the dataset.
        Default:
        ```
        {
            "video_paths": List[str],
            "video_pts": List[str],
            "video_fps": List[float],
        }
        ```
        """
        if (self.metadata_dir / f"{split}.pt").exists():
            return
        video_paths = sorted(list((self.save_dir / split).glob("**/*.mp4")), key=str)
        dl: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            _VideoTimestampsDataset(video_paths),
            batch_size=16,
            num_workers=4,
            collate_fn=_collate_fn,
        )
        video_pts: List[torch.Tensor] = (
            []
        )  # each entry is a tensor of shape (num_frames, )
        video_fps: List[float] = []
        valid_video_paths: List[Path] = []

        with tqdm(total=len(dl), desc=f"Building metadata for {split}", position=0) as pbar:
            for batch in dl:
                pbar.update(1)
                batch_pts, batch_fps, batch_valid_video_path = list(zip(*batch))
                batch_pts = [
                    torch.as_tensor(pts, dtype=torch.long).cpu() for pts in tqdm(batch_pts, desc="Converting to tensor", position=1)
                ]
                video_pts.extend(batch_pts)
                video_fps.extend(batch_fps)
                valid_video_paths.extend(batch_valid_video_path)

        metadata = {
            "video_paths": valid_video_paths,
            "video_pts": video_pts,
            "video_fps": video_fps,
        }
        torch.save(metadata, self.metadata_dir / f"{split}.pt")

    def subsample(
        self,
        metadata: List[Dict[str, Any]],
        filter_fn: Callable[[Dict[str, Any]], bool],
        filter_msg: str,
    ) -> List[Dict[str, Any]]:
        """
        Subsample the dataset with the given filter function
        """
        before_len = len(metadata)
        metadata = [
            video_metadata for video_metadata in metadata if filter_fn(video_metadata)
        ]
        after_len = len(metadata)
        rank_zero_print(
            cyan(
                f"{self.split}: {after_len} / {before_len} videos will be used after filtering out {filter_msg}"
            ),
        )
        return metadata

    def augment_dataset(self) -> None:
        """
        Augment the dataset
        """
        # pylint: disable=assignment-from-none
        augmentation = self._build_data_augmentation()
        if augmentation is not None:
            self.metadata = self._augment_dataset(self.metadata, *augmentation)

    def _augment_dataset(
        self,
        metadata: List[Dict[str, Any]],
        augment_fn: Callable[[Dict[str, Any]], List[Dict[str, Any]]],
        augment_msg: str,
    ) -> List[Dict[str, Any]]:
        """
        Augment the dataset (corresponds to metadata) with the given augment function

        Args:
            metadata: list of video metadata - stands for the dataset
            augment_fn: function that takes video metadata and returns a list of augmented video metadata
        """
        before_len = len(metadata)
        metadata = [
            augmented_video_metadata
            for video_metadata in metadata
            for augmented_video_metadata in augment_fn(video_metadata)
        ]
        after_len = len(metadata)
        rank_zero_print(
            cyan(
                f"{self.split}: {before_len} -> {after_len} videos after augmenting with {augment_msg}"
            ),
        )
        return metadata

    def _build_data_augmentation(
        self,
    ) -> Optional[Tuple[Callable[[Dict[str, Any]], List[Dict[str, Any]]], str]]:
        """
        Build a data augmentation (composed of augment_fn and augment_msg) that will be applied to the dataset

        if None, no data augmentation will be applied
        """
        return None

    def load_metadata(self) -> List[Dict[str, Any]]:
        """
        Load metadata from metadata_dir
        """
        try:
            metadata = torch.load(
                self.metadata_dir / f"{self.split}.pt", weights_only=False
            )
        except Exception as e:
            rank_zero_print(red(f"Error loading metadata: {e}, please check whether you have included {self.split} into _ALL_SPLITS_ in corresponding dataset class"))
            raise e
        return [
            {key: metadata[key][i] for key in metadata.keys()}
            for i in range(len(metadata["video_paths"]))
        ]

    def video_length(self, video_metadata: Dict[str, Any]) -> int:
        """
        Return the length of the video at idx
        """
        return len(video_metadata["video_pts"])

    def video_metadata_to_latent_path(self, video_metadata: Dict[str, Any]) -> Path:
        """
        Convert video_path to latent_path
        """
        return (
            self.latent_dir / video_metadata["video_paths"].relative_to(self.save_dir)
        ).with_suffix(".pt")
        
    def video_metadata_to_latent_depth_path(self, video_metadata: Dict[str, Any]) -> Path:
        """
        Convert video_path to latent_path
        """
        p = self.latent_dir / video_metadata["video_paths"].relative_to(self.save_dir)
        return p.with_stem(p.stem + "_depth").with_suffix(".pt")


    def get_latent_paths(self, split: SPLIT) -> List[Path]:
        """
        Return list of latent paths for the given split
        """
        return sorted(list((self.latent_dir / split).glob("**/*.pt")), key=str)

    def load_video(
        self,
        video_metadata: Dict[str, Any],
        start_frame: int,
        end_frame: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Load video from video_idx with given start_frame and end_frame (exclusive)
        if end_frame is None, load until the end of the video
        return shape: (T, C, H, W)
        """
        if end_frame is None:
            end_frame = self.video_length(video_metadata)
        video_path, video_pts = (
            video_metadata["video_paths"],
            video_metadata["video_pts"],
        )

        indices = video_pts[start_frame:end_frame].tolist()
        vr = VideoReader(str(video_path), ctx=cpu(0))
        
        video = torch.Tensor(vr.get_batch(indices).asnumpy())
        
        return video.permute(0, 3, 1, 2) / 255.0

    def exclude_short_videos(
        self, metadata: List[Dict[str, Any]], min_frames: int
    ) -> List[Dict[str, Any]]:
        """
        Exclude videos that are shorter than n_frames
        """
        return self.subsample(
            metadata,
            lambda video_metadata: self.video_length(video_metadata) >= min_frames,
            f"videos shorter than {min_frames} frames",
        )

        
    @classmethod
    def get_splits(cls):
        return cls._ALL_SPLITS
    
    
class BaseSimpleVideoDataset(BaseVideoDataset):
    """
    Base class for simple video datasets
    that load full videos with given resolution
    Also provides latent_path where latent should be saved
    """

    def __init__(self, cfg: DictConfig, split: SPLIT = "training"):
        super().__init__(cfg, split)
        self.latent_dir.mkdir(exist_ok=True, parents=True)
        # filter videos to only include the ones that have not been preprocessed
        self.metadata = self.exclude_videos_with_latents(self.metadata)
        self.max_frames = cfg.max_frames
        if self.max_frames:
            self.metadata = self.exclude_short_videos(self.metadata, self.max_frames)
        rank_zero_print(
            cyan(
                f"{self.split}: {len(self.metadata)}" + f"each contains {self.max_frames} frames" if self.max_frames else ""
            )
        )

    def exclude_videos_with_latents(
        self, metadata: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        latent_paths = set(self.get_latent_paths(self.split))

        return self.subsample(
            metadata,
            lambda video_metadata: self.video_metadata_to_latent_path(video_metadata)
            not in latent_paths,
            "videos that have already been preprocessed to latents",
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        loads video together with the path where latent should be saved
        """
        video_metadata = self.metadata[idx]
        video = self.load_video(video_metadata, 0, self.max_frames)

        return (
            self.transform(video),
            self.video_metadata_to_latent_path(video_metadata).as_posix(),
        )


class BaseAdvancedVideoDataset(BaseVideoDataset):
    """
    Base class for video dataset
    that load video clips with given resolution and frame skip
    Videos may be of variable lengths.
    """

    cumulative_sizes: List[int]
    idx_remap: List[int]

    def __init__(
        self,
        cfg: DictConfig,
        split: SPLIT = "training",
        current_epoch: Optional[int] = None,
        purpose: Literal["training", "validation", "test"] = "training",
    ):
        super().__init__(cfg, split)
        self.use_preprocessed_latents = (
            cfg.latent.enable and cfg.latent.type.startswith("pre_")
        )
        self.current_subepoch = current_epoch
        self.subdataset_size = cfg.subdataset_size

        self.purpose = purpose
        
        # check if we are running preprocess latent dataset. if so, we simulate the original 
        # basesimplevideodataset class
        self.is_latent_preprocessing_expt = cfg.latent.is_preprocessing_expt

        if not self.is_latent_preprocessing_expt and self.use_preprocessed_latents and not self.latent_dir.exists():
            raise ValueError(
                f"Preprocess the video to latents first and save them in {self.latent_dir}"
            )

        self.external_cond_dim = cfg.external_cond_dim * (
            cfg.frame_skip if cfg.external_cond_stack else 1
        )
        # This `self.n_frames` is the number of frames take to make an item, not the real frames in the item
        self.n_frames = (
            1
            + ((cfg.max_frames if purpose == "training" else cfg.n_frames) - 1) # cfg.n_frames's default value is dataset.max_frames
            * cfg.frame_skip
        )
        self.frame_skip = cfg.frame_skip

        if self.is_latent_preprocessing_expt:
            self.metadata = self.exclude_videos_with_latents(self.metadata)
        elif self.use_preprocessed_latents:
            self.metadata = self.exclude_videos_without_latents(self.metadata)

        if self.purpose == "training" or cfg.filter_min_len is None:
            self.filter_min_len = self.n_frames
        else:
            self.filter_min_len = cfg.filter_min_len

        self.metadata = self.choose_by_path() 
        self.metadata = self.exclude_short_videos(self.metadata, self.filter_min_len)
        

        if hasattr(self.cfg, f"num_{self.split}_clips"):
            self.num_clips = getattr(self.cfg, f"num_{self.split}_clips")
            rank_zero_print(cyan(f"Using {self.num_clips} clips for {self.split}"))
        else:
            self.num_clips = None

        if not self.is_latent_preprocessing_expt and self.num_clips is None and self.purpose == "validation":
            rank_zero_print(cyan(f"Using default number ({self.cfg.num_default_clips}) of videos for {self.split}"))
            self.num_clips = self.cfg.num_default_clips

        self.on_before_prepare_clips()
        # When overriding __init__ in a subclass,
        # no more dataset filtering should be performed after prepare_clips is called (i.e. after super().__init__())
        # Instead, override on_before_prepare_clips to perform additional modifications before prepare_clips
        self.prepare_clips()
    
    def choose_by_path(self) -> List[Dict[str, Any]]:
        """
        Choose videos by path
        """
        if self.cfg.path_list is not None:
            return self.subsample(self.metadata, lambda x: str(x["video_paths"]) in [item[0] for item in self.cfg.path_list], "videos by path list")
        return self.metadata 

    @property
    def use_subdataset(self) -> bool:
        """
        Check if subdataset strategy is enabled
        """
        return (
            self.split in self.cfg.training_splits
            and self.subdataset_size is not None
            and self.current_subepoch is not None
        )

    @property
    def use_split_subdataset(self) -> bool:
        """
        Check if using deterministic subdataset for evaluation
        """
        
        if self.num_clips is not None and self.purpose == "training":
            rank_zero_print(cyan(f"Using split subdataset for training, with num_clips = {self.num_clips} which will cause a great decrease in the amount of training dataset. Please make sure this is intended !"))
        return self.num_clips is not None # and self.purpose == "validation"

    def on_before_prepare_clips(self) -> None:
        """
        Additional setup before preparing clips (e.g. excluding invalid videos)
        """
        return

    def prepare_clips(self) -> None:
        """
        Compute cumulative sizes for the dataset and update self.cumulative_sizes
        Shuffle the dataset with a fixed seed
        """
        num_clips = torch.as_tensor(
            [
                max(self.video_length(video_metadata) - self.n_frames + 1, 1)
                for video_metadata in self.metadata
            ]
        )
        self.cumulative_sizes = num_clips.cumsum(0).tolist()
        self.idx_remap = self._build_idx_remap()

    def _build_idx_remap(self) -> List[int]:
        """
        Deterministically build idx_remap for the dataset, which maps the indices of the current dataset to the absolute indices of the full dataset
        - If use_subdataset is True, idx_remap remaps the subdataset indices (ranging from 0 to self.subdataset_size) to the full dataset indices (ranging from 0 to self.cumulative_sizes[-1])
        - If use_split_subdataset is True, idx_remap remaps the indices (range from 0 to self.num_clips) to the full dataset indices (ranging from 0 to self.cumulative_sizes[-1]), where each index corresponds to a randomly chosen clip from randomly chosen num_validation_clips videos
        - Otherwise, idx_remap is a deterministic shuffle of 0 to self.__len__()
        """
        if self.use_subdataset:
            # assign deterministic sequence of indices to each subepoch
            def idx_to_epoch_and_idx(idx: int) -> Tuple[int, int]:
                effective_idx = idx + self.subdataset_size * self.current_subepoch
                return divmod(effective_idx, self.cumulative_sizes[-1])

            start_epoch, start_idx_in_epoch = idx_to_epoch_and_idx(0)
            end_epoch, end_idx_in_epoch = idx_to_epoch_and_idx(self.subdataset_size - 1)
            assert (
                0 <= end_epoch - start_epoch <= 1
            ), "Subdataset size should be <= dataset size"

            epoch_to_shuffled_indices: Dict[int, List[int]] = {}
            for epoch in range(start_epoch, end_epoch + 1):
                indices = list(range(self.cumulative_sizes[-1]))
                random.seed(epoch)
                random.shuffle(indices)
                epoch_to_shuffled_indices[epoch] = indices

            if start_epoch == end_epoch:
                idx_remap = epoch_to_shuffled_indices[start_epoch][
                    start_idx_in_epoch : end_idx_in_epoch + 1
                ]
            else:
                idx_remap = (
                    epoch_to_shuffled_indices[start_epoch][start_idx_in_epoch:]
                    + epoch_to_shuffled_indices[end_epoch][: end_idx_in_epoch + 1]
                )
            assert (
                len(idx_remap) == self.subdataset_size
            ), "Something went wrong while remapping subdataset indices"
            return idx_remap
        elif self.use_split_subdataset:
            # deterministically choose one clip per video for evaluation
            # raise a warning if num_validation_clips > num_videos
            rank_zero_print(f"Using {self.split} subdataset of size {self.num_clips}")
            if self.num_clips > len(self.cumulative_sizes):
                rank_zero_print(
                    cyan(
                        f"There are less clips ({len(self.cumulative_sizes)}) in the dataset than the number of requested evaluation clips ({self.num_clips})"
                    )
                )
            random.seed(RANDOME_SEED)
            idx_remap = []
            # each video has self.cumulative_sizes[i+1] - self.cumulative_sizes[i] clips, randomly choose one
            for video_idx, (start, end) in enumerate(zip(
                [0] + self.cumulative_sizes[:-1], self.cumulative_sizes
            )):
                if self.cfg.path_list is not None:
                    start_frame = self.match_clip_by_video_path(str(self.metadata[video_idx]["video_paths"]))
                    idx_remap.append(start_frame) # no random choice, just use the start frame in self.cfg.path_list
                else:
                    idx_remap.append(random.randrange(start, end))
            random.shuffle(idx_remap)
            return idx_remap[: self.num_clips]

        else:
            # shuffle but keep the same order for each epoch, so validation sample is diverse yet deterministic
            idx_remap = list(range(self.__len__()))
            return idx_remap

    def exclude_videos_without_latents(
        self, metadata: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        latent_paths = set(self.get_latent_paths(self.split))
        use_depth = getattr(self.cfg, "use_depth", False)
        return self.subsample(
            metadata,
            lambda video_metadata: (
                self.video_metadata_to_latent_path(video_metadata) in latent_paths and
                (not use_depth or self.video_metadata_to_latent_depth_path(video_metadata) in latent_paths)
            ),
            "videos without latents" + (" or depth latents" if use_depth else ""),
        )

    def exclude_videos_with_latents(
        self, metadata: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        latent_paths = set(self.get_latent_paths(self.split))
        use_depth = getattr(self.cfg, "use_depth", False)
        return self.subsample(
            metadata,
            lambda video_metadata: (
                self.video_metadata_to_latent_path(video_metadata) not in latent_paths or
                (use_depth and self.video_metadata_to_latent_depth_path(video_metadata) not in latent_paths)
            ),
            "videos that have already been preprocessed to latents" + (" or depth latents" if use_depth else ""),
        )

    def get_clip_location(self, idx: int) -> Tuple[int, int]:
        idx = self.idx_remap[idx]
        video_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if video_idx == 0:
            clip_idx = idx
        else:
            clip_idx = idx - self.cumulative_sizes[video_idx - 1]
        return video_idx, clip_idx

    def load_latent(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ) -> torch.Tensor:
        latent_path = self.video_metadata_to_latent_path(video_metadata)
        return torch.load(latent_path, weights_only=False)[start_frame:end_frame]

    def load_depth_latent(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ):
        latent_depth_path = self.video_metadata_to_latent_depth_path(video_metadata)
        return torch.load(latent_depth_path, weights_only=False)[start_frame:end_frame]

    @abstractmethod
    def load_cond(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ) -> torch.Tensor:
        raise NotImplementedError

    def load_video_and_cond(
        self,
        video_metadata: Dict[str, Any],
        start_frame: int,
        end_frame: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load video and conditions from video_idx with given start_frame and end_frame (exclusive)
        """
        video = self.load_video(video_metadata, start_frame, end_frame)
        cond = self.load_cond(video_metadata, start_frame, end_frame)
        return video, cond

    def __len__(self) -> int:
        return (
            self.subdataset_size
            if self.use_subdataset
            else (
                min(self.num_clips, len(self.cumulative_sizes))
                if self.use_split_subdataset
                else self.cumulative_sizes[-1]
            )
        )

    def align_conds_w_frames(self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int, video_length: int) -> Tuple[int, int]:
        """
        Align frames and actions based on the dataset action correspondence.
        params:
            video_metadata: video metadata
            start_frame: start frame
            end_frame: end frame
            video_length: the length of the complete video, which is different from the clip length
        return:
            cond: aligned condition
        """

        cond_alignment = self.cfg.cond_alignment
        data_cond_alignment = self.cfg.data_cond_alignment
        clip_length = end_frame - start_frame

        # f1 f2 f3 f4
        # t -> t+1
        # a12 a23 a34
        # t-1 -> t
        # a01 a12 a23

        # Default: action frames align with video frames
        start_action, end_action = start_frame, end_frame
        
        # Adjust action frames when alignments differ
        if cond_alignment != data_cond_alignment:
            if cond_alignment == "t-1->t" and data_cond_alignment == "t->t+1":
                start_action = max(start_frame - 1, 0)
                end_action = max(end_frame - 1, 0)
            elif cond_alignment == "t->t+1" and data_cond_alignment == "t-1->t":
                start_action = min(start_frame + 1, video_length - 1)
                end_action = min(end_frame + 1, video_length - 1)
            else:
                raise ValueError(f"Invalid dataset action correspondence: {data_cond_alignment}, {cond_alignment}")
        

        # Load conditions (actions)
        cond = self.load_cond(video_metadata, start_action, end_action)
        if isinstance(cond, np.ndarray):
            cond = torch.from_numpy(cond)
        if len(cond) < clip_length: # this only happens when dataset action correspondence and action correspondence are different
            if cond_alignment == "t-1->t": # left pad with zeros
                if cond.ndim == 1:
                    cond = torch.nn.functional.pad(cond, (clip_length - len(cond), 0)).contiguous()
                else:
                    cond = torch.nn.functional.pad(cond, (0, 0, clip_length - len(cond), 0)).contiguous()
            elif cond_alignment == "t->t+1": # right pad with zeros
                if cond.ndim == 1:
                    cond = torch.nn.functional.pad(cond, (0, clip_length - len(cond))).contiguous()
                else:
                    cond = torch.nn.functional.pad(cond, (0, 0, 0, clip_length - len(cond))).contiguous()
            else:
                raise ValueError(f"Invalid action correspondence: {cond_alignment}")

        return cond

    def match_clip_by_video_path(self, video_path: Path) -> Tuple[int, int]:
        """
        expected cfg.path_list: [
        [("video_path", "[start_frame, end_frame]")],
        [("video_path", "[start_frame, end_frame]")],
        ...
        ]
        """
        if self.cfg.path_list is None:
            raise ValueError("cfg.path_list is None")
        try:
            paths = [item[0] for item in self.cfg.path_list]
            if video_path in paths:
                return self.cfg.path_list[paths.index(str(video_path))][1][0] # [("video_path", "[start_frame, end_frame]")] -> start_frame
            else:
                raise ValueError(f"Video path {video_path} not found in cfg.path_list")
        except Exception as e:
            rank_zero_print(red(f"Error matching clip by video path: {e}"))
            raise e
        


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_idx, clip_idx = self.get_clip_location(idx)
        video_metadata = self.metadata[video_idx]
        video_length = self.video_length(video_metadata)
        start_frame, end_frame = clip_idx, min(clip_idx + self.n_frames, video_length)

        video, latent, cond = None, None, None
        if self.use_preprocessed_latents:
            latent = self.load_latent(video_metadata, start_frame, end_frame)

        if self.use_preprocessed_latents and self.split == "training":
            # do not load video if we are training with latents
            if self.external_cond_dim > 0:
                cond = self.load_cond(video_metadata, start_frame, end_frame)

        else:
            if self.external_cond_dim > 0:
                video, cond = self.load_video_and_cond(
                    video_metadata, start_frame, end_frame
                )
            else:
                # load video only
                video = self.load_video(video_metadata, start_frame, end_frame)

        lens = [len(x) for x in (video, cond, latent) if x is not None]
        assert len(set(lens)) == 1, f"video, cond, latent must have the same length, video {len(video)}, cond {len(cond)}"
        pad_len = self.n_frames - lens[0]

        nonterminal = torch.ones(self.n_frames, dtype=torch.bool)
        if pad_len > 0:
            if video is not None: # (T, C, H, W)
                video = F.pad(video, (0, 0, 0, 0, 0, 0, 0, pad_len)).contiguous()
            if latent is not None: # (T, C, H, W)
                latent = F.pad(latent, (0, 0, 0, 0, 0, 0, 0, pad_len)).contiguous()
            if cond is not None: # (T, dim)
                cond = F.pad(cond, (0, 0, 0, pad_len)).contiguous()
            nonterminal[-pad_len:] = 0
            # 3D memory doesn't need to be padded

        if self.frame_skip > 1:
            if video is not None:
                video = video[:: self.frame_skip]
            if latent is not None:
                latent = latent[:: self.frame_skip]
            nonterminal = nonterminal[:: self.frame_skip]
        if cond is not None:
            cond = self._process_external_cond(cond)

        output = {
            "videos": self.transform(video) if video is not None else None,
            "latents": latent,
            "conds": cond.float() if cond is not None else None,
            "nonterminal": nonterminal,
        }
            
        return {key: value for key, value in output.items() if value is not None}

    def _process_external_cond(self, external_cond: torch.Tensor) -> torch.Tensor:
        """
        Post-processes external condition.
        Args:
            external_cond: (T, *) tensor, T = self.n_frames
        Returns:
            processed_cond: (T', *) tensor, T' = number of frames after frame skip
        By default:
            shifts external condition by self.frame_skip - 1
            so that each frame has condition corresponding to
            current and previous frames
            then stacks the conditions for skipping frames
        """
        if self.frame_skip == 1:
            return external_cond
        external_cond = F.pad(external_cond, (0, 0, self.frame_skip - 1, 0), value=0.0)
        return rearrange(external_cond, "(t fs) d -> t (fs d)", fs=self.frame_skip)


