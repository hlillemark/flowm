from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
from omegaconf import DictConfig
from torchvision import transforms
from typing_extensions import override
from omegaconf import OmegaConf
import hydra
from hydra import initialize, compose
from utils.distributed_utils import rank_zero_print 

from tqdm import tqdm
from datasets.video.utils import collate_fn_skip_none

from .base_video import (
    BaseAdvancedVideoDataset,
    SPLIT,
)

# Optional backends for reading video
_HAS_DECORD = False
try:
    from decord import VideoReader, cpu as decord_cpu
    _HAS_DECORD = True
except Exception:
    pass

try:
    # torchvision fallback
    from torchvision.io import read_video, read_video_timestamps
    _HAS_TORCHVISION_VIDEO = True
except Exception:
    _HAS_TORCHVISION_VIDEO = False

def _to_grayscale_chw_uint8(frames: np.ndarray) -> np.ndarray:
    """Convert frames from (T, H, W, C) RGB/gray to (T, 1, H, W) uint8 grayscale."""
    if frames.ndim != 4:
        raise ValueError(f"Expected (T, H, W, C), got {frames.shape}")
    T, H, W, C = frames.shape
    if C == 1:
        gray = frames[..., 0]
    elif C == 3:
        # Luma-ish conversion; uint8-safe by working in float then casting back
        gray = (0.2989 * frames[..., 0] + 0.5870 * frames[..., 1] + 0.1140 * frames[..., 2]).astype(np.float32)
    else:
        # Fallback: average channels
        gray = frames.mean(axis=-1).astype(np.float32)
    if gray.dtype != np.float32:
        gray = gray.astype(np.float32)
    gray = np.clip(gray, 0, 255)
    gray = gray.astype(np.uint8)
    # (T, 1, H, W)
    return gray[:, None, :, :]


def _load_video_segment(path: Path, start: int, end: int) -> torch.Tensor:
    """Return frames [start:end] as float in [0,1], shape (T, 1, H, W)."""
    if _HAS_DECORD:
        vr = VideoReader(str(path), ctx=decord_cpu())
        end = min(end, len(vr))
        idx = np.arange(start, end)
        if len(idx) == 0:
            return torch.empty(0, 1, 0, 0)
        batch = vr.get_batch(idx).asnumpy()  # (T, H, W, C) uint8
        gray = _to_grayscale_chw_uint8(batch)  # (T, 1, H, W) uint8
        return torch.from_numpy(gray).float() / 255.0
    elif _HAS_TORCHVISION_VIDEO:
        # torchvision doesn't have exact frame index slicing here.
        # We'll read full, then slice by index (ok for small 28x28 clips).
        vid, _, _ = read_video(str(path), pts_unit="sec")
        # vid: (T, H, W, C) uint8
        vid = vid.numpy()
        gray = _to_grayscale_chw_uint8(vid)
        gray = torch.from_numpy(gray).float() / 255.0
        return gray[start:end]
    else:
        raise ImportError("Neither decord nor torchvision video I/O is available.")


def _probe_num_frames(path: Path) -> int:
    if _HAS_DECORD:
        vr = VideoReader(str(path), ctx=decord_cpu())
        return len(vr)
    elif _HAS_TORCHVISION_VIDEO:
        pts, _ = read_video_timestamps(str(path), pts_unit="sec")
        return len(pts)
    else:
        raise ImportError("Neither decord nor torchvision video I/O is available.")



class _VideoTimestampsDataset:
    """
    Dataset used to parallelize the reading of the timestamps
    of a list of videos, given their paths in the filesystem.

    Used in VideoClips and defined at top level, so it can be
    pickled when forking.
    """

    def __init__(self, video_paths: List[str]) -> None:
        self.video_paths = video_paths

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Tuple[List[int], Optional[float]]:

        return validate_item_and_get_info(self.video_paths[idx])    


def validate_item_and_get_info(video_path: Path):
    """Returns a list of PTS (presentation timestamps) for each frame,
       and the video fps (if available). Updated to load MP4 videos."""

    try:
        from decord import VideoReader, cpu
        
        # Load MP4 video using decord
        vr = VideoReader(str(video_path), ctx=cpu())
        num_frames = len(vr)
        fps = vr.get_avg_fps()
        timestamps = [i for i in range(0, num_frames)]
        return timestamps, fps, video_path
    
    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
        return None


class MNISTWorldVideoDataset(BaseAdvancedVideoDataset):
    """
    MNIST World dataset (static world with moving camera), or MNIST World Dynamic
    (moving digits). Expects data saved under cfg.save_dir with MP4 videos and
    companion .pt files per video containing a dict with:
      - actions: Long/Int Tensor [T, 2]
      - additional metadata keys (e.g., digit_labels, digit_positions, ...)

    Folder structure:
      {save_dir}/
        ├── training/
        │     ├── example_000000.mp4
        │     ├── example_000000.pt
        │     └── ...
        ├── validation/
        └── test/
    """

    _ALL_SPLITS = [
        "static_training", "static_validation", "static_training_200", "static_validation_200", 
        "dynamic_training", "dynamic_validation", "dynamic_training_200", "dynamic_validation_200", 
        "dynamic_training_smallworld", "dynamic_validation_smallworld", "dynamic_validation_smallworld_200", 
        "dynamic_training_smallworld_no_em", "dynamic_validation_smallworld_no_em", "dynamic_validation_smallworld_no_em_200",
    ]

    def __init__(self, cfg: DictConfig, split: str = "training", purpose: str = "training"):
        super().__init__(cfg, split=split, purpose=purpose)

    def build_transform(self):
        # Resize frames to target resolution
        return transforms.Resize(self.resolution, antialias=True)
    
    @override
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
        video_paths = sorted(
            [
                p for p in (self.save_dir / split).glob("**/*.mp4")
                if (not p.name.endswith("_fullworld.mp4") and not p.name.endswith("_fullworld_withcam.mp4"))
            ],
            key=str
        )
        dl: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            _VideoTimestampsDataset(video_paths),
            batch_size=16,
            num_workers=4,
            collate_fn=collate_fn_skip_none,
            pin_memory=True,  
            persistent_workers=True 
        )
        video_pts: List[torch.Tensor] = (
            []
        )  # each entry is a tensor of shape (num_frames, )
        video_fps: List[float] = []
        valid_video_paths: List[Path] = []

        if len(dl) == 0:
            rank_zero_print(f"No videos found in {self.save_dir / split}")
            return
        
        with tqdm(total=len(dl), desc=f"Building metadata for {split}", position = 0) as pbar:
            for batch in dl:
                pbar.update(1)
                batch_pts, batch_fps,   batch_valid_video_path = list(zip(*batch))
                batch_pts = [
                    torch.as_tensor(pts, dtype=torch.long).cpu() for pts in batch_pts
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

    def _pt_path_from_video(self, video_path: Path) -> Path:
        return video_path.with_suffix(".pt")

    def _load_pt(self, video_path: Path) -> Dict[str, Any]:
        pt_path = self._pt_path_from_video(video_path)
        data = torch.load(pt_path, map_location=torch.device("cpu"), weights_only=False)
        return data

    def load_cond(self, video_metadata: Dict[str, Any], start_frame: int, end_frame: Optional[int] = None) -> torch.Tensor:
        """
        Load actions as external condition. Returns Tensor [T, 2].
        """
        if end_frame is None:
            end_frame = self.video_length(video_metadata)
        video_path: Path = video_metadata["video_paths"]
        meta = self._load_pt(video_path)
        actions = meta.get("actions", None)
        if actions is None:
            raise KeyError(f"Missing 'actions' in metadata: {self._pt_path_from_video(video_path)}")
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        actions = torch.as_tensor(actions)
        return actions[start_frame:end_frame]

    def load_all_metadata(self, video_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load all per-video metadata from the companion .pt file except 'actions'.
        Returned values should be tensors or collatable types.
        """
        video_path: Path = video_metadata["video_paths"]
        meta = self._load_pt(video_path)
        return {k: v for k, v in meta.items() if k != "actions"}
    
    @override
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Compute clip location
        video_idx, clip_idx = self.get_clip_location(idx)
        video_metadata = self.metadata[video_idx]
        video_length = self.video_length(video_metadata)
        start_frame, end_frame = clip_idx, min(clip_idx + self.n_frames, video_length)
        video = self.load_video(video_metadata, start_frame, end_frame)
        cond = self.align_conds_w_frames(video_metadata, start_frame, end_frame, video_length)

        # Length handling/padding to n_frames
        lens = [len(x) for x in (video, cond) if x is not None]
        assert len(set(lens)) == 1, f"video and cond must have same length, got {len(video)} vs {len(cond)}"
        pad_len = self.n_frames - lens[0]

        nonterminal = torch.ones(self.n_frames, dtype=torch.bool)
        if pad_len > 0:
            # pad video: (T, C, H, W)
            video = torch.nn.functional.pad(video, (0, 0, 0, 0, 0, 0, 0, pad_len)).contiguous()
            # pad cond: (T, 2)
            cond = torch.nn.functional.pad(cond, (0, 0, 0, pad_len)).contiguous()
            nonterminal[-pad_len:] = 0

        # Frame skip processing
        if self.frame_skip > 1:
            video = video[:: self.frame_skip]
            cond = cond[:: self.frame_skip]
            nonterminal = nonterminal[:: self.frame_skip]
        cond = self._process_external_cond(cond)

        # Transform video
        video = self.transform(video)

        # Load static metadata for this video (not frame-specific)
        extra_meta = self.load_all_metadata(video_metadata)
        # Ensure tensor types for collate
        for k, v in list(extra_meta.items()):
            if isinstance(v, np.ndarray):
                extra_meta[k] = torch.from_numpy(v)

        output: Dict[str, Any] = {
            "videos": video,
            "conds": cond.float(),
            "nonterminal": nonterminal,
            "metadata": {
                "path": str(video_metadata["video_paths"]),
                "clip": [start_frame, end_frame],
            },
        }

            
        # Attach any available metadata (e.g., digit_labels, digit_positions, etc.)
        # Exclude actions which are already in conds
        for k, v in extra_meta.items():
            if k == "actions":
                continue
            output[k] = v

        return output
    
    def download_dataset(self):
        pass

def main():
    with initialize(config_path="../../configurations"):
        cfg: DictConfig = compose(config_name="config", overrides=["dataset=mnist_world"])
        print("Using dataset:", cfg.dataset)
    
    # Pretty-print config
    print("\n=== Loaded Config ===")
    print(OmegaConf.to_yaml(cfg.dataset))

    dataset = MNISTWorldVideoDataset(cfg.dataset, split="static_validation", purpose="training")

    # Debug information about video lengths and clip calculations
    print("\n=== DEBUG INFO ===")
    print(f"cfg.n_frames: {cfg.dataset.n_frames}")
    print(f"dataset.n_frames: {dataset.n_frames}")
    print(f"dataset.frame_skip: {dataset.frame_skip}")
    print(f"dataset.use_split_subdataset: {dataset.use_split_subdataset}")
    print(f"dataset.purpose: {dataset.purpose}")
    print(f"Total dataset length: {len(dataset)}")

    if len(dataset) > 0:
        print("\nLoading first item...")
        batch = dataset[0]
        print(f"Batch keys: {list(batch.keys())}")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"{key}: {value}")
    else:
        print("Dataset is empty - no valid scenes found")


if __name__ == "__main__":
    main()
