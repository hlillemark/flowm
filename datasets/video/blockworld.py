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

from tqdm import tqdm
from datasets.video.utils import collate_fn_skip_none
import glob

from .base_video import (
    BaseAdvancedVideoDataset,
    SPLIT,
)

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


class BlockWorldVideoDataset(BaseAdvancedVideoDataset):
    """
    BlockWorld dataset with support for latent loading.

    Folder structure:
      {save_dir}/
        ├── training/
        │     ├── {blocknum}/
        │     │     ├── 0000_rgb.mp4
        │     │     ├── 0000_depth.mp4
        │     │     └── 0000_actions.pt
        ├── validation/
        └── test/

    actions list, from putnext.py
    | Num | Action                      |
    |-----|-----------------------------|
    | 0   | turn left                   |
    | 1   | turn right                  |
    | 2   | move forward                |
    | 3   | -not assigned-              |
    | 4   | noop                        |
    
    Latent loading support:
      - Set cfg.latent.enable = True to enable latent loading
      - Set cfg.latent.type to start with "pre_" (e.g., "pre_vae") for preprocessed latents
      - Latents are expected to be saved in {save_dir}_latent_{resolution}/ directory
      - Latent files should have the same structure as the original video files
      - When using latents for training, video loading is skipped to improve performance
    """

    _ALL_SPLITS = [
        "sunday_v2_training", "sunday_v2_validation", 
        "sunday_v2_static_training", "sunday_v2_static_validation", 
        "tex_training", "tex_validation",
        ]

    def __init__(self, cfg: DictConfig, split: str = "training", purpose: str = "training"):
        super().__init__(cfg, split=split, purpose=purpose)
        self.use_depth = getattr(cfg, "use_depth", False)

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
        # Only index RGB videos (exclude depth videos like *_depth.mp4)
        root = self.save_dir / split
        video_paths = [
            p for p in sorted(
                (Path(p) for p in glob.glob(str(root / "**/*.mp4"), recursive=True)),
                key=str,
            )
            if p.name.endswith("_rgb.mp4")
        ]
        dl: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            _VideoTimestampsDataset(video_paths),
            batch_size=max(1, min(16, len(video_paths))),
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
        # actions/metadata are saved as {base_stem}_actions.pt alongside the video
        # if video is named like 0000_rgb.mp4, strip suffix "_rgb" to get 0000_actions.pt
        stem = video_path.stem
        if stem.endswith("_rgb"):
            stem = stem[:-4]
        return video_path.with_name(f"{stem}_actions.pt")

    def _depth_path_from_video(self, video_path: Path) -> Path:
        # match 0000_rgb.mp4 -> 0000_depth.mp4
        stem = video_path.stem
        if stem.endswith("_rgb"):
            stem = stem[:-4]
        return video_path.with_name(f"{stem}_depth.mp4")

    def _load_pt(self, video_path: Path) -> Dict[str, Any]:
        pt_path = self._pt_path_from_video(video_path)
        data = torch.load(pt_path, map_location=torch.device("cpu"), weights_only=False)
        return data

    def load_cond(self, video_metadata: Dict[str, Any], start_frame: int, end_frame: Optional[int] = None) -> torch.Tensor:
        """
        Load condition as concatenation of agent_pos and delta_dir. Returns Tensor [T, D].
        """
        
        if end_frame is None:
            end_frame = self.video_length(video_metadata)
        video_path: Path = video_metadata["video_paths"]
        meta = self._load_pt(video_path)
        
        actions = meta['actions'][start_frame:end_frame]
        if self.cond_loading_style == "one_hot":
            actions = torch.eye(self.external_cond_dim)[actions]
        elif self.cond_loading_style == "action_int":
            actions = actions
        else:
            raise RuntimeError(f"Action cond loading style {self.cond_loading_style} not recognized in blockworld dataset.")
        return actions

    def load_all_metadata(self, video_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load all per-video metadata from the companion .pt file, excluding fields used for cond.
        Returned values should be tensors or collatable types.
        """
        video_path: Path = video_metadata["video_paths"]
        meta = self._load_pt(video_path)
        return {k: v for k, v in meta.items() if k not in ("actions", "agent_pos", "delta_dir", "delta_xz")}

    def load_depth(self, video_metadata: Dict[str, Any], start_frame: int, end_frame: Optional[int] = None) -> Optional[torch.Tensor]:
        """
        Load depth frames from the corresponding *_depth.mp4.
        Returns a tensor of shape (T, 3, H, W) after repeating single-channel depth to 3 channels.
        """
        if not self.use_depth:
            return None
        if end_frame is None:
            end_frame = self.video_length(video_metadata)
        video_path: Path = video_metadata["video_paths"]
        depth_path = self._depth_path_from_video(video_path)
        if not depth_path.exists():
            # Depth not available; return None to keep RGB-only flow working
            return None
        # Use same indices as RGB
        indices = video_metadata["video_pts"][start_frame:end_frame].tolist()
        from decord import VideoReader, cpu
        vr = VideoReader(str(depth_path), ctx=cpu())
        depth_np = vr.get_batch(indices).asnumpy()  # (T, H, W, C)
        depth_t = torch.from_numpy(depth_np).float() / 255.0
        # Convert to single-channel then repeat to 3 channels to match convention
        if depth_t.dim() == 4 and depth_t.shape[-1] > 1:
            depth_t = depth_t.mean(dim=-1, keepdim=True)
        depth_t = depth_t.permute(0, 3, 1, 2).contiguous()  # (T, 1, H, W)
        depth_t = depth_t.repeat(1, 3, 1, 2)  # (T, 3, H, W)
        return depth_t

    @override
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Compute clip location
        video_idx, clip_idx = self.get_clip_location(idx)
        video_metadata = self.metadata[video_idx]
        video_length = self.video_length(video_metadata)
        start_frame, end_frame = clip_idx, min(clip_idx + self.n_frames, video_length)

        # Initialize variables
        video, depth, cond, latent, depth_latent = None, None, None, None, None

        # Load latents if using preprocessed latents
        if self.use_preprocessed_latents:
            latent = self.load_latent(video_metadata, start_frame, end_frame)
            if self.use_depth:
                depth_latent = self.load_depth_latent(video_metadata, start_frame, end_frame)

        # Load video and depth if not using latents or if purpose is not training
        if self.use_preprocessed_latents and self.purpose == "training":
            # Skip loading the videos and depth videos if we are training and using preprocessed latents
            pass 
        else:
            # Load video and conditions
            video = self.load_video(video_metadata, start_frame, end_frame)
            depth = self.load_depth(video_metadata, start_frame, end_frame)
        
        cond = self.align_conds_w_frames(video_metadata, start_frame, end_frame, video_length)
        
        # Determine length and padding based on what we have
        if latent is not None:
            lens = [len(x) for x in (latent, depth_latent) if x is not None]
            if depth_latent is not None:
                assert len(set(lens)) == 1, f"latent, depth_latent must have the same length, latent {len(latent)}, depth_latent {len(depth_latent)}"
            pad_len = self.n_frames - lens[0]
        else:
            lens = [len(x) for x in (video, cond, depth) if x is not None]
            assert len(set(lens)) == 1, f"video/cond/depth must have same length, got {[len(x) for x in (video, cond, depth) if x is not None]}"
            pad_len = self.n_frames - lens[0]

        nonterminal = torch.ones(self.n_frames, dtype=torch.bool)
        if pad_len > 0:
            # pad video: (T, C, H, W)
            if video is not None:
                video = torch.nn.functional.pad(video, (0, 0, 0, 0, 0, 0, 0, pad_len)).contiguous()
            # pad cond: (T, D)
            if cond is not None:
                cond = torch.nn.functional.pad(cond, (0, 0, 0, pad_len)).contiguous()
            if depth is not None:
                depth = torch.nn.functional.pad(depth, (0, 0, 0, 0, 0, 0, 0, pad_len)).contiguous()
            # pad latents: (T, C, H, W)
            if latent is not None:
                latent = torch.nn.functional.pad(latent, (0, 0, 0, 0, 0, 0, 0, pad_len)).contiguous()
            if depth_latent is not None:
                depth_latent = torch.nn.functional.pad(depth_latent, (0, 0, 0, 0, 0, 0, 0, pad_len)).contiguous()
            nonterminal[-pad_len:] = 0

        # Frame skip processing
        if self.frame_skip > 1:
            if video is not None:
                video = video[:: self.frame_skip]
            if cond is not None:
                cond = cond[:: self.frame_skip]
            if depth is not None:
                depth = depth[:: self.frame_skip]
            if latent is not None:
                latent = latent[:: self.frame_skip]
            if depth_latent is not None:
                depth_latent = depth_latent[:: self.frame_skip]
            nonterminal = nonterminal[:: self.frame_skip]

        # Process external condition
        if cond is not None:
            cond = self._process_external_cond(cond)

        # Transform video and depth
        if video is not None:
            video = self.transform(video)
        if depth is not None:
            depth = self.transform(depth)

        # Load static metadata for this video (not frame-specific)
        extra_meta = self.load_all_metadata(video_metadata)
        # Ensure tensor types for collate
        for k, v in list(extra_meta.items()):
            if isinstance(v, np.ndarray):
                extra_meta[k] = torch.from_numpy(v)

        output: Dict[str, Any] = {
            "nonterminal": nonterminal,
        }
        
        # Add video data
        if video is not None:
            output["videos"] = video
        if cond is not None:
            output["conds"] = cond.float()
        if depth is not None:
            output["depths"] = depth
            
        # Add latent data
        if latent is not None:
            output["latents"] = latent
        if depth_latent is not None:
            output["depth_latents"] = depth_latent
            
        # Attach any available metadata (e.g., digit_labels, digit_positions, etc.)
        # Exclude actions which are already in conds
        for k, v in extra_meta.items():
            if k == "actions":
                continue
            output[k] = v

        output["metadata"] = {
            "path": str(video_metadata["video_paths"]),
            "clip": [start_frame, end_frame],
        }

        # Handle preprocessing experiments
        if self.is_latent_preprocessing_expt:
            return output, self.video_metadata_to_latent_path(video_metadata).as_posix(), self.video_metadata_to_latent_depth_path(video_metadata).as_posix()
        else:
            return output
    
    def download_dataset(self):
        pass

def main():
    with initialize(config_path="../../configurations"):
        cfg: DictConfig = compose(config_name="config", overrides=["dataset=blockworld"])
        print("Using dataset:", cfg.dataset)
    
    # Pretty-print config
    print("\n=== Loaded Config ===")
    print(OmegaConf.to_yaml(cfg.dataset))

    dataset = BlockWorldVideoDataset(cfg.dataset, split="validation", purpose="training")

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
