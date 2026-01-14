#!/usr/bin/env python3
"""
Upload mnist_world dataset to Hugging Face Hub.

Structure:
- Organization: one dataset repo for mnist_world
- Configs: dynamic_po, static_po, dynamic_fo, dynamic_fo_no_sm
- Splits: train, train_200, validation, validation_200, etc.

Usage:
    python upload_mnist_world_to_hf.py --org your-org --config dynamic_po
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

try:
    from datasets import Dataset, DatasetDict
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Missing dependency 'datasets'. Install with: pip install datasets"
    ) from e

try:
    from huggingface_hub import CommitOperationAdd, HfApi
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Missing dependency 'huggingface_hub'. Install with: pip install huggingface_hub"
    ) from e


# Mapping from config name to directory patterns
CONFIG_TO_PATTERNS = {
    "dynamic_po": {
        "train": "dynamic_training",
        "train_200": "dynamic_training_200",
        "validation": "dynamic_validation",
        "validation_200": "dynamic_validation_200",
    },
    "static_po": {
        "train": "static_training",
        "train_200": "static_training_200",
        "validation": "static_validation",
        "validation_200": "static_validation_200",
    },
    "dynamic_fo": {
        "train": "dynamic_training_ws80",
        "validation": "dynamic_validation_ws80",
        "validation_200": "dynamic_validation_ws80_200",
    },
    "dynamic_fo_no_sm": {
        "train": "dynamic_training_smallworld_no_em",
        "validation": "dynamic_validation_smallworld_no_em",
        "validation_200": "dynamic_validation_smallworld_no_em_200",
    },
}


def _iter_files_recursive(root: Path, *, follow_symlink_dirs: bool) -> Iterable[Path]:
    """Yield all files under a directory (recursively).

    If follow_symlink_dirs=True, symlinked directories will be traversed.
    """
    import os

    for dirpath, _, filenames in os.walk(root, followlinks=follow_symlink_dirs):
        for fn in filenames:
            p = Path(dirpath) / fn
            try:
                if p.is_file():
                    yield p
            except OSError:
                # Broken symlink or permission issue; skip.
                continue


def _safe_tar_add_filter(tarinfo):
    """Filter for tarfile.add() to only include desired file types.

    Keeps directories; keeps only .pt and .mp4 files.
    """
    if tarinfo.isdir():
        return tarinfo
    name = tarinfo.name.lower()
    if name.endswith(".pt") or name.endswith(".mp4"):
        return tarinfo
    return None


def build_episode_shards_for_split(
    *,
    config_name: str,
    data_root: Path,
    split_dir: str,
    shard_root: Path,
    dry_run: bool,
    overwrite: bool,
    max_episodes: Optional[int],
    episode_ids: Optional[Set[str]],
) -> Tuple[int, int]:
    """Create one .tar shard per episode directory.

    Shard layout (local):
        {shard_root}/{config_name}/{split_dir}/{episode_id}.tar

    Tar contents are stored with arcnames like:
        {split_dir}/{episode_id}/...

    Returns:
        (created, skipped_existing)
    """
    import tarfile

    split_path = data_root / split_dir
    if not split_path.exists():
        print(f"Warning: {split_path} does not exist, skipping sharding")
        return 0, 0

    # We intentionally mirror the desired Hub layout:
    #   <config_name>/<split_dir>/<episode_id>.tar
    # so the repo browser looks clean (configs at top-level; then splits).
    out_dir = shard_root / config_name / split_dir
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    episode_dirs = sorted([d for d in split_path.iterdir() if d.is_dir()])
    if episode_ids:
        available_ids = {d.name for d in episode_dirs}
        missing = sorted(list(episode_ids - available_ids))
        if missing:
            preview = ", ".join(missing[:10])
            more = "" if len(missing) <= 10 else f" (+{len(missing) - 10} more)"
            print(
                f"  Note: split '{split_dir}' is missing requested episode_ids: {preview}{more}"
            )
        episode_dirs = [d for d in episode_dirs if d.name in episode_ids]
    if max_episodes is not None:
        episode_dirs = episode_dirs[:max_episodes]

    created = 0
    skipped = 0
    print(f"Sharding split '{split_dir}': {len(episode_dirs)} episode dirs -> {out_dir}")

    for i, episode_dir in enumerate(episode_dirs, start=1):
        tar_path = out_dir / f"{episode_dir.name}.tar"
        if tar_path.exists() and not overwrite:
            skipped += 1
            continue
        if dry_run:
            created += 1
            continue

        tmp_path = tar_path.with_suffix(".tar.tmp")
        if tmp_path.exists():
            tmp_path.unlink()

        # Store paths in-tar so extraction at repo root recreates expected layout.
        arcname = f"{split_dir}/{episode_dir.name}"
        with tarfile.open(tmp_path, mode="w") as tf:
            tf.add(str(episode_dir), arcname=arcname, filter=_safe_tar_add_filter)
        tmp_path.replace(tar_path)
        created += 1
        if i % 50 == 0:
            print(f"  Sharded {i}/{len(episode_dirs)} episodes")

    return created, skipped


def ensure_dataset_repo(api: HfApi, repo_id: str, private: bool) -> None:
    """Create the dataset repo if needed."""
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=private)


def upload_split_folder(
    api: HfApi,
    repo_id: str,
    data_root: Path,
    split_dir: str,
    *,
    dry_run: bool,
    commit_message: str,
    follow_symlink_dirs: bool,
    raw_upload_mode: str = "auto",
    num_workers: int = 16,
    large_report_every: int = 60,
    max_files: Optional[int] = None,
    batch_size: int = 500,
) -> Tuple[Optional[int], int]:
    """Upload a split directory (e.g., dynamic_training) into the dataset repo.

    Files are uploaded to the repo under the same relative path as on disk.
    For example: data_root/dynamic_training/122/000.pt -> dynamic_training/122/000.pt

    Returns:
        (uploaded_commits_or_none, skipped_files)
    """
    split_path = data_root / split_dir
    if not split_path.exists():
        print(f"Warning: {split_path} does not exist, skipping raw upload")
        return 0, 0

    # Prefer a resumable uploader for large folders.
    # NOTE: upload_large_folder cannot set path_in_repo; to keep the desired
    # layout (split_dir/** in the repo), we upload from data_root and filter via
    # allow_patterns.
    mode = raw_upload_mode
    if mode == "auto":
        if (not follow_symlink_dirs) and (max_files is None) and (not dry_run) and hasattr(api, "upload_large_folder"):
            mode = "large"
        elif (not follow_symlink_dirs) and (max_files is None) and (not dry_run):
            mode = "folder"
        else:
            mode = "batched"

    print(
        f"Resolved raw upload mode for split '{split_dir}': {mode} "
        f"(requested={raw_upload_mode}, follow_symlink_dirs={follow_symlink_dirs}, max_files={max_files}, dry_run={dry_run})"
    )

    if mode == "large":
        if not hasattr(api, "upload_large_folder"):
            print(
                "Warning: huggingface_hub in this environment does not expose 'upload_large_folder'. "
                "Falling back to 'upload_folder'. Consider upgrading huggingface_hub."
            )
            mode = "folder"
        else:
            print(
                f"Uploading raw split '{split_dir}' with a resumable uploader (multi-commit)...\n"
                f"  - root: {data_root}\n"
                f"  - include: {split_dir}/**/*.pt, {split_dir}/**/*.mp4\n"
                f"  - workers: {num_workers}"
            )
            if dry_run:
                return None, 0

            api.upload_large_folder(
                repo_id=repo_id,
                repo_type="dataset",
                folder_path=str(data_root),
                allow_patterns=[
                    f"{split_dir}/**/*.pt",
                    f"{split_dir}/**/*.mp4",
                ],
                num_workers=num_workers,
                print_report=True,
                print_report_every=large_report_every,
            )
            # upload_large_folder creates multiple commits; do not pretend we know the count.
            return None, 0

    # Fast path: single commit upload using Hub helper.
    # This is much faster than per-file commits, but less resilient for huge folders.
    if mode == "folder":
        if dry_run:
            return 1, 0
        print(f"Uploading raw folder '{split_dir}' with a single commit...")
        api.upload_folder(
            folder_path=str(split_path),
            path_in_repo=split_dir,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message,
        )
        return 1, 0

    # Robust (but not resumable) path: enumerate files (optionally following
    # symlinked dirs) and batch them into commits.
    if max_files is None:
        all_files = list(_iter_files_recursive(split_path, follow_symlink_dirs=follow_symlink_dirs))
    else:
        # IMPORTANT: stop early so a small test doesn't require scanning the whole tree.
        all_files = []
        for p in _iter_files_recursive(split_path, follow_symlink_dirs=follow_symlink_dirs):
            all_files.append(p)
            if len(all_files) >= max_files:
                break

    print(f"Uploading raw files for split '{split_dir}': {len(all_files)} files")

    if dry_run:
        return len(all_files), 0

    uploaded_commits = 0
    skipped_files = 0
    batch: list[CommitOperationAdd] = []
    for i, local_path in enumerate(all_files, start=1):
        path_in_repo = str(local_path.relative_to(data_root).as_posix())
        try:
            batch.append(CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=str(local_path)))
        except Exception as e:
            skipped_files += 1
            print(f"  Warning: cannot stage {local_path} -> {path_in_repo}: {e}")
            continue

        if len(batch) >= batch_size:
            api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                operations=batch,
                commit_message=f"{commit_message} (batch ending at {i}/{len(all_files)})",
            )
            uploaded_commits += 1
            batch = []

    if batch:
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=batch,
            commit_message=f"{commit_message} (final batch)",
        )
        uploaded_commits += 1

    return uploaded_commits, skipped_files


def load_episode_data(episode_dir: Path) -> Dict:
    """Load all data from an episode directory.
    
    Args:
        episode_dir: Path to episode directory (e.g., dynamic_training/122)
    
    Returns:
        Dictionary with episode metadata and file paths
    """
    episode_id = episode_dir.name
    
    # Get all .pt and .mp4 files
    pt_files = sorted(episode_dir.glob("*.pt"))
    mp4_files = sorted(episode_dir.glob("*.mp4"))
    
    episode_path = str(episode_dir.relative_to(episode_dir.parent.parent))

    return {
        "episode_id": episode_id,
        "num_frames": len(pt_files),
        "episode_path": episode_path,
        # Store relative paths for flexibility
        "data_files": [str(f.relative_to(episode_dir.parent.parent)) for f in pt_files],
        "video_files": [str(f.relative_to(episode_dir.parent.parent)) for f in mp4_files],
        # Optional convenience for sharded hosting: users can download/extract shards
        # to materialize data_files/video_files paths.
        # Shards are stored under <config_name>/<split>/<episode_id>.tar (see sharding uploader).
        # Note: load_episode_data() doesn't know config_name; create_dataset_for_split() fills this.
        "shard_file": None,
    }


def create_dataset_for_split(data_root: Path, split_dir: str, *, config_name: str) -> Dataset:
    """Create a dataset for a single split.
    
    Args:
        data_root: Root directory (e.g., /data/.../mnist_world)
        split_dir: Directory name (e.g., "dynamic_training")
    
    Returns:
        Dataset object
    """
    split_path = data_root / split_dir
    
    if not split_path.exists():
        print(f"Warning: {split_path} does not exist, skipping")
        return None
    
    # Get all episode directories
    episode_dirs = sorted([d for d in split_path.iterdir() if d.is_dir()])
    
    print(f"Loading {len(episode_dirs)} episodes from {split_dir}...")
    
    # Load all episodes
    episodes = []
    for i, episode_dir in enumerate(episode_dirs):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(episode_dirs)} episodes")
        
        episode_data = load_episode_data(episode_dir)
        # Fill shard_file using the Hub layout: <config>/<split>/<episode>.tar
        # episode_path is like "dynamic_training/122".
        episode_data["shard_file"] = f"{config_name}/{episode_data['episode_path']}.tar"
        episodes.append(episode_data)
    
    print(f"Loaded {len(episodes)} episodes")
    
    # Create dataset
    return Dataset.from_list(episodes)


def create_dataset_dict(data_root: Path, config_name: str) -> DatasetDict:
    """Create a DatasetDict for a specific config.
    
    Args:
        data_root: Root directory (e.g., /data/.../mnist_world)
        config_name: Config name (e.g., "dynamic_po")
    
    Returns:
        DatasetDict with all splits
    """
    if config_name not in CONFIG_TO_PATTERNS:
        raise ValueError(f"Unknown config: {config_name}")
    
    patterns = CONFIG_TO_PATTERNS[config_name]
    
    dataset_dict = {}
    for split_name, dir_name in patterns.items():
        print(f"\nProcessing split: {split_name} (dir: {dir_name})")
        ds = create_dataset_for_split(data_root, dir_name, config_name=config_name)
        
        if ds is not None:
            dataset_dict[split_name] = ds
    
    return DatasetDict(dataset_dict)


def upload_to_hub(
    dataset_dict: DatasetDict,
    repo_id: str,
    config_name: str,
    private: bool = False
):
    """Upload dataset to Hugging Face Hub.
    
    Args:
        dataset_dict: DatasetDict to upload
        repo_id: Repository ID (e.g., "your-org/mnist-world")
        config_name: Config name (e.g., "dynamic_po")
        private: Whether to make the dataset private
    """
    print(f"\n{'='*60}")
    print(f"Uploading to: {repo_id}")
    print(f"Config: {config_name}")
    print(f"Splits: {list(dataset_dict.keys())}")
    print(f"Private: {private}")
    print(f"{'='*60}\n")
    
    # Upload
    dataset_dict.push_to_hub(
        repo_id=repo_id,
        config_name=config_name,
        private=private,
        commit_message=f"Add {config_name} configuration"
    )
    
    print(f"\n✓ Successfully uploaded {config_name} to {repo_id}")


def create_readme(repo_id: str, org_name: str) -> str:
    """Create a README for the dataset."""
    
    return f"""---
license: mit
task_categories:
- video-classification
- reinforcement-learning
tags:
- world-models
- video
- sequential
pretty_name: MNIST World Dataset
size_categories:
- 10K<n<100K
---

# MNIST World Dataset

A collection of episodic video datasets for world model research, featuring moving MNIST digits in various configurations.

## Dataset Structure

This dataset contains multiple configurations with different observability and environment settings:

### Configurations

| Config | Description | Splits |
|--------|-------------|--------|
| `dynamic_po` | **Dynamic Partially Observable** (default, highest priority) | train, train_200, validation, validation_200 |
| `static_po` | Static Partially Observable | train, train_200, validation, validation_200 |
| `dynamic_fo` | Dynamic Fully Observable (world size 80) | train, validation, validation_200 |
| `dynamic_fo_no_sm` | Dynamic Fully Observable without Small World | train, validation, validation_200 |

### Data Format

Each example contains:
- `episode_id`: Unique identifier for the episode
- `num_frames`: Number of frames in the episode
- `episode_path`: Relative path to episode directory
- `data_files`: List of paths to .pt files (state data)
- `video_files`: List of paths to .mp4 files (videos)

## Usage

```python
from datasets import load_dataset

# Load dynamic partially observable (default, highest priority)
ds = load_dataset("{repo_id}", "dynamic_po", split="train")

# Load validation set
val_ds = load_dataset("{repo_id}", "dynamic_po", split="validation")

# Load other configurations
static_ds = load_dataset("{repo_id}", "static_po", split="train")
fo_ds = load_dataset("{repo_id}", "dynamic_fo", split="train")

# Access episode data
episode = ds[0]
print(f"Episode {{episode['episode_id']}} has {{episode['num_frames']}} frames")
```

## Dataset Statistics

### Priority Configuration: dynamic_po

This is the primary configuration used in our research.

- **Training episodes**: ~1000+ episodes
- **Validation episodes**: ~200+ episodes
- **Episode length**: Variable
- **Frame rate**: 30 fps
- **Resolution**: 64x64

## Citation

If you use this dataset, please cite:

```bibtex
@misc{{mnist_world,
  title={{MNIST World Dataset}},
  author={{Your Name}},
  year={{2026}},
  publisher={{Hugging Face}},
  howpublished={{\\url{{https://huggingface.co/datasets/{repo_id}}}}}
}}
```

## License

MIT License
"""


def main():
    parser = argparse.ArgumentParser(description="Upload mnist_world dataset to HF Hub")
    parser.add_argument(
        "--org",
        type=str,
        default="flowm123",
        help="Organization name (default: flowm123)"
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=list(CONFIG_TO_PATTERNS.keys()),
        default="dynamic_po",
        help="Configuration to upload (default: dynamic_po, highest priority)"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/mnist_world",
        help="Root directory of mnist_world data"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private"
    )
    parser.add_argument(
        "--upload-raw",
        action="store_true",
        help="Also upload raw .pt/.mp4 files into the dataset repo (recommended)"
    )
    parser.add_argument(
        "--shard-raw",
        action="store_true",
        help=(
            "Upload raw data as per-episode .tar shards under <config>/<split>/<episode>.tar. "
            "Greatly reduces file count and avoids Hub rate limits. Users must extract after download."
        ),
    )
    parser.add_argument(
        "--shard-root",
        type=str,
        default=None,
        help=(
            "Where to write shard files locally (default: <data-root>/_hf_shards). "
            "Shards will be created under <shard-root>/<config>/<split>/..."
        ),
    )
    parser.add_argument(
        "--shard-overwrite",
        action="store_true",
        help="Rebuild shard .tar files even if they already exist",
    )
    parser.add_argument(
        "--max-episodes-per-split",
        type=int,
        default=None,
        help="For debugging: only shard/upload the first N episode folders per split",
    )
    parser.add_argument(
        "--episode-ids",
        type=str,
        default=None,
        help="Comma-separated episode IDs to shard/upload (e.g. '115,136'). Overrides ordering.",
    )
    parser.add_argument(
        "--follow-symlink-dirs",
        action="store_true",
        help="Follow symlinked directories when uploading raw files (slower)"
    )
    parser.add_argument(
        "--raw-upload-mode",
        type=str,
        choices=["auto", "large", "folder", "batched"],
        default="auto",
        help=(
            "How to upload raw files. 'large' uses a resumable multi-commit uploader "
            "(recommended for big folders). 'folder' uses a single-commit folder upload. "
            "'batched' enumerates files and creates commits in batches. 'auto' picks a mode."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of workers for resumable large-folder uploads (only affects --raw-upload-mode=large/auto)",
    )
    parser.add_argument(
        "--large-report-every",
        type=int,
        default=60,
        help="Seconds between progress reports for --raw-upload-mode=large (set smaller to see frequent updates)",
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Upload all configs (and their splits) instead of a single --config"
    )
    parser.add_argument(
        "--max-files-per-split",
        type=int,
        default=None,
        help="For debugging: only upload the first N files per split"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't upload anything; just print what would happen"
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip creating/pushing the index dataset (raw upload only). Useful for quick upload tests."
    )
    parser.add_argument(
        "--create-readme",
        action="store_true",
        help="Create README.md file only"
    )
    
    args = parser.parse_args()
    
    repo_id = f"{args.org}/mnist-world"
    data_root = Path(args.data_root)
    
    if not data_root.exists():
        print(f"Error: Data root does not exist: {data_root}")
        return
    
    # Create README if requested
    if args.create_readme:
        readme_content = create_readme(repo_id, args.org)
        readme_path = data_root / "README.md"
        readme_path.write_text(readme_content)
        print(f"✓ Created README at {readme_path}")
        return
    
    api = HfApi()

    shard_root = Path(args.shard_root) if args.shard_root else (data_root / "_hf_shards")
    episode_ids: Optional[Set[str]] = None
    if args.episode_ids:
        episode_ids = {s.strip() for s in args.episode_ids.split(",") if s.strip()}

    if args.dry_run:
        print("Dry run mode - not uploading to Hub")
        print(f"Would use repo: {repo_id}")
    else:
        ensure_dataset_repo(api, repo_id=repo_id, private=args.private)

    configs_to_upload = list(CONFIG_TO_PATTERNS.keys()) if args.all_configs else [args.config]
    uploaded_raw_dirs: Set[str] = set()

    for config_name in configs_to_upload:
        print(f"\n{'='*60}")
        print(f"Config: {config_name}")
        print(f"Repo:   {repo_id}")
        print(f"Raw:    {args.upload_raw}")
        print(f"{'='*60}\n")

        patterns = CONFIG_TO_PATTERNS[config_name]

        # Upload raw files (optional) so others can actually download them.
        if args.upload_raw:
            uploaded_total: Optional[int] = 0
            skipped_total = 0
            for split_name, dir_name in patterns.items():
                if dir_name in uploaded_raw_dirs:
                    print(f"Skipping raw upload for '{dir_name}' (already uploaded)")
                    continue
                if args.shard_raw:
                    print(f"\nPreparing shards for '{dir_name}' ({config_name}/{split_name})")
                    created, skipped = build_episode_shards_for_split(
                        config_name=config_name,
                        data_root=data_root,
                        split_dir=dir_name,
                        shard_root=shard_root,
                        dry_run=args.dry_run,
                        overwrite=args.shard_overwrite,
                        max_episodes=args.max_episodes_per_split,
                        episode_ids=episode_ids,
                    )
                    print(f"Shards ready for '{dir_name}': created={created}, skipped_existing={skipped}")

                    if args.dry_run:
                        up = None
                        sk = 0
                    else:
                        # If nothing was selected/built, skip the upload entirely.
                        if (created + skipped) == 0:
                            print(
                                f"No shards to upload for split '{dir_name}' (created=0 and none existing). Skipping."
                            )
                        else:
                            print(
                                f"Uploading shards for split '{dir_name}' from {shard_root}/{config_name}/{dir_name} ..."
                            )
                            api.upload_large_folder(
                                repo_id=repo_id,
                                repo_type="dataset",
                                folder_path=str(shard_root),
                                # Some glob implementations don't match files directly under
                                # <config>/<split>/ with the pattern "**/*.tar" (i.e., it may
                                # require at least one subdirectory). Include both patterns.
                                allow_patterns=[
                                    f"{config_name}/{dir_name}/*.tar",
                                    f"{config_name}/{dir_name}/**/*.tar",
                                ],
                                num_workers=max(1, int(args.num_workers)),
                                print_report=True,
                                print_report_every=args.large_report_every,
                            )
                        up = None
                        sk = 0
                else:
                    commit_msg = f"Upload raw files: {dir_name} ({config_name}/{split_name})"
                    up, sk = upload_split_folder(
                        api,
                        repo_id=repo_id,
                        data_root=data_root,
                        split_dir=dir_name,
                        dry_run=args.dry_run,
                        commit_message=commit_msg,
                        follow_symlink_dirs=args.follow_symlink_dirs,
                        raw_upload_mode=args.raw_upload_mode,
                        num_workers=args.num_workers,
                        large_report_every=args.large_report_every,
                        max_files=args.max_files_per_split,
                    )

                if up is None:
                    uploaded_total = None
                elif uploaded_total is not None:
                    uploaded_total += up
                skipped_total += sk
                uploaded_raw_dirs.add(dir_name)
            if uploaded_total is None:
                print(
                    f"\nRaw upload summary for {config_name}: commits=(multiple), file_failures={skipped_total}"
                )
            else:
                print(
                    f"\nRaw upload summary for {config_name}: commits={uploaded_total}, file_failures={skipped_total}"
                )

        if args.skip_index:
            print("\nSkipping index dataset creation/push (--skip-index)")
            continue

        # Create and push the index dataset.
        print(f"\nCreating index dataset for config: {config_name}")
        dataset_dict = create_dataset_dict(data_root, config_name)

        print(f"\n{'='*60}")
        print("Index Dataset Statistics:")
        print(f"{'='*60}")
        for split_name, ds in dataset_dict.items():
            print(f"  {split_name}: {len(ds)} episodes")
        print(f"{'='*60}\n")

        if args.dry_run:
            print(f"Dry run: would push index dataset for config '{config_name}' to {repo_id}")
        else:
            upload_to_hub(
                dataset_dict=dataset_dict,
                repo_id=repo_id,
                config_name=config_name,
                private=args.private,
            )
    
    print(f"\n{'='*60}")
    print("Next steps:")
    print(f"{'='*60}")
    print("1. Recommended full upload:")
    print(f"   python {__file__} --org {args.org} --upload-raw --all-configs")
    print("1. Recommended full upload (sharded raw + index):")
    print(
        f"   python {__file__} --org {args.org} --upload-raw --shard-raw --all-configs --num-workers 2 --large-report-every 30"
    )

    print("\n2. Upload a single config (sharded raw + index):")
    print(
        f"   python {__file__} --org {args.org} --upload-raw --shard-raw --config dynamic_po --num-workers 2 --large-report-every 30"
    )

    print("\n3. Quick test: shard/upload a single episode only (raw only):")
    print(
        f"   python {__file__} --org {args.org} --upload-raw --shard-raw --config dynamic_po --episode-ids 115 --skip-index --num-workers 2 --large-report-every 10"
    )
    print(f"   https://huggingface.co/datasets/{repo_id}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
