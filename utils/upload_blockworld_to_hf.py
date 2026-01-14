#!/usr/bin/env python3
"""Upload Blockworld dataset to Hugging Face Hub.

We mirror the MNIST World uploader structure so usage and Hub layout stay consistent.

Hub layout (raw shards mode):
  <config>/<split_dir>/<episode_id>.tar

Index dataset layout (Datasets viewer):
  <config_name>/{train,validation}-*.parquet

Configs (per user request):
  - dynamic (dynamic blockworld) [highest priority]
  - tex (textured blockworld)
  - static (static blockworld)

Splits:
  - training
  - validation

Example:
  # Upload only index for all configs (fast)
  python utils/upload_blockworld_to_hf.py --org <org> --all-configs

  # Upload raw shards + index for one config
  python utils/upload_blockworld_to_hf.py --org <org> --upload-raw --shard-raw --config dynamic
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

try:
    from datasets import Dataset, DatasetDict
except Exception as e:  # pragma: no cover
    raise ImportError("Missing dependency 'datasets'. Install with: pip install datasets") from e

try:
    from huggingface_hub import CommitOperationAdd, HfApi
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Missing dependency 'huggingface_hub'. Install with: pip install huggingface_hub"
    ) from e


# Mapping from config name to split directories under --data-root
CONFIG_TO_PATTERNS: Dict[str, Dict[str, str]] = {
    # Highest priority (dynamic)
    "dynamic": {
        "train": "sunday_v2_training",
        "validation": "sunday_v2_validation",
    },
    # Medium priority (textured)
    "tex": {
        "train": "tex_training",
        "validation": "tex_validation",
    },
    # Medium priority (static)
    "static": {
        "train": "sunday_v2_static_training",
        "validation": "sunday_v2_static_validation",
    },
}


def _iter_files_recursive(root: Path, *, follow_symlink_dirs: bool) -> Iterable[Path]:
    """Yield all files under a directory (recursively)."""
    import os

    for dirpath, _, filenames in os.walk(root, followlinks=follow_symlink_dirs):
        for fn in filenames:
            p = Path(dirpath) / fn
            try:
                if p.is_file():
                    yield p
            except OSError:
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

    Local shard layout:
        {shard_root}/{config_name}/{split_dir}/{episode_id}.tar

    Tar contents use arcnames:
        {split_dir}/{episode_id}/...

    Returns:
        (created, skipped_existing)
    """

    import tarfile

    split_path = data_root / split_dir
    if not split_path.exists():
        print(f"Warning: {split_path} does not exist, skipping sharding")
        return 0, 0

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
            print(f"  Note: split '{split_dir}' is missing requested episode_ids: {preview}{more}")
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

        arcname = f"{split_dir}/{episode_dir.name}"
        with tarfile.open(tmp_path, mode="w") as tf:
            tf.add(str(episode_dir), arcname=arcname, filter=_safe_tar_add_filter)
        tmp_path.replace(tar_path)
        created += 1

        if i % 50 == 0:
            print(f"  Sharded {i}/{len(episode_dirs)} episodes")

    return created, skipped


def ensure_dataset_repo(api: HfApi, repo_id: str, private: bool) -> None:
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
    """Upload a split directory into the dataset repo."""

    split_path = data_root / split_dir
    if not split_path.exists():
        print(f"Warning: {split_path} does not exist, skipping raw upload")
        return 0, 0

    mode = raw_upload_mode
    if mode == "auto":
        if (
            (not follow_symlink_dirs)
            and (max_files is None)
            and (not dry_run)
            and hasattr(api, "upload_large_folder")
        ):
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
                "Falling back to 'upload_folder'."
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
            return None, 0

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

    # Batched per-file commits
    if max_files is None:
        all_files = list(_iter_files_recursive(split_path, follow_symlink_dirs=follow_symlink_dirs))
    else:
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
    """Build a lightweight index record for one episode directory."""

    episode_id = episode_dir.name

    pt_files = sorted(episode_dir.glob("*.pt"))
    mp4_files = sorted(episode_dir.glob("*.mp4"))

    rgb_mp4 = sorted(episode_dir.glob("*_rgb.mp4"))
    map_mp4 = sorted(episode_dir.glob("*_map_2d.mp4"))

    # For Blockworld, many episodes store per-step media as many small mp4 files.
    # Use *_rgb.mp4 count as the closest proxy for "num_frames" when available.
    if len(rgb_mp4) > 0:
        num_frames = len(rgb_mp4)
    elif len(mp4_files) > 0:
        num_frames = len(mp4_files)
    else:
        num_frames = len(pt_files)

    episode_path = str(episode_dir.relative_to(episode_dir.parent.parent))

    return {
        "episode_id": episode_id,
        "num_frames": num_frames,
        "num_pt_files": len(pt_files),
        "num_mp4_files": len(mp4_files),
        "num_rgb_mp4": len(rgb_mp4),
        "num_map_2d_mp4": len(map_mp4),
        "episode_path": episode_path,
        "data_files": [str(f.relative_to(episode_dir.parent.parent)) for f in pt_files],
        "video_files": [str(f.relative_to(episode_dir.parent.parent)) for f in mp4_files],
        # Filled by create_dataset_for_split() once config_name is known.
        "shard_file": None,
    }


def create_dataset_for_split(data_root: Path, split_dir: str, *, config_name: str) -> Optional[Dataset]:
    split_path = data_root / split_dir
    if not split_path.exists():
        print(f"Warning: {split_path} does not exist, skipping")
        return None

    episode_dirs = sorted([d for d in split_path.iterdir() if d.is_dir()])
    print(f"Loading {len(episode_dirs)} episodes from {split_dir}...")

    episodes = []
    for i, episode_dir in enumerate(episode_dirs):
        if i % 200 == 0:
            print(f"  Processed {i}/{len(episode_dirs)} episodes")
        episode_data = load_episode_data(episode_dir)
        episode_data["shard_file"] = f"{config_name}/{episode_data['episode_path']}.tar"
        episodes.append(episode_data)

    print(f"Loaded {len(episodes)} episodes")
    return Dataset.from_list(episodes)


def create_dataset_dict(data_root: Path, config_name: str) -> DatasetDict:
    if config_name not in CONFIG_TO_PATTERNS:
        raise ValueError(f"Unknown config: {config_name}")

    patterns = CONFIG_TO_PATTERNS[config_name]
    dataset_dict: Dict[str, Dataset] = {}
    for split_name, dir_name in patterns.items():
        print(f"\nProcessing split: {split_name} (dir: {dir_name})")
        ds = create_dataset_for_split(data_root, dir_name, config_name=config_name)
        if ds is not None:
            dataset_dict[split_name] = ds

    return DatasetDict(dataset_dict)


def upload_to_hub(dataset_dict: DatasetDict, repo_id: str, config_name: str, private: bool) -> None:
    print(f"\nPushing index dataset for config '{config_name}' to {repo_id} ...")
    dataset_dict.push_to_hub(repo_id, config_name=config_name, private=private)
    print("✓ Index push complete")


def create_readme(repo_id: str, org: str) -> str:
    return f"""# Blockworld Dataset\n\nThis repository hosts the **Blockworld** dataset in multiple configs.\n\n## Configs\n- `dynamic` (dynamic)\n- `tex` (textured)\n- `static` (static)\n\n## Splits\nEach config contains `train` and `validation` splits.\n\n## Raw data hosting\nRaw episode data is uploaded as per-episode tar shards under:\n\n- `<config>/<split_dir>/<episode_id>.tar`\n\nAfter downloading the dataset repository, extract tar shards at the repo root to recreate the original folder structure.\n\n## How to extract shards\nUse the companion script:\n\n- `python utils/unpack_blockworld_shards.py --repo-root /path/to/downloaded/repo`\n\nRepo: https://huggingface.co/datasets/{repo_id}\n"""


def main() -> None:
    p = argparse.ArgumentParser(description="Upload Blockworld dataset to HF Hub")
    p.add_argument(
        "--org",
        type=str,
        default="flowm123",
        help="Organization/user name (default: flowm123)",
    )
    p.add_argument(
        "--config",
        type=str,
        choices=list(CONFIG_TO_PATTERNS.keys()),
        default="dynamic",
        help="Config to upload (default: dynamic)",
    )
    p.add_argument(
        "--data-root",
        type=str,
        default="data/blockworld",
        help="Root directory of blockworld data",
    )
    p.add_argument("--private", action="store_true", help="Make the dataset private")
    p.add_argument("--upload-raw", action="store_true", help="Also upload raw .pt/.mp4 files")
    p.add_argument(
        "--shard-raw",
        action="store_true",
        help=(
            "Upload raw data as per-episode .tar shards under <config>/<split>/<episode>.tar. "
            "Greatly reduces file count and avoids Hub rate limits."
        ),
    )
    p.add_argument(
        "--shard-root",
        type=str,
        default=None,
        help=(
            "Where to write shard files locally (default: <data-root>/_hf_shards). "
            "Shards will be created under <shard-root>/<config>/<split>/..."
        ),
    )
    p.add_argument("--shard-overwrite", action="store_true", help="Rebuild shard .tar files even if they exist")
    p.add_argument(
        "--max-episodes-per-split",
        type=int,
        default=None,
        help="For debugging: only shard/upload the first N episode folders per split",
    )
    p.add_argument(
        "--episode-ids",
        type=str,
        default=None,
        help="Comma-separated episode IDs to shard/upload (e.g. '115,136')",
    )
    p.add_argument(
        "--all-configs",
        action="store_true",
        help="Upload all configs (and their splits) instead of a single --config",
    )
    p.add_argument("--dry-run", action="store_true", help="Don't upload anything")
    p.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip creating/pushing the index dataset (raw upload only)",
    )
    p.add_argument(
        "--create-readme",
        action="store_true",
        help="Create README.md file only (at --data-root/README.md)",
    )

    # Raw non-sharded upload mode controls (kept consistent with MNIST World uploader)
    p.add_argument("--follow-symlink-dirs", action="store_true", help="Follow symlinked directories when uploading")
    p.add_argument(
        "--raw-upload-mode",
        type=str,
        choices=["auto", "large", "folder", "batched"],
        default="auto",
        help="How to upload raw files when not using --shard-raw",
    )
    p.add_argument("--num-workers", type=int, default=16, help="Workers for resumable uploads")
    p.add_argument(
        "--large-report-every",
        type=int,
        default=60,
        help="Seconds between progress reports for resumable uploads",
    )
    p.add_argument(
        "--max-files-per-split",
        type=int,
        default=None,
        help="For debugging: only upload first N files per split (non-sharded mode)",
    )

    args = p.parse_args()

    repo_id = f"{args.org}/blockworld"
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise SystemExit(f"Error: data root does not exist: {data_root}")

    if args.create_readme:
        readme = create_readme(repo_id, args.org)
        out = data_root / "README.md"
        out.write_text(readme)
        print(f"✓ Created README at {out}")
        return

    shard_root = Path(args.shard_root) if args.shard_root else (data_root / "_hf_shards")

    episode_ids: Optional[Set[str]] = None
    if args.episode_ids:
        episode_ids = {s.strip() for s in args.episode_ids.split(",") if s.strip()}

    api = HfApi()
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
                        if (created + skipped) == 0:
                            print(f"No shards to upload for split '{dir_name}'. Skipping.")
                        else:
                            print(f"Uploading shards for split '{dir_name}' from {shard_root}/{config_name}/{dir_name} ...")
                            api.upload_large_folder(
                                repo_id=repo_id,
                                repo_type="dataset",
                                folder_path=str(shard_root),
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
                print(f"\nRaw upload summary for {config_name}: commits=(multiple), file_failures={skipped_total}")
            else:
                print(f"\nRaw upload summary for {config_name}: commits={uploaded_total}, file_failures={skipped_total}")

        if args.skip_index:
            print("\nSkipping index dataset creation/push (--skip-index)")
            continue

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
            upload_to_hub(dataset_dict, repo_id=repo_id, config_name=config_name, private=args.private)

    print(f"\n{'='*60}")
    print("Next steps:")
    print(f"{'='*60}")
    print("1. Upload all configs (index only):")
    print(f"   python {__file__} --org {args.org} --all-configs")
    print("\n2. Upload a single config (sharded raw + index):")
    print(
        f"   python {__file__} --org {args.org} --upload-raw --shard-raw --config dynamic --num-workers 2 --large-report-every 30"
    )
    print("\n3. Quick test: shard/upload a single episode only (raw only):")
    print(
        f"   python {__file__} --org {args.org} --upload-raw --shard-raw --config dynamic --episode-ids 0 --skip-index --num-workers 2 --large-report-every 10"
    )
    print(f"   https://huggingface.co/datasets/{repo_id}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
