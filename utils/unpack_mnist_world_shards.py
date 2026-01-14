#!/usr/bin/env python3
"""Unpack MNIST World per-episode shards.

This script is meant to be used after downloading the dataset repo contents from
Hugging Face (e.g., via `snapshot_download` or `git clone`). It extracts shard
archives stored under either of these layouts:

    <config>/<split>/<episode_id>.tar
    shards/<split>/<episode_id>.tar   (legacy)

Each tar contains paths like:

  <split>/<episode_id>/*.pt
  <split>/<episode_id>/*.mp4

So extracting at the dataset root recreates the original on-disk structure and
existing training scripts can keep using paths like `dynamic_training/115/000.pt`.

Example:
  python utils/unpack_mnist_world_shards.py --repo-root /path/to/mnist-world

Safety:
  Uses a safe extraction routine to prevent path traversal attacks.
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path
from typing import Iterable


def iter_shard_files(repo_root: Path) -> Iterable[Path]:
    # Prefer the legacy location if present.
    shards_dir = repo_root / "shards"
    found: list[Path] = []
    if shards_dir.exists():
        found.extend(list(shards_dir.rglob("*.tar")))

    # Also support the newer "<config>/<split>/*.tar" layout by scanning known config
    # directories if they exist.
    for cfg in ["dynamic_po", "static_po", "dynamic_fo", "dynamic_fo_no_sm"]:
        cfg_dir = repo_root / cfg
        if cfg_dir.exists():
            found.extend(list(cfg_dir.rglob("*.tar")))

    # De-duplicate and sort for stable behavior.
    return sorted({p.resolve() for p in found})


def _is_within_directory(directory: Path, target: Path) -> bool:
    try:
        directory = directory.resolve()
        target = target.resolve()
    except FileNotFoundError:
        # Target may not exist yet; use parents based check.
        directory = directory.resolve()
        target = (directory / target).resolve()
    return str(target).startswith(str(directory) + "/") or target == directory


def safe_extract(tf: tarfile.TarFile, path: Path) -> None:
    for member in tf.getmembers():
        member_path = path / member.name
        if not _is_within_directory(path, member_path):
            raise RuntimeError(f"Blocked path traversal attempt in tar member: {member.name}")
    tf.extractall(path)


def main() -> None:
    p = argparse.ArgumentParser(description="Unpack MNIST World shard tar files")
    p.add_argument(
        "--repo-root",
        type=str,
        required=True,
        help="Path to the downloaded dataset repository root (the folder containing shards/)",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Where to extract. Default: same as --repo-root",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="For testing: only extract first N shard files",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files (tarfile will overwrite by default).",
    )

    args = p.parse_args()
    repo_root = Path(args.repo_root)
    out_root = Path(args.output_root) if args.output_root else repo_root

    shard_files = list(iter_shard_files(repo_root))
    if args.limit is not None:
        shard_files = shard_files[: args.limit]

    if len(shard_files) == 0:
        print(
            "No .tar shards found. Looked under:\n"
            f"  - {repo_root / 'shards'}\n"
            f"  - {repo_root / '<config>'} (dynamic_po/static_po/dynamic_fo/dynamic_fo_no_sm)"
        )
        return

    print(f"Found {len(shard_files)} shard files")
    print(f"Extracting into: {out_root}")

    # NOTE: tar extraction overwrites by default. We keep the flag for explicitness.
    _ = args.overwrite

    for i, tar_path in enumerate(shard_files, start=1):
        if i % 50 == 0:
            print(f"  Extracted {i}/{len(shard_files)}")
        with tarfile.open(tar_path, mode="r") as tf:
            safe_extract(tf, out_root)

    print("Done.")


if __name__ == "__main__":
    main()
