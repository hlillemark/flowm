#!/usr/bin/env python3
"""Unpack Blockworld per-episode shards.

This script is used after downloading the dataset repo contents from Hugging Face.
It extracts shard archives stored under either of these layouts:

  <config>/<split>/<episode_id>.tar
  shards/<split>/<episode_id>.tar   (legacy)

Each tar contains paths like:

  <split>/<episode_id>/*.pt
  <split>/<episode_id>/*.mp4

So extracting at the dataset root recreates the original on-disk structure.

Example:
  python utils/unpack_blockworld_shards.py --repo-root /path/to/blockworld

Safety:
  Uses a safe extraction routine to prevent path traversal attacks.
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path
from typing import Iterable


def iter_shard_files(repo_root: Path) -> Iterable[Path]:
    found: list[Path] = []

    shards_dir = repo_root / "shards"
    if shards_dir.exists():
        found.extend(list(shards_dir.rglob("*.tar")))

    # New layout: config directories at repo root.
    for cfg in ["sunday_v2", "tex", "sunday_v2_static"]:
        cfg_dir = repo_root / cfg
        if cfg_dir.exists():
            found.extend(list(cfg_dir.rglob("*.tar")))

    return sorted({p.resolve() for p in found})


def _is_within_directory(directory: Path, target: Path) -> bool:
    try:
        directory = directory.resolve()
        target = target.resolve()
    except FileNotFoundError:
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
    p = argparse.ArgumentParser(description="Unpack Blockworld shard tar files")
    p.add_argument(
        "--repo-root",
        type=str,
        required=True,
        help="Path to the downloaded dataset repository root",
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
            f"  - {repo_root / '<config>'} (sunday_v2/tex/sunday_v2_static)"
        )
        return

    print(f"Found {len(shard_files)} shard files")
    print(f"Extracting into: {out_root}")

    for i, tar_path in enumerate(shard_files, start=1):
        if i % 50 == 0:
            print(f"  Extracted {i}/{len(shard_files)}")
        with tarfile.open(tar_path, mode="r") as tf:
            safe_extract(tf, out_root)

    print("Done.")


if __name__ == "__main__":
    main()
