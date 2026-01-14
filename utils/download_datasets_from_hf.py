#!/usr/bin/env python3
"""Download sharded datasets from Hugging Face and unshard (untar) into ./data.

This script is a companion to:
  - utils/upload_blockworld_to_hf.py
  - utils/upload_mnist_world_to_hf.py

Those uploaders store raw data as per-episode tar shards under layouts like:

  <config>/<split_dir>/<episode_id>.tar

Each tar contains paths like:

  <split_dir>/<episode_id>/*.pt
  <split_dir>/<episode_id>/*.mp4

So extracting shards into the desired dataset root recreates the original on-disk
folder structure used by training (e.g. data/blockworld/tex_training/0/0000_rgb.mp4).

Key features:
  - Download only selected dataset/config/split/episode_ids for quick tests.
  - Parallel downloads (HF snapshot_download workers).
  - Parallel extraction (thread pool).
  - Keeps the downloaded shard repo snapshot on disk for re-use.
  
For configs and splits, see below for the mapping and valid options.

Examples:
  # Download + extract both datasets (defaults: org=flowm123)
  python utils/download_datasets_from_hf.py

  # Just Blockworld tex+static, only validation split, only first 20 shards (quick test)
  python utils/download_datasets_from_hf.py --dataset blockworld --configs tex,static --splits validation --limit-tars 20

  # MNIST World dynamic_po, only validation_200
  python utils/download_datasets_from_hf.py --dataset mnist_world --configs dynamic_po --splits validation_200

  # Download only (no extraction)
  python utils/download_datasets_from_hf.py --dataset blockworld --download-only

  # Extract only from an already-downloaded snapshot
  python utils/download_datasets_from_hf.py --dataset blockworld --extract-only
"""

from __future__ import annotations

import argparse
import contextlib
import os
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from pathlib import PurePosixPath
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


# --- Dataset repos and layout helpers -------------------------------------------------

DATASET_REPOS = {
    "blockworld": "blockworld",
    "mnist_world": "mnist-world",
}


# Blockworld shards on HF are stored under top-level config dirs:
#   dynamic/, tex/, static/
# with split dirs inside each config dir.
# We also accept aliases (sunday_v2, sunday_v2_static) and normalize them.
BLOCKWORLD_CONFIG_ALIASES: Dict[str, str] = {
    "dynamic": "dynamic",
    "sunday_v2": "dynamic",
    "tex": "tex",
    "static": "static",
    "sunday_v2_static": "static",
}

# Split-dir mapping mirrors utils/upload_blockworld_to_hf.py
BLOCKWORLD_SPLIT_DIRS_BY_CONFIG: Dict[str, Dict[str, str]] = {
    "dynamic": {
        "train": "sunday_v2_training",
        "validation": "sunday_v2_validation",
    },
    "tex": {
        "train": "tex_training",
        "validation": "tex_validation",
    },
    "static": {
        "train": "sunday_v2_static_training",
        "validation": "sunday_v2_static_validation",
    },
}

# MNIST World config dirs and split dirs (from uploader).
MNIST_CONFIG_DIRS: Dict[str, List[str]] = {
    "dynamic_po": ["dynamic_po"],
    "static_po": ["static_po"],
    "dynamic_fo": ["dynamic_fo"],
    "dynamic_fo_no_sm": ["dynamic_fo_no_sm"],
}

MNIST_SPLITS_BY_CONFIG: Dict[str, Dict[str, str]] = {
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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_download_root() -> Path:
    # Keep snapshots under data/ so everything is self-contained.
    return _repo_root() / "data" / "_hf_snapshots"


def _default_extract_root(dataset: str) -> Path:
    return _repo_root() / "data" / dataset


def _get_tqdm():
    """Return tqdm callable if available, else None."""

    with contextlib.suppress(Exception):
        from tqdm.auto import tqdm  # type: ignore

        return tqdm
    return None


def _matches_any(path: str, patterns: Sequence[str]) -> bool:
    p = PurePosixPath(path)
    for pat in patterns:
        if p.match(pat):
            return True
    return False


def _split_csv(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


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


def _iter_tar_files(root: Path) -> Iterable[Path]:
    yield from root.rglob("*.tar")


def _build_allow_patterns(
    *,
    dataset: str,
    configs: Sequence[str],
    splits: Sequence[str],
    episode_ids: Optional[Set[str]],
) -> List[str]:
    patterns: List[str] = []

    # Always grab repo metadata files if present.
    patterns += ["README.md", "dataset_infos.json", ".gitattributes"]

    if dataset == "blockworld":
        # Normalize config names to the actual HF repo layout.
        if not configs:
            configs = ["dynamic", "tex", "static"]

        norm_configs: List[str] = []
        for c in configs:
            norm_configs.append(BLOCKWORLD_CONFIG_ALIASES.get(c, c))
        norm_configs = sorted(set(norm_configs))

        if not splits:
            splits = ["train", "validation"]

        for cfg in norm_configs:
            split_map = BLOCKWORLD_SPLIT_DIRS_BY_CONFIG.get(cfg)
            if split_map is None:
                # Allow advanced users to specify raw layout directly.
                # In that case, treat cfg as a literal top-level directory name.
                split_map = {}

            for split_key in splits:
                split_dir = split_map.get(split_key, split_key)
                if episode_ids:
                    for eid in episode_ids:
                        patterns.append(f"{cfg}/{split_dir}/{eid}.tar")
                else:
                    patterns.append(f"{cfg}/{split_dir}/*.tar")
                    patterns.append(f"{cfg}/{split_dir}/**/*.tar")

        return patterns

    if dataset == "mnist_world":
        if not configs:
            configs = ["dynamic_po"]
        if not splits:
            splits = ["train", "validation"]

        for cfg in configs:
            cfg_dirs = MNIST_CONFIG_DIRS.get(cfg, [cfg])
            split_map = MNIST_SPLITS_BY_CONFIG.get(cfg, {})
            for cfg_dir in cfg_dirs:
                for split_key in splits:
                    split_dir = split_map.get(split_key)
                    if split_dir is None:
                        # Allow user to specify raw split_dir names too.
                        split_dir = split_key

                    if episode_ids:
                        for eid in episode_ids:
                            patterns.append(f"{cfg_dir}/{split_dir}/{eid}.tar")
                    else:
                        patterns.append(f"{cfg_dir}/{split_dir}/*.tar")
                        patterns.append(f"{cfg_dir}/{split_dir}/**/*.tar")

        return patterns

    raise ValueError(f"Unknown dataset: {dataset}")


def _download_snapshot(
    *,
    repo_id: str,
    local_dir: Path,
    allow_patterns: List[str],
    revision: Optional[str],
    token: Optional[str],
    max_workers: int,
    dry_run: bool,
) -> None:
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install requirements and retry.\n"
            f"Original error: {e}"
        )

    if dry_run:
        print(f"[DRY-RUN] Would download: {repo_id}")
        print(f"  -> {local_dir}")
        print("  allow_patterns:")
        for p in allow_patterns:
            print(f"    - {p}")
        return

    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
        max_workers=max(1, int(max_workers)),
        token=token,
    )


def _download_files_with_progress(
    *,
    repo_id: str,
    local_dir: Path,
    allow_patterns: List[str],
    revision: Optional[str],
    token: Optional[str],
    max_workers: int,
    dry_run: bool,
) -> None:
    """Download selected repo files with an explicit progress bar.

    This is slower than snapshot_download for large repos sometimes, but provides
    a clear overall progress bar and reliable parallelism.
    """

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install requirements and retry.\n"
            f"Original error: {e}"
        )

    api = HfApi()
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    wanted = [f for f in repo_files if _matches_any(f, allow_patterns)]
    wanted = sorted(set(wanted))

    if dry_run:
        print(f"[DRY-RUN] Would download (file mode): {repo_id}")
        print(f"  -> {local_dir}")
        print(f"  matched_files: {len(wanted)}")
        for f in wanted[:50]:
            print(f"    - {f}")
        if len(wanted) > 50:
            print(f"    ... (+{len(wanted) - 50} more)")
        return

    local_dir.mkdir(parents=True, exist_ok=True)
    tqdm = _get_tqdm()

    def _dl_one(filename: str) -> None:
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            revision=revision,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            token=token,
        )

    workers = max(1, int(max_workers))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_dl_one, f): f for f in wanted}
        if tqdm is not None:
            with tqdm(total=len(futs), desc="Downloading", unit="file") as pbar:
                for fut in as_completed(futs):
                    fname = futs[fut]
                    try:
                        fut.result()
                    except Exception as e:
                        raise RuntimeError(f"Failed to download {repo_id}:{fname}: {e}") from e
                    pbar.update(1)
        else:
            # Fallback: periodic prints
            for i, fut in enumerate(as_completed(futs), start=1):
                fname = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    raise RuntimeError(f"Failed to download {repo_id}:{fname}: {e}") from e
                if (i % 50) == 0 or i == len(futs):
                    print(f"  Downloaded {i}/{len(futs)} files")


def _extract_tars(
    *,
    tar_files: Sequence[Path],
    output_root: Path,
    num_workers: int,
    dry_run: bool,
) -> None:
    if len(tar_files) == 0:
        print("No .tar files to extract.")
        return

    print(f"Extracting {len(tar_files)} tar files into: {output_root}")

    if dry_run:
        for p in tar_files[:20]:
            print(f"  [DRY-RUN] would extract: {p}")
        if len(tar_files) > 20:
            print(f"  ... (+{len(tar_files) - 20} more)")
        return

    output_root.mkdir(parents=True, exist_ok=True)

    started = time.time()
    tqdm = _get_tqdm()

    def _extract_one(tar_path: Path) -> None:
        with tarfile.open(tar_path, mode="r") as tf:
            safe_extract(tf, output_root)

    workers = max(1, int(num_workers))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_extract_one, p): p for p in tar_files}
        if tqdm is not None:
            with tqdm(total=len(futs), desc="Extracting", unit="tar") as pbar:
                for fut in as_completed(futs):
                    p = futs[fut]
                    try:
                        fut.result()
                    except Exception as e:
                        raise RuntimeError(f"Failed to extract {p}: {e}") from e
                    pbar.update(1)
        else:
            for i, fut in enumerate(as_completed(futs), start=1):
                p = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    raise RuntimeError(f"Failed to extract {p}: {e}") from e
                if (i % 25) == 0 or i == len(tar_files):
                    elapsed = time.time() - started
                    print(f"  Progress: {i}/{len(tar_files)} ({elapsed:.1f}s)")


def main() -> None:
    p = argparse.ArgumentParser(description="Download HF datasets and unshard into ./data")
    p.add_argument(
        "--org",
        type=str,
        default="flowm123",
        help="HF org/user (default: flowm123)",
    )
    p.add_argument(
        "--dataset",
        type=str,
        choices=["all", "blockworld", "mnist_world"],
        default="all",
        help="Which dataset to download (default: all)",
    )
    p.add_argument(
        "--blockworld-repo",
        type=str,
        default=DATASET_REPOS["blockworld"],
        help="HF dataset repo name for Blockworld (default: blockworld)",
    )
    p.add_argument(
        "--mnist-repo",
        type=str,
        default=DATASET_REPOS["mnist_world"],
        help="HF dataset repo name for MNIST World (default: mnist-world)",
    )
    p.add_argument(
        "--configs",
        type=str,
        default=None,
        help=(
            "Comma-separated configs to download (dataset-specific). "
            "Blockworld examples: dynamic,tex,static. MNIST examples: dynamic_po,static_po,dynamic_fo. "
            "Default: blockworld=dynamic,tex,static; mnist_world=dynamic_po"
        ),
    )
    p.add_argument(
        "--splits",
        type=str,
        default=None,
        help=(
            "Comma-separated split keys. Blockworld: train,validation. "
            "MNIST: train,validation,train_200,validation_200,... (or raw split dir names)."
        ),
    )
    p.add_argument(
        "--episode-ids",
        type=str,
        default=None,
        help="Comma-separated episode IDs to download/extract (e.g. '0,115,136')",
    )
    p.add_argument(
        "--limit-tars",
        type=int,
        default=None,
        help="For quick tests: only extract first N tar files after download/filtering",
    )
    p.add_argument(
        "--download-root",
        type=str,
        default=str(_default_download_root()),
        help="Where to store downloaded repo snapshots (default: data/_hf_snapshots)",
    )
    p.add_argument(
        "--data-root",
        type=str,
        default=str(_repo_root() / "data"),
        help="Base data dir (default: ./data)",
    )
    p.add_argument(
        "--download-workers",
        type=int,
        default=8,
        help="Parallel download workers (HF snapshot_download max_workers; default: 8)",
    )
    p.add_argument(
        "--download-method",
        type=str,
        choices=["snapshot", "files"],
        default="snapshot",
        help=(
            "How to download from HF. 'snapshot' is usually fastest. "
            "'files' provides an explicit overall progress bar. (default: snapshot)"
        ),
    )
    p.add_argument(
        "--extract-workers",
        type=int,
        default=8,
        help="Parallel extraction workers (default: 8)",
    )
    p.add_argument(
        "--parallel-repos",
        action="store_true",
        help="Download/extract Blockworld and MNIST World concurrently (when --dataset all)",
    )
    p.add_argument("--download-only", action="store_true", help="Only download shards, do not extract")
    p.add_argument("--extract-only", action="store_true", help="Only extract from existing snapshot, do not download")
    p.add_argument("--dry-run", action="store_true", help="Print actions without downloading/extracting")
    p.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional dataset repo revision/tag/commit",
    )
    args = p.parse_args()

    data_root = Path(args.data_root)
    download_root = Path(args.download_root)
    
    os.makedirs(data_root, exist_ok=True)

    episode_ids = set(_split_csv(args.episode_ids)) if args.episode_ids else None
    user_configs = _split_csv(args.configs)
    user_splits = _split_csv(args.splits)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    def run_one(dataset: str) -> None:
        if dataset == "blockworld":
            repo_id = f"{args.org}/{args.blockworld_repo}"
            configs = user_configs if user_configs else ["dynamic", "tex", "static"]
            splits = user_splits
            snapshot_dir = download_root / "blockworld"
            extract_root = data_root / "blockworld"
        elif dataset == "mnist_world":
            repo_id = f"{args.org}/{args.mnist_repo}"
            configs = user_configs if user_configs else ["dynamic_po"]
            splits = user_splits
            snapshot_dir = download_root / "mnist_world"
            extract_root = data_root / "mnist_world"
        else:
            raise ValueError(dataset)

        allow_patterns = _build_allow_patterns(
            dataset=dataset,
            configs=configs,
            splits=splits,
            episode_ids=episode_ids,
        )

        print("\n" + "=" * 70)
        print(f"Dataset: {dataset}")
        print(f"Repo:    {repo_id}")
        print(f"Snap:    {snapshot_dir}")
        print(f"Extract: {extract_root}")
        print(f"DL wkrs: {args.download_workers} (method={args.download_method})")
        print(f"EX wkrs: {args.extract_workers}")
        print("=" * 70)

        if not args.extract_only:
            if args.download_method == "snapshot":
                _download_snapshot(
                    repo_id=repo_id,
                    local_dir=snapshot_dir,
                    allow_patterns=allow_patterns,
                    revision=args.revision,
                    token=token,
                    max_workers=args.download_workers,
                    dry_run=args.dry_run,
                )
            else:
                _download_files_with_progress(
                    repo_id=repo_id,
                    local_dir=snapshot_dir,
                    allow_patterns=allow_patterns,
                    revision=args.revision,
                    token=token,
                    max_workers=args.download_workers,
                    dry_run=args.dry_run,
                )

        if args.download_only:
            print("Skipping extraction (--download-only)")
            return

        # Collect tar files from the snapshot.
        tar_files = sorted(_iter_tar_files(snapshot_dir))

        # Apply an additional episode-id filter at extraction-time (defensive).
        if episode_ids:
            tar_files = [p for p in tar_files if p.stem in episode_ids]

        if args.limit_tars is not None:
            tar_files = tar_files[: max(0, int(args.limit_tars))]

        _extract_tars(
            tar_files=tar_files,
            output_root=extract_root,
            num_workers=args.extract_workers,
            dry_run=args.dry_run,
        )

    if args.dataset == "all":
        datasets = ["blockworld", "mnist_world"]
    else:
        datasets = [args.dataset]

    if args.parallel_repos and args.dataset == "all" and (not args.dry_run) and len(datasets) > 1:
        with ThreadPoolExecutor(max_workers=2) as ex:
            futs = [ex.submit(run_one, d) for d in datasets]
            for fut in as_completed(futs):
                fut.result()
    else:
        for d in datasets:
            run_one(d)

    print("\nDone.")


if __name__ == "__main__":
    main()
