#!/usr/bin/env python3
"""Download released model checkpoints from Hugging Face Hub.

This is the companion to `utils/upload_models_to_hf.py`.

It downloads the curated "paper artifact" model repos:
  - <org>/<mnist_repo>      (default: mnistworld-models)
  - <org>/<blockworld_repo> (default: blockworld-models)

The script prefers `MODEL_INDEX.json` (written by the uploader) to know exactly
which files to download. If that file is missing, it falls back to downloading
all files in the repo.

By default files are saved under:
  downloaded_checkpoints/hf_models/<repo_name>/<path_in_repo>

Examples:
  # Download everything into the default folder
  python utils/download_models_from_hf.py --org flowm123

  # Download to a custom directory
  python utils/download_models_from_hf.py --org flowm123 --out /path/to/downloaded_checkpoints

  # Dry-run (show what would be downloaded)
  python utils/download_models_from_hf.py --org flowm123 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional


class RepoSpec(NamedTuple):
    repo_id: str
    repo_name: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_out_dir() -> Path:
    return _repo_root() / "downloaded_checkpoints" / "hf_models"


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text())


def _safe_rel_path(path_in_repo: str) -> Path:
    # Keep a strict relative path (no traversal).
    rel = Path(path_in_repo)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Refusing suspicious repo path: {path_in_repo}")
    return rel


def _iter_files_from_model_index(model_index: Dict[str, Any]) -> List[str]:
    items = model_index.get("items", [])
    paths: List[str] = []
    for it in items:
        p = it.get("path")
        if isinstance(p, str) and p.strip():
            paths.append(p)
    # De-dup while preserving order
    seen = set()
    out: List[str] = []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _ensure_hf_cache_writable() -> None:
    """Best-effort: keep behavior consistent with the uploader.

    This avoids failures on clusters where HF_HOME points to a shared read-only path.
    """

    # Respect user-provided overrides.
    if os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get("HF_HUB_CACHE"):
        return
    if os.environ.get("HF_HOME"):
        return

    # Default cache is usually fine; do not aggressively override here.
    # (Uploader needs locks for writes; downloads are usually okay.)


def _download_repo(
    spec: RepoSpec,
    out_root: Path,
    *,
    revision: Optional[str],
    dry_run: bool,
    use_model_index: bool,
    num_workers: int,
) -> None:
    _ensure_hf_cache_writable()

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install requirements and retry.\n"
            f"Original error: {e}"
        )

    api = HfApi()

    repo_out = out_root / spec.repo_name
    if not dry_run:
        repo_out.mkdir(parents=True, exist_ok=True)

    files_to_get: List[str] = []
    model_index_local: Optional[Path] = None

    if use_model_index:
        try:
            model_index_local = Path(
                hf_hub_download(
                    repo_id=spec.repo_id,
                    repo_type="model",
                    filename="MODEL_INDEX.json",
                    revision=revision,
                )
            )
            idx = _load_json(model_index_local)
            files_to_get = _iter_files_from_model_index(idx)
        except Exception:
            # Fall back to listing the repo.
            files_to_get = []

    if not files_to_get:
        files_to_get = [
            f
            for f in api.list_repo_files(repo_id=spec.repo_id, repo_type="model", revision=revision)
            # Skip any LFS lockfiles / git internals; keep content only.
            if not f.startswith(".git/")
        ]

    # Always include README and MODEL_INDEX (if present) for convenience.
    for extra in ["README.md", "MODEL_INDEX.json"]:
        if extra not in files_to_get:
            files_to_get.append(extra)

    print(f"\nRepo: {spec.repo_id}")
    print(f"  -> {repo_out}")
    print(f"  files: {len(files_to_get)}")

    manifest: Dict[str, Any] = {
        "repo_id": spec.repo_id,
        "repo_name": spec.repo_name,
        "revision": revision,
        "files": [],
    }

    def _download_one(path_in_repo: str) -> Dict[str, str]:
        rel = _safe_rel_path(path_in_repo)
        dest = repo_out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        local = hf_hub_download(
            repo_id=spec.repo_id,
            repo_type="model",
            filename=path_in_repo,
            revision=revision,
            local_dir=str(repo_out),
            local_dir_use_symlinks=False,
        )
        return {"path_in_repo": path_in_repo, "local_path": local}

    if dry_run:
        for path_in_repo in files_to_get:
            rel = _safe_rel_path(path_in_repo)
            dest = repo_out / rel
            print(f"  [DRY-RUN] {path_in_repo} -> {dest}")
            manifest["files"].append({"path_in_repo": path_in_repo, "local_path": str(dest)})
    else:
        workers = max(1, int(num_workers))
        started = time.time()
        failures: List[str] = []
        results: List[Dict[str, str]] = []

        # Parallel per-file downloads. Threading is appropriate here because downloads are IO-bound.
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_download_one, p): p for p in files_to_get}
            for i, fut in enumerate(as_completed(futs), start=1):
                p = futs[fut]
                try:
                    results.append(fut.result())
                except Exception as e:
                    failures.append(p)
                    print(f"  Warning: failed to download '{p}': {e}")
                if (i % 10) == 0 or i == len(files_to_get):
                    elapsed = time.time() - started
                    print(f"  Progress: {i}/{len(files_to_get)} files ({elapsed:.1f}s)")

        # Keep manifest stable/deterministic.
        results.sort(key=lambda d: d["path_in_repo"])
        manifest["files"].extend(results)
        if failures:
            manifest["failures"] = failures

    if not dry_run:
        (repo_out / "DOWNLOAD_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> None:
    p = argparse.ArgumentParser(description="Download released model checkpoints from Hugging Face")
    p.add_argument(
        "--org",
        type=str,
        default="flowm123",
        help="HF organization/user (default: flowm123)",
    )
    p.add_argument(
        "--mnist-repo",
        type=str,
        default="mnistworld-models",
        help="Model repo name for MNIST World (default: mnistworld-models)",
    )
    p.add_argument(
        "--blockworld-repo",
        type=str,
        default="blockworld-models",
        help="Model repo name for Blockworld (default: blockworld-models)",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(_default_out_dir()),
        help="Output directory root (default: downloaded_checkpoints/hf_models)",
    )
    p.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional git revision / tag / commit hash to download from",
    )
    p.add_argument("--dry-run", action="store_true", help="Print what would be downloaded")
    p.add_argument(
        "--no-model-index",
        action="store_true",
        help="Ignore MODEL_INDEX.json and download all files listed in the repo",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel download workers per repo (default: 8)",
    )
    p.add_argument(
        "--parallel-repos",
        action="store_true",
        help="Download MNIST + Blockworld repos concurrently",
    )
    p.add_argument(
        "--only",
        type=str,
        choices=["all", "mnist", "blockworld"],
        default="all",
        help="Download only one repo (default: all)",
    )

    args = p.parse_args()

    out_root = Path(args.out).expanduser()

    repos: List[RepoSpec] = []
    if args.only in ("all", "mnist"):
        repos.append(RepoSpec(f"{args.org}/{args.mnist_repo}", args.mnist_repo))
    if args.only in ("all", "blockworld"):
        repos.append(RepoSpec(f"{args.org}/{args.blockworld_repo}", args.blockworld_repo))

    print("Downloading model repos:")
    for r in repos:
        print(f"  - {r.repo_id}")

    if args.parallel_repos and len(repos) > 1 and (not args.dry_run):
        # Parallelize across repos (coarse-grained). Each repo already uses per-file parallelism.
        with ThreadPoolExecutor(max_workers=min(2, len(repos))) as ex:
            futs = [
                ex.submit(
                    _download_repo,
                    r,
                    out_root,
                    revision=args.revision,
                    dry_run=args.dry_run,
                    use_model_index=not args.no_model_index,
                    num_workers=args.num_workers,
                )
                for r in repos
            ]
            for fut in as_completed(futs):
                fut.result()
    else:
        for r in repos:
            _download_repo(
                r,
                out_root,
                revision=args.revision,
                dry_run=args.dry_run,
                use_model_index=not args.no_model_index,
                num_workers=args.num_workers,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
