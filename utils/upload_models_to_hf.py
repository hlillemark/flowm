#!/usr/bin/env python3
"""Upload selected checkpoints to Hugging Face Hub (model repos).

This script is intended for "paper artifact" uploads: a small curated subset of
checkpoints from `configurations/ckpt_map/default.yaml`.

By user request, the default repo names are:
  - mnistworld-models
  - blockworld-models

Typical usage:
  python utils/upload_models_to_hf.py --org <org> --dry-run
  python utils/upload_models_to_hf.py --org <org>

Notes (cluster environments):
  If Hugging Face cache points to a shared read-only location, uploads can fail
  due to lockfile permissions. This script auto-falls-back to a user-writable
  cache (e.g. ~/.cache/huggingface or /tmp/huggingface) before importing
  `huggingface_hub`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple


def _default_hf_home() -> Path:
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache_home).expanduser() if xdg_cache_home else (Path.home() / ".cache")
    return base / "huggingface"


def _hub_cache_from_env_or_home() -> Tuple[Optional[Path], Optional[Path]]:
    hub_cache_env = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get("HF_HUB_CACHE")
    if hub_cache_env:
        return Path(hub_cache_env).expanduser(), None

    hf_home_env = os.environ.get("HF_HOME")
    if hf_home_env:
        hf_home = Path(hf_home_env).expanduser()
        return hf_home / "hub", hf_home

    hf_home = _default_hf_home()
    return hf_home / "hub", hf_home


def _ensure_writable_dir(p: Path) -> bool:
    try:
        p.mkdir(parents=True, exist_ok=True)
        probe = p / ".write_test"
        probe.write_text("ok")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _configure_hf_cache_if_needed() -> None:
    hub_cache, _hf_home = _hub_cache_from_env_or_home()
    if hub_cache is None:
        return

    locks_dir = hub_cache / ".locks"
    if _ensure_writable_dir(locks_dir):
        return

    fallback_home = _default_hf_home()
    fallback_hub = fallback_home / "hub"

    if not _ensure_writable_dir(fallback_hub / ".locks"):
        tmp_base = Path(tempfile.gettempdir()) / "huggingface"
        fallback_home = tmp_base
        fallback_hub = tmp_base / "hub"
        _ensure_writable_dir(fallback_hub / ".locks")

    os.environ["HF_HOME"] = str(fallback_home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(fallback_hub)
    os.environ["HF_HUB_CACHE"] = str(fallback_hub)

    print(
        "Warning: Hugging Face cache is not writable at the configured location. "
        f"Falling back to: HF_HOME={fallback_home}",
        file=sys.stderr,
    )


_configure_hf_cache_if_needed()

# Imports that may read HF env vars.
try:
    from huggingface_hub import CommitOperationAdd, HfApi
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: huggingface_hub. Install requirements and retry.\n"
        f"Original error: {e}"
    )

try:
    from omegaconf import DictConfig, OmegaConf
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: omegaconf. Install requirements and retry.\n"
        f"Original error: {e}"
    )


class ModelSpec(NamedTuple):
    """A single file to upload."""

    local_path: Path
    path_in_repo: str
    name: str
    meta: Dict[str, Any]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_ckpt_map(path: Path) -> DictConfig:
    cfg = OmegaConf.load(str(path))
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected DictConfig from {path}, got {type(cfg)}")
    return cfg


def _get_by_dotted_key(cfg: DictConfig, dotted: str) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, DictConfig):
            raise KeyError(f"Key '{dotted}' is invalid at '{part}'")
        if part not in cur:
            raise KeyError(f"Key '{dotted}' missing at '{part}'")
        cur = cur[part]
    return cur


def _get_first_by_dotted_keys(cfg: DictConfig, dotted_keys: Sequence[str]) -> Any:
    """Return the first value that exists among multiple dotted-key candidates."""

    last_err: Optional[Exception] = None
    for k in dotted_keys:
        try:
            return _get_by_dotted_key(cfg, k)
        except Exception as e:
            last_err = e
            continue
    raise KeyError(f"None of the keys exist: {list(dotted_keys)}") from last_err


def _resolve_local(repo_root: Path, maybe_rel: str) -> Path:
    p = Path(maybe_rel)
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def _make_model_index(specs: Sequence[ModelSpec]) -> Dict[str, Any]:
    # Keep this intentionally minimal; users can still browse files on the hub.
    return {
        "format": "model/model-index@v1",
        "items": [
            {
                "name": s.name,
                "path": s.path_in_repo,
                **s.meta,
            }
            for s in specs
        ],
    }


def _create_readme(title: str, repo_id: str, specs: Sequence[ModelSpec]) -> str:
    lines = [
        f"# {title}",
        "",
        "This repository hosts a curated set of checkpoints used for experiments.",
        "",
        "## Contents",
    ]
    for s in specs:
        lines.append(f"- `{s.path_in_repo}` — {s.name}")
    lines += [
        "",
        f"Hub: https://huggingface.co/{repo_id}",
        "",
    ]
    return "\n".join(lines)


def _build_mnist_specs(cfg: DictConfig, repo_root: Path) -> List[ModelSpec]:
    specs: List[ModelSpec] = []

    # MNIST World: DFoT dynamic_po
    dfot_po = _get_by_dotted_key(cfg, "algorithm.model_weights.dfot.mnist.dynamic_po")
    specs.append(
        ModelSpec(
            _resolve_local(repo_root, str(dfot_po)),
            "mnistworld/dfot/dynamic_po.ckpt",
            "mnistworld dfot dynamic_po",
            {"dataset": "mnist_world", "method": "dfot", "variant": "dynamic_po"},
        )
    )

    # MNIST World: DFoT-SSM dynamic_po
    ssm_po = _get_by_dotted_key(cfg, "algorithm.model_weights.ssm.mnist.dynamic_po")
    specs.append(
        ModelSpec(
            _resolve_local(repo_root, str(ssm_po)),
            "mnistworld/dfot-ssm/dynamic_po.ckpt",
            "mnistworld dfot-ssm dynamic_po",
            {"dataset": "mnist_world", "method": "dfot-ssm", "variant": "dynamic_po"},
        )
    )

    # MNIST World: FlowM (FERNN) checkpoint requested as .pt
    fernn_flowm = _get_by_dotted_key(cfg, "algorithm.model_weights.fernn.mnist.dynamic_po_flowm")
    specs.append(
        ModelSpec(
            _resolve_local(repo_root, str(fernn_flowm)),
            "mnistworld/flowm-fernn/dynamic_po_flowm.pt",
            "mnistworld flowm (fernn) dynamic_po_flowm",
            {"dataset": "mnist_world", "method": "flowm-fernn", "variant": "dynamic_po_flowm"},
        )
    )

    # MNIST World VAE
    mnist_vae = _get_first_by_dotted_keys(
        cfg,
        [
            "algorithm.vae.pretrained_dir.mnist_world",
            "vae.pretrained_dir.mnist_world",
        ],
    )
    specs.append(
        ModelSpec(
            _resolve_local(repo_root, str(mnist_vae)),
            "mnistworld/vae/mnist_world_vae.ckpt",
            "mnistworld vae",
            {"dataset": "mnist_world", "method": "vae"},
        )
    )

    return specs


def _build_blockworld_specs(cfg: DictConfig, repo_root: Path) -> List[ModelSpec]:
    specs: List[ModelSpec] = []

    def add_triplet(config_name: str, dotted_suffix: str) -> None:
        # FlowM
        flowm = _get_by_dotted_key(cfg, f"algorithm.model_weights.flowm.blockworld.flowm.{dotted_suffix}")
        specs.append(
            ModelSpec(
                _resolve_local(repo_root, str(flowm)),
                f"blockworld/{config_name}/flowm/{dotted_suffix}.ckpt",
                f"blockworld {config_name} flowm {dotted_suffix}",
                {"dataset": "blockworld", "config": config_name, "method": "flowm"},
            )
        )

        # DFoT
        dfot = _get_by_dotted_key(cfg, f"algorithm.model_weights.dfot.blockworld.{dotted_suffix}")
        specs.append(
            ModelSpec(
                _resolve_local(repo_root, str(dfot)),
                f"blockworld/{config_name}/dfot/{dotted_suffix}.ckpt",
                f"blockworld {config_name} dfot {dotted_suffix}",
                {"dataset": "blockworld", "config": config_name, "method": "dfot"},
            )
        )

        # DFoT-SSM
        ssm = _get_by_dotted_key(cfg, f"algorithm.model_weights.ssm.blockworld.{dotted_suffix}")
        specs.append(
            ModelSpec(
                _resolve_local(repo_root, str(ssm)),
                f"blockworld/{config_name}/dfot-ssm/{dotted_suffix}.ckpt",
                f"blockworld {config_name} dfot-ssm {dotted_suffix}",
                {"dataset": "blockworld", "config": config_name, "method": "dfot-ssm"},
            )
        )

    # User-facing config names:
    # - dynamic  -> v2_dynamic
    # - static   -> v2_static
    # - tex      -> tex
    add_triplet(config_name="dynamic", dotted_suffix="v2_dynamic")
    add_triplet(config_name="static", dotted_suffix="v2_static")
    add_triplet(config_name="tex", dotted_suffix="tex")

    # Blockworld VAEs
    bw_vae = _get_first_by_dotted_keys(
        cfg,
        [
            "algorithm.vae.pretrained_dir.blockworld",
            "vae.pretrained_dir.blockworld",
        ],
    )
    specs.append(
        ModelSpec(
            _resolve_local(repo_root, str(bw_vae)),
            "blockworld/vae/blockworld_vae.ckpt",
            "blockworld vae",
            {"dataset": "blockworld", "method": "vae", "config": "dynamic/static"},
        )
    )

    bw_tex_vae = _get_first_by_dotted_keys(
        cfg,
        [
            "algorithm.vae.pretrained_dir.blockworld_tex",
            "vae.pretrained_dir.blockworld_tex",
        ],
    )
    specs.append(
        ModelSpec(
            _resolve_local(repo_root, str(bw_tex_vae)),
            "blockworld/vae/blockworld_tex_vae.ckpt",
            "blockworld tex vae",
            {"dataset": "blockworld", "method": "vae", "config": "tex"},
        )
    )

    return specs


def _validate_specs(specs: Sequence[ModelSpec]) -> List[str]:
    missing: List[str] = []
    for s in specs:
        if not s.local_path.exists():
            missing.append(f"missing: {s.local_path} (for {s.path_in_repo})")
    return missing


def _commit_repo(
    api: HfApi,
    repo_id: str,
    title: str,
    specs: Sequence[ModelSpec],
    private: bool,
    dry_run: bool,
    skip_index: bool,
) -> None:
    if dry_run:
        print(f"[DRY-RUN] Would create/update repo: {repo_id} (private={private})")
    else:
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    ops: List[CommitOperationAdd] = []

    readme = _create_readme(title=title, repo_id=repo_id, specs=specs)
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme.encode("utf-8")))

    if not skip_index:
        model_index = _make_model_index(specs)
        ops.append(
            CommitOperationAdd(
                path_in_repo="MODEL_INDEX.json",
                path_or_fileobj=json.dumps(model_index, indent=2, sort_keys=True).encode("utf-8"),
            )
        )

    for s in specs:
        ops.append(CommitOperationAdd(path_in_repo=s.path_in_repo, path_or_fileobj=str(s.local_path)))

    if dry_run:
        print(f"[DRY-RUN] Would upload {len(specs)} checkpoint files (+ README/index) to {repo_id}:")
        for s in specs:
            print(f"  - {s.local_path} -> {s.path_in_repo}")
        return

    api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=ops,
        commit_message=f"Upload curated checkpoints ({title})",
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--org", default="flowm123", help="HF org/user to upload under (default: flowm123)")
    p.add_argument(
        "--ckpt-map",
        type=str,
        default="configurations/ckpt_map/default.yaml",
        help="Path to ckpt map yaml (default: configurations/ckpt_map/default.yaml)",
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
    p.add_argument("--private", action="store_true", help="Create repos as private")
    p.add_argument("--dry-run", action="store_true", help="Print actions without uploading")
    p.add_argument("--skip-mnist", action="store_true", help="Skip uploading MNIST World repo")
    p.add_argument("--skip-blockworld", action="store_true", help="Skip uploading Blockworld repo")
    p.add_argument("--skip-index", action="store_true", help="Do not upload MODEL_INDEX.json")

    args = p.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    api = HfApi(token=token)

    repo_root = _repo_root()
    ckpt_map_path = _resolve_local(repo_root, args.ckpt_map)
    if not ckpt_map_path.exists():
        raise SystemExit(f"ckpt map not found: {ckpt_map_path}")

    cfg = _load_ckpt_map(ckpt_map_path)

    if not args.skip_mnist:
        mnist_specs = _build_mnist_specs(cfg, repo_root)
        missing = _validate_specs(mnist_specs)
        if missing:
            print("MNIST World: missing local files:", file=sys.stderr)
            for m in missing:
                print(f"  - {m}", file=sys.stderr)
            raise SystemExit(2)

        mnist_repo_id = f"{args.org}/{args.mnist_repo}"
        _commit_repo(
            api=api,
            repo_id=mnist_repo_id,
            title="MNIST World Models",
            specs=mnist_specs,
            private=args.private,
            dry_run=args.dry_run,
            skip_index=args.skip_index,
        )

    if not args.skip_blockworld:
        bw_specs = _build_blockworld_specs(cfg, repo_root)
        missing = _validate_specs(bw_specs)
        if missing:
            print("Blockworld: missing local files:", file=sys.stderr)
            for m in missing:
                print(f"  - {m}", file=sys.stderr)
            raise SystemExit(2)

        bw_repo_id = f"{args.org}/{args.blockworld_repo}"
        _commit_repo(
            api=api,
            repo_id=bw_repo_id,
            title="Blockworld Models",
            specs=bw_specs,
            private=args.private,
            dry_run=args.dry_run,
            skip_index=args.skip_index,
        )


if __name__ == "__main__":
    main()
