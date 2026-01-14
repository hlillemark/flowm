#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./download_ckpts.sh [options]

Options:
  --org ORG                 Hugging Face org/user (default: flowm123)
  --out DIR                 Output root (default: downloaded_checkpoints/hf_models)
  --mnist-repo NAME         Default: mnistworld-models
  --blockworld-repo NAME    Default: blockworld-models
  --revision REV            Optional revision/tag/commit
  --dry-run                 Print what would be downloaded
  --no-model-index          Ignore MODEL_INDEX.json; download all repo files
  --num-workers N            Parallel download workers per repo (default: 8)
  --parallel-repos           Download MNIST + Blockworld repos concurrently
  --conda-env ENV           Run via: conda run -n ENV python ... (default: wmm2)
  --no-conda                Run via: python3 ... (ignore conda)

Notes:
  - If ./.env exists, it will be sourced (exporting HF_TOKEN if you set it).
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load .env if present (best effort). This is useful for private repos (HF_TOKEN).
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

ORG="flowm123"
OUT=""
MNIST_REPO=""
BLOCKWORLD_REPO=""
REVISION=""
DRY_RUN=0
NO_MODEL_INDEX=0
NUM_WORKERS=""
PARALLEL_REPOS=0

USE_CONDA=1
CONDA_ENV="wmm2"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --org)
      ORG="$2"; shift 2 ;;
    --out)
      OUT="$2"; shift 2 ;;
    --mnist-repo)
      MNIST_REPO="$2"; shift 2 ;;
    --blockworld-repo)
      BLOCKWORLD_REPO="$2"; shift 2 ;;
    --revision)
      REVISION="$2"; shift 2 ;;
    --dry-run)
      DRY_RUN=1; shift 1 ;;
    --no-model-index)
      NO_MODEL_INDEX=1; shift 1 ;;
    --num-workers)
      NUM_WORKERS="$2"; shift 2 ;;
    --parallel-repos)
      PARALLEL_REPOS=1; shift 1 ;;
    --conda-env)
      CONDA_ENV="$2"; USE_CONDA=1; shift 2 ;;
    --no-conda)
      USE_CONDA=0; shift 1 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

ARGS=("$REPO_ROOT/utils/download_models_from_hf.py" "--org" "$ORG")

if [[ -n "$OUT" ]]; then
  ARGS+=("--out" "$OUT")
fi
if [[ -n "$MNIST_REPO" ]]; then
  ARGS+=("--mnist-repo" "$MNIST_REPO")
fi
if [[ -n "$BLOCKWORLD_REPO" ]]; then
  ARGS+=("--blockworld-repo" "$BLOCKWORLD_REPO")
fi
if [[ -n "$REVISION" ]]; then
  ARGS+=("--revision" "$REVISION")
fi
if [[ $DRY_RUN -eq 1 ]]; then
  ARGS+=("--dry-run")
fi
if [[ $NO_MODEL_INDEX -eq 1 ]]; then
  ARGS+=("--no-model-index")
fi
if [[ -n "$NUM_WORKERS" ]]; then
  ARGS+=("--num-workers" "$NUM_WORKERS")
fi
if [[ $PARALLEL_REPOS -eq 1 ]]; then
  ARGS+=("--parallel-repos")
fi

cd "$REPO_ROOT"

run_py() {
  if [[ $USE_CONDA -eq 1 ]] && command -v conda >/dev/null 2>&1; then
    conda run -n "$CONDA_ENV" --no-capture-output python "$@"
  else
    python3 "$@"
  fi
}

# If requested, do coarse-grained parallel by launching two separate processes
# (one per repo). This can be faster on high-latency networks.
if [[ $PARALLEL_REPOS -eq 1 ]] && [[ $DRY_RUN -eq 0 ]]; then
  run_py "${ARGS[@]}" --only mnist &
  PID1=$!
  run_py "${ARGS[@]}" --only blockworld &
  PID2=$!
  wait "$PID1"
  wait "$PID2"
else
  run_py "${ARGS[@]}"
fi
