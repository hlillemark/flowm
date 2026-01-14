#!/usr/bin/env bash
set -euo pipefail

# Download + unshard datasets from Hugging Face into ./data.
#
# Usage examples:
#   ./download_datasets.sh
#   ./download_datasets.sh --dataset blockworld --configs tex,static --splits validation --limit-tars 50 --download-workers 16 --extract-workers 16
#   ./download_datasets.sh --dataset mnist_world --configs dynamic_po --splits validation_200
#
# Env:
#   HF_TOKEN (optional but recommended for speed/rate limits)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source .env if present (so HF_TOKEN can be set there)
if [[ -f "$ROOT_DIR/.env" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
fi

PYTHON_BIN="python3"

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  PYTHON_BIN="python"
fi

exec "$PYTHON_BIN" "$ROOT_DIR/utils/download_datasets_from_hf.py" "$@"
