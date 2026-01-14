#!/usr/bin/env bash
set -e
set -o pipefail

# Example usage: 
# ./setup_flowm_env.sh --download-ssm
# ./setup_flowm_env.sh --cuda13
# ./setup_flowm_env.sh --cuda12.4

DOWNLOAD_SSM=false
CUDA_VERSION=""
for arg in "$@"; do
  case "$arg" in
    --download-ssm)    DOWNLOAD_SSM=true ;;
    --no-download-ssm) DOWNLOAD_SSM=false ;;
    --cuda13) CUDA_VERSION="13.0" ;;
    --cuda12.4) CUDA_VERSION="12.4" ;;
  esac
done

# Default and validate CUDA version flag
if [[ -z "$CUDA_VERSION" ]]; then
  CUDA_VERSION="12.4"
fi
case "$CUDA_VERSION" in
  13.0|12.4) ;;
  *)
    echo "ERROR: Unsupported --cuda-version '$CUDA_VERSION'. Allowed: 13.0 or 12.4. Or don't specify to default to settings for 12.4" >&2
    exit 1
    ;;
esac

echo "Using CUDA version flag: $CUDA_VERSION"

ENV_NAME="flowm_test"


# ---- robust conda initialization (for non-interactive scripts) ----
init_conda() {
  # 0) If conda isn't even on PATH, we can't proceed
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: 'conda' not found on PATH." >&2
    echo "  - If you're on an HPC, load your conda module (e.g., 'module load Mambaforge')." >&2
    echo "  - Otherwise install Miniconda/Mambaforge and reopen your shell." >&2
    exit 1
  fi

  # 1) Preferred: initialize via conda's shell hook (most robust)
  # This avoids hardcoding ~/miniconda3/... paths.
  if eval "$(conda shell.bash hook 2>/dev/null)"; then
    return 0
  fi

  # 2) Fallback: try common conda.sh locations
  local candidates=(
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/mambaforge/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
    "/opt/conda/etc/profile.d/conda.sh"
  )

  for f in "${candidates[@]}"; do
    if [[ -f "$f" ]]; then
      # shellcheck source=/dev/null
      source "$f"
      return 0
    fi
  done

  # 3) Last resort: explain how to fix
  echo "ERROR: Found 'conda' but couldn't initialize it for 'conda activate'." >&2
  echo "Tried: 'eval \$(conda shell.bash hook)' and common conda.sh paths." >&2
  echo "Fix options:" >&2
  echo "  - Run: conda init bash  (then restart your shell)" >&2
  echo "  - Or edit this script to source your conda.sh explicitly." >&2
  exit 1
}
# -------------------------------------------------------------------


# 1. Create the conda environment
echo "Creating conda environment '$ENV_NAME'..."

# source ~/miniconda3/etc/profile.d/conda.sh
init_conda
conda create -n $ENV_NAME python=3.12 -y
conda activate $ENV_NAME

echo "Conda environment '$ENV_NAME' created and activated."

# Note: Additional cluster specific setup may be required.
# For example, on Harvard’s Kempner cluster, uncomment the lines below before running, and comment out the three lines above.

# module load Mambaforge
# module load cuda/12.4.1-fasrc01
# module load gcc/9.5.0-fasrc01
# mamba create -n $ENV_NAME python=3.12 -y
# mamba activate $ENV_NAME

# 2. Install PyTorch with CUDA support

pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128

echo "Installed PyTorch with CUDA support."

# 3. Run pip installs in order (flash attention and others require torch)
pip install -r requirements.txt
echo "Installed base requirements."

# 4. Install PyTorch with CUDA nvcc for Mamba and Conv1d
# conda install -y -c "nvidia/label/cuda-12.9.1" cuda-toolkit=12.9.1 cuda-nvcc
conda install -y -c nvidia cuda-toolkit=12.8 cuda-nvcc=12.8


# 5. Install Flash Attention (requires torch to be installed first)
if [[ "$CUDA_VERSION" == "13.0" ]]; then
  # Do this if cuda version 13
  pip install --no-build-isolation flash_attn==2.7.4.post1
elif [[ "$CUDA_VERSION" == "12.4" ]]; then
  # Do this if cuda version 12.4
  pip install --no-build-isolation flash-attn==2.8.3
else
  echo "ERROR: Unsupported CUDA version for FlashAttention: $CUDA_VERSION" >&2
  exit 1
fi
echo "Installed flash attention requirements."

# 6. Optionally download and install the SSM package
if $DOWNLOAD_SSM; then
    echo "DOWNLOAD_SSM is true, downloading"
    pip install mamba-ssm==2.2.6.post3 causal-conv1d==1.5.3.post1
else
  echo "DOWNLOAD_SSM is false, skipping"
fi

echo "All done! The environment '$ENV_NAME' is ready."
