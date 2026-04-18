#!/usr/bin/env bash
# =============================================================================
# setup_trellis.sh — Set up TRELLIS.2 for Stage 2
# =============================================================================
# Clones microsoft/TRELLIS.2 into external/TRELLIS.2 and installs its
# dependencies into the main project venv (.venv, Python 3.13).
#
# Usage:
#   ./scripts/setup_trellis.sh [--skip-clone]
#
# Options:
#   --skip-clone   Skip git clone if external/TRELLIS.2 already exists.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EXTERNAL_DIR="$PROJECT_DIR/external"
TRELLIS_DIR="$EXTERNAL_DIR/TRELLIS.2"
VENV="$PROJECT_DIR/.venv"

SKIP_CLONE=0
for arg in "$@"; do
    [[ "$arg" == "--skip-clone" ]] && SKIP_CLONE=1
done

echo "=== TRELLIS.2 Setup ==="
echo "Project: $PROJECT_DIR"
echo "Target:  $TRELLIS_DIR"

mkdir -p "$EXTERNAL_DIR"

# ── 1. Clone ──────────────────────────────────────────────────────────────────
if [[ $SKIP_CLONE -eq 0 ]]; then
    if [[ -d "$TRELLIS_DIR" ]]; then
        echo "[setup_trellis] TRELLIS.2 already cloned at $TRELLIS_DIR"
    else
        echo "[setup_trellis] Cloning microsoft/TRELLIS.2 …"
        git clone https://github.com/microsoft/TRELLIS.2.git "$TRELLIS_DIR"
    fi
else
    echo "[setup_trellis] Skipping clone (--skip-clone)"
fi

# ── 2. Activate venv (or use uv) ─────────────────────────────────────────────
echo "[setup_trellis] Installing TRELLIS.2 dependencies into main venv …"
cd "$TRELLIS_DIR"

# Install torch + CUDA first (required before other deps)
uv pip install --python "$VENV/bin/python" \
    "torch>=2.5.0" "torchvision>=0.20.0" \
    --index-url https://download.pytorch.org/whl/cu124

# Install build tools first
uv pip install --python "$VENV/bin/python" "ninja" "setuptools" "wheel"

# Install TRELLIS.2 components
echo "[setup_trellis] Installing o-voxel …"
uv pip install --python "$VENV/bin/python" ./o-voxel --no-build-isolation

# Install additional TRELLIS.2 runtime dependencies
uv pip install --python "$VENV/bin/python" \
    "imageio>=2.34" \
    "imageio-ffmpeg" \
    "trimesh>=4.4" \
    "einops" \
    "scipy" \
    "opencv-python-headless" \
    "easydict" \
    "pillow>=10.0" \
    "tqdm" \
    "gradio==6.0.1" \
    "tensorboard" \
    "pandas" \
    "lpips" \
    "zstandard" \
    "kornia" \
    "timm" \
    "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"

# Optional, for flash attention and nvdiffrast you might need to install them manually if required, but let's stick to the necessary ones.


# ── 3. Patch pipeline for 10GB VRAM support ──────────────────────────────────
PIPELINE_FILE="$TRELLIS_DIR/trellis2/pipelines/trellis2_image_to_3d.py"
echo "[setup_trellis] Checking pipeline file for 10GB VRAM patch …"

if [[ -f "$PIPELINE_FILE" ]]; then
    # Check if already patched
    if grep -q "skip_rembg" "$PIPELINE_FILE"; then
        echo "[setup_trellis] Pipeline already patched (skip_rembg found)."
    else
        echo "[setup_trellis] Applying 10GB VRAM patch to pipeline …"
        python3 "$SCRIPT_DIR/patch_trellis_pipeline.py" "$PIPELINE_FILE"
    fi
else
    echo "[setup_trellis] WARNING: Pipeline file not found at $PIPELINE_FILE"
    echo "  Check TRELLIS.2 repo structure — it may have been restructured."
fi

echo ""
echo "=== TRELLIS.2 setup complete ==="
echo "Run Stage 2 with: uv run python -m src.stage2_trellis_wrapper --input ..."
