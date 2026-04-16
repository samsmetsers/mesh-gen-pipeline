#!/usr/bin/env bash
# =============================================================================
# setup_puppeteer.sh — Set up Puppeteer for Stage 4 auto-rigging
# =============================================================================
# Clones Seed3D/Puppeteer into external/Puppeteer and creates a dedicated
# Python 3.10 venv at .venv_puppeteer/ with required dependencies.
#
# Puppeteer requires Python 3.10.13 with PyTorch 2.1.1 specifically.
# flash-attn is installed from a prebuilt wheel (no compilation required).
# pytorch3d is skipped (not needed by the skeleton/skinning/export stages).
#
# Usage:
#   ./scripts/setup_puppeteer.sh [--skip-clone]
#
# Options:
#   --skip-clone    Skip git clone if external/Puppeteer already exists.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EXTERNAL_DIR="$PROJECT_DIR/external"
PUPPETEER_DIR="$EXTERNAL_DIR/Puppeteer"
VENV="$PROJECT_DIR/.venv_puppeteer"

SKIP_CLONE=0
for arg in "$@"; do
    [[ "$arg" == "--skip-clone" ]] && SKIP_CLONE=1
done

echo "=== Puppeteer Setup ==="
echo "Project:  $PROJECT_DIR"
echo "Target:   $PUPPETEER_DIR"
echo "Venv:     $VENV"

mkdir -p "$EXTERNAL_DIR"

# ── 1. Clone ──────────────────────────────────────────────────────────────────
if [[ $SKIP_CLONE -eq 0 ]]; then
    if [[ -d "$PUPPETEER_DIR" ]]; then
        echo "[setup_puppeteer] Puppeteer already cloned at $PUPPETEER_DIR"
    else
        echo "[setup_puppeteer] Cloning Seed3D/Puppeteer …"
        git clone https://github.com/Seed3D/Puppeteer.git "$PUPPETEER_DIR"
    fi
else
    echo "[setup_puppeteer] Skipping clone (--skip-clone)"
fi

# ── 2. Create Python 3.10 venv ────────────────────────────────────────────────
if [[ ! -d "$VENV" ]]; then
    echo "[setup_puppeteer] Creating Python 3.10 venv at $VENV …"
    uv venv "$VENV" --python 3.10
else
    echo "[setup_puppeteer] Venv already exists at $VENV"
fi

PYTHON="$VENV/bin/python"

# ── 3. Install PyTorch 2.1.1 (specific version required for Puppeteer) ────────
echo "[setup_puppeteer] Installing PyTorch 2.1.1+cu118 …"
uv pip install --python "$PYTHON" \
    "torch==2.1.1+cu118" \
    "torchvision==0.16.1+cu118" \
    --index-url https://download.pytorch.org/whl/cu118

# ── 4. Install Puppeteer requirements ─────────────────────────────────────────
echo "[setup_puppeteer] Installing Puppeteer requirements …"
cd "$PUPPETEER_DIR"
if [[ -f "requirements.txt" ]]; then
    uv pip install --python "$PYTHON" "cython<3.0.0" "numpy<2.0.0" "setuptools" "pybind11"
    uv pip install --python "$PYTHON" --no-build-isolation -r requirements.txt
    uv pip install --python "$PYTHON" "huggingface_hub"
fi

# ── 5. Install torch-scatter (required by skinning/PartField) ─────────────────
echo "[setup_puppeteer] Installing torch-scatter for skinning …"
uv pip install --python "$PYTHON" \
    torch-scatter \
    --find-links "https://data.pyg.org/whl/torch-2.1.1+cu118.html"

# ── 6. Create Michelangelo symlink in skinning/third_partys/ ──────────────────
# skinning_models/models.py imports third_partys.Michelangelo, but Michelangelo
# only lives under skeleton/third_partys/. Create a symlink so it resolves.
MICH_SRC="$PUPPETEER_DIR/skeleton/third_partys/Michelangelo"
MICH_LINK="$PUPPETEER_DIR/skinning/third_partys/Michelangelo"
if [[ -d "$MICH_SRC" ]] && [[ ! -e "$MICH_LINK" ]]; then
    echo "[setup_puppeteer] Symlinking Michelangelo into skinning/third_partys/ …"
    ln -s "$MICH_SRC" "$MICH_LINK"
elif [[ -e "$MICH_LINK" ]]; then
    echo "[setup_puppeteer] Michelangelo symlink already exists"
fi

# ── 7. pytorch3d ─────────────────────────────────────────────────────────────
# pytorch3d is only used by Puppeteer's animation/ module which this pipeline
# does not invoke (we use Blender for animation in Stage 5).  Compiling from
# source requires nvcc matching torch's CUDA (11.8), which fails when the
# system has a newer CUDA toolkit.  Skip it.
echo "[setup_puppeteer] Skipping pytorch3d (not used by skeleton/skinning/export stages)."

# ── 8. Install flash-attn (prebuilt wheel for cu118 + torch 2.1 + py310) ──────
# Compiling from source requires nvcc version to match torch's CUDA (11.8),
# but the system may have a newer nvcc.  Use the official prebuilt wheel instead.
echo "[setup_puppeteer] Installing flash-attn 2.6.3 (prebuilt wheel, cu118+torch2.1) …"
uv pip install --python "$PYTHON" \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3%2Bcu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

# ── 9. Download Puppeteer checkpoints ────────────────────────────────────────
CKPT_DIR="$PUPPETEER_DIR/skeleton_ckpts"
CKPT_FILE="$CKPT_DIR/puppeteer_skeleton_w_diverse_pose.pth"
mkdir -p "$CKPT_DIR"

if [[ -f "$CKPT_FILE" ]]; then
    echo "[setup_puppeteer] Checkpoint already exists at $CKPT_FILE"
else
    echo "[setup_puppeteer] Downloading Puppeteer checkpoint from HuggingFace …"
    # Model is at: https://huggingface.co/Seed3D/Puppeteer
    "$PYTHON" -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='Seed3D/Puppeteer',
    filename='skeleton_ckpts/puppeteer_skeleton_w_diverse_pose.pth',
    local_dir='$PUPPETEER_DIR',
)
print(f'Downloaded to: {path}')
path2 = hf_hub_download(
    repo_id='Seed3D/Puppeteer',
    filename='skinning_ckpts/puppeteer_skin_w_diverse_pose_depth1.pth',
    local_dir='$PUPPETEER_DIR',
)
print(f'Downloaded to: {path2}')
"
fi

echo ""
echo "=== Puppeteer setup complete ==="
echo ""
echo "NOTE: Michelangelo shapevae-256.ckpt may also be required:"
echo "  Place it at: $PUPPETEER_DIR/third_partys/Michelangelo/checkpoints/"
echo ""
echo "Stage 4 will auto-detect the Puppeteer venv."
