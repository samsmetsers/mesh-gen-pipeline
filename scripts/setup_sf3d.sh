#!/usr/bin/env bash
# scripts/setup_sf3d.sh — Install Stable Fast 3D (image-to-3D backend, Stage 2)
#
# SF3D is the primary Stage 2 backend, replacing TRELLIS.2 which is blocked by
# a CUDA version mismatch (system CUDA 13.2 vs PyTorch cu124).
#
# SF3D advantages over TRELLIS.2:
#   - No compiled CUDA extensions (avoids nvcc version mismatch entirely)
#   - ~6-7 GB VRAM on RTX 3080 (fits 10 GB budget)
#   - Game-ready UV-unwrapped PBR GLB output
#   - ~0.5 s inference time
#
# Requirements: Python venv at .venv (run `uv sync` first), CUDA GPU for inference.
# HuggingFace login required for stabilityai/stable-fast-3d:
#   huggingface-cli login
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SF3D_DIR="$REPO_ROOT/external/stable-fast-3d"
VENV="$REPO_ROOT/.venv"
PIP="$VENV/bin/pip"

echo "=== Setting up Stable Fast 3D ==="

# 1. Clone if not present
if [ ! -d "$SF3D_DIR/.git" ]; then
    echo "[1/5] Cloning Stability-AI/stable-fast-3d …"
    git clone --depth=1 https://github.com/Stability-AI/stable-fast-3d.git "$SF3D_DIR"
else
    echo "[1/5] stable-fast-3d already cloned at $SF3D_DIR"
fi

# 2. Install core requirements (fix gpytoolbox version: 0.2.0 has no wheels)
echo "[2/5] Installing SF3D Python requirements …"
"$PIP" install \
    einops==0.7.0 \
    jaxtyping==0.2.31 \
    omegaconf==2.3.0 \
    "transformers==4.42.3" \
    open_clip_torch==2.24.0 \
    "trimesh[all]==4.4.1" \
    "numpy==1.26.4" \
    "huggingface-hub>=0.23.4" \
    "rembg[gpu]==2.0.57" \
    pynanoinstantmeshes==0.0.3 \
    "gpytoolbox==0.3.3"   # 0.2.0 has no pre-built wheels

# 3. Install texture_baker (local C++/CUDA package) with CUDA disabled to avoid
#    CUDA version mismatch (system CUDA 13.2 vs PyTorch cu124).
#    Pure C++ baking is ~5% slower but fully functional.
echo "[3/5] Building texture_baker (CPU/C++ mode, USE_CUDA=0) …"
cd "$SF3D_DIR/texture_baker"
USE_CUDA=0 "$PIP" install --no-build-isolation -e .
cd "$REPO_ROOT"

# 4. Install uv_unwrapper (pure C++, no CUDA needed)
echo "[4/5] Building uv_unwrapper …"
cd "$SF3D_DIR/uv_unwrapper"
"$PIP" install --no-build-isolation -e .
cd "$REPO_ROOT"

# 5. Install the sf3d package itself (no setup.py, but needs __init__.py importable)
echo "[5/5] Installing sf3d as editable package …"
"$PIP" install -e "$SF3D_DIR" --no-deps 2>/dev/null || true
# SF3D has no pyproject.toml, so add to PYTHONPATH in .env instead
if ! grep -q "stable-fast-3d" "$REPO_ROOT/.env" 2>/dev/null; then
    echo "PYTHONPATH=\"$SF3D_DIR:\${PYTHONPATH:-}\"" >> "$REPO_ROOT/.env"
    echo "  Added $SF3D_DIR to .env PYTHONPATH"
fi

echo ""
echo "=== SF3D setup complete ==="
echo ""
echo "Next step: download the model weights."
echo "  1. Login to HuggingFace: huggingface-cli login"
echo "  2. Accept model terms at: https://huggingface.co/stabilityai/stable-fast-3d"
echo "  3. Weights download automatically on first run."
echo ""
echo "Test with:"
echo "  uv run python main.py --prompt 'test character' -n test --mock"
