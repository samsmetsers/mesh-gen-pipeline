#!/usr/bin/env bash
# =============================================================================
# setup_flux.sh — Install FLUX.2-Klein-9B dependencies for Stage 1
# =============================================================================
# Adds torch (CUDA), diffusers, transformers, accelerate, and bitsandbytes
# to the main project venv (.venv, Python 3.13).
#
# On first run, Stage 1 will quantize the FLUX.2-Klein-9B transformer to NF4
# and cache it at .cache/quantized/ (~4.5 GB). Subsequent runs load from cache.
#
# Usage:
#   ./scripts/setup_flux.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV="$PROJECT_DIR/.venv"

echo "=== FLUX.2-Klein-9B Setup ==="
echo "Project: $PROJECT_DIR"
echo "Venv:    $VENV"

# ── Install torch + CUDA ──────────────────────────────────────────────────────
echo "[setup_flux] Installing PyTorch with CUDA 12.4 …"
uv pip install --python "$VENV/bin/python" \
    "torch>=2.5.0" \
    "torchvision>=0.20.0" \
    --index-url https://download.pytorch.org/whl/cu124

# ── Install diffusers / transformers ─────────────────────────────────────────
echo "[setup_flux] Installing diffusers, transformers, accelerate …"
uv pip install --python "$VENV/bin/python" \
    "diffusers>=0.30.0" \
    "transformers>=4.44.0" \
    "accelerate>=0.33.0" \
    "sentencepiece" \
    "pillow>=10.0"

# ── Install bitsandbytes (NF4 quantization) ───────────────────────────────────
echo "[setup_flux] Installing bitsandbytes (NF4 quantization support) …"
uv pip install --python "$VENV/bin/python" "bitsandbytes>=0.43.0"

echo ""
echo "=== FLUX setup complete ==="
echo ""
echo "First run will quantize FLUX.2-Klein-9B transformer to NF4 (~5 min)."
echo "Quantized weights cached at: .cache/quantized/"
echo ""
echo "Run Stage 1: uv run python -m src.stage1_vision_prior 'your prompt' -n name"
