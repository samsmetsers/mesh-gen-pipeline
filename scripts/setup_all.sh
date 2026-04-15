#!/usr/bin/env bash
# =============================================================================
# setup_all.sh — Full pipeline setup
# =============================================================================
# Installs all dependencies for the complete mesh-gen-pipeline.
#
# Usage:
#   ./scripts/setup_all.sh [--skip-compile] [--mock-only]
#
# Options:
#   --skip-compile  Skip flash-attn / pytorch3d compilation (stages 1-3 only).
#   --mock-only     Only install main venv deps (no Puppeteer setup).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

SKIP_COMPILE=0
MOCK_ONLY=0
for arg in "$@"; do
    [[ "$arg" == "--skip-compile" ]] && SKIP_COMPILE=1
    [[ "$arg" == "--mock-only" ]]    && MOCK_ONLY=1
done

cd "$PROJECT_DIR"

echo "=== Mesh-Gen-Pipeline Full Setup ==="
echo "Project: $PROJECT_DIR"
echo ""

# ── 1. Install base Python deps (pydantic, requests, pytest, etc.) ────────────
echo "[setup_all] Installing base project dependencies …"
uv sync

if [[ $MOCK_ONLY -eq 0 ]]; then
    # ── 2. Install ML/3D deps for Stage 2 & 3 ────────────────────────────────
    echo "[setup_all] Installing PyTorch, Diffusers, Trimesh, PyMeshLab for Stages 2 and 3 …"
    uv pip install --python .venv/bin/python -e ".[all]"

    # Puppeteer for Stage 4
    COMPILE_FLAG=""
    [[ $SKIP_COMPILE -eq 1 ]] && COMPILE_FLAG="--skip-compile"
    echo "[setup_all] Setting up Puppeteer …"
    bash "$SCRIPT_DIR/setup_puppeteer.sh" $COMPILE_FLAG
else
    echo "[setup_all] Mock-only mode: skipping ML dependencies and Puppeteer setup."
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Test the full pipeline (mock mode):"
echo "  uv run python main.py --prompt 'Game-ready prehistoric shaman character' --mock"
echo ""
echo "Run for real (requires GPU):"
echo "  uv run python main.py --prompt 'Game-ready prehistoric shaman character' -n shaman"