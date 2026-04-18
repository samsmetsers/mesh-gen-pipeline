#!/usr/bin/env bash
# =============================================================================
# setup_all.sh — Full pipeline setup
# =============================================================================
# Installs all dependencies for the complete mesh-gen-pipeline.
#
# Usage:
#   ./scripts/setup_all.sh [--mock-only]
#
# Options:
#   --mock-only     Only install main venv deps (no new models setup).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MOCK_ONLY=0
for arg in "$@"; do
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

    echo "[setup_all] Setting up UniRig …"
    bash "$SCRIPT_DIR/setup_unirig.sh"

    echo "[setup_all] Setting up P3-SAM …"
    bash "$SCRIPT_DIR/setup_p3sam.sh"

    echo "[setup_all] Setting up MotionGPT3 …"
    bash "$SCRIPT_DIR/setup_motiongpt.sh"
else
    echo "[setup_all] Mock-only mode: skipping ML dependencies and model setup."
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Test the full pipeline (mock mode):"
echo "  uv run python main.py --prompt 'Game-ready prehistoric shaman character' --mock"
echo ""
echo "Run for real (requires GPU):"
echo "  uv run python main.py --prompt 'Game-ready prehistoric shaman character' -n shaman"
