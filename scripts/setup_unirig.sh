#!/bin/bash
set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -d ".venv_unirig" ]; then
    uv venv .venv_unirig --python 3.11
fi
source .venv_unirig/bin/activate
uv pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Install flash-attn with no-build-isolation if needed, or skip if it's too problematic
uv pip install -r external/UniRig/requirements.txt --no-build-isolation || true
