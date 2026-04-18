#!/bin/bash
set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -d ".venv_p3sam" ]; then
    uv venv .venv_p3sam --python 3.10
fi
source .venv_p3sam/bin/activate
uv pip install setuptools torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install viser fpsample trimesh numba gradio
cd external/Hunyuan3D-Part/P3-SAM/utils/chamfer3D
python setup.py install || true
cd ../../../../..
mkdir -p external/Hunyuan3D-Part/P3-SAM/weights
if [ ! -f "external/Hunyuan3D-Part/P3-SAM/weights/p3sam.safetensors" ]; then
    # Try the correct URL from HuggingFace
    wget -O external/Hunyuan3D-Part/P3-SAM/weights/p3sam.safetensors https://huggingface.co/tencent/Hunyuan3D-Part/resolve/main/p3sam.safetensors || echo "HuggingFace download failed, please check the URL"
fi
