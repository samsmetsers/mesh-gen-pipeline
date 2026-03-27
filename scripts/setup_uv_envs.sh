#!/bin/bash
set -e

# setup_uv_envs.sh - Create and populate uv virtualenvs to replace conda

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"

# Ensure uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

cd "$PROJECT_ROOT"

echo "Creating root environment..."
uv venv .venv --python 3.13 --clear
. .venv/bin/activate
uv pip install -r requirements.txt

echo "Installing TRELLIS.2 CUDA extensions into root environment..."
# Build flags for RTX 30xx (SM 8.6); adjust TORCH_CUDA_ARCH_LIST for other GPUs
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export ALLOW_CUDA_VERSION_MISMATCH=1
export MAX_JOBS="${MAX_JOBS:-2}"

# ── FlexGEMM ──────────────────────────────────────────────────────────────────
echo "  Building FlexGEMM..."
uv pip install git+https://github.com/JeffreyXiang/FlexGEMM.git \
    --no-build-isolation --python .venv/bin/python

# ── CuMesh (with CUDA 13 compatibility patches applied automatically) ──────────
echo "  Building CuMesh (patching for CUDA 13 CUB API changes)..."
rm -rf /tmp/_cumesh_build
git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/_cumesh_build --recursive

# Patch 1: CUDA 13 no longer implicitly converts nullptr to void* in templates
find /tmp/_cumesh_build/src -name "*.cu" -o -name "*.h" -o -name "*.cuh" | \
    xargs sed -i 's/nullptr, temp_storage_bytes/(void*)nullptr, temp_storage_bytes/g'

# Patch 2: CUDA 13 CCCL removed implicit 4-arg in-place DeviceScan calls;
#           add explicit d_out = d_in to all in-place ExclusiveSum/InclusiveSum calls
python3 - <<'PYEOF'
fixes = {
    '/tmp/_cumesh_build/src/shared.h': [
        ("        cu_new_ids,\n        N\n    ));",
         "        cu_new_ids,\n        cu_new_ids,\n        N\n    ));"),
    ],
    '/tmp/_cumesh_build/src/clean_up.cu': [
        ("        cu_vertex_is_referenced, V+1\n    ));",
         "        cu_vertex_is_referenced,\n        cu_vertex_is_referenced,\n        V+1\n    ));"),
        ("        cu_loop_bound_loop_ids,\n        E\n    ));",
         "        cu_loop_bound_loop_ids,\n        cu_loop_bound_loop_ids,\n        E\n    ));"),
        ("        cu_new_loop_bound_loop_ids,\n        new_num_loop_boundaries\n    ));",
         "        cu_new_loop_bound_loop_ids,\n        cu_new_loop_bound_loop_ids,\n        new_num_loop_boundaries\n    ));"),
    ],
    '/tmp/_cumesh_build/src/simplify.cu': [
        ("        ctx.vertices_map.ptr, V+1\n    ));",
         "        ctx.vertices_map.ptr,\n        ctx.vertices_map.ptr,\n        V+1\n    ));"),
        ("        ctx.faces_map.ptr, F+1\n    ));",
         "        ctx.faces_map.ptr,\n        ctx.faces_map.ptr,\n        F+1\n    ));"),
    ],
    '/tmp/_cumesh_build/src/connectivity.cu': [
        ("        this->loop_boundaries_offset.ptr,\n        this->num_bound_loops + 1\n    ));",
         "        this->loop_boundaries_offset.ptr,\n        this->loop_boundaries_offset.ptr,\n        this->num_bound_loops + 1\n    ));"),
    ],
    '/tmp/_cumesh_build/src/remesh/svox2vert.cu': [
        ("ExclusiveSum((void*)nullptr, temp_storage_bytes, num_vertices, M + 1);",
         "ExclusiveSum((void*)nullptr, temp_storage_bytes, num_vertices, num_vertices, M + 1);"),
        ("ExclusiveSum(d_temp_storage, temp_storage_bytes, num_vertices, M + 1);",
         "ExclusiveSum(d_temp_storage, temp_storage_bytes, num_vertices, num_vertices, M + 1);"),
    ],
}
for filepath, replacements in fixes.items():
    text = open(filepath).read()
    for old, new in replacements:
        count = text.count(old)
        text = text.replace(old, new)
        print(f"  Patched {count} occurrence(s) in {filepath.split('/')[-1]}")
    open(filepath, 'w').write(text)
print("  All DeviceScan in-place patches applied.")
PYEOF

uv pip install /tmp/_cumesh_build --no-build-isolation --python .venv/bin/python
rm -rf /tmp/_cumesh_build

# ── nvdiffrast ────────────────────────────────────────────────────────────────
echo "  Building nvdiffrast..."
rm -rf /tmp/_nvdiffrast_build
git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/_nvdiffrast_build
uv pip install /tmp/_nvdiffrast_build --no-build-isolation --python .venv/bin/python
rm -rf /tmp/_nvdiffrast_build

# ── o-voxel ───────────────────────────────────────────────────────────────────
# cumesh and flex_gemm already installed above; skip their git re-download
echo "  Building o-voxel..."
uv pip install "$PROJECT_ROOT/TRELLIS.2/o-voxel" \
    --no-build-isolation --no-deps --python .venv/bin/python

# ── flash-attn ────────────────────────────────────────────────────────────────
# Only needed if ATTN_BACKEND is set to flash_attn (pipeline defaults to sdpa).
# Compiling from source is slow (~30-60 min). Set SKIP_FLASH_ATTN=1 to skip.
if [ "${SKIP_FLASH_ATTN:-0}" != "1" ]; then
    echo "  Building flash-attn from source (this may take 30-60 min)..."
    echo "  Set SKIP_FLASH_ATTN=1 to skip (pipeline uses sdpa by default)."
    uv pip install flash-attn --no-build-isolation --python .venv/bin/python || \
        echo "  WARNING: flash-attn build failed. Pipeline will use sdpa backend."
else
    echo "  Skipping flash-attn (SKIP_FLASH_ATTN=1). Pipeline uses sdpa."
fi

echo "Creating PartSAM environment (3.11)..."
uv venv .venv_PartSAM --python 3.11 --clear
. .venv_PartSAM/bin/activate
# From scripts/setup_stage4.sh logic
uv pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install lightning==2.2 h5py yacs trimesh scikit-image loguru boto3 \
            mesh2sdf tetgen pymeshlab plyfile einops libigl polyscope \
            potpourri3d simple_parsing arrgh open3d safetensors \
            hydra-core omegaconf accelerate timm igraph ninja vtk
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu124.html

# ── pointops (required by PartSAM) ────────────────────────────────────────────
# pointops is a CUDA extension from SAMPart3D. PyTorch cu124 was compiled with
# CUDA 12.4, but the system may have CUDA 13.x. Temporarily relax the major-
# version check in torch's cpp_extension so the build proceeds.
echo "  Building pointops (from extern/SAMPart3D/libs/pointops)..."
CPP_EXT="$PROJECT_ROOT/.venv_PartSAM/lib/python3.11/site-packages/torch/utils/cpp_extension.py"
cp "$CPP_EXT" "${CPP_EXT}.bak"
sed -i 's/raise RuntimeError(CUDA_MISMATCH_MESSAGE/warnings.warn(CUDA_MISMATCH_MESSAGE/' "$CPP_EXT"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export MAX_JOBS="${MAX_JOBS:-2}"
uv pip install "$PROJECT_ROOT/extern/SAMPart3D/libs/pointops" \
    --no-build-isolation --python "$PROJECT_ROOT/.venv_PartSAM/bin/python" || \
    echo "  WARNING: pointops build failed. PartSAM will use geometric fallback."

# ── torkit3d (required by PartSAM) ───────────────────────────────────────────
echo "  Building torkit3d (from https://github.com/Jiayuan-Gu/torkit3d)..."
uv pip install git+https://github.com/Jiayuan-Gu/torkit3d.git \
    --no-build-isolation --python "$PROJECT_ROOT/.venv_PartSAM/bin/python" || \
    echo "  WARNING: torkit3d build failed. PartSAM will use geometric fallback."

# ── apex (required by PartSAM for FusedLayerNorm) ────────────────────────────
echo "  Building apex (NVIDIA)..."
uv pip install git+https://github.com/NVIDIA/apex.git \
    --no-build-isolation --python "$PROJECT_ROOT/.venv_PartSAM/bin/python" || \
    echo "  WARNING: apex build failed."

mv "${CPP_EXT}.bak" "$CPP_EXT"

echo "Creating unirig (Puppeteer) environment (3.10)..."
uv venv .venv_unirig --python 3.10 --clear
. .venv_unirig/bin/activate
# Pre-install build and common dependencies
uv pip install setuptools cython==0.29.36 fvcore iopath
# tetgen needs cython and setuptools available at build time
uv pip install tetgen==0.5.2 --no-build-isolation
if [ -f "extern/Puppeteer/requirements.txt" ]; then
    # Filter out tetgen and cython from requirements to avoid re-attempting build isolation
    grep -vE "^(tetgen|cython)" extern/Puppeteer/requirements.txt > unirig_reqs.tmp
    uv pip install -r unirig_reqs.tmp
    rm unirig_reqs.tmp
fi
uv pip install flash-attn==2.6.3 --no-build-isolation || true
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.1+cu118.html || true
# PyTorch3D is hard to install via uv/pip sometimes, might need prebuilt wheel
uv pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt211/download.html || true

echo "Creating riganything (Blender/bpy) environment (3.13)..."
uv venv .venv_riganything --python 3.13 --clear
. .venv_riganything/bin/activate
uv pip install bpy

echo "Creating sampart3d environment (3.10)..."
uv venv .venv_sampart3d --python 3.10 --clear
. .venv_sampart3d/bin/activate
if [ -f "extern/SAMPart3D/requirements.txt" ]; then
    uv pip install -r extern/SAMPart3D/requirements.txt
fi

echo "All uv environments created successfully!"
