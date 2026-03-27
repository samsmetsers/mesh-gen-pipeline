# mesh-gen-pipeline

Automated text-to-3D character pipeline. Takes a text prompt and produces a rigged, animated, game-ready FBX and GLB asset.

```
Prompt → Reference image → Raw 3D mesh → Refined mesh → Rigged + animated character
```

## Pipeline stages

| Stage | Model | Environment | Input → Output |
|-------|-------|-------------|----------------|
| 1 | FLUX (FLUX.2-klein-9B) | `.venv` | Text → reference PNG |
| 2 | TRELLIS.2 (4B) | `.venv` | PNG → raw OBJ + GLB |
| 3 | PyMeshLab | `.venv` | Raw OBJ → repaired + texture-preserved GLB |
| 4a | Skip (Rig-Aware fallback) | `.venv` | N/A (Props handled in Stage 4d) |
| 4b | Puppeteer | `.venv_unirig` | Refined GLB → rigged_body.fbx + joints.json |
| 4c | Procedural motion synthesis | `.venv` | joints.json → walk/idle/attack animations |
| 4d | Blender (bpy) | `.venv_riganything` | Rig + motions + textures + Rigid Weighting → final FBX/GLB |

## Requirements

- Linux (tested on WSL2)
- CUDA-capable GPU (tested on RTX 3080 / SM 8.6)
- CUDA 12+ toolkit installed (`nvcc` on PATH)
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

```bash
# 1. Clone external repositories (TRELLIS.2, Puppeteer, SAMPart3D)
bash scripts/setup_external_repos.sh

# 2. Install all environments and build CUDA extensions
bash scripts/setup_uv_envs.sh
```

This will:
- Create `.venv` (Python 3.13) for stages 1–3 and motion synthesis
- Build and install CUDA extensions: `cumesh`, `o_voxel`, `flex_gemm`, `nvdiffrast`
- Create `.venv_PartSAM`, `.venv_unirig`, `.venv_riganything` for stage 4

> **CUDA 13 note:** The setup script automatically patches CuMesh for CUDA 13 CCCL API changes (CUB DeviceScan in-place call signatures).

> **Flash-attn:** Optional. Pipeline defaults to PyTorch `sdpa` backend. Set `SKIP_FLASH_ATTN=1` to skip the slow flash-attn build.

### GPU architecture

The default `TORCH_CUDA_ARCH_LIST` is `8.6` (RTX 30xx). Change before running setup if needed:

```bash
export TORCH_CUDA_ARCH_LIST="8.9"   # RTX 40xx
bash scripts/setup_uv_envs.sh
```

## Usage

```bash
# Full pipeline (all stages)
uv run main.py --prompt "A warrior elf with a sword" --output_name elf

# Specific stages only
uv run main.py --prompt "A shaman with a wooden staff" --output_name shaman --stage 3 4

# Override face target (default: 10000)
uv run main.py --prompt "A dragon" --output_name dragon --target_faces 8000
```

### Output layout

```
output/
└── shaman/
    ├── shaman_final.fbx          # rigged + animated, game-ready
    ├── shaman_final.glb          # same, GLTF format
    └── intermediate/
        ├── shaman_reference.png  # stage 1 output
        ├── shaman_raw.obj        # stage 2 output
        ├── shaman_raw.glb        # stage 2 output (with UV texture)
        ├── shaman_refined.glb    # stage 3 output (repaired, high-poly)
        ├── rigged_body.fbx       # stage 4b output (Puppeteer)
        ├── joints.json           # stage 4b output (skeleton hierarchy)
        └── motions/              # stage 4c output (animation NPYs)
```

## Models

Models are downloaded automatically from HuggingFace on first run:

| Model | HuggingFace ID | Used in |
|-------|----------------|---------|
| FLUX.2-klein-9B | `black-forest-labs/FLUX.2-klein-9B` | Stage 1 |
| TRELLIS.2-4B | `microsoft/TRELLIS.2-4B` | Stage 2 |
| DINOv3-ViT-L | `facebook/dinov3-vitl16-pretrain-lvd1689m` | Stage 2 |
| BiRefNet (rembg) | `ZhengPeng7/BiRefNet` | Stage 2 |
| Puppeteer skeleton | bundled in `extern/Puppeteer/` | Stage 4b |
| Puppeteer skinning | bundled in `extern/Puppeteer/` | Stage 4b |
| Michelangelo VAE | bundled in `extern/Puppeteer/` | Stage 4b |

## Quality check

```bash
uv run scripts/check_quality.py output/shaman/
```

Checks mesh integrity, material presence, rig validity, and animation data.

## Stage details

### Stage 3 — Mesh Optimization

Runs in the root `.venv`. Two phases:
1. PyMeshLab in-place repair (close holes, merge seam vertices, remove non-manifold geometry).
2. Watertightness check — full reconstruction only if >2% non-manifold edges.

**Decimation is disabled** to preserve maximum geometric detail and texture fidelity from the TRELLIS generation.

### Stage 4a — Segmentation (Skipped)

The pipeline no longer physically separates props into different files. This avoids VRAM-heavy segmentation models and preserves fused geometric seams. Props are instead handled via rigid weighting in Stage 4d.

### Stage 4b — Rigging

Puppeteer predicts a skeleton and skinning weights for humanoid characters. If Puppeteer fails, a heuristic T-pose humanoid skeleton is generated from the mesh bounding box so downstream stages can continue.

### Stage 4c — Motion Synthesis

Fully procedural, no external model required. Detects character type (biped/quadruped/flying) and class (mage/staff, melee, archer) from the text prompt, then generates appropriate animation tracks:

| Character | Animations generated |
|-----------|----------------------|
| Biped mage/shaman | idle, walk, staff raise attack |
| Biped melee fighter | idle, walk, sword slash |
| Biped archer/ranger | idle, walk, bow aim+shoot |
| Biped (generic) | idle, walk, lunge |
| Quadruped | idle, diagonal trot |
| Flying creature | idle (perched), wing-flap locomotion |

### Stage 4d — Assembly & Rigid Weighting

Blender (via `bpy`) imports the rigged FBX and applies:
1. **Rig-Aware Rigid Weighting**: Automatically detects vertices belonging to props (extending past hands) and forces their weights to 100% hand-bone attachment. This treats fused weapons as rigid extensions.
2. **Material Setup**: Prioritizes original UV textures with a fallback to vertex colors.
3. **NLA Animation**: Applies retargeted animation tracks and exports the final asset.

## License

The pipeline orchestration code in this repository is released under the **MIT License** — see [LICENSE](LICENSE).

> **Non-commercial use only.** The complete running pipeline depends on third-party components with additional restrictions. In particular:
>
> - **FLUX.1-dev** ([license](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md)) — non-commercial, research and personal use only.
> - **nvdiffrast** ([license](https://github.com/NVlabs/nvdiffrast/blob/main/LICENSE.txt)) — NVIDIA Source Code License; non-commercial research use only.
> - **PyMeshLab** (GPL-3.0) and **Blender/bpy** (GPL-2.0+) — copyleft; source files that directly import these are subject to their respective GPL licenses.
>
> You are responsible for complying with the licenses of all third-party components you install. See [LICENSE](LICENSE) for the full list.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: No module named 'cumesh'` | CUDA extensions not built | Run `bash scripts/setup_uv_envs.sh` |
| `OSError: libcudart.so.11.0` in stage 4b | `torch-scatter` built for wrong CUDA | `uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.1+cu121.html --python .venv_unirig/bin/python --reinstall` |
| `ModuleNotFoundError: No module named 'pkg_resources'` in stage 4b | `setuptools` too new | `uv pip install "setuptools<60" --python .venv_unirig/bin/python --reinstall` |
| Stage 4b rigging fails, pipeline continues | Puppeteer model incompatibility | Heuristic skeleton fallback is used; output is still generated but without AI-predicted skinning |
| White/grey model in final GLB | No TRELLIS colour source | Re-run from stage 2 to regenerate `shaman_raw.glb` |
| `[SPARSE] Attention backend: flash_attn` then ImportError | flash-attn not installed | Set `ATTN_BACKEND=sdpa` or run with `SKIP_FLASH_ATTN=1` during setup |
