# Mesh Generation & Animation Pipeline

## Project Overview
Autonomous pipeline for generating rigged, animated, game-ready 3D characters from text prompts. Integrates AI models for vision priors, 3D generation, mesh optimization, and automated rigging.

## Environment & Infrastructure

The project uses **uv** for dependency management across isolated virtualenvs.

### Virtual Environments
Located in the project root:
- `.venv`: **Root Environment** (Python 3.13) — Stages 1, 2, 3, and motion synthesis (4c).
- `.venv_unirig`: **Puppeteer** (Python 3.10) — Stage 4b (Humanoid Rigging).
- `.venv_riganything`: **Blender/bpy** (Python 3.13) — Stage 4d (Final Assembly & Rigid Weighting).
- `.venv_PartSAM`: **PartSAM** (Python 3.11) — Stage 4a (optional segmentation, currently skipped).
- `.venv_sampart3d`: **SAMPart3D** (Python 3.10) — Alternative segmentation backend.

---

## Pipeline Architecture

```
Stage 1  .venv          Text → reference image (FLUX)
Stage 2  .venv          Image → raw OBJ/GLB (TRELLIS.2)
Stage 3  .venv          Raw OBJ → repaired GLB (PyMeshLab)
Stage 4a skipped        Props handled via rigid weighting in 4d
Stage 4b .venv_unirig   Refined GLB → rigged_body.fbx + joints.json (Puppeteer)
Stage 4c .venv          joints.json → motion NPYs (procedural synthesis)
Stage 4d .venv_rig...   Rig + motions + textures → final FBX/GLB (Blender/bpy)
```

### Stage 4c — Procedural Motion Synthesis (`src/stage4_motion_synthesis.py`)

Character-type routing from prompt text. No external model required.

| Type detected | Animations |
|---------------|------------|
| Biped mage/shaman (staff) | idle, walk, staff-raise attack |
| Biped melee fighter | idle, walk, sword slash |
| Biped archer/ranger | idle, walk, bow aim+shoot |
| Biped generic | idle, walk, upper-body lunge |
| Quadruped | idle, diagonal trot |
| Flying creature | idle (perched), wing-flap |

**Skeleton classification** uses hierarchy subtree propagation — correctly handles Puppeteer rigs where root sits at feet level.

**Motion retargeting** (`src/stage4_assemble_final.py`) applies rotations as `rotation_euler = (ch0, ch1, ch2)` without axis-remapping. Channel 0 = primary bend axis.

---

## Directory Structure
- `src/`: Core pipeline scripts (one per stage).
- `extern/`: Third-party repos — Puppeteer (UniRig), PartSAM, SAMPart3D.
- `scripts/`: Setup, quality check, and validation utilities.
- `output/`: Results by `output_name`, intermediates in `output/{name}/intermediate/`.
- `TRELLIS.2/`: Local copy of the TRELLIS.2 3D generation engine.

---

## How to Run

```bash
# 1. Clone external repos (first time only)
bash scripts/setup_external_repos.sh

# 2. Create all virtual environments
bash scripts/setup_uv_envs.sh

# 3. Run full pipeline
uv run main.py --prompt "A mystic shaman with a staff" --output_name shaman

# Run specific stages
uv run main.py --prompt "A warrior with a sword" --output_name warrior --stage 4

# Quality check
uv run scripts/check_quality.py output/shaman/
```

---

## Maintenance Notes
- **Adding dependencies:** `uv pip install <pkg> --python .<venv>/bin/python`.
- **Debugging stage 4b (Puppeteer):** Check `.venv_unirig` — `setuptools<60` required, `torch-scatter` must match cu118.
- **Debugging stage 4d (assembly):** Uses Blender's `bpy`; run with `.venv_riganything/bin/python`.
- **Animation looks wrong:** Check `_classify_skeleton` output (roles/sides logged to stdout). If joints are misclassified, inspect `joints.json` hierarchy.
- **CUDA architecture:** Default `TORCH_CUDA_ARCH_LIST=8.6` (RTX 30xx). Export before running setup for other GPUs.
