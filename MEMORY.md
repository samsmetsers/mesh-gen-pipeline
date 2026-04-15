# Project Memory Root

Welcome to the Mesh-Gen-Pipeline Knowledge Graph. This is the central source of truth for all agents (Gemini, Claude, etc.) working on this repository.

NOTE: We only have 1 pipeline. Stage 2 uses FLUX.2-klein-4B (Text→Image) → BiRefNet (rembg) → TRELLIS.2-4B (Image→3D).

## Table of Contents
- [Current State](#current-state)
- [Project Goals](#project-goals)
- [Pipeline Architecture](memory/pipeline/architecture.md)
- [Pipeline Best Practices](memory/pipeline/best_practices.md)
- [Research & Models](memory/research/models.md)
- [Setup Guide](memory/setup/setup_guide.md)

## Current State
- **Status:** All 5 stages implemented and tested (121 tests passing).
- **Date:** 2026-04-15
- **Recent Actions (2026-04-15 tenth pass — Stage 4 robust rigging & axis fix):**
  - **Fixed Stage 4 Axis Alignment:**
    - Root cause: Pre-converting GLB to OBJ via trimesh was losing the Y-to-Z up conversion that Blender's GLTF importer handles natively. This caused characters to be "lying down" (Y-up) in Blender's Z-up world, making the vertical rigger slice them "front-to-back" instead of "head-to-toe".
    - Fix: Removed `_glb_to_obj` pre-conversion. `blender_auto_rig.py` now imports the GLB directly via `bpy.ops.import_scene.gltf`.
  - **Improved Arm/Limb Detection:**
    - Root cause: Centroid-based limb detection was pulled towards large props (staffs, shields).
    - Fix 1: Switched from centroid to **median-based cluster detection**. Median is robust to outliers like staffs sticking out further than the arm.
    - Fix 2: Replaced Z-sorting with **Euclidean distance from shoulder attachment**. This correctly handles bent arms, raised arms, and combat poses where Z-sorting is meaningless.
    - Fix 3: Improved anatomical proportions for neck, head, and feet.
  - **Stage 3 "Holes" fix:**
    - Increased `maxholesize` in `meshing_close_holes` from 30 to 100 to better handle TRELLIS.2's open bases or thin fused parts.
  - **Validation:** Re-ran Shaman example. Skeleton now correctly aligns with the 1.0m height (Z-up), and arms correctly track the limbs despite the staff prop.
- **Earlier (2026-04-15 thirteenth pass — Fixed shininess and smoothness):**
  - **Removed GLB "Shininess":**
    - Root cause: Blender's OBJ importer defaults to a glossy Principled BSDF when PBR properties are missing from the MTL.
    - Fix: Added material overrides in `scripts/obj_to_glb.py` and `scripts/blender_auto_rig.py` to force Roughness to 0.8 and Metallic/Specular to 0.0, restoring the matte "uniform" look of Stage 2.
  - **Improved "Smoothness" and Shading:**
    - Fix 1: Increased Taubin smoothing iterations from 3 to 10 in Stage 3 to better eliminate decimation artifacts.
    - Fix 2: Added `ms.compute_normal_per_vertex()` in Stage 3 to ensure smooth vertex normals across the decimated mesh, avoiding a facetted appearance.
- **Earlier (2026-04-15 twelfth pass — Fixed GLB holes with Blender converter):**
  - **Fixed Stage 3 "Holes" and "Faulty Faces" in GLB:**
    - Root cause: `trimesh.load()` and its GLTF exporter were merging vertices and deleting "degenerate" faces even with `process=False`. This caused geometry mismatches and visible holes in the `refined.glb` compared to the `refined.obj`.
    - Fix: Replaced `trimesh` with a custom Blender-based conversion script (`scripts/obj_to_glb.py`) for the final Stage 3 export. Blender's `wm.obj_import` and `export_scene.gltf` provide a bit-perfect geometry transfer, ensuring the `refined.glb` exactly matches the quality of the `refined.obj`.
    - Added a `trimesh` fallback in Stage 3 to ensure the pipeline remains runnable if Blender's background mode fails.
- **Earlier (2026-04-15 eleventh pass — Fixed Stage 3/4 GLB holes & rigging):**
  - **Fixed Stage 3 "Holes" in GLB:**
    - Root cause: `trimesh.load()` was merging vertices and deleting "degenerate" faces by default when exporting the refined OBJ to GLB.
    - Fix: Added `process=False` to `trimesh.load()` in Stage 3.
  - **Improved Stage 4 Robustness:**
    - Fix 1: Stage 3 now permanently saves the refined OBJ (`<name>_refined.obj`).
    - Fix 2: Stage 4 now prefers the refined OBJ over GLB for rigging. OBJ is geometrically more reliable and imported natively by Blender.
    - Fix 3: Added **Auto-Orientation** to `blender_auto_rig.py`. If a mesh is imported "lying down" (Y-up or X-up from a misaligned source), it is automatically rotated to Z-up before landmark detection.
  - **Test count: 86 → 86** (Contract tests updated for new `Stage3Output.refined_obj_path`).
- **Earlier (2026-04-15 tenth pass — Stage 4 robust rigging & axis fix):**
  - **Fixed Stage 3 to actually hit the target face count and preserve UV/textures:**
    - Root cause: `run_stage3` was loading `raw.obj` (2.3M faces, NO UV) instead of `raw.glb` (986k faces, UV atlas). TRELLIS.2's raw.obj is an untextured geometry dump (`trimesh.Trimesh(..., process=False).export()`).
    - Fix 1: Input preference switched to GLB-first. `_repair_and_decimate` now pre-converts GLB → OBJ+MTL+PNG via trimesh before PyMeshLab (PyMeshLab can't save embedded GLB texture blobs directly).
    - Fix 2: `preserveboundary=False` — UV seams were being classified as boundary edges by QEC, blocking decimation at ~101k instead of 12k.
    - Fix 3: `qualitythr` lowered from 0.3 to 0.05 (pass 1) / 0.01 (pass 2). Two-pass decimation added for cases where pass 1 stalls >20% above target.
    - Fix 4: `subprocess.run(..., timeout=600)` added to Blender call in Stage 4. Without timeout, WSL OOM-kills the session when ARMATURE_AUTO hangs on a dense mesh.
    - **Test count: 81 → 86** (4 new Stage 3 tests + 1 Stage 2 test corrected).
  - **Additional fix (2026-04-15, after first real run):**
    - PyMeshLab `meshing_decimation_quadric_edge_collapse_with_texture` raises "inconsistent tex coordinates" when some faces in the trimesh-converted OBJ lack UV. Added try/except fallback to standard QEC. Standard QEC + `save_textures=True` preserves the UV atlas (just no seam-aware collapses). Decimation still hits target: 962k → 11,999 faces.
    - Cleanup fix: trimesh writes MTL/PNG with material names (`material.mtl`, `material_0.png`), NOT `_s3_in_tmp.*`. Changed cleanup strategy to snapshot directory before conversion and delete all new files immediately after `ms.load_new_mesh()` (not in finally block, which isn't reached on QEC crash).
  - **Full pipeline test (shaman):** All 5 stages completed. Final: 6.9MB GLB, 11,999 faces, 20-joint rig, idle/walk/staff_summon animations.
- **Earlier (seventh pass — Stage 2 Image→3D replaced with TRELLIS.2-4B):**
  - **Replaced Hunyuan3D 2.0 with TRELLIS.2-4B:**
    - WSL crash root cause: 24 parallel CUDA compiler jobs (24 cores × ~1-2GB RAM each) → OOM. Fix: `MAX_JOBS=2`.
    - Built and installed all CUDA extensions: nvdiffrast v0.4.0, CuMesh (patched for CUDA 13.2 CUB API), FlexGEMM, o-voxel (patched pyproject.toml + Eigen submodule).
    - Three distinct CUDA 13.2 CUB breaking changes fixed in CuMesh: `nullptr→(void*)nullptr`, count args type, 4-arg in-place `ExclusiveSum` ambiguity (all converted to 5-arg explicit form).
    - Applied `pipeline_512_only=True` + `skip_rembg=True` patch to `trellis2_image_to_3d.py` via `scripts/patch_trellis_pipeline.py` (fits in 10GB VRAM).
    - `_remove_background()` now uses TRELLIS.2's bundled BiRefNet (was Hunyuan3D's rembg).
    - `_image_to_3d()` now: loads `Trellis2ImageTo3DPipeline`, runs `pipeline.run(image, pipeline_type='512', preprocess_image=False)`, calls `mesh.simplify(16M)`, exports raw OBJ via trimesh, then calls `o_voxel.postprocess.to_glb()` for PBR texture baking.
    - TRELLIS.2 model files stored at `~/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/`.
    - CUDA extensions compiled with `MAX_JOBS=2`: nvdiffrast, CuMesh, FlexGEMM, o-voxel all in `/tmp/trellis_ext/`. Patched CuMesh source at `/tmp/trellis_ext/CuMesh/`.
  - **Earlier (sixth pass):** Replaced SD 3.5 Medium with FLUX.2-klein-4B for Text→Image.
  - **Earlier (fifth pass):** Replaced SDXL Lightning with SD 3.5 Medium (fixed costume description parsing, multi-view bug).
  - **Earlier (fourth pass):** Improved prompts, 5k/12k/30k presets, Taubin smoothing, 3-tuple prompt return.
- **Earlier (third pass):** Stage 3/4 texture preservation, ARMATURE_AUTO+ENVELOPE fallback rigging, semi-voxel prompt wrapper.
- **Earlier (second/first pass):** Initial fixes for Stage 2, Stage 4 numpy/GLTF2 issues.

## Project Goals
Develop a free, state-of-the-art, prompt-to-3D-rigged-game-character pipeline.
- **Hardware:** NVIDIA RTX 3080 (10 GB VRAM)
- **Modularity:** Executable by stage; each stage has `--mock` for GPU-free testing
- **Constraints:**
  1. Low poly count (mobile/standard/high presets: 5k/12k/30k faces)
  2. Rigged with 21-joint skeleton (Puppeteer or heuristic fallback)
  3. Props/weapons fused into character mesh via ComfyUI text-to-3D generation.
  4. Animations inferred from prompt (idle/walk/attack)

## Quick Start
```bash
# Install base deps (no GPU)
uv sync

# Full real setup (GPU required):
./scripts/setup_all.sh

# Test the full pipeline (no GPU, mock mode):
uv run python main.py --prompt "Game-ready prehistoric shaman character wearing tribal furs and a dinosaur skull mask, holding a wooden staff with glowing mushrooms, in a combat stance, low-poly, stylized, mobile-optimized." --mock -n shaman

# Real run (after setup):
uv run python main.py --prompt "..." -n shaman
```
