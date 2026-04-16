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
- **Status:** All 5 stages implemented and tested (176 tests passing). Puppeteer fully verified end-to-end.
- **Date:** 2026-04-16
- **Recent Actions (2026-04-16 twenty-first pass — Direct GLB export + no transform_apply warping):**
  - **Root cause of remaining see-through + material issues**: The FBX→GLB round-trip (`_convert_fbx_to_glb`) was fundamentally lossy — FBX Phong can't represent PBR materials, and the FBX importer's negative-scale coordinate flip corrupted winding even after `transform_apply`.
  - **Fix**: Export GLB directly from `puppeteer_blend_export.py` in the same Blender session that imported the OBJ, bypassing FBX entirely for the GLB output. PBR materials, correct winding order (OBJ import is a pure rotation), and correct bone axes are all preserved.
  - **`scripts/puppeteer_blend_export.py`**: Added `--output-glb` arg; after FBX export, also calls `bpy.ops.export_scene.gltf()` in the same session; normals+shade-smooth+material fixes applied before both exports.
  - **`scripts/puppeteer_runner.py`**: Passes `--output-glb` to the Blender call.
  - **`src/stage4_auto_rig.py`** (`_run_puppeteer`): Passes `--output-glb` to runner; falls back to `_convert_fbx_to_glb` only if direct GLB export is missing (belt-and-suspenders).
  - **`scripts/fbx_to_glb.py`**: Now only used as fallback (heuristic Blender path / if direct export fails).
  - **Mesh warping during animations fixed**: `blender_animate.py`'s `_fix_scene_meshes_and_materials` previously called `bpy.ops.object.transform_apply(rotation=True, scale=True)`. For GLTF-imported skinned meshes this baked the Y→Z rotation into vertex positions while bone positions remained unchanged → vertices moved off bones → warping. Fix: completely removed `transform_apply` from `blender_animate.py`.
  - **176 tests passing** (9 new: `TestPuppeteerBlendExportGlb` × 7 + `TestStage4PuppeteerWiring.test_puppeteer_path_passes_output_glb_to_runner` + `test_no_transform_apply_on_skinned_mesh`)
- **Recent Actions (2026-04-16 twentieth pass — Fix see-through mesh + flexible animations):**
  - **See-through root cause identified**: FBX import applies a negative scale on one axis for Y-up → Z-up coordinate correction. `normals_make_consistent` alone doesn't fix this because the winding is flipped in world space by the scale transform. Fix chain: (1) `transform_apply(scale=True)` bakes negative scale into vertex positions, correcting winding order; (2) `normals_make_consistent(inside=False)` fixes any residual inward faces; (3) `use_backface_culling=False` on all materials → `doubleSided:true` in GLTF as belt-and-suspenders.
  - **`scripts/fbx_to_glb.py`** updated with `transform_apply` + double-sided.
  - **Flexible animation system (`blender_animate.py` rewritten)**:
    - Old system hardcoded heuristic-rigger bone names (`spine_02`, `upper_arm_l`, etc.) → zero keyframes on Puppeteer skeletons.
    - New system: `classify_skeleton(joints)` analyses joint positions and hierarchy geometry to map each joint to a semantic role (`pelvis`, `spine_mid`, `head`, `left_upper_arm`, `right_forearm`, `left_thigh`, etc.) with no name assumptions.
    - All animation builders (`build_idle`, `build_walk`, `build_attack_*`) use semantic roles via `rot(arm, 'role_name', roles, frame, ...)` helper.
    - Verified on shaman: 20/20 roles mapped correctly. `joint14→pelvis`, `joint17→spine_mid`, `joint21→head`, `joint8→left_upper_arm`, `joint25→right_upper_arm`, `joint11→left_thigh`, etc.
  - **167 tests passing** (6 new: `TestFbxToGlbScript.test_applies_transform_before_normals`, `test_sets_doublesided`, `TestBlenderAnimateFixup.test_uses_semantic_roles_not_hardcoded_names`, `test_classify_skeleton_pure_python`, `test_classify_skeleton_maps_shaman`, `test_sets_doublesided`)
- **Recent Actions (2026-04-16 nineteenth pass — Fix see-through / glistening final GLB):**
  - **Root cause (partial):** `_convert_fbx_to_glb()` was a bare inline `--python-expr` — no normals, no shade smooth, no material fix.
  - **`scripts/fbx_to_glb.py`** created: normals + shade smooth + material roughness fix.
  - **HC-Laplacian fix**: called with no arguments (beta param not supported by installed pymeshlab).
  - **161 tests passing**
- **Recent Actions (2026-04-16 eighteenth pass — Proper rigging + smooth meshes):**
  - **34-joint rig achieved** (`rigging_method=puppeteer`, 34 joints, all 3 Puppeteer sub-stages working)
  - **Root causes of 2-joint degenerate skeleton (all fixed):**
    1. `--apply_marching_cubes` was missing from skeleton Stage 1 call. Semi-voxel/blocky meshes produce degenerate point clouds; MC smooths the surface first → 2 joints → 27-34 joints
    2. `flash_attn` not installed (CUDA 13.2 mismatch with torch's CUDA 11.8). Fixed by installing prebuilt wheel: `flash_attn-2.6.3+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl` from `Dao-AILab/flash-attention` GitHub releases
    3. Puppeteer's `export.py` was stripping UV/textures (used `from_pydata()` without UV). Replaced by `scripts/puppeteer_blend_export.py` which uses Blender's OBJ importer (preserves UV+materials) and applies skin weights via position-based vertex mapping
    4. MTL/PNG sidecars not copied alongside OBJ into tmp_puppeteer/examples. Fixed in `puppeteer_runner.py` to copy all `.mtl/.png/.jpg` files
    5. `pytorch3d` compilation skipped (not needed by skeleton/skinning/export; only by animation/ which we don't use)
  - **Mesh smoothing improvements:**
    - Stage 3: Taubin smoothing increased from 10 → 30 iterations; HC-Laplacian smoothing added as second pass (no-param call)
    - `obj_to_glb.py`, `puppeteer_blend_export.py`, `blender_auto_rig.py`: shade-smooth enabled on all exported meshes (stored as smooth vertex normals in GLB)
  - **146 tests passing**
- **Recent Actions (2026-04-15 seventeenth pass — Puppeteer fully working end-to-end):**
  - **Verified Puppeteer runs all 3 sub-stages successfully** (`rigging_method=puppeteer` confirmed in stage4_output.json)
  - **Runtime bugs discovered and fixed (all in external/Puppeteer + scripts):**
    1. `skeletongen.py`: `flash_attn` optional — falls back to `_ATTN_IMPL="eager"` when not compiled
    2. `skeleton_opt.py`: Added eager 4D causal mask via `_prepare_4d_causal_attention_mask` (was `raise ValueError`)
    3. `export.py`: Blender argparse fix — parse `sys.argv` after `--` separator (Blender injects own flags before script args)
    4. `skinning/third_partys/Michelangelo`: Added symlink → `skeleton/third_partys/Michelangelo` (runner auto-creates; setup script creates)
    5. `torch-scatter` added to `.venv_puppeteer` install step in `setup_puppeteer.sh`
    6. `puppeteer_runner.py`: Strip `VIRTUAL_ENV`, `VIRTUAL_ENV_PROMPT`, `PYTHONPATH` from env to prevent `.venv` (Python 3.13) contaminating `.venv_puppeteer` (Python 3.10) torchrun workers
- **Recent Actions (2026-04-15 sixteenth pass — Puppeteer properly wired as primary rigger):**
  - **Fixed Puppeteer Integration (properly this time):**
    - Root cause 1: `run_stage4()` passed `puppeteer_dir=""` (empty string) to `_run_puppeteer()`, so Puppeteer could never find its checkpoints. Fix: added `_PUPPETEER_DIR = _PROJECT_ROOT / "external" / "Puppeteer"` as a module-level constant; removed `puppeteer_dir` parameter entirely from `run_stage4()` and `run_pipeline()`.
    - Root cause 2: `puppeteer_runner.py` looked for skinning checkpoint at `puppeteer_dir/skinning_ckpts/` but the actual location is `puppeteer_dir/skinning/skinning/skinning_ckpts/`. Fix: added `_find_checkpoint()` helper with prioritised candidate paths for both skeleton and skinning.
    - Root cause 3: Runner used system `torchrun` (not in PATH). Fix: derive `torchrun_bin = Path(sys.executable).parent / "torchrun"` from the venv that invokes the script.
    - Root cause 4: Dead validation block in `run_stage4()` calling `scripts/validate_rigging_enhanced.py` (which never existed). Removed.
  - **New tests added (24 new tests → 145 total):**
    - `TestPuppeteerRunner` — structural checks on `puppeteer_runner.py` (venv torchrun path, checkpoint discovery strings, joint_token/seq_shuffle/post_filter flags, cleanup).
    - `TestPuppeteerInstall` — verifies all Puppeteer artefacts on disk (venv, torchrun, both checkpoints, Michelangelo ckpt, script files).
    - `TestStage4PuppeteerWiring` — verifies `run_stage4` signature has no `puppeteer_dir` param, auto-detect constants present, Puppeteer attempted before Blender fallback.
  - **Cleaned up:**
    - Removed `--puppeteer-dir` CLI argument from `main.py` (auto-detected at module level now).
    - Removed dead `validate_rigging_enhanced.py` invocation.
- **Earlier (2026-04-15 fourteenth pass — Robust Agnostic Rigging):**
  - **Improved Rigging Heuristic Accuracy:**
    - Fix 1: Added **Character Centering**. The mesh is now automatically centered at X=0, Y=0 before analysis. This ensures the spine and pelvis are perfectly vertical and symmetrical.
    - Fix 2: Switched to **Density-Based Limb Detection**. Instead of simple spatial slices, the rigger now uses median-based clusters and 3D distance analysis to find the "core" of the limbs, making it much more robust to large props (staffs, shields) and non-standard poses.
    - Fix 3: Refined **Humanoid Proportions**. Adjusted bone heights and shoulder attachment points to better match common humanoid archetypes while remaining agnostic to the specific character design.
    - Fix 4: Improved **Symmetry Handling**. Legs are now forced to be symmetrical around the character's center, preventing "leg drift" caused by asymmetric accessories.
  - **Updated Rigging Tests:**
    - Adjusted `test_stage4_auto_rig.py` to reflect the new centering logic and tighter symmetry requirements.
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
