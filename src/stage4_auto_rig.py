"""
Stage 4: Auto-Rigging
=====================
Primary rigging path: Puppeteer (ML-based, Seed3D/Puppeteer).
Fallback:            Blender heuristic auto-rigger.

Puppeteer pipeline (three sub-stages):
  1. Skeleton generation — SkeletonGPT predicts humanoid joints from a point cloud.
  2. Skinning weights    — SkinningNet assigns vertex weights to predicted joints.
  3. FBX export          — Blender export.py bakes mesh + rig into FBX.

Blender heuristic fallback:
  - Cross-section analysis → landmark detection → procedural 20-bone armature.
  - Used when Puppeteer venv is not installed or inference fails.

Output:
  - <name>_rigged.fbx
  - <name>_final.glb
  - joints.json
  - stage4_output.json
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from src.stage3_mesh_optimization import Stage3Output


class Stage4Output(BaseModel):
    fbx_path: str = Field(description="Path to the rigged FBX file.")
    glb_path: str = Field(description="Path to the rigged GLB file.")
    joints_path: str = Field(description="Path to joints.json skeleton description.")
    joint_count: int = Field(description="Number of joints in the rig.")
    output_name: str = Field(description="Short identifier used for file naming.")
    rigging_method: str = Field(description="Method used: 'puppeteer' or 'blender_auto'.")


_PROJECT_ROOT   = Path(__file__).parent.parent
_PUPPETEER_DIR  = _PROJECT_ROOT / "external" / "Puppeteer"
_PUPPETEER_VENV = _PROJECT_ROOT / ".venv_puppeteer"


def _run_puppeteer(
    input_mesh: str,
    output_fbx: str,
    output_glb: str,
    joints_path: str,
) -> int:
    """
    Run Puppeteer (ML-based rigging) via its dedicated Python 3.10 venv.

    The runner calls puppeteer_blend_export.py which exports BOTH the FBX
    (rigged character for DCC tools) and a GLB (for Stage 5) directly from
    the same Blender session that imported the refined OBJ.  Exporting GLB
    directly — rather than converting FBX→GLB afterwards — preserves:
      • PBR materials (FBX uses lossy Phong; the OBJ/MTL → GLTF path is lossless)
      • Correct winding order (OBJ import is a pure rotation; no scale flip)
      • Correct bone axes (no FBX coordinate re-transform on import)

    Returns the number of joints written to joints_path.
    """
    python_bin = str(_PUPPETEER_VENV / "bin" / "python")
    if not os.path.exists(python_bin):
        raise RuntimeError(
            "Puppeteer venv not found at .venv_puppeteer/. "
            "Run scripts/setup_puppeteer.sh first."
        )
    if not _PUPPETEER_DIR.exists():
        raise RuntimeError(
            f"Puppeteer repo not found at {_PUPPETEER_DIR}. "
            "Run scripts/setup_puppeteer.sh first."
        )

    script_path = _PROJECT_ROOT / "scripts" / "puppeteer_runner.py"

    cmd = [
        python_bin,
        str(script_path),
        "--input",         str(Path(input_mesh).resolve()),
        "--output",        str(Path(output_fbx).resolve()),
        "--output-glb",    str(Path(output_glb).resolve()),
        "--joints",        str(Path(joints_path).resolve()),
        "--puppeteer-dir", str(_PUPPETEER_DIR.resolve()),
    ]

    print(f"[Stage 4] Running Puppeteer: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-3000:], file=sys.stderr)
        raise RuntimeError(f"Puppeteer failed with code {result.returncode}")

    print(result.stdout[-2000:])

    # GLB is produced directly by puppeteer_blend_export.py (no FBX→GLB round-trip).
    # _convert_fbx_to_glb is kept as a fallback for the heuristic Blender path only.
    if not Path(output_glb).exists():
        print("[Stage 4] WARNING: direct GLB export missing, falling back to FBX→GLB conversion.")
        _convert_fbx_to_glb(output_fbx, output_glb)

    if Path(joints_path).exists():
        return len(json.loads(Path(joints_path).read_text()))
    return 0


def _convert_fbx_to_glb(fbx_path: str, glb_path: str) -> None:
    """Helper to convert the rigged FBX back to GLB for Stage 5.

    Uses scripts/fbx_to_glb.py which fixes three issues that the old inline
    one-liner had:
      1. Inverted normals (see-through mesh due to back-face culling).
      2. Flat / faceted shading (shade-smooth not preserved across FBX).
      3. Glistening material (FBX Phong → Principled BSDF maps with low roughness).
    """
    script_path = _PROJECT_ROOT / "scripts" / "fbx_to_glb.py"
    cmd = [
        "blender", "--background",
        "--python", str(script_path),
        "--",
        "--input",  str(Path(fbx_path).resolve()),
        "--output", str(Path(glb_path).resolve()),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"fbx_to_glb failed (code {result.returncode}):\n"
            f"{result.stderr[-2000:]}"
        )


def _run_blender_rigger(
    input_mesh: str,
    output_fbx: str,
    output_glb: str,
    joints_path: str,
) -> int:
    """
    Invoke Blender in background mode to rig the mesh.

    Uses the input OBJ or GLB. Blender's importers handle the Y-to-Z up
    conversion for both formats, ensuring the character is upright.
    Preferring OBJ from Stage 3 avoids "holes" introduced by trimesh's
    GLB export.

    Returns the number of joints.
    """
    script_path = Path(__file__).parent.parent / "scripts" / "blender_auto_rig.py"

    cmd = [
        "blender",
        "--background",
        "--python", str(script_path),
        "--",
        "--input", str(Path(input_mesh).resolve()),
        "--output-fbx", str(Path(output_fbx).resolve()),
        "--output-glb", str(Path(output_glb).resolve()),
        "--joints", str(Path(joints_path).resolve()),
    ]

    # Hard timeout to prevent an unresponsive Blender process from hanging
    # indefinitely and consuming all WSL RAM (bone-heat weighting on dense
    # meshes can stall for hours).  600 s is generous for a 12 k-face mesh;
    # reduce Stage 3 quality preset if this fires on your hardware.
    _BLENDER_TIMEOUT = 600
    print(f"[Stage 4] Running Blender Auto-Rigger (timeout={_BLENDER_TIMEOUT}s): {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_BLENDER_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Blender auto-rigger timed out after {_BLENDER_TIMEOUT} s. "
            "The input mesh may be too dense for ARMATURE_AUTO. "
            "Re-run Stage 3 with --quality mobile (5 k faces) and retry."
        )

    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        raise RuntimeError(f"Blender auto-rigger failed with code {result.returncode}")

    if Path(joints_path).exists():
        return len(json.loads(Path(joints_path).read_text()))
    return 0


def run_stage4(
    stage3_output: Stage3Output,
    output_dir: str = "output",
    mock: bool = False,
) -> Stage4Output:
    """
    Run Stage 4: auto-rig the optimised mesh using Blender.
    """
    name = stage3_output.output_name
    out_root = Path(output_dir) / name
    intermediate = out_root / "intermediate"
    intermediate.mkdir(parents=True, exist_ok=True)

    fbx_path     = str(out_root / f"{name}_rigged.fbx")
    glb_path     = str(out_root / f"{name}_final.glb")
    joints_path  = str(intermediate / "joints.json")

    if mock:
        print("[Stage 4] Mock mode: skipping actual Blender rigging.")
        # Just create dummy files
        Path(fbx_path).write_text("Mock FBX")
        Path(glb_path).write_text("Mock GLB")
        dummy_joints = [{"name": "root", "parent": None, "position": [0,0,0]}]
        Path(joints_path).write_text(json.dumps(dummy_joints))
        return Stage4Output(
            fbx_path=os.path.abspath(fbx_path),
            glb_path=os.path.abspath(glb_path),
            joints_path=os.path.abspath(joints_path),
            joint_count=1,
            output_name=name,
            rigging_method="mock"
        )

    # Prefer the refined OBJ over the GLB for rigging.
    # Why: Stage 3's trimesh-exported GLB can sometimes have holes or flipped
    # normals (especially on complex characters with props). The refined OBJ
    # (saved directly from PyMeshLab) is geometrically more reliable and is
    # imported natively by Blender with correct axis-handling.
    input_mesh = stage3_output.refined_obj_path
    if not Path(input_mesh).exists():
        input_mesh = stage3_output.refined_glb_path

    # Prefer Puppeteer (ML-based) — pose-independent, trained skeleton prediction.
    # Fall back to the Blender heuristic only if Puppeteer is not installed or fails.
    try:
        print("[Stage 4] Attempting Puppeteer rigging …")
        joint_count = _run_puppeteer(
            input_mesh=input_mesh,
            output_fbx=fbx_path,
            output_glb=glb_path,
            joints_path=joints_path,
        )
        rig_method = "puppeteer"
    except Exception as e:
        print(f"[Stage 4] Puppeteer failed ({e}); falling back to Blender heuristic …")
        joint_count = _run_blender_rigger(
            input_mesh=input_mesh,
            output_fbx=fbx_path,
            output_glb=glb_path,
            joints_path=joints_path,
        )
        rig_method = "blender_auto"

    print(f"[Stage 4] Rigging complete: {joint_count} joints generated via {rig_method}.")

    return Stage4Output(
        fbx_path=os.path.abspath(fbx_path),
        glb_path=os.path.abspath(glb_path),
        joints_path=os.path.abspath(joints_path),
        joint_count=joint_count,
        output_name=name,
        rigging_method=rig_method,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 4: Auto-Rigging (Blender)")
    parser.add_argument("--input", "-i", type=str, help="Path to stage3_output.json")
    parser.add_argument("--output-name", "-n", type=str, default="character")
    parser.add_argument("--output-dir", "-o", type=str, default="output")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if args.input:
        data = json.loads(Path(args.input).read_text())
        s3 = Stage3Output(**data)
    else:
        parser.error("--input is required.")

    result = run_stage4(
        s3,
        output_dir=args.output_dir,
        mock=args.mock,
    )

    json_path = Path(args.output_dir) / s3.output_name / "intermediate" / "stage4_output.json"
    json_path.write_text(result.model_dump_json(indent=2))
    print(f"\n[Stage 4] Complete. Output JSON: {json_path}")
    print(result.model_dump_json(indent=2))
