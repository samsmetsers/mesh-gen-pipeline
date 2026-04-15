"""
Stage 4: Auto-Rigging (Blender Heuristic)
===========================================
Replaces the broken/heavy Puppeteer ML model with a lightweight, robust,
VRAM-free Blender-based auto-rigger.

Process:
  1. Imports the optimized GLB into Blender.
  2. Calculates the bounding box of the mesh to determine proportions.
  3. Procedurally generates a standard 21-bone humanoid Armature scaled
     to the mesh's exact dimensions.
  4. Binds the mesh to the Armature using Blender's Automatic Weights
     (bone heat/envelope skinning).
  5. Exports the rigged character as FBX and GLB.
  6. Saves the skeleton hierarchy as joints.json.

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
from pathlib import Path

from pydantic import BaseModel, Field

from src.stage3_mesh_optimization import Stage3Output


class Stage4Output(BaseModel):
    fbx_path: str = Field(description="Path to the rigged FBX file.")
    glb_path: str = Field(description="Path to the rigged GLB file.")
    joints_path: str = Field(description="Path to joints.json skeleton description.")
    joint_count: int = Field(description="Number of joints in the rig.")
    output_name: str = Field(description="Short identifier used for file naming.")
    rigging_method: str = Field(description="Method used: 'blender_auto'", default="blender_auto")


def _glb_to_obj(glb_path: str, obj_path: str) -> None:
    """
    Convert a GLB mesh to OBJ using trimesh, preserving UV maps and materials.

    Blender's system-Python may lack numpy (required by its GLTF importer),
    so we pre-convert to OBJ which Blender can import natively.

    Exporting the scene directly (rather than concatenating geometries) preserves
    UV texture coordinates and material definitions in the OBJ+MTL output.
    """
    import trimesh  # type: ignore[import]

    scene_or_mesh = trimesh.load(glb_path, force="scene")
    # Export the scene directly — this writes OBJ + MTL + texture image files
    # in the same directory, preserving UV coords and materials.
    # Using concatenate() was discarding visual data; direct export keeps it.
    scene_or_mesh.export(obj_path)

    if hasattr(scene_or_mesh, "geometry"):
        face_count = sum(
            m.faces.shape[0] for m in scene_or_mesh.geometry.values()
            if hasattr(m, "faces")
        )
    else:
        face_count = scene_or_mesh.faces.shape[0]
    print(f"[Stage 4] Converted GLB → OBJ: {obj_path} ({face_count} faces)")


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
    puppeteer_dir: str = "", # kept for backwards signature compatibility but unused
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

    joint_count = _run_blender_rigger(
        input_mesh=input_mesh,
        output_fbx=fbx_path,
        output_glb=glb_path,
        joints_path=joints_path,
    )
    
    print(f"[Stage 4] Rigging complete: {joint_count} joints generated.")

    return Stage4Output(
        fbx_path=os.path.abspath(fbx_path),
        glb_path=os.path.abspath(glb_path),
        joints_path=os.path.abspath(joints_path),
        joint_count=joint_count,
        output_name=name,
        rigging_method="blender_auto",
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
