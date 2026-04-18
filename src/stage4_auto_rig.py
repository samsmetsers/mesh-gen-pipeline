"""
Stage 4: Auto-Rigging (Next-Gen Modular Stack)
==============================================
Primary rigging path: UniRig (Autoregressive Transformer) + P3-SAM + Rigodotify

Pipeline (four sub-stages):
  1. Part Segmentation (P3-SAM) - Decomposes mesh to identify props/weapons.
  2. Autoregressive Rigging (UniRig) - Generates topologically valid skeleton and skinning weights. Includes "Grip" bones.
  3. Standardization (Blender headless) - Renames bones, applies Rigodotify for Unity/Godot standards, adds twist bones.
  4. Export - Final FBX and GLB.

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
    rigging_method: str = Field(description="Method used: 'unirig'.")

_PROJECT_ROOT = Path(__file__).parent.parent
_UNIRIG_VENV = _PROJECT_ROOT / ".venv_unirig"
_P3SAM_VENV = _PROJECT_ROOT / ".venv_p3sam"

def _run_p3sam(input_mesh: str, output_masks: str) -> None:
    print(f"[Stage 4] Running P3-SAM segmentation on {input_mesh}")
    python_bin = str(_P3SAM_VENV / "bin" / "python")
    script_path = _PROJECT_ROOT / "scripts" / "p3sam_inference.py"
    
    cmd = [
        python_bin, str(script_path),
        "--input", str(Path(input_mesh).resolve()),
        "--output", str(Path(output_masks).resolve())
    ]
    subprocess.run(cmd, check=True)

def _run_unirig(input_mesh: str, masks_path: str, output_glb: str, joints_path: str) -> int:
    print(f"[Stage 4] Running UniRig autoregressive rigging...")
    python_bin = str(_UNIRIG_VENV / "bin" / "python")
    script_path = _PROJECT_ROOT / "scripts" / "unirig_inference.py"
    
    cmd = [
        python_bin, str(script_path),
        "--input", str(Path(input_mesh).resolve()),
        "--output-glb", str(Path(output_glb).resolve()),
        "--joints-path", str(Path(joints_path).resolve())
    ]
    subprocess.run(cmd, check=True)
    
    with open(joints_path, "r") as f:
        joints = json.load(f)
    return len(joints)

def _run_headless_standardization(input_glb: str, output_fbx: str, output_glb: str,
                                   masks_path: str | None = None,
                                   textured_glb: str | None = None) -> None:
    print(f"[Stage 4] Running Headless Blender Standardization (Rigodotify, Twist Bones, Grip Bones)")
    script_path = _PROJECT_ROOT / "scripts" / "blender_standardize.py"
    cmd = [
        "blender", "--background", "--python", str(script_path), "--",
        "--input", str(Path(input_glb).resolve()),
        "--output-fbx", str(Path(output_fbx).resolve()),
        "--output-glb", str(Path(output_glb).resolve()),
    ]
    if masks_path:
        cmd.extend(["--masks", str(Path(masks_path).resolve())])
    if textured_glb:
        cmd.extend(["--textured-glb", str(Path(textured_glb).resolve())])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Blender standardization failed: {result.stderr}")

def run_stage4(stage3_output: Stage3Output, output_dir: str = "output") -> Stage4Output:
    name = stage3_output.output_name
    out_root = Path(output_dir) / name
    intermediate = out_root / "intermediate"
    intermediate.mkdir(parents=True, exist_ok=True)

    fbx_path = str(out_root / f"{name}_rigged.fbx")
    glb_path = str(out_root / f"{name}_final.glb")
    masks_path = str(intermediate / "masks.json")
    unirig_glb = str(intermediate / f"{name}_unirig.glb")
    joints_path = str(intermediate / "joints.json")

    input_mesh = stage3_output.refined_obj_path
    if not Path(input_mesh).exists():
        input_mesh = stage3_output.refined_glb_path

    _run_p3sam(input_mesh, masks_path)
    joint_count = _run_unirig(input_mesh, masks_path, unirig_glb, joints_path)
    _run_headless_standardization(
        unirig_glb, fbx_path, glb_path, masks_path,
        textured_glb=stage3_output.refined_glb_path,
    )

    print(f"[Stage 4] Rigging complete: {joint_count} joints generated via unirig.")

    return Stage4Output(
        fbx_path=os.path.abspath(fbx_path),
        glb_path=os.path.abspath(glb_path),
        joints_path=os.path.abspath(joints_path),
        joint_count=joint_count,
        output_name=name,
        rigging_method="unirig",
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 4: Auto-Rigging (Next-Gen)")
    parser.add_argument("--input", "-i", type=str, help="Path to stage3_output.json")
    parser.add_argument("--output-name", "-n", type=str, default="character")
    parser.add_argument("--output-dir", "-o", type=str, default="output")
    args = parser.parse_args()

    if args.input:
        data = json.loads(Path(args.input).read_text())
        s3 = Stage3Output(**data)
    else:
        parser.error("--input is required.")

    result = run_stage4(s3, output_dir=args.output_dir)
    json_path = Path(args.output_dir) / s3.output_name / "intermediate" / "stage4_output.json"
    json_path.write_text(result.model_dump_json(indent=2))
    print(f"\n[Stage 4] Complete. Output JSON: {json_path}")
    print(result.model_dump_json(indent=2))
