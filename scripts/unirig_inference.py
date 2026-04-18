import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
import time
import shutil

# Add UniRig src to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
UNIRIG_DIR = PROJECT_ROOT / "external" / "UniRig"
UNIRIG_PYTHON = str(PROJECT_ROOT / ".venv_unirig" / "bin" / "python")

def run_cmd(cmd, cwd=str(UNIRIG_DIR), check=True):
    print(f"Executing: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(UNIRIG_DIR) + ":" + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=check, cwd=cwd, env=env)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-glb", required=True)
    parser.add_argument("--joints-path", required=True)
    args = parser.parse_args()

    # Step 0: Prepare dedicated workspace
    input_path = Path(args.input).resolve()
    base_name = input_path.stem
    workspace = PROJECT_ROOT / "tmp_unirig_workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    
    # Copy input to workspace to ensure paths are clean
    mesh_in_workspace = workspace / input_path.name
    shutil.copy(input_path, mesh_in_workspace)
    
    # UniRig's dataset_inference directory is often used as default
    dataset_dir = workspace / "dataset"
    dataset_dir.mkdir()
    shutil.copy(mesh_in_workspace, dataset_dir / mesh_in_workspace.name)

    # Step 1: Extract Mesh
    print(f"[UniRig] Extracting mesh from {mesh_in_workspace}")
    current_time = time.strftime("%Y_%m_%d_%H_%M_%S")
    extract_cmd = [
        UNIRIG_PYTHON, "-m", "src.data.extract",
        "--config", "configs/data/quick_inference.yaml",
        "--require_suffix", "obj,fbx,FBX,dae,glb,gltf,vrm",
        "--input", str(mesh_in_workspace),
        "--output_dir", str(workspace),
        "--force_override", "true",
        "--faces_target_count", "50000",
        "--num_runs", "1",
        "--id", "0",
        "--time", current_time
    ]
    # extract.py tends to SIGSEGV at Blender shutdown even though raw_data.npz
    # is written successfully. Verify the artefact exists instead of trusting rc.
    run_cmd(extract_cmd, check=False)
    expected_npz = workspace / base_name / "raw_data.npz"
    if not expected_npz.exists():
        raise RuntimeError(f"UniRig extract failed: {expected_npz} not produced")
    
    # Step 2: Skeleton Prediction
    print("[UniRig] Predicting skeleton...")
    skeleton_task = "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml"
    skeleton_cmd = [
        UNIRIG_PYTHON, "run.py",
        "--task", skeleton_task,
        "--input", str(mesh_in_workspace),
        "--npz_dir", str(workspace),
        "--output_dir", str(workspace)
    ]
    run_cmd(skeleton_cmd)

    # Step 3: Skinning Prediction
    print("[UniRig] Predicting skinning weights...")
    skin_task = "configs/task/quick_inference_unirig_skin.yaml"
    skin_cmd = [
        UNIRIG_PYTHON, "run.py",
        "--task", skin_task,
        "--input", str(mesh_in_workspace),
        "--npz_dir", str(workspace),
        "--output_dir", str(workspace)
    ]
    run_cmd(skin_cmd)

    # Step 4: Merge results
    print("[UniRig] Merging results into final mesh...")
    # UniRig's merge script might need the original mesh as source
    # and the directory containing the results as target.
    # Results are in workspace/base_name/
    # source = rigged FBX from skinning step; target = original mesh geometry.
    skin_fbx = workspace / base_name / "result_fbx.fbx"
    if not skin_fbx.exists():
        raise RuntimeError(f"UniRig skin step did not produce {skin_fbx}")
    merge_cmd = [
        UNIRIG_PYTHON, "-m", "src.inference.merge",
        "--require_suffix", "obj,fbx,FBX,dae,glb,gltf,vrm",
        "--num_runs", "1",
        "--id", "0",
        "--source", str(skin_fbx),
        "--target", str(mesh_in_workspace),
        "--output", str(Path(args.output_glb).resolve()),
    ]
    run_cmd(merge_cmd, check=False)
    if not Path(args.output_glb).exists():
        raise RuntimeError(f"UniRig merge failed: {args.output_glb} not produced")

    # Step 5: Export joints to JSON by reading predict_skeleton.npz directly.
    print("[UniRig] Exporting joints to JSON...")
    import numpy as np
    npz_path = workspace / base_name / "predict_skeleton.npz"
    data = np.load(npz_path, allow_pickle=True)
    joint_pos = data["joints"]
    parents_arr = data["parents"] if "parents" in data.files else None
    names_arr = data["names"] if "names" in data.files else None
    joints = []
    for i in range(len(joint_pos)):
        if names_arr is not None and len(names_arr) == len(joint_pos):
            name = str(names_arr[i])
        else:
            name = f"joint_{i}"
        if parents_arr is not None and len(parents_arr) == len(joint_pos):
            p = parents_arr[i]
            parent = None
            if p is not None and not (isinstance(p, float) and np.isnan(p)):
                try:
                    pi = int(p)
                    if pi >= 0:
                        parent = (str(names_arr[pi]) if names_arr is not None and len(names_arr) == len(joint_pos) else pi)
                except (TypeError, ValueError):
                    parent = None
        else:
            parent = None
        joints.append({
            "name": name,
            "parent": parent,
            "position": [float(x) for x in joint_pos[i]],
        })
    with open(args.joints_path, "w") as f:
        json.dump(joints, f, indent=2)

    # Clean up workspace after successful run.
    try:
        if workspace.exists():
            shutil.rmtree(workspace)
    except Exception as exc:
        print(f"[UniRig] Workspace cleanup warning: {exc}")

    print(f"[UniRig] Complete. GLB: {args.output_glb}, Joints: {args.joints_path}")

if __name__ == "__main__":
    main()
