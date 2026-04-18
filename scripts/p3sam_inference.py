import os
import sys
import argparse
import json
import torch
from pathlib import Path
import trimesh

# Add P3-SAM demo dir to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
P3SAM_DIR = PROJECT_ROOT / "external" / "Hunyuan3D-Part" / "P3-SAM"
P3SAM_DEMO_DIR = P3SAM_DIR / "demo"

sys.path.append(str(P3SAM_DEMO_DIR))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ckpt", default=str(P3SAM_DIR / "weights" / "p3sam.safetensors"))
    args = parser.parse_args()

    input_path = str(Path(args.input).resolve())
    output_path = str(Path(args.output).resolve())
    ckpt_path = str(Path(args.ckpt).resolve())

    print(f"[P3-SAM] Segmenting {input_path}")
    
    # Load mesh
    mesh = trimesh.load(input_path, force='mesh')
    
    # We need to change dir to demo dir so P3-SAM can find its internal modules (sys.path.append('..') in auto_mask.py)
    # But only AFTER we have loaded the mesh and resolved everything.
    os.chdir(str(P3SAM_DEMO_DIR))
    from auto_mask import AutoMask

    # Initialize AutoMask
    auto_mask = AutoMask(ckpt_path)
    
    # Predict AABB and face_ids
    # save_path is used for intermediate results
    temp_out = Path(output_path).parent / "p3sam_temp"
    temp_out.mkdir(parents=True, exist_ok=True)
    
    aabb, face_ids, mesh = auto_mask.predict_aabb(
        mesh,
        save_path=str(temp_out),
        point_num=50000,
        prompt_num=100,
        threshold=0.95,
        post_process=0,
        show_info=1,
        prompt_bs=4,
        is_parallel=False,
    )
    
    # face_ids is a numpy array of labels for each face
    # We want to identify "weapon" vs "body"
    # P3-SAM doesn't necessarily label them "weapon". It labels them by ID.
    
    # aabb has shape [num_valid_ids, 2, 3] indexed in sorted unique-id order, excluding -1/-2.
    import numpy as np
    parts = []
    valid_sorted_ids = [int(i) for i in sorted(set(int(v) for v in face_ids)) if int(i) not in (-1, -2)]
    for pos, uid in enumerate(valid_sorted_ids):
        indices = [int(i) for i, val in enumerate(face_ids) if int(val) == uid]
        part_aabb = None
        if pos < len(aabb):
            box = np.asarray(aabb[pos]).reshape(-1).tolist()
            part_aabb = box  # [min_x, min_y, min_z, max_x, max_y, max_z]
        parts.append({
            "id": uid,
            "face_indices": indices,
            "aabb": part_aabb,
        })
    
    output_data = {
        "parts": parts,
        "input_mesh": input_path
    }
    
    os.chdir(str(PROJECT_ROOT))
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # Clean up temp intermediate directory (we only need the masks JSON).
    try:
        import shutil
        if temp_out.exists():
            shutil.rmtree(temp_out)
    except Exception as exc:
        print(f"[P3-SAM] Temp cleanup warning: {exc}")

    print(f"[P3-SAM] Masks saved to {output_path}")

if __name__ == "__main__":
    main()
