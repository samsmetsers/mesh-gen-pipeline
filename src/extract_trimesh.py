import os
import argparse
import numpy as np
import trimesh
import sys

# Add UniRig to path to import RawData if needed, but we can also just save the npz directly
# based on the UniRig/src/data/raw_data.py structure.

def extract_mesh(input_path, output_dir, target_count=50000):
    """
    Extracts mesh data using trimesh instead of Blender.
    Saves to the format expected by UniRig (raw_data.npz).
    """
    print(f"Extracting {input_path} using trimesh...")
    
    # Load mesh
    mesh = trimesh.load(input_path)
    
    # UniRig expects a single mesh. If it's a scene, merge it.
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    
    # Simplify if needed
    if len(mesh.faces) > target_count:
        print(f"Simplifying from {len(mesh.faces)} to {target_count} faces using fast_simplification...")
        import fast_simplification
        new_vertices, new_faces = fast_simplification.simplify(mesh.vertices, mesh.faces, target_count=target_count)
        mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)

    # Prepare data for npz
    # Based on UniRig's RawData structure
    data = {
        'vertices': np.array(mesh.vertices, dtype=np.float32),
        'vertex_normals': np.array(mesh.vertex_normals, dtype=np.float32),
        'faces': np.array(mesh.faces, dtype=np.int64),
        'face_normals': np.array(mesh.face_normals, dtype=np.float32),
        # Placeholders for skeleton data (not present in raw mesh)
        'joints': None,
        'tails': None,
        'skin': None,
        'parents': None,
        'names': None,
        'matrix_local': None,
    }
    
    # UniRig expects raw_data.npz to be in a folder named after the asset
    base_mesh_name = os.path.basename(input_path).replace(".glb", "").replace(".obj", "")
    asset_dir = os.path.join(output_dir, base_mesh_name)
    os.makedirs(asset_dir, exist_ok=True)
    
    save_path = os.path.join(asset_dir, "raw_data.npz")
    
    # We must save exactly as np.savez expects for Asset.from_raw
    np.savez(save_path, **data)
    print(f"Extracted data saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--faces_target_count", type=int, default=50000)
    args = parser.parse_args()
    
    extract_mesh(args.input, args.output_dir, args.faces_target_count)
