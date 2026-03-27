"""
Stage 4a: Automated 3D Decomposition via PartSAM.
Identifies and separates props (weapons, hats, etc.) from the main character body.
"""
import os
import sys
import shutil
import subprocess
import argparse
import trimesh
import numpy as np

def _find_partsam_python():
    """Return Python interpreter for PartSAM uv venv."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    py = os.path.join(project_root, ".venv_PartSAM", "bin", "python")
    if os.path.isfile(py):
        return py
    return "python" # Fallback

def segment_partsam(input_mesh_path, output_dir, prompt="character"):
    print(f"--- PartSAM: Automated 3D Decomposition ---")
    print(f"  Input: {input_mesh_path}")
    
    project_root = os.getcwd()
    partsam_dir = os.path.join(project_root, "extern", "PartSAM")
    partsam_python = _find_partsam_python()
    
    if not os.path.exists(partsam_dir):
        print(f"  Error: PartSAM repository not found at {partsam_dir}")
        return None, None

    # 1. Prepare data directory for PartSAM
    # PartSAM eval_everypart.py reads from dataset.root_dir
    data_dir = os.path.join(partsam_dir, "data_input")
    os.makedirs(data_dir, exist_ok=True)
    
    # Copy input mesh (PartSAM likes .obj or .ply)
    basename = "input_mesh"
    input_ext = os.path.splitext(input_mesh_path)[1]
    work_mesh = os.path.join(data_dir, f"{basename}{input_ext}")
    shutil.copy(input_mesh_path, work_mesh)
    
    # 2. Run PartSAM Headless
    print(f"  Executing PartSAM 'Segment-Every-Part' mode...")
    
    # We use xvfb-run for headless execution if available, otherwise just python
    cmd = []
    if shutil.which("xvfb-run"):
        cmd += ["xvfb-run", "-a"]
    
    cmd += [
        partsam_python, "evaluation/eval_everypart.py",
        f"dataset.root_dir={data_dir}",
        "eval_params.ckpt_path=pretrained/model.safetensors"
    ]
    
    try:
        # Run inside PartSAM dir
        subprocess.run(cmd, cwd=partsam_dir, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"  PartSAM execution failed: {e.stderr.decode()}")
        # Fallback to simple component analysis if PartSAM fails
        return _geometric_fallback(input_mesh_path, output_dir)

    # 3. Find results
    # PartSAM saves to results/{id}.ply
    result_ply = os.path.join(partsam_dir, "results", f"{basename}.ply")
    if not os.path.exists(result_ply):
        print(f"  Error: PartSAM output not found at {result_ply}")
        return _geometric_fallback(input_mesh_path, output_dir)

    # 4. Load result and decompose
    mesh = trimesh.load(result_ply, process=False)
    
    # PartSAM outputs a mesh where parts are distinguished by vertex colors or face labels.
    # If it's a single mesh, we split it.
    # Usually, eval_everypart.py might save multiple parts if modified, 
    # but the default seems to be a colored mesh.
    
    # Split by connected components first, then by color/label within components.
    components = mesh.split(only_watertight=False)
    
    # Group components into Body and Props
    # Logic: Largest continuous segmented structure is Body.
    # Others are Props.
    
    components = sorted(components, key=lambda c: c.area, reverse=True)
    
    if not components:
        print("  No components found in PartSAM output.")
        return _geometric_fallback(input_mesh_path, output_dir)

    body_parts = [components[0]]
    prop_parts = components[1:]
    
    # Heuristic: if a component is very large (>30% of largest), it's likely part of the body (e.g. head/limbs)
    # unless it's spatially separated and elongated.
    final_body_parts = [components[0]]
    final_prop_parts = []
    
    largest_area = components[0].area
    for comp in components[1:]:
        # If it's large relative to the main body, it's probably part of the character
        if comp.area > largest_area * 0.15:
            final_body_parts.append(comp)
        else:
            # Check aspect ratio
            extents = comp.extents
            aspect = max(extents) / (min(extents) + 1e-6)
            # If elongated and small, definitely a prop (like a staff)
            if aspect > 3.0:
                final_prop_parts.append(comp)
            else:
                # Small but roundish? Might be a button or a small detail, keep in body
                final_body_parts.append(comp)

    body_mesh = trimesh.util.concatenate(final_body_parts)
    prop_mesh = trimesh.util.concatenate(final_prop_parts) if final_prop_parts else None

    # 5. Export results
    body_path = os.path.join(output_dir, "body.obj")
    props_path = os.path.join(output_dir, "props.obj")
    
    body_mesh.export(body_path)
    print(f"  Exported body: {body_path} ({len(body_mesh.vertices)} verts)")
    
    if prop_mesh:
        prop_mesh.export(props_path)
        print(f"  Exported props: {props_path} ({len(prop_mesh.vertices)} verts)")
    else:
        # Create empty props.obj or just don't create it?
        # The pipeline expects it might not exist.
        if os.path.exists(props_path): os.remove(props_path)
        print("  No props detected.")

    return body_path, props_path

def _geometric_fallback(input_mesh_path, output_dir):
    """
    Fallback when PartSAM is unavailable.

    Strategy:
    1. Split into connected components.
    2. The largest component always becomes body.
    3. A secondary component is treated as a "prop" only when it is clearly
       isolated (spatially distant from the main body centroid) AND elongated
       (aspect ratio > 2.5, consistent with a staff/sword/rod).
    4. All remaining small/attached pieces are merged back into body — they
       are typically clothing fragments, accessory details, or body geometry
       that was split by UV-island seams during Stage-3 decimation.
    """
    print("  Running geometric fallback...")
    mesh = trimesh.load(input_mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(geoms) if geoms else mesh

    components = mesh.split(only_watertight=False)
    components = sorted(components, key=lambda c: c.area, reverse=True)

    if not components:
        body_path = os.path.join(output_dir, "body.obj")
        mesh.export(body_path)
        return body_path, None

    body_comp = components[0]
    body_centroid = body_comp.centroid
    body_diag = float(np.linalg.norm(body_comp.extents))

    body_parts = [body_comp]
    prop_parts  = []

    for comp in components[1:]:
        extents = comp.extents
        # Aspect ratio: how elongated is this component?
        aspect = float(np.max(extents)) / (float(np.min(extents)) + 1e-6)
        # Spatial isolation: distance from body centroid relative to body size
        dist = float(np.linalg.norm(comp.centroid - body_centroid))
        isolation = dist / (body_diag + 1e-6)

        # Only classify as prop if: large enough to be a real object (not noise),
        # clearly elongated (consistent with staff/weapon), AND spatially separate.
        # Tiny degenerate fragments (< 50 verts) are always merged into body.
        is_large_enough = len(comp.vertices) >= 50
        if is_large_enough and aspect > 2.5 and isolation > 0.25:
            prop_parts.append(comp)
            print(f"    → prop component: verts={len(comp.vertices)} aspect={aspect:.1f} isolation={isolation:.2f}")
        else:
            # Merge back: clothing fragment, body detail, UV-split piece
            body_parts.append(comp)

    body_mesh = trimesh.util.concatenate(body_parts)
    prop_mesh  = trimesh.util.concatenate(prop_parts) if prop_parts else None

    body_path  = os.path.join(output_dir, "body.obj")
    props_path = os.path.join(output_dir, "props.obj")

    body_mesh.export(body_path)
    print(f"  Body: {len(body_mesh.vertices)} verts ({len(body_parts)} components merged)")
    if prop_mesh:
        prop_mesh.export(props_path)
        print(f"  Props: {len(prop_mesh.vertices)} verts ({len(prop_parts)} components)")
    else:
        if os.path.exists(props_path):
            os.remove(props_path)
        print("  No distinct props detected — all fragments merged into body.")
        props_path = None

    return body_path, props_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt", default="character")
    args = parser.parse_args()
    segment_partsam(args.input, args.output_dir, args.prompt)
