"""
Stage 4a (Alt): SAMPart3D-based 3D Part Segmentation

Uses SAMPart3D to segment an input mesh into body and props parts, preserving
vertex colors in the output GLB files.  Falls back to geometric analysis if
SAMPart3D is unavailable or fails.

Interface:
    from src.stage4_sampart3d_segment import segment_sampart3d
    body_path, props_path = segment_sampart3d(input_mesh_path, output_dir, prompt="character")

Outputs:
    body.glb   — main character body
    props.glb  — prop objects (staff, weapons, etc.) or None if none found
"""
import os
import sys
import shutil
import subprocess
import tempfile
import json
import argparse
import numpy as np


# ─── Python Resolver ──────────────────────────────────────────────────────

def _find_sampart3d_python():
    """Return Python interpreter for SAMPart3D uv venv."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    py = os.path.join(project_root, ".venv_sampart3d", "bin", "python")
    if os.path.isfile(py):
        return py
    return sys.executable  # fallback


# ─── SAMPart3D Inference Runner ───────────────────────────────────────────

def _run_sampart3d_inference(input_mesh_path, sampart3d_dir, sampart3d_python,
                              work_dir, scale=1.0):
    """
    Run SAMPart3D inference via its launch/eval.py entry point.

    SAMPart3D's eval pipeline requires:
      1. Blender-rendered 16 views of the object (render_{nnnn}.webp + depth_{nnnn}.exr)
      2. A meta.json with camera intrinsics, transforms, and mesh normalisation params
      3. A GLB mesh file

    We first render the 16 views headlessly via Blender (using
    tools/blender_render_16views.py), then run the eval.

    Returns path to the results directory on success, or None on failure.
    """
    # ── Convert input mesh to GLB for SAMPart3D ───────────────────────────
    try:
        import trimesh
        mesh = trimesh.load(input_mesh_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            geoms = list(mesh.geometry.values())
            if geoms:
                mesh = trimesh.util.concatenate(geoms)
            else:
                print("  SAMPart3D: empty scene in input mesh.")
                return None
        glb_path = os.path.join(work_dir, "input_mesh.glb")
        mesh.export(glb_path)
    except Exception as e:
        print(f"  SAMPart3D: mesh conversion failed: {e}")
        return None

    # ── Render 16 views via Blender ───────────────────────────────────────
    render_dir = os.path.join(work_dir, "render")
    os.makedirs(render_dir, exist_ok=True)

    blender_script = os.path.join(sampart3d_dir, "tools", "blender_render_16views.py")
    if not os.path.isfile(blender_script):
        print(f"  SAMPart3D: blender render script not found: {blender_script}")
        return None

    blender_bin = shutil.which("blender")
    if not blender_bin:
        print("  SAMPart3D: 'blender' not found in PATH — cannot render views.")
        return None

    render_cmd = [
        blender_bin, "--background", "--python", blender_script,
        "--",
        "--mesh_path",  glb_path,
        "--output_dir", render_dir,
        "--n_views",    "16",
    ]

    print(f"  SAMPart3D: rendering 16 views...")
    try:
        subprocess.run(
            render_cmd,
            cwd=sampart3d_dir,
            check=True,
            capture_output=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        print("  SAMPart3D: blender render timed out.")
        return None
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace") if e.stderr else ""
        print(f"  SAMPart3D: blender render failed: {stderr[-400:]}")
        return None

    # Verify render produced meta.json
    meta_json = os.path.join(render_dir, "meta.json")
    if not os.path.exists(meta_json):
        print(f"  SAMPart3D: blender render did not produce meta.json at {render_dir}")
        return None

    # ── Build a minimal config override for eval ──────────────────────────
    cfg_override_path = os.path.join(work_dir, "eval_cfg_override.py")
    checkpoint_path   = os.path.join(sampart3d_dir, "checkpoints", "model_best.pth")
    base_cfg          = os.path.join(sampart3d_dir, "configs", "sampart3d",
                                     "sampart3d-trainmlp-render16views.py")
    results_dir       = os.path.join(work_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Write a small Python config override that points to our specific object
    cfg_content = f"""
_base_ = ["{base_cfg}"]
weight        = "{checkpoint_path}"
save_path     = "{work_dir}"
data_root     = "{os.path.dirname(render_dir)}"
mesh_root     = "{os.path.dirname(glb_path)}"
oid           = "input_mesh"
label         = ""
val_scales_list = [{scale}]
mesh_voting   = True
"""
    with open(cfg_override_path, "w") as f:
        f.write(cfg_content)

    # ── Run SAMPart3D eval ────────────────────────────────────────────────
    eval_script = os.path.join(sampart3d_dir, "launch", "eval.py")
    eval_cmd = [
        sampart3d_python, eval_script,
        "--config-file", cfg_override_path,
        "--num-gpus", "1",
        "options",
        f"weight={checkpoint_path}",
    ]

    print(f"  SAMPart3D: running eval...")
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = sampart3d_dir + ":" + env.get("PYTHONPATH", "")
        subprocess.run(
            eval_cmd,
            cwd=sampart3d_dir,
            check=True,
            capture_output=True,
            timeout=600,
            env=env,
        )
    except subprocess.TimeoutExpired:
        print("  SAMPart3D: eval timed out.")
        return None
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace") if e.stderr else ""
        print(f"  SAMPart3D: eval failed: {stderr[-600:]}")
        return None

    # Results are saved under {work_dir}/results/model_best/mesh_{scale}.npy
    results_subdir = os.path.join(work_dir, "results", "model_best")
    mesh_label_path = os.path.join(results_subdir, f"mesh_{scale}.npy")
    if not os.path.exists(mesh_label_path):
        print(f"  SAMPart3D: expected result not found: {mesh_label_path}")
        return None

    return {"glb_path": glb_path, "labels_path": mesh_label_path, "mesh": None}


# ─── Post-Processing: Labels → Body / Props ───────────────────────────────

def _labels_to_body_props(mesh, labels, output_dir):
    """
    Given per-face part labels from SAMPart3D, split the mesh into body and props.

    Strategy:
      - The label group corresponding to the largest connected region of faces
        is treated as the body.  Additional groups that are spatially attached or
        large relative to the body are also merged into it.
      - Small, elongated groups (e.g. staff) are classified as props.

    Returns (body_path, props_path) where paths point to .glb files.
    """
    try:
        import trimesh
    except ImportError:
        return None, None

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0]  # ignore -1 (unlabelled)

    if len(unique_labels) == 0:
        return None, None

    # Group faces by label and measure each group's area
    parts = {}
    for lbl in unique_labels:
        face_mask = labels == lbl
        sub_mesh = mesh.submesh([np.where(face_mask)[0]], append=True)
        parts[lbl] = sub_mesh

    # Sort by area descending
    sorted_parts = sorted(parts.items(), key=lambda kv: kv[1].area, reverse=True)
    largest_area = sorted_parts[0][1].area

    body_parts = [sorted_parts[0][1]]
    prop_parts = []

    for lbl, sub in sorted_parts[1:]:
        extents = sub.extents
        if extents.min() < 1e-6:
            aspect = float('inf')
        else:
            aspect = extents.max() / extents.min()

        if sub.area > largest_area * 0.15:
            # Large region → body
            body_parts.append(sub)
        elif aspect > 3.0:
            # Small + elongated → prop (staff, sword…)
            prop_parts.append(sub)
        else:
            # Small + compact → small body detail
            body_parts.append(sub)

    body_mesh = trimesh.util.concatenate(body_parts)
    prop_mesh  = trimesh.util.concatenate(prop_parts) if prop_parts else None

    body_path  = os.path.join(output_dir, "body.glb")
    props_path = os.path.join(output_dir, "props.glb") if prop_mesh else None

    body_mesh.export(body_path)
    print(f"  SAMPart3D: body exported ({len(body_mesh.vertices)} verts) → {body_path}")

    if prop_mesh:
        prop_mesh.export(props_path)
        print(f"  SAMPart3D: props exported ({len(prop_mesh.vertices)} verts) → {props_path}")
    else:
        print("  SAMPart3D: no props detected.")

    return body_path, props_path


# ─── Geometric Fallback ───────────────────────────────────────────────────

def _geometric_fallback(input_mesh_path, output_dir):
    """
    Fallback segmentation using connected component analysis and heuristics.
    Saves body.glb and props.glb (or body.obj / props.obj if trimesh GLB
    export is unavailable).
    """
    print("  Geometric fallback: connected component analysis...")
    try:
        import trimesh
    except ImportError:
        print("  ERROR: trimesh not available.")
        return None, None

    try:
        mesh = trimesh.load(input_mesh_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            geoms = list(mesh.geometry.values())
            if not geoms:
                return None, None
            mesh = trimesh.util.concatenate(geoms)
    except Exception as e:
        print(f"  Geometric fallback: mesh load failed: {e}")
        return None, None

    components = mesh.split(only_watertight=False)
    if not components:
        # Mesh has no disconnected components — export as-is
        body_path = os.path.join(output_dir, "body.glb")
        mesh.export(body_path)
        return body_path, None

    components = sorted(components, key=lambda c: c.area, reverse=True)
    largest_area = components[0].area

    body_parts = [components[0]]
    prop_parts = []

    for comp in components[1:]:
        extents = comp.extents
        aspect = (extents.max() / extents.min()) if extents.min() > 1e-6 else float('inf')
        if comp.area > largest_area * 0.15:
            body_parts.append(comp)
        elif aspect > 3.0:
            prop_parts.append(comp)
        else:
            body_parts.append(comp)

    body_mesh = trimesh.util.concatenate(body_parts)
    prop_mesh  = trimesh.util.concatenate(prop_parts) if prop_parts else None

    body_path  = os.path.join(output_dir, "body.glb")
    props_path = os.path.join(output_dir, "props.glb") if prop_mesh else None

    body_mesh.export(body_path)
    if prop_mesh:
        prop_mesh.export(props_path)

    print(f"  Fallback: body={len(body_mesh.vertices)} verts, "
          f"props={'yes' if prop_mesh else 'no'}")
    return body_path, props_path


# ─── Public Interface ─────────────────────────────────────────────────────

def segment_sampart3d(input_mesh_path, output_dir, prompt="character"):
    """
    Segment input_mesh_path into body and props using SAMPart3D.

    Parameters
    ----------
    input_mesh_path : str
        Path to the input mesh (.obj, .glb, .ply, etc.)
    output_dir : str
        Directory where body.glb and props.glb will be written.
    prompt : str
        Unused for SAMPart3D (kept for API compatibility with segment_partsam).

    Returns
    -------
    (body_path, props_path) : (str, str | None)
        Paths to the exported body and props GLB files.
        props_path is None if no props are detected.
    """
    print("--- SAMPart3D: Part Segmentation ---")
    print(f"  Input: {input_mesh_path}")
    os.makedirs(output_dir, exist_ok=True)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sampart3d_dir = os.path.join(project_root, "extern", "SAMPart3D")

    if not os.path.isdir(sampart3d_dir):
        print(f"  SAMPart3D directory not found: {sampart3d_dir}")
        return _geometric_fallback(input_mesh_path, output_dir)

    checkpoint_path = os.path.join(sampart3d_dir, "checkpoints", "model_best.pth")
    if not os.path.isfile(checkpoint_path):
        print(f"  SAMPart3D checkpoint not found: {checkpoint_path}")
        return _geometric_fallback(input_mesh_path, output_dir)

    sampart3d_python = _find_sampart3d_python()

    # Run inference in a temporary work directory so we don't clutter
    # the SAMPart3D repo with per-run artefacts.
    with tempfile.TemporaryDirectory(prefix="sampart3d_") as work_dir:
        result = _run_sampart3d_inference(
            input_mesh_path, sampart3d_dir, sampart3d_python,
            work_dir, scale=1.0
        )

        if result is None:
            print("  SAMPart3D inference failed — falling back to geometric analysis.")
            return _geometric_fallback(input_mesh_path, output_dir)

        # Load the mesh and the predicted per-face labels
        try:
            import trimesh
            glb_path     = result["glb_path"]
            labels_path  = result["labels_path"]

            mesh   = trimesh.load(glb_path, process=False)
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

            labels = np.load(labels_path)  # int array [n_faces]

            if len(labels) != len(mesh.faces):
                print(f"  SAMPart3D: label count mismatch "
                      f"({len(labels)} labels vs {len(mesh.faces)} faces) — fallback.")
                return _geometric_fallback(input_mesh_path, output_dir)

            return _labels_to_body_props(mesh, labels, output_dir)

        except Exception as e:
            print(f"  SAMPart3D post-processing failed: {e}")
            import traceback; traceback.print_exc()
            return _geometric_fallback(input_mesh_path, output_dir)


# ─── CLI Entry Point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAMPart3D segmentation: splits a mesh into body and props."
    )
    parser.add_argument("--input",      required=True, help="Input mesh path")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--prompt",     default="character",
                        help="Character type hint (unused, kept for API compat)")
    args = parser.parse_args()

    body, props = segment_sampart3d(args.input, args.output_dir, args.prompt)
    print(f"Body:  {body}")
    print(f"Props: {props}")
