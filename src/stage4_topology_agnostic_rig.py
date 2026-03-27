"""
Stage 4b: Topology-Agnostic Rigging (Hybrid: Puppeteer & AnyTop)
Hybrid rigging approach based on character morphology.

- Puppeteer: NeurIPS 2025 Spotlight. Optimized for humanoid rigging on chaotic AI meshes.
- AnyTop: Treats skeleton as a DAG for diverse creature topologies.
"""
import os
import sys
import subprocess
import argparse
import json
import trimesh
import numpy as np
import shutil
from datetime import datetime

def run_puppeteer_engine(input_mesh, output_dir):
    """
    Execute Puppeteer inference for humanoid characters.
    1. Skeleton prediction
    2. Skinning prediction
    3. Export to FBX
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    puppeteer_dir = os.path.join(project_root, "extern", "Puppeteer")
    
    puppeteer_python = os.path.join(project_root, ".venv_unirig", "bin", "python")
    if not os.path.isfile(puppeteer_python): puppeteer_python = sys.executable

    print(f"  Executing Puppeteer Humanoid Engine...")
    
    # 1. Setup temporary workspace
    tmp_work = os.path.join(output_dir, "puppeteer_tmp")
    os.makedirs(tmp_work, exist_ok=True)
    tmp_mesh_dir = os.path.join(tmp_work, "mesh")
    os.makedirs(tmp_mesh_dir, exist_ok=True)
    
    # Puppeteer expects .obj
    base_name = "input"
    mesh_obj = os.path.join(tmp_mesh_dir, f"{base_name}.obj")
    mesh = trimesh.load(input_mesh, process=False)
    if isinstance(mesh, trimesh.Scene):
        geoms = list(mesh.geometry.values())
        mesh = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]
    mesh.export(mesh_obj)

    try:
        # 2. Skeleton Generation
        print("    Generating skeleton...")
        skel_dir = os.path.join(puppeteer_dir, "skeleton")
        subprocess.run([
            puppeteer_python, "demo.py",
            "--input_dir", os.path.abspath(tmp_mesh_dir),
            "--pretrained_weights", "skeleton_ckpts/puppeteer_skeleton_w_diverse_pose.pth",
            "--output_dir", os.path.abspath(tmp_work),
            "--save_name", "skel_results",
            "--input_pc_num", "8192",
            "--joint_token", "--seq_shuffle"
        ], cwd=skel_dir, check=True)

        # 3. Prepare for Skinning
        skel_file = os.path.join(tmp_work, "skel_results", f"{base_name}_pred.txt")
        skel_target_dir = os.path.join(tmp_work, "skeletons")
        os.makedirs(skel_target_dir, exist_ok=True)
        shutil.copy(skel_file, os.path.join(skel_target_dir, f"{base_name}.txt"))

        # 4. Skinning Weight Prediction
        print("    Generating skinning weights...")
        skin_dir = os.path.join(puppeteer_dir, "skinning")
        subprocess.run([
            puppeteer_python, "-m", "torch.distributed.run", "--nproc_per_node=1", "--master_port=10011",
            "main.py",
            "--num_workers", "1", "--batch_size", "1", "--generate", "--save_skin_npy",
            "--pretrained_weights", "skinning_ckpts/puppeteer_skin_w_diverse_pose_depth1.pth",
            "--input_skel_folder", os.path.abspath(skel_target_dir),
            "--mesh_folder", os.path.abspath(tmp_mesh_dir),
            "--post_filter", "--depth", "1",
            "--save_folder", os.path.abspath(os.path.join(tmp_work, "skin_results"))
        ], cwd=skin_dir, check=True)

        # 5. Export to FBX
        print("    Exporting rigged character...")
        final_rig_txt = os.path.join(tmp_work, "skin_results", "generate", f"{base_name}_skin.txt")
        output_fbx = os.path.join(output_dir, "rigged_body.fbx")
        
        # Use riganything env which has bpy
        bpy_python = os.path.join(project_root, ".venv_riganything", "bin", "python")
        if not os.path.isfile(bpy_python): bpy_python = sys.executable

        # Puppeteer's export.py has a bug where it defaults to 4 groups per vertex
        # even if fewer are predicted. We'll patch it on the fly or call it carefully.
        # Actually, let's just make sure the environment is correct.
        
        subprocess.run([
            bpy_python, os.path.abspath(os.path.join(puppeteer_dir, "export.py")),
            "--mesh", os.path.abspath(mesh_obj),
            "--rig", os.path.abspath(final_rig_txt),
            "--output", os.path.abspath(output_fbx)
        ], cwd=puppeteer_dir, check=True)

        return output_fbx
    except Exception as e:
        print(f"  Puppeteer failure: {e}")
        return None

def run_anytop_rig_engine(input_mesh, output_dir):
    """Placeholder for AnyTop creature rigging."""
    return None

def extract_joints_from_rig(rig_path, joints_json):
    """Extract joint positions and hierarchy from a rigged file (FBX/GLB)."""
    print(f"  Extracting joints from {rig_path}...")
    bpy_python = find_bpy_python()
    
    # Correct extension handling
    is_glb = rig_path.lower().endswith(('.glb', '.gltf'))
    import_cmd = f'bpy.ops.import_scene.gltf(filepath="{os.path.abspath(rig_path)}")' if is_glb else f'bpy.ops.import_scene.fbx(filepath="{os.path.abspath(rig_path)}")'

    script = f"""
import bpy
import json
import os

bpy.ops.wm.read_factory_settings(use_empty=True)
{import_cmd}

armature = next((o for o in bpy.data.objects if o.type == 'ARMATURE'), None)
if not armature:
    print("No armature found")
    exit(1)

data = {{
    "joints": {{}},
    "hierarchy": {{}},
    "root": None
}}

for bone in armature.data.bones:
    data["joints"][bone.name] = [float(c) for c in bone.head_local]
    if bone.parent:
        data["hierarchy"].setdefault(bone.parent.name, []).append(bone.name)
    else:
        data["root"] = bone.name

with open("{os.path.abspath(joints_json)}", "w") as f:
    json.dump(data, f)
"""
    tmp_script = "extract_joints_tmp.py"
    with open(tmp_script, "w") as f: f.write(script)
    
    try:
        subprocess.run([bpy_python, tmp_script], check=True, capture_output=True)
        print(f"  Joints extracted to {joints_json}")
    except subprocess.CalledProcessError as e:
        print(f"  Joint extraction failed: {e.stderr.decode()}")
    finally:
        if os.path.exists(tmp_script): os.remove(tmp_script)
def find_bpy_python():
    """Finds a Python interpreter with bpy installed."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(project_root, ".venv_riganything", "bin", "python"),
        os.path.join(project_root, ".venv_unirig", "bin", "python"),
        sys.executable
    ]
    for py in candidates:
        if os.path.isfile(py):
            try:
                subprocess.run([py, "-c", "import bpy"], capture_output=True, check=True)
                return py
            except: pass
    return sys.executable

def _is_skeleton_degenerate(joints_data):
    """Returns (is_degenerate, reason_str).

    A skeleton is degenerate when Puppeteer technically succeeds but produces
    an unusable rig — e.g. a pure spine chain with no arms or legs.
    Two checks:
      1. Too few joints (< 10).
      2. Poor lateral spread: the second-largest axis range is < 20% of height,
         meaning all joints are nearly collinear (no arms/legs branching out).
    """
    joints = joints_data.get("joints", {})
    if len(joints) < 10:
        return True, f"too few joints ({len(joints)} < 10)"
    positions = np.array(list(joints.values()))
    ranges = positions.max(0) - positions.min(0)
    height = float(ranges.max())
    lateral = float(np.sort(ranges)[-2])   # second-largest spread
    if height > 0 and lateral / height < 0.20:
        return True, f"poor lateral spread ({lateral / height:.2f} < 0.20)"
    return False, ""


def topology_agnostic_rig(body_obj_path, output_dir, strategy=None):
    """Main implementation of Step 2."""
    os.makedirs(output_dir, exist_ok=True)

    mesh = trimesh.load(body_obj_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        geoms = list(mesh.geometry.values())
        mesh = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]

    if strategy is None:
        if mesh.extents[1] > mesh.extents[0] * 0.9:
            strategy = "Puppeteer"
        else:
            strategy = "AnyTop"

    print(f"--- {strategy}: Topology-Agnostic Rigging ---")

    rig_path = None
    if strategy == "Puppeteer":
        rig_path = run_puppeteer_engine(body_obj_path, output_dir)

    if rig_path is None:
        print(f"  Attempting AnyTop fallback...")
        rig_path = run_anytop_rig_engine(body_obj_path, output_dir)

    if rig_path:
        joints_json = os.path.join(output_dir, "joints.json")
        extract_joints_from_rig(rig_path, joints_json)

        # Validate skeleton quality — Puppeteer can succeed but produce a
        # degenerate spine-only rig (e.g. 5 joints, no arms/legs).
        try:
            with open(joints_json) as f:
                jdata = json.load(f)
            degenerate, reason = _is_skeleton_degenerate(jdata)
        except Exception:
            degenerate, reason = False, ""

        if not degenerate:
            return rig_path, joints_json

        print(f"  Skeleton is degenerate ({reason}) — falling through to heuristic.")

    # Last resort: generate a heuristic T-pose skeleton from the mesh bounds
    print("  Building heuristic T-pose skeleton.")
    joints_json = os.path.join(output_dir, "joints.json")
    output_fbx  = os.path.join(output_dir, "rigged_body.fbx")
    _generate_heuristic_skeleton(mesh, body_obj_path, joints_json, output_fbx)
    fbx_ok = os.path.exists(output_fbx)
    return (output_fbx if fbx_ok else None), joints_json


def _generate_heuristic_skeleton(mesh, mesh_source_path, joints_json, output_fbx):
    """Build a humanoid T-pose skeleton from the mesh bounding box.

    Writes joints.json and, via a bpy subprocess, rigged_body.fbx with
    automatic heat-diffusion skin weights so the assembly stage has a
    real deformable rig to work with.
    """
    mn, mx = mesh.bounds
    cx = (mn[0] + mx[0]) / 2
    by = mn[1]   # bottom y
    ty = mx[1]   # top y
    h  = ty - by

    joints = {
        "hips":          [cx,           by + h * 0.53, 0.0],
        "spine":         [cx,           by + h * 0.62, 0.0],
        "chest":         [cx,           by + h * 0.72, 0.0],
        "neck":          [cx,           by + h * 0.83, 0.0],
        "head":          [cx,           by + h * 0.92, 0.0],
        "left_shoulder": [cx - h*0.18,  by + h * 0.72, 0.0],
        "left_arm":      [cx - h*0.30,  by + h * 0.65, 0.0],
        "left_forearm":  [cx - h*0.40,  by + h * 0.55, 0.0],
        "left_hand":     [cx - h*0.46,  by + h * 0.45, 0.0],
        "right_shoulder":[cx + h*0.18,  by + h * 0.72, 0.0],
        "right_arm":     [cx + h*0.30,  by + h * 0.65, 0.0],
        "right_forearm": [cx + h*0.40,  by + h * 0.55, 0.0],
        "right_hand":    [cx + h*0.46,  by + h * 0.45, 0.0],
        "left_upleg":    [cx - h*0.09,  by + h * 0.48, 0.0],
        "left_leg":      [cx - h*0.09,  by + h * 0.28, 0.0],
        "left_foot":     [cx - h*0.09,  by + h * 0.04, 0.0],
        "right_upleg":   [cx + h*0.09,  by + h * 0.48, 0.0],
        "right_leg":     [cx + h*0.09,  by + h * 0.28, 0.0],
        "right_foot":    [cx + h*0.09,  by + h * 0.04, 0.0],
    }
    hierarchy = {
        "hips":           ["spine", "left_upleg", "right_upleg"],
        "spine":          ["chest"],
        "chest":          ["neck", "left_shoulder", "right_shoulder"],
        "neck":           ["head"],
        "left_shoulder":  ["left_arm"],
        "left_arm":       ["left_forearm"],
        "left_forearm":   ["left_hand"],
        "right_shoulder": ["right_arm"],
        "right_arm":      ["right_forearm"],
        "right_forearm":  ["right_hand"],
        "left_upleg":     ["left_leg"],
        "left_leg":       ["left_foot"],
        "right_upleg":    ["right_leg"],
        "right_leg":      ["right_foot"],
    }
    with open(joints_json, "w") as f:
        json.dump({"joints": joints, "hierarchy": hierarchy, "root": "hips"}, f, indent=2)
    print(f"  Heuristic skeleton written to {joints_json}")

    # Also build a rigged FBX so the assembly stage has a deformable mesh
    _build_heuristic_rig_fbx(mesh_source_path, output_fbx)


def _build_heuristic_rig_fbx(mesh_source_path, output_fbx):
    """Use bpy to create a properly weighted heuristic FBX rig.

    The key design decision: bone positions are computed INSIDE the bpy script
    from the mesh's actual Blender-space bounds (Z-up), NOT from the pre-computed
    Y-up positions in joints.json.  This prevents the "skeleton lying flat"
    problem where the heuristic Y-up positions are misinterpreted as Z-up depth
    values by Blender's OBJ importer (which converts Y-up OBJ → Z-up Blender).
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bpy_python = os.path.join(project_root, ".venv_riganything", "bin", "python")
    if not os.path.isfile(bpy_python):
        print("  bpy not found — heuristic FBX generation skipped.")
        return

    # Export mesh to a temporary OBJ (reliable across all Blender importers)
    tmp_obj = output_fbx.replace(".fbx", "_heuristic_mesh.obj")
    try:
        m = trimesh.load(mesh_source_path, process=False)
        if isinstance(m, trimesh.Scene):
            geoms = list(m.geometry.values())
            m = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]
        m.export(tmp_obj)
    except Exception as e:
        print(f"  trimesh mesh pre-export failed ({e}); using source path directly.")
        tmp_obj = mesh_source_path

    bpy_script = r"""
import bpy, sys

bpy.ops.wm.read_factory_settings(use_empty=True)

mesh_path  = @@MESH_PATH@@
output_fbx = @@OUTPUT_FBX@@

# ── Import mesh ───────────────────────────────────────────────────────────────
imported_ok = False
if mesh_path.endswith('.obj'):
    for op in (lambda: bpy.ops.wm.obj_import(filepath=mesh_path),
               lambda: bpy.ops.import_scene.obj(filepath=mesh_path)):
        try:
            op(); imported_ok = True; break
        except Exception: pass
elif mesh_path.endswith(('.glb', '.gltf')):
    try:
        bpy.ops.import_scene.gltf(filepath=mesh_path); imported_ok = True
    except Exception as e:
        print(f"GLTF import failed: {e}")
else:
    try:
        bpy.ops.import_scene.fbx(filepath=mesh_path); imported_ok = True
    except Exception as e:
        print(f"FBX import failed: {e}")

if not imported_ok:
    print("Mesh import failed"); sys.exit(1)

mesh_objs = [o for o in bpy.data.objects if o.type == 'MESH']
if not mesh_objs:
    print("No mesh after import"); sys.exit(1)
bpy.ops.object.select_all(action='DESELECT')
for o in mesh_objs: o.select_set(True)
bpy.context.view_layer.objects.active = mesh_objs[0]
if len(mesh_objs) > 1:
    bpy.ops.object.join()
body_mesh = bpy.context.view_layer.objects.active

# ── Compute heuristic joint positions from the mesh in Blender's world space ──
# Blender uses Z-up after the OBJ importer applies the Y-up→Z-up conversion.
# We measure the mesh here so the skeleton is ALWAYS aligned with the mesh.
coords = [body_mesh.matrix_world @ v.co for v in body_mesh.data.vertices]
xs = [c.x for c in coords]; ys = [c.y for c in coords]; zs = [c.z for c in coords]
cx = (min(xs) + max(xs)) / 2   # lateral center (X axis)
cy = (min(ys) + max(ys)) / 2   # depth center   (Y axis)
bz = min(zs)                   # feet level     (Z axis)
tz = max(zs)                   # head level     (Z axis)
h  = max(tz - bz, 1e-3)

# Joints in Blender Z-up space: X=lateral, Y=depth(center), Z=height
joints = {
    "hips":           (cx,          cy, bz + h * 0.53),
    "spine":          (cx,          cy, bz + h * 0.62),
    "chest":          (cx,          cy, bz + h * 0.72),
    "neck":           (cx,          cy, bz + h * 0.83),
    "head":           (cx,          cy, bz + h * 0.92),
    "left_shoulder":  (cx - h*0.18, cy, bz + h * 0.72),
    "left_arm":       (cx - h*0.30, cy, bz + h * 0.65),
    "left_forearm":   (cx - h*0.40, cy, bz + h * 0.55),
    "left_hand":      (cx - h*0.46, cy, bz + h * 0.45),
    "right_shoulder": (cx + h*0.18, cy, bz + h * 0.72),
    "right_arm":      (cx + h*0.30, cy, bz + h * 0.65),
    "right_forearm":  (cx + h*0.40, cy, bz + h * 0.55),
    "right_hand":     (cx + h*0.46, cy, bz + h * 0.45),
    "left_upleg":     (cx - h*0.09, cy, bz + h * 0.48),
    "left_leg":       (cx - h*0.09, cy, bz + h * 0.28),
    "left_foot":      (cx - h*0.09, cy, bz + h * 0.04),
    "right_upleg":    (cx + h*0.09, cy, bz + h * 0.48),
    "right_leg":      (cx + h*0.09, cy, bz + h * 0.28),
    "right_foot":     (cx + h*0.09, cy, bz + h * 0.04),
}
hierarchy = {
    "hips":           ["spine", "left_upleg", "right_upleg"],
    "spine":          ["chest"],
    "chest":          ["neck", "left_shoulder", "right_shoulder"],
    "neck":           ["head"],
    "left_shoulder":  ["left_arm"],
    "left_arm":       ["left_forearm"],
    "left_forearm":   ["left_hand"],
    "right_shoulder": ["right_arm"],
    "right_arm":      ["right_forearm"],
    "right_forearm":  ["right_hand"],
    "left_upleg":     ["left_leg"],
    "left_leg":       ["left_foot"],
    "right_upleg":    ["right_leg"],
    "right_leg":      ["right_foot"],
}
root_name = "hips"

# ── Build armature ────────────────────────────────────────────────────────────
arm_data = bpy.data.armatures.new('HeuristicArm')
arm_obj  = bpy.data.objects.new('HeuristicArm', arm_data)
bpy.context.collection.objects.link(arm_obj)
bpy.context.view_layer.objects.active = arm_obj
bpy.ops.object.mode_set(mode='EDIT')

bone_map = {}
def add_bone(name, parent_name=None):
    pos      = joints[name]
    children = hierarchy.get(name, [])
    b        = arm_data.edit_bones.new(name)
    b.head   = pos
    if children:
        cp     = joints[children[0]]
        b.tail = cp
        if list(b.head) == list(b.tail):
            b.tail = (pos[0], pos[1], pos[2] + 0.05)
    elif parent_name and parent_name in bone_map:
        pb   = bone_map[parent_name]
        diff = (pos[0]-pb.head.x, pos[1]-pb.head.y, pos[2]-pb.head.z)
        L    = max((diff[0]**2+diff[1]**2+diff[2]**2)**0.5, 0.02)
        b.tail = (pos[0]+diff[0]/L*0.05, pos[1]+diff[1]/L*0.05, pos[2]+diff[2]/L*0.05)
    else:
        b.tail = (pos[0], pos[1], pos[2] + 0.08)
    if parent_name and parent_name in bone_map:
        b.parent      = bone_map[parent_name]
        b.use_connect = False
    bone_map[name] = b
    for ch in children:
        if ch in joints:
            add_bone(ch, name)

add_bone(root_name)
bpy.ops.object.mode_set(mode='OBJECT')
print(f"  Heuristic armature built: {len(bone_map)} bones, "
      f"height={h:.3f}, cx={cx:.3f}, bz={bz:.3f}, tz={tz:.3f}")

# ── Parent with automatic weights ─────────────────────────────────────────────
bpy.ops.object.select_all(action='DESELECT')
body_mesh.select_set(True)
arm_obj.select_set(True)
bpy.context.view_layer.objects.active = arm_obj
try:
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    print("  Auto weights applied.")
except Exception as e:
    print(f"  Auto weights failed ({e}), trying envelope weights.")
    try:
        bpy.ops.object.parent_set(type='ARMATURE_ENVELOPE')
    except Exception as e2:
        print(f"  Envelope weights also failed ({e2}).")

# ── Export FBX ────────────────────────────────────────────────────────────────
bpy.ops.export_scene.fbx(
    filepath=output_fbx,
    use_selection=False,
    add_leaf_bones=False,
    bake_anim=False,
)
print("Heuristic FBX exported:", output_fbx)
"""

    bpy_script = (bpy_script
                  .replace("@@MESH_PATH@@",  repr(os.path.abspath(tmp_obj)))
                  .replace("@@OUTPUT_FBX@@", repr(os.path.abspath(output_fbx))))

    tmp_script = output_fbx.replace(".fbx", "_heuristic_build.py")
    with open(tmp_script, "w") as f:
        f.write(bpy_script)

    try:
        subprocess.run([bpy_python, tmp_script], check=True)
        print(f"  Heuristic rigged FBX written to {output_fbx}")
    except Exception as e:
        print(f"  Heuristic FBX build failed: {e}")
    finally:
        if os.path.exists(tmp_script):
            os.remove(tmp_script)
        if tmp_obj != mesh_source_path and os.path.exists(tmp_obj):
            os.remove(tmp_obj)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    topology_agnostic_rig(args.input, args.output_dir)
