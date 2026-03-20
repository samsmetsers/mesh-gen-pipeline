import sys
# Add conda env site-packages to path (specifically the 3.12 one for blender)
sys.path.append('/home/samsmetsers/miniconda3/envs/blender_env/lib/python3.12/site-packages')
import bpy
import numpy as np
import os
import argparse
import math
from collections import defaultdict

def create_t_pose(armature_obj):
    print("Creating T-Pose animation...")
    if not armature_obj.animation_data:
        armature_obj.animation_data_create()
    
    action = bpy.data.actions.new(name="TPose")
    armature_obj.animation_data.action = action
    
    for bone in armature_obj.pose.bones:
        bone.rotation_mode = 'XYZ'
        bone.rotation_euler = (0, 0, 0)
        bone.location = (0, 0, 0)
        bone.keyframe_insert(data_path="rotation_euler", frame=1)
        bone.keyframe_insert(data_path="location", frame=1)
    
    track = armature_obj.animation_data.nla_tracks.new()
    track.name = "TPose"
    track.strips.new("TPose", 1, action)

def create_walk_cycle(armature_obj):
    print("Creating Walking animation...")
    action = bpy.data.actions.new(name="WalkCycle")
    armature_obj.animation_data.action = action
    
    frames = 40
    for frame in range(1, frames + 1):
        # Time t from 0 to 2pi
        t = (frame - 1) / (frames - 1) * 2 * math.pi
        
        for bone in armature_obj.pose.bones:
            bone.rotation_mode = 'XYZ'
            name = bone.name.lower()
            rot = [0.0, 0.0, 0.0]
            loc = [0.0, 0.0, 0.0]
            
            # Phase shift for left side
            phase = math.pi if "_l_" in name or "left" in name else 0.0
            
            # Very simple procedural walk
            if "upperleg" in name or "thigh" in name:
                rot[0] = math.radians(25 * math.sin(t + phase))
            elif "lowerleg" in name or "calf" in name:
                # Knee bend (only backwards)
                rot[0] = math.radians(35 * (math.sin(t + phase - math.pi/2) * 0.5 + 0.5))
            elif "upperarm" in name:
                rot[0] = math.radians(-20 * math.sin(t + phase))
            elif "lowerarm" in name:
                rot[1] = math.radians(15 * (math.sin(t + phase) * 0.5 + 0.5))
            elif "hips" in name or "pelvis" in name:
                loc[2] = 0.03 * abs(math.sin(2 * t)) # Bounce
                
            bone.rotation_euler = tuple(rot)
            if any(loc): bone.location = tuple(loc)
            
            bone.keyframe_insert(data_path="rotation_euler", frame=frame)
            if any(loc): bone.keyframe_insert(data_path="location", frame=frame)

    track = armature_obj.animation_data.nla_tracks.new()
    track.name = "WalkCycle"
    track.strips.new("WalkCycle", 1, action)

def assemble_rigged_character(mesh_path, skel_npz_path, skin_npz_path, output_path, test_anim=True):
    print(f"Assembling character: {mesh_path}")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # Load mesh
    if mesh_path.endswith('.glb'):
        bpy.ops.import_scene.gltf(filepath=mesh_path)
    elif mesh_path.endswith('.obj'):
        bpy.ops.wm.obj_import(filepath=mesh_path)
    else:
        raise ValueError(f"Unsupported mesh format: {mesh_path}")
        
    character_obj = None
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            character_obj = obj
            break
    
    if not character_obj:
        raise ValueError("No mesh found in imported file")
    
    # Apply transforms
    bpy.context.view_layer.objects.active = character_obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    
    # Load data
    skel_data = np.load(skel_npz_path, allow_pickle=True)
    skin_data = np.load(skin_npz_path, allow_pickle=True)
    
    norm_vertices = skel_data['vertices']
    joints_raw = skel_data['joints']
    tails_raw = skel_data['tails']
    parents_raw = skel_data['parents']
    names_raw = skel_data['names']
    
    # Names should be strings
    names = [str(n) for n in names_raw]
    
    # world_vertices are in Blender space
    world_vertices = np.array([v.co for v in character_obj.data.vertices])
    
    def get_bounds(pts):
        return np.min(pts, axis=0), np.max(pts, axis=0)
    
    w_min, w_max = get_bounds(world_vertices)
    n_min, n_max = get_bounds(norm_vertices)
    
    w_center = (w_max + w_min) / 2
    n_center = (n_max + n_min) / 2
    w_span = w_max - w_min
    n_span = n_max - n_min
    
    n_height_axis = np.argmax(n_span)
    w_height_axis = np.argmax(w_span)
    
    print(f"Detected Height Axis: Normalized={n_height_axis}, World={w_height_axis}")
    
    def align_pt(pt):
        p = pt - n_center
        res = np.zeros(3)
        # Handle coordinate swap if necessary (Y-up to Z-up)
        if n_height_axis == 1 and w_height_axis == 2:
            res[0] = p[0]
            res[1] = -p[2]
            res[2] = p[1]
        else:
            res = p
        res = res * (np.max(w_span) / np.max(n_span))
        res = res + w_center
        return res

    joints = np.array([align_pt(j) for j in joints_raw])
    tails_orig = np.array([align_pt(t) for t in tails_raw])

    # Convert parents to list
    parents = []
    children = defaultdict(list)
    for i, p in enumerate(parents_raw):
        if p == -1 or p is None or (isinstance(p, np.ndarray) and p.size == 0):
            parents.append(None)
        else:
            p_idx = int(p)
            parents.append(p_idx)
            children[p_idx].append(i)
            
    # Fix bone tails to avoid "little balls"
    # A bone should go from joints[i] to its child or its provided tail
    tails = np.zeros_like(joints)
    for i in range(len(joints)):
        if len(children[i]) == 1:
            # Connect to unique child
            tails[i] = joints[children[i][0]]
        else:
            # Use provided tail or small extrusion
            tails[i] = tails_orig[i]
            # If tail is still exactly head, extrude
            if np.allclose(tails[i], joints[i]):
                tails[i][2] += 0.05 * np.max(w_span)

    # High poly skinning data
    high_poly_skin = skin_data['skin']
    
    # Create Armature
    bpy.ops.object.armature_add(enter_editmode=True)
    armature_obj = bpy.context.view_layer.objects.active
    armature_obj.name = "Armature"
    edit_bones = armature_obj.data.edit_bones
    
    if 'Bone' in edit_bones:
        edit_bones.remove(edit_bones['Bone'])
    
    print("Creating bones...")
    for i in range(len(joints)):
        bone = edit_bones.new(names[i])
        bone.head = tuple(joints[i])
        bone.tail = tuple(tails[i])
        # Force a minimum length to avoid "balls"
        direction = np.array(bone.tail) - np.array(bone.head)
        if np.linalg.norm(direction) < 1e-4:
            bone.tail[2] += 0.01
            
    for i in range(len(joints)):
        if parents[i] is not None:
            edit_bones[names[i]].parent = edit_bones[names[parents[i]]]
            
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Parent mesh to armature
    for o in bpy.data.objects: o.select_set(False)
    character_obj.select_set(True)
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.parent_set(type='ARMATURE_NAME')
    
    # Assign weights
    print("Assigning vertex weights...")
    for i, name in enumerate(names):
        vgroup = character_obj.vertex_groups.new(name=name)
        weights = high_poly_skin[:, i]
        non_zero_indices = np.where(weights > 0.001)[0]
        # Optimized: add in chunks if possible, but bpy is slow anyway
        for v_idx in non_zero_indices:
            vgroup.add([int(v_idx)], float(weights[v_idx]), 'REPLACE')
            
    if test_anim:
        create_t_pose(armature_obj)
        create_walk_cycle(armature_obj)

    print(f"Exporting to {output_path}")
    bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLB', export_animations=True)
    print("Assembly complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--skel_npz", required=True)
    parser.add_argument("--skin_npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--test_anim", action="store_true")
    
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
        args = parser.parse_args(argv)
        assemble_rigged_character(args.mesh, args.skel_npz, args.skin_npz, args.output, args.test_anim)
