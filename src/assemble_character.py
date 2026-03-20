import sys
# Add conda env site-packages to path
sys.path.append('/home/samsmetsers/miniconda3/envs/blender_env/lib/python3.12/site-packages')
import bpy
import numpy as np
import os
import argparse
import math
from mathutils import Vector, kdtree
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
        bone.keyframe_insert(data_path="rotation_euler", frame=1)
    # Use NLA to ensure export
    track = armature_obj.animation_data.nla_tracks.new()
    track.name = "TPose"
    track.strips.new("TPose", 1, action)

def create_walk_cycle(armature_obj):
    print("Creating Walking animation...")
    action = bpy.data.actions.new(name="WalkCycle")
    armature_obj.animation_data.action = action
    frames = 40
    for frame in range(1, frames + 1):
        t = (frame - 1) / (frames - 1) * 2 * math.pi
        for bone in armature_obj.pose.bones:
            bone.rotation_mode = 'XYZ'
            name = bone.name.lower()
            rot = [0.0, 0.0, 0.0]
            phase = math.pi if "_l_" in name or "left" in name else 0.0
            if "upperleg" in name or "thigh" in name:
                rot[0] = math.radians(30 * math.sin(t + phase))
            elif "lowerleg" in name or "calf" in name:
                rot[0] = math.radians(40 * (math.sin(t + phase - math.pi/2) * 0.5 + 0.5))
            elif "upperarm" in name:
                rot[0] = math.radians(-25 * math.sin(t + phase))
            elif "lowerarm" in name:
                rot[1] = math.radians(20 * (math.sin(t + phase) * 0.5 + 0.5))
            bone.rotation_euler = tuple(rot)
            bone.keyframe_insert(data_path="rotation_euler", frame=frame)
    track = armature_obj.animation_data.nla_tracks.new()
    track.name = "WalkCycle"
    track.strips.new("WalkCycle", 1, action)

def assemble_rigged_character(mesh_path, skel_npz_path, skin_npz_path, output_path, test_anim=True):
    print(f"Assembling character: {mesh_path}")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # Import Mesh
    if mesh_path.endswith('.glb'):
        bpy.ops.import_scene.gltf(filepath=mesh_path)
    else:
        bpy.ops.wm.obj_import(filepath=mesh_path)
        
    character_obj = next((o for o in bpy.data.objects if o.type == 'MESH'), None)
    if not character_obj: raise ValueError("No mesh found")
    
    # Clean transform
    bpy.context.view_layer.objects.active = character_obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    
    # Load UniRig data
    # skel_npz contains vertices (normalized mesh)
    # skin_npz contains predicted skinning weights for those vertices
    skel_data = np.load(skel_npz_path, allow_pickle=True)
    skin_data = np.load(skin_npz_path, allow_pickle=True)
    
    # Get spatial alignment info
    norm_vertices = skel_data['vertices'] # UniRig space
    world_vertices = np.array([v.co for v in character_obj.data.vertices])
    
    w_min, w_max = world_vertices.min(axis=0), world_vertices.max(axis=0)
    n_min, n_max = norm_vertices.min(axis=0), norm_vertices.max(axis=0)
    w_center, n_center = (w_max + w_min)/2, (n_max + n_min)/2
    w_span, n_span = w_max - w_min, n_max - n_min
    n_height_axis, w_height_axis = np.argmax(n_span), np.argmax(w_span)
    
    scale_factor = np.max(w_span) / np.max(n_span)
    
    def align_to_world(pt):
        p = pt - n_center
        res = np.zeros(3)
        if n_height_axis == 1 and w_height_axis == 2:
            res[0], res[1], res[2] = p[0], -p[2], p[1]
        else: res = p
        return res * scale_factor + w_center

    # Align skeleton to mesh
    joints = np.array([align_to_world(j) for j in skel_data['joints']])
    tails_orig = np.array([align_to_world(t) for t in skel_data['tails']])
    names = [str(n) for n in skel_data['names']]
    parents_raw = skel_data['parents']
    
    children = defaultdict(list)
    for i, p in enumerate(parents_raw):
        if p != -1 and p is not None: children[int(p)].append(i)

    # Create Armature
    bpy.ops.object.armature_add(enter_editmode=True)
    armature_obj = bpy.context.active_object
    armature_obj.name = "Armature"
    
    # Improved visibility
    armature_obj.show_in_front = True
    armature_obj.display_type = 'WIRE'
    armature_obj.data.display_type = 'OCTAHEDRAL'
    
    edit_bones = armature_obj.data.edit_bones
    if 'Bone' in edit_bones: edit_bones.remove(edit_bones['Bone'])
    
    for i in range(len(joints)):
        bone = edit_bones.new(names[i])
        bone.head = tuple(joints[i])
        # Calculate proper tail
        if len(children[i]) == 1: 
            bone.tail = tuple(joints[children[i][0]])
        else: 
            bone.tail = tuple(tails_orig[i])
            
        # Ensure bone is not a "ball"
        direction = np.array(bone.tail) - np.array(bone.head)
        if np.linalg.norm(direction) < 1e-4:
            bone.tail[2] += 0.05 * np.max(w_span)
            
    for i, p in enumerate(parents_raw):
        if p != -1 and p is not None: edit_bones[names[i]].parent = edit_bones[names[int(p)]]
            
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # --- INTERNAL RESKINNING ---
    print("Mapping weights to Blender vertices...")
    # NOTE: In predict_skin.npz, vertices are stored as 'vertices' 
    # BUT in predict_skeleton.npz they are also 'vertices'
    # The error was skin_data['vertices'] missing? Let's check keys
    print(f"Skin NPZ keys: {list(skin_data.keys())}")
    
    # Use vertices from skel_data if missing in skin_data
    if 'vertices' in skin_data:
        sampled_vertices_norm = skin_data['vertices']
    else:
        sampled_vertices_norm = skel_data['vertices']
        
    sampled_vertices_world = np.array([align_to_world(v) for v in sampled_vertices_norm])
    
    kd = kdtree.KDTree(len(sampled_vertices_world))
    for i, v in enumerate(sampled_vertices_world):
        kd.insert(Vector(v), i)
    kd.balance()
    
    # Use 'skin' key for weights
    sampled_skin = skin_data['skin'] # (S, J)
    for i, name in enumerate(names):
        vgroup = character_obj.vertex_groups.new(name=name)
    
    for i, v in enumerate(character_obj.data.vertices):
        hits = kd.find_n(v.co, 3)
        if not hits: continue
        total_dist_inv = sum(1.0 / (h[2] + 1e-6) for h in hits)
        for name_idx, bone_name in enumerate(names):
            final_weight = sum((sampled_skin[h[1], name_idx] / (h[2] + 1e-6)) for h in hits) / total_dist_inv
            if final_weight > 0.01:
                character_obj.vertex_groups[bone_name].add([i], final_weight, 'REPLACE')

    # Link Armature
    for o in bpy.data.objects: o.select_set(False)
    character_obj.select_set(True)
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.parent_set(type='ARMATURE_NAME')
    
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
