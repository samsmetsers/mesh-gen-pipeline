"""
Retarget SMPL 22-joint motion (from MotionGPT3) onto a UniRig skeleton.

Input motion format:
  `.npy` file from MotionGPT3, shape (T, 22, 3) or (1, T, 22, 3), containing
  SMPL joint *world positions* in metres (HumanML3D convention; Y-up).

Retargeting strategy (Hierarchical FK with 3D Frames):
  To prevent "windmilling" and "twisted torsos", shortest-arc rotation is not
  enough. We extract full 3D coordinate frames (using cross products of joint
  positions) for key structural bones (Hips, Chest) in both the rest pose and 
  the posed state. 
  For remaining bones (e.g. Spine1, Limbs), we use shortest-arc rotation,
  inheriting the robust roll from their parents.
"""
import bpy
import sys
import argparse
import numpy as np
from mathutils import Vector, Quaternion, Matrix

SMPL_TO_UNIRIG = {
    0:  "Hips",
    1:  "LeftUpperLeg",   2:  "RightUpperLeg",
    3:  "Spine",          4:  "LeftLowerLeg",
    5:  "RightLowerLeg",  6:  "Spine1",
    7:  "LeftFoot",       8:  "RightFoot",
    9:  "Spine2",         10: "LeftToes",
    11: "RightToes",      12: "Neck",
    13: "LeftShoulder",   14: "RightShoulder",
    15: "Head",           16: "LeftUpperArm",
    17: "RightUpperArm",  18: "LeftLowerArm",
    19: "RightLowerArm",  20: "LeftHand",
    21: "RightHand",
}
UNIRIG_TO_SMPL = {v: k for k, v in SMPL_TO_UNIRIG.items()}

# Primary target children for shortest-arc rotation
SMPL_TARGETS = {
    1: 4,  2: 5,  3: 6,  4: 7,  5: 8,  6: 9,  7: 10, 8: 11,
    12: 15, 13: 16, 14: 17, 16: 18, 17: 19, 18: 20, 19: 21,
}

# 3D Frame extractors (Center, Up, Right, Left)
FRAMES = {
    "Hips": (0, 3, 2, 1),
    "Spine2": (9, 12, 14, 13)
}

def _load_motion(npy_path):
    arr = np.load(npy_path)
    if arr.ndim == 4:
        arr = arr[0]  # (1, T, 22, 3) → (T, 22, 3)
    if arr.ndim != 3 or arr.shape[1] != 22 or arr.shape[2] != 3:
        raise RuntimeError(f"Unexpected motion shape {arr.shape}; expected (T,22,3)")
    return arr.astype(np.float32)

def to_blender_coords(v):
    """
    Convert SMPL (HumanML3D) coordinates to Blender coordinates.
    SMPL: +X = Left, +Y = Up, +Z = Forward
    Blender: +X = Right, +Y = Back, +Z = Up
    Mapping:
      Blender.X = -SMPL.X  (Left to Right)
      Blender.Y = -SMPL.Z  (Forward to Back)
      Blender.Z = SMPL.Y   (Up to Up)
    """
    return Vector((-v[0], -v[2], v[1]))

def make_frame(up, right):
    up = up.normalized()
    right = right.normalized()
    # In Blender, Y is backward. Z is up. X is right.
    # Up (Z) cross Right (X) = Backward (Y).
    back = up.cross(right)
    if back.length < 1e-5:
        return None
    back = back.normalized()
    right = back.cross(up).normalized()
    m = Matrix()
    m[0][0], m[1][0], m[2][0] = right.x, right.y, right.z
    m[0][1], m[1][1], m[2][1] = back.x, back.y, back.z
    m[0][2], m[1][2], m[2][2] = up.x, up.y, up.z
    return m.to_3x3()

def get_blender_frame(armature, c_name, u_name, r_name, l_name):
    try:
        c = armature.data.bones[c_name].head_local
        u = armature.data.bones[u_name].head_local
        r = armature.data.bones[r_name].head_local
        l = armature.data.bones[l_name].head_local
        return make_frame(u - c, r - l)
    except KeyError:
        return None

def get_smpl_frame(motion_t, c_idx, u_idx, r_idx, l_idx):
    c = to_blender_coords(motion_t[c_idx])
    u = to_blender_coords(motion_t[u_idx])
    r = to_blender_coords(motion_t[r_idx])
    l = to_blender_coords(motion_t[l_idx])
    return make_frame(u - c, r - l)

def _min_rotation(src, dst):
    dot = src.dot(dst)
    if dot > 0.99999:
        return Quaternion((1, 0, 0, 0))
    if dot < -0.99999:
        axis = src.cross(Vector((1, 0, 0)))
        if axis.length < 1e-6:
            axis = src.cross(Vector((0, 1, 0)))
        axis.normalize()
        return Quaternion(axis, np.pi)
    axis = src.cross(dst)
    axis.normalize()
    angle = np.arccos(max(-1.0, min(1.0, dot)))
    return Quaternion(axis, angle)

def retarget_action(armature, motion, action_name):
    T = motion.shape[0]
    
    if not armature.animation_data:
        armature.animation_data_create()
    action = bpy.data.actions.new(name=action_name)
    armature.animation_data.action = action

    for bone in armature.pose.bones:
        bone.rotation_mode = 'QUATERNION'

    smpl_root_rest = to_blender_coords(motion[0, 0])
    
    # Topological sort of bones
    bones_ordered = []
    def add_bone(b):
        if b not in bones_ordered:
            if b.parent:
                add_bone(b.parent)
            bones_ordered.append(b)
    for b in armature.data.bones:
        add_bone(b)

    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')

    hips_name = "Hips"
    if "Hips" not in armature.pose.bones:
        roots = [b for b in armature.pose.bones if b.parent is None]
        if roots:
            hips_name = roots[0].name

    for t in range(T):
        frame = t + 1
        motion_t = motion[t]
        pose_rotations = {}

        hips = armature.pose.bones.get(hips_name)
        if hips:
            delta = to_blender_coords(motion_t[0]) - smpl_root_rest
            hips.location = delta
            hips.keyframe_insert("location", frame=frame)

        for b in bones_ordered:
            b_name = b.name
            if b.parent:
                Parent_Pose_Obj = pose_rotations[b.parent.name]
                Parent_Rest_Obj = b.parent.matrix_local.to_3x3()
            else:
                Parent_Pose_Obj = Matrix.Identity(3)
                Parent_Rest_Obj = Matrix.Identity(3)

            R_C_rest = b.matrix_local.to_3x3()
            L_rest = Parent_Rest_Obj.inverted() @ R_C_rest
            Base_Pose_Obj = Parent_Pose_Obj @ L_rest
            
            Pose_Obj = Base_Pose_Obj
            Pose_Local = Matrix.Identity(3)
            
            smpl_idx = UNIRIG_TO_SMPL.get(b_name)
            if smpl_idx is not None:
                applied_3d_frame = False
                
                # 1. Try structural 3D frame
                if b_name in FRAMES:
                    c_idx, u_idx, r_idx, l_idx = FRAMES[b_name]
                    f_rest = get_blender_frame(armature, SMPL_TO_UNIRIG[c_idx], SMPL_TO_UNIRIG[u_idx], SMPL_TO_UNIRIG[r_idx], SMPL_TO_UNIRIG[l_idx])
                    f_posed = get_smpl_frame(motion_t, c_idx, u_idx, r_idx, l_idx)
                    if f_rest is not None and f_posed is not None:
                        R_diff = f_posed @ f_rest.inverted()
                        Pose_Obj = R_diff @ R_C_rest
                        Pose_Local = Base_Pose_Obj.inverted() @ Pose_Obj
                        applied_3d_frame = True
                
                # 2. Fallback to shortest-arc rotation
                if not applied_3d_frame and smpl_idx in SMPL_TARGETS:
                    child_idx = SMPL_TARGETS[smpl_idx]
                    c_name = SMPL_TO_UNIRIG.get(child_idx)
                    if c_name and c_name in armature.data.bones:
                        v_target = to_blender_coords(motion_t[child_idx]) - to_blender_coords(motion_t[smpl_idx])
                        if v_target.length > 1e-6:
                            v_target.normalize()
                            rest_dir = armature.data.bones[c_name].head_local - armature.data.bones[b_name].head_local
                            if rest_dir.length > 1e-6:
                                rest_dir.normalize()
                                v_local = R_C_rest.inverted() @ rest_dir
                                v_base = Base_Pose_Obj @ v_local
                                R_diff = _min_rotation(v_base, v_target).to_matrix()
                                Pose_Obj = R_diff @ Base_Pose_Obj
                                Pose_Local = Base_Pose_Obj.inverted() @ Pose_Obj

            pose_rotations[b_name] = Pose_Obj
            pb = armature.pose.bones.get(b_name)
            if pb:
                pb.rotation_quaternion = Pose_Local.to_quaternion()
                pb.keyframe_insert("rotation_quaternion", frame=frame)

    bpy.ops.object.mode_set(mode='OBJECT')
    
    track = armature.animation_data.nla_tracks.new()
    track.name = action_name
    track.strips.new(action.name, int(action.frame_range[0]), action)
    return action

def main():
    argv = sys.argv
    if "--" not in argv:
        return
    args = argv[argv.index("--") + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glb", required=True, help="Rigged GLB")
    parser.add_argument("--motion-npy", action='append', required=True, help="name=path/to/motion.npy")
    parser.add_argument("--output-glb", required=True, help="Animated GLB")
    parsed = parser.parse_args(args)

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=parsed.input_glb)

    armature = next((o for o in bpy.data.objects if o.type == 'ARMATURE'), None)
    if armature is None:
        raise RuntimeError("No armature in input GLB")

    if armature.animation_data:
        armature.animation_data_clear()

    print(f"[retarget] Processing {len(parsed.motion_npy)} motions...")
    
    for item in parsed.motion_npy:
        if "=" not in item:
            continue
        name, path = item.split("=", 1)
        motion = _load_motion(path)
        print(f"[retarget] Applying '{name}' (T={motion.shape[0]})")
        retarget_action(armature, motion, name)

    bpy.ops.export_scene.gltf(
        filepath=parsed.output_glb,
        export_format='GLB',
        export_apply=False,
        export_animations=True,
        export_skins=True,
        export_morph=True,
        export_image_format='AUTO',
        export_materials='EXPORT',
        export_texcoords=True,
    )
    print(f"[retarget] Wrote animated GLB → {parsed.output_glb}")

if __name__ == "__main__":
    main()
