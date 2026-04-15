"""
Blender Animation Script — generates idle / walk / attack NLA clips.

Invoked by Stage 5 via:
  blender --background --python scripts/blender_animate.py -- \
      --input  <rigged.glb>   \
      --joints <joints.json>  \
      --output <animated.glb> \
      --attack-type <type>

Frame layout (24 fps, 2 s each):
  idle    :   0 –  47  (subtle breathing + subtle sway)
  walk    :  48 –  95  (biped walk cycle, arms opposing legs)
  attack  :  96 – 143  (attack motion chosen by --attack-type)

Joint naming convention (from Stage 4 heuristic skeleton):
  root, pelvis, spine_01/02/03, neck, head,
  shoulder_l/r, upper_arm_l/r, lower_arm_l/r, hand_l/r,
  thigh_l/r, shin_l/r, foot_l/r

Blender version: 4.0.2 (uses bpy API with NLA)
"""

import argparse
import json
import math
import sys

import bpy


# ---------------------------------------------------------------------------
# Parse CLI args (after the -- separator blender passes)
# ---------------------------------------------------------------------------

def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    p = argparse.ArgumentParser()
    p.add_argument("--input",       required=True)
    p.add_argument("--joints",      required=True)
    p.add_argument("--output",      required=True)
    p.add_argument("--attack-type", default="generic_strike",
                   dest="attack_type")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def euler(rx=0.0, ry=0.0, rz=0.0):
    """Return a tuple of (rx, ry, rz) in radians from degrees."""
    return (math.radians(rx), math.radians(ry), math.radians(rz))


def insert_bone_rotation(armature_obj, bone_name: str, frame: int,
                          rx=0.0, ry=0.0, rz=0.0):
    """Insert a rotation keyframe on the named bone (euler XYZ)."""
    if bone_name not in armature_obj.pose.bones:
        return  # bone may not exist in all skeletons — skip gracefully
    pb = armature_obj.pose.bones[bone_name]
    pb.rotation_mode = "XYZ"
    pb.rotation_euler = euler(rx, ry, rz)
    pb.keyframe_insert("rotation_euler", frame=frame)


def insert_bone_location(armature_obj, bone_name: str, frame: int,
                          x=0.0, y=0.0, z=0.0):
    """Insert a location keyframe on the named bone."""
    if bone_name not in armature_obj.pose.bones:
        return
    pb = armature_obj.pose.bones[bone_name]
    pb.location = (x, y, z)
    pb.keyframe_insert("location", frame=frame)


# ---------------------------------------------------------------------------
# Animation builders
# ---------------------------------------------------------------------------

def build_idle(arm, fs=0, fe=47):
    """
    Idle: subtle sine-wave breathing (spine_02 Y rotation ±2°) +
    gentle head sway (head Z rotation ±1.5°) + pelvis micro-bob.
    Period = 48 frames @ 24fps = 2s.
    """
    half = (fe - fs) // 2

    # Breathing: spine expands up then down
    insert_bone_rotation(arm, "spine_02", fs,     rx=0)
    insert_bone_rotation(arm, "spine_02", fs + half, rx=2)
    insert_bone_rotation(arm, "spine_02", fe,     rx=0)

    # Subtle head sway
    insert_bone_rotation(arm, "head", fs,          rz=0)
    insert_bone_rotation(arm, "head", fs + half,   rz=1.5)
    insert_bone_rotation(arm, "head", fe,           rz=0)

    # Pelvis micro-bob (down then back to rest)
    insert_bone_location(arm, "pelvis", fs,        y=0)
    insert_bone_location(arm, "pelvis", fs + half, y=-0.01)
    insert_bone_location(arm, "pelvis", fe,        y=0)

    # Arms hang naturally (slight elbow bend for organic look)
    for side in ("l", "r"):
        sign = -1 if side == "l" else 1
        insert_bone_rotation(arm, f"lower_arm_{side}", fs, rx=5 * sign)
        insert_bone_rotation(arm, f"lower_arm_{side}", fe, rx=5 * sign)


def build_walk(arm, fs=48, fe=95):
    """
    Walk cycle: 4 key poses over 48 frames (2 strides @ 24fps).
    - Foot plants, leg drive, opposing arm swing.
    """
    q = (fe - fs) // 4
    # t0=contact, t1=down, t2=pass, t3=up
    t = [fs, fs + q, fs + 2 * q, fs + 3 * q, fe]

    # Pelvis bob (down at t1, t3 for weight transfer)
    for ti, bob in zip(t, [0, -0.03, 0, -0.03, 0]):
        insert_bone_location(arm, "pelvis", ti, y=bob)

    # Spine slight counter-rotation to pelvis
    for ti, rx in zip(t, [0, 3, 0, -3, 0]):
        insert_bone_rotation(arm, "spine_02", ti, rz=rx)

    # Legs: thigh and shin rotations for stepping (XZ plane)
    # Left leg leads first stride; right follows second
    leg_keys = {
        "thigh_l":  [0, 30, 0, -30, 0],    # forward / neutral / back
        "shin_l":   [0, -45, 0, 0, 0],      # knee bend
        "foot_l":   [10, -10, 0, 10, 10],   # heel / toe
        "thigh_r":  [0, -30, 0, 30, 0],
        "shin_r":   [0, 0, 0, -45, 0],
        "foot_r":   [10, 10, 0, -10, 10],
    }
    for bone, vals in leg_keys.items():
        for ti, rx in zip(t, vals):
            insert_bone_rotation(arm, bone, ti, rx=rx)

    # Arms opposing legs (shoulder rotation around X)
    arm_keys = {
        "upper_arm_l": [0, -25, 0, 25, 0],   # forward when right foot leads
        "upper_arm_r": [0, 25, 0, -25, 0],
    }
    for bone, vals in arm_keys.items():
        for ti, rx in zip(t, vals):
            insert_bone_rotation(arm, bone, ti, rx=rx)


def build_attack_staff_summon(arm, fs=96, fe=143):
    """
    Staff Summon (shaman/mage):
    Wind-up → raise staff overhead → slam down + flash → recover.
    """
    q = (fe - fs) // 4
    t = [fs, fs + q, fs + 2 * q, fs + 3 * q, fe]

    # Torso leans back in wind-up then snaps forward
    torso = [0, -10, -15, 10, 0]
    for ti, rx in zip(t, torso):
        insert_bone_rotation(arm, "spine_02", ti, rx=rx)

    # Right arm (weapon arm) raises overhead
    r_arm = [0, -60, -120, -90, 0]
    for ti, rx in zip(t, r_arm):
        insert_bone_rotation(arm, "upper_arm_r", ti, rx=rx)

    # Left arm extends forward / outward for magical effect
    l_arm = [0, 30, 60, 20, 0]
    for ti, rx in zip(t, l_arm):
        insert_bone_rotation(arm, "upper_arm_l", ti, rx=rx)

    # Head tilts back looking up at apex, then forward at target
    head_vals = [0, -15, -30, 15, 0]
    for ti, rx in zip(t, head_vals):
        insert_bone_rotation(arm, "head", ti, rx=rx)

    # Elbow bend on weapon arm for natural arc
    elbow_r = [0, -30, -60, -20, 0]
    for ti, rx in zip(t, elbow_r):
        insert_bone_rotation(arm, "lower_arm_r", ti, rx=rx)


def build_attack_sword_slash(arm, fs=96, fe=143):
    """Overhead sword/axe slash: raise → overhead → diagonal downstrike → recover."""
    q = (fe - fs) // 4
    t = [fs, fs + q, fs + 2 * q, fs + 3 * q, fe]

    # Torso rotation (wind-up right, follow-through left)
    spine_rz = [0, 15, 0, -20, 0]
    for ti, rz in zip(t, spine_rz):
        insert_bone_rotation(arm, "spine_02", ti, rz=rz)

    # Right arm overhead then slam
    r_arm_rx = [0, -100, -140, 30, 0]
    for ti, rx in zip(t, r_arm_rx):
        insert_bone_rotation(arm, "upper_arm_r", ti, rx=rx)

    # Left arm balance (extends opposite)
    l_arm_rx = [0, 20, 40, -10, 0]
    for ti, rx in zip(t, l_arm_rx):
        insert_bone_rotation(arm, "upper_arm_l", ti, rx=rx)

    # Right elbow straightens on downstroke
    elbow_r = [0, -45, -90, 0, 0]
    for ti, rx in zip(t, elbow_r):
        insert_bone_rotation(arm, "lower_arm_r", ti, rx=rx)

    # Step into the swing
    insert_bone_rotation(arm, "thigh_r", fs + q, rx=20)
    insert_bone_rotation(arm, "thigh_r", fs + 2 * q, rx=0)


def build_attack_bow_fire(arm, fs=96, fe=143):
    """Draw bow and fire: draw → full draw → release → follow-through."""
    q = (fe - fs) // 4
    t = [fs, fs + q, fs + 2 * q, fs + 3 * q, fe]

    # Torso side-on to target
    for ti, rz in zip(t, [0, 30, 30, 10, 0]):
        insert_bone_rotation(arm, "spine_02", ti, rz=rz)

    # Left arm (bow arm) extends forward
    for ti, rx in zip(t, [0, -60, -70, -60, 0]):
        insert_bone_rotation(arm, "upper_arm_l", ti, rx=rx)

    # Right arm (draw arm) pulls back
    for ti, rx in zip(t, [0, -30, -80, -10, 0]):
        insert_bone_rotation(arm, "upper_arm_r", ti, rx=rx)

    # Right elbow bends on draw
    for ti, rx in zip(t, [0, -30, -90, -10, 0]):
        insert_bone_rotation(arm, "lower_arm_r", ti, rx=rx)

    # Head turns to aim
    for ti, rz in zip(t, [0, 30, 30, 20, 0]):
        insert_bone_rotation(arm, "head", ti, rz=rz)


def build_attack_claw_swipe(arm, fs=96, fe=143):
    """Lunge + claw swipe: crouch → lunge forward → swipe across → recover."""
    q = (fe - fs) // 4
    t = [fs, fs + q, fs + 2 * q, fs + 3 * q, fe]

    # Crouch/coil (spine bends forward)
    for ti, rx in zip(t, [0, -20, -5, 5, 0]):
        insert_bone_rotation(arm, "spine_02", ti, rx=rx)

    # Both arms sweep: right arm sweeps across
    for ti, rz in zip(t, [0, -30, 60, 10, 0]):
        insert_bone_rotation(arm, "upper_arm_r", ti, rz=rz)

    # Left arm swipes up
    for ti, rz in zip(t, [0, 20, -40, -10, 0]):
        insert_bone_rotation(arm, "upper_arm_l", ti, rz=rz)

    # Legs: lunge step
    for ti, rx in zip(t, [0, 15, 30, 5, 0]):
        insert_bone_rotation(arm, "thigh_l", ti, rx=rx)
    for ti, rx in zip(t, [0, -10, -20, -5, 0]):
        insert_bone_rotation(arm, "thigh_r", ti, rx=rx)


def build_attack_generic_strike(arm, fs=96, fe=143):
    """Generic forward punch/strike: wind-up → extend → recover."""
    q = (fe - fs) // 4
    t = [fs, fs + q, fs + 2 * q, fs + 3 * q, fe]

    for ti, rx in zip(t, [0, -10, 5, 5, 0]):
        insert_bone_rotation(arm, "spine_02", ti, rx=rx)

    # Right arm punch: pull back then extend
    for ti, rx in zip(t, [0, 10, -80, -30, 0]):
        insert_bone_rotation(arm, "upper_arm_r", ti, rx=rx)

    # Elbow extension on punch
    for ti, rx in zip(t, [0, -90, 0, -20, 0]):
        insert_bone_rotation(arm, "lower_arm_r", ti, rx=rx)

    # Left guard arm
    for ti, rx in zip(t, [0, -30, -20, -30, 0]):
        insert_bone_rotation(arm, "upper_arm_l", ti, rx=rx)


_ATTACK_BUILDERS = {
    "staff_summon":   build_attack_staff_summon,
    "sword_slash":    build_attack_sword_slash,
    "bow_fire":       build_attack_bow_fire,
    "claw_swipe":     build_attack_claw_swipe,
    "generic_strike": build_attack_generic_strike,
}


# ---------------------------------------------------------------------------
# NLA track builder
# ---------------------------------------------------------------------------

def bake_action_to_nla(armature_obj, action, track_name: str, start_frame: int):
    """Push the given action into an NLA track at the specified start frame."""
    if armature_obj.animation_data is None:
        armature_obj.animation_data_create()
    track = armature_obj.animation_data.nla_tracks.new()
    track.name = track_name
    strip = track.strips.new(name=action.name, start=start_frame, action=action)
    strip.name = track_name
    return strip


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"[blender_animate] Input GLB:  {args.input}")
    print(f"[blender_animate] Joints:     {args.joints}")
    print(f"[blender_animate] Output GLB: {args.output}")
    print(f"[blender_animate] Attack:     {args.attack_type}")

    # Clear default scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import rigged GLB
    bpy.ops.import_scene.gltf(filepath=args.input)

    # Find the armature object
    armature_obj = None
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE":
            armature_obj = obj
            break

    if armature_obj is None:
        print("[blender_animate] WARNING: no armature found; creating proxy armature from joints.json")
        # Create a minimal armature from joints.json so animations still work
        joints = json.loads(open(args.joints).read())
        bpy.ops.object.armature_add(enter_editmode=True)
        armature_obj = bpy.context.object
        armature_obj.name = "Armature_proxy"
        # We only need the armature object to exist — keyframes on missing bones
        # are silently skipped via the guard in insert_bone_rotation/location
        bpy.ops.object.editmode_toggle()

    bpy.context.view_layer.objects.active = armature_obj

    # --- Build idle action ---
    bpy.ops.object.mode_set(mode="POSE")
    idle_action = bpy.data.actions.new("idle")
    armature_obj.animation_data_create()
    armature_obj.animation_data.action = idle_action
    build_idle(armature_obj)

    # --- Build walk action ---
    walk_action = bpy.data.actions.new("walk")
    armature_obj.animation_data.action = walk_action
    build_walk(armature_obj)

    # --- Build attack action ---
    attack_action = bpy.data.actions.new("attack")
    armature_obj.animation_data.action = attack_action
    attack_builder = _ATTACK_BUILDERS.get(args.attack_type, build_attack_generic_strike)
    attack_builder(armature_obj)

    bpy.ops.object.mode_set(mode="OBJECT")

    # Push each action into its own NLA track
    bake_action_to_nla(armature_obj, idle_action, "idle", 0)
    bake_action_to_nla(armature_obj, walk_action, "walk", 48)
    bake_action_to_nla(armature_obj, attack_action, "attack", 96)

    # Detach active action so all NLA tracks play on export
    armature_obj.animation_data.action = None

    # Export animated GLB
    print(f"[blender_animate] Exporting animated GLB to {args.output}")
    bpy.ops.export_scene.gltf(
        filepath=args.output,
        export_format="GLB",
        export_animations=True,
        export_nla_strips=True,
        export_current_frame=False,
    )
    print("[blender_animate] Done.")


if __name__ == "__main__":
    main()
