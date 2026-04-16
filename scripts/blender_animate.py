"""
Blender Animation Script — generates idle / walk / attack NLA clips.

Works with ANY humanoid skeleton produced by Puppeteer (or the heuristic
rigger) regardless of joint count or naming convention.

Invoked by Stage 5 via:
  blender --background --python scripts/blender_animate.py -- \
      --input  <rigged.glb>   \
      --joints <joints.json>  \
      --output <animated.glb> \
      --attack-type <type>

How the flexible rig system works
-----------------------------------
joints.json is a list of dicts:  {name, position[x,y,z], parent|null}
The positions are in Y-up world space (same coordinate system as the OBJ
file that was passed into Puppeteer).  We analyse the geometry to classify
each joint into a semantic role:

  pelvis / spine_lower / spine_mid / spine_upper / neck / head
  left_shoulder / left_upper_arm / left_forearm / left_wrist
  right_shoulder / right_upper_arm / right_forearm / right_wrist
  left_thigh / left_shin / left_foot
  right_thigh / right_shin / right_foot

Animation keyframes are then addressed via these roles, so the same
animation code works whether the skeleton has 20 or 34 joints and
regardless of what those joints are named.

Frame layout (24 fps, 2 s each):
  idle    :   0 –  47
  walk    :  48 –  95
  attack  :  96 – 143

Blender version: 4.0.2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from typing import Optional

import bpy  # type: ignore


# ---------------------------------------------------------------------------
# Parse CLI args
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
    p.add_argument("--attack-type", default="generic_strike", dest="attack_type")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Semantic skeleton classifier
# ---------------------------------------------------------------------------

def classify_skeleton(joints: list) -> dict:
    """
    Analyse joint positions / hierarchy and return a role→joint_name mapping.

    Algorithm (position-based, no name assumptions):
      1.  Find root (no parent) → pelvis.
      2.  Trace the spine: follow the chain of children that stay near the
          character's X-centre and go upward.  The last joint(s) above the
          shoulders become neck / head.
      3.  Trace legs: children of the root that go downward on each side.
      4.  Trace arms: children of upper-spine joints that go sideways.
    """
    by_name   = {j['name']: j for j in joints}
    children  = defaultdict(list)
    root_name = None

    for j in joints:
        parent = j.get('parent')
        if parent is None:
            root_name = j['name']
        else:
            children[parent].append(j['name'])

    if root_name is None:
        # fallback: lowest-y joint
        root_name = min(joints, key=lambda j: j['position'][1])['name']

    pos = {j['name']: j['position'] for j in joints}

    x_vals = [p[0] for p in pos.values()]
    y_vals = [p[1] for p in pos.values()]
    x_center   = (min(x_vals) + max(x_vals)) / 2.0
    body_height = max(y_vals) - min(y_vals)
    center_thr  = body_height * 0.22   # joints within this X-distance = "centre"

    # ── Spine ────────────────────────────────────────────────────────────────
    def trace_up(start: str) -> list:
        chain = [start]
        current = start
        for _ in range(30):
            ups = [
                c for c in children[current]
                if pos[c][1] > pos[current][1]
                and abs(pos[c][0] - x_center) < center_thr
            ]
            if not ups:
                break
            nxt = max(ups, key=lambda c: pos[c][1])
            chain.append(nxt)
            current = nxt
        return chain

    spine = trace_up(root_name)

    # ── Legs ─────────────────────────────────────────────────────────────────
    def find_leg_start(side_sign: float) -> Optional[str]:
        candidates = [
            c for c in children[root_name]
            if pos[c][1] < pos[root_name][1]  # below root
        ]
        sided = [c for c in candidates
                 if side_sign * (pos[c][0] - x_center) > center_thr * 0.1]
        if sided:
            return max(sided, key=lambda c: side_sign * pos[c][0])
        # Fallback: any below-root child on the correct side
        if candidates:
            return max(candidates, key=lambda c: side_sign * pos[c][0])
        return None

    def trace_down(start: Optional[str]) -> list:
        if start is None:
            return []
        chain = [start]
        current = start
        for _ in range(10):
            downs = [c for c in children[current]
                     if pos[c][1] < pos[current][1]]
            if not downs:
                break
            nxt = min(downs, key=lambda c: pos[c][1])
            chain.append(nxt)
            current = nxt
        return chain

    left_leg  = trace_down(find_leg_start(-1.0))
    right_leg = trace_down(find_leg_start(+1.0))

    # ── Arms ─────────────────────────────────────────────────────────────────
    spine_set = set(spine)

    def find_arm_start(anchor: str, side_sign: float) -> Optional[str]:
        lateral = [
            c for c in children[anchor]
            if c not in spine_set
            and side_sign * (pos[c][0] - x_center) > center_thr * 0.2
        ]
        if not lateral:
            return None
        return max(lateral, key=lambda c: side_sign * pos[c][0])

    def trace_arm(start: Optional[str], side_sign: float) -> list:
        if start is None:
            return []
        chain = [start]
        current = start
        for _ in range(6):
            # Continue laterally; also allow dropping (forearm hangs down)
            cands = [c for c in children[current] if c not in spine_set]
            if not cands:
                break
            nxt = max(cands, key=lambda c: side_sign * pos[c][0])
            # Stop when we stop moving laterally (finger fan-out)
            if side_sign * (pos[nxt][0] - pos[current][0]) <= 0:
                break
            chain.append(nxt)
            current = nxt
        return chain

    left_arm_start  = None
    right_arm_start = None
    for sp in reversed(spine):
        if left_arm_start is None:
            left_arm_start  = find_arm_start(sp, -1.0)
        if right_arm_start is None:
            right_arm_start = find_arm_start(sp, +1.0)
        if left_arm_start and right_arm_start:
            break

    left_arm  = trace_arm(left_arm_start,  -1.0)
    right_arm = trace_arm(right_arm_start, +1.0)

    # ── Build role map ────────────────────────────────────────────────────────
    roles: dict = {}
    roles['root']   = root_name
    roles['pelvis'] = root_name

    if len(spine) >= 2:
        roles['spine_lower'] = spine[1]
    if len(spine) >= 3:
        roles['spine_mid']   = spine[len(spine) // 2]
    if len(spine) >= 2:
        roles['spine_upper'] = spine[-1]

    # Neck and head: continue upward from spine tip, outside the spine chain
    spine_tip = spine[-1] if spine else root_name
    above_tip = [
        c for c in children[spine_tip]
        if pos[c][1] > pos[spine_tip][1] and c not in spine_set
    ]
    if above_tip:
        neck = max(above_tip, key=lambda c: pos[c][1])
        roles['neck'] = neck
        above_neck = [c for c in children[neck] if pos[c][1] > pos[neck][1]]
        if above_neck:
            roles['head'] = max(above_neck, key=lambda c: pos[c][1])
        else:
            roles['head'] = neck
    else:
        # spine tip itself is the head area
        roles['head'] = spine_tip

    # Legs
    for side, chain in (('left', left_leg), ('right', right_leg)):
        if len(chain) >= 1:
            roles[f'{side}_thigh'] = chain[0]
        if len(chain) >= 2:
            roles[f'{side}_shin']  = chain[1]
        if len(chain) >= 3:
            roles[f'{side}_foot']  = chain[-1]

    # Arms  (chain[0]=shoulder, [1]=upper_arm, [2]=forearm, [3]=wrist)
    for side, chain in (('left', left_arm), ('right', right_arm)):
        if len(chain) >= 1:
            roles[f'{side}_shoulder']  = chain[0]
        if len(chain) >= 2:
            roles[f'{side}_upper_arm'] = chain[1]
        if len(chain) >= 3:
            roles[f'{side}_forearm']   = chain[2]
        if len(chain) >= 4:
            roles[f'{side}_wrist']     = chain[3]

    return roles


# ---------------------------------------------------------------------------
# Keyframe helpers
# ---------------------------------------------------------------------------

def _bone(arm, role: str, roles: dict):
    """Return the pose bone for a semantic role, or None if absent."""
    name = roles.get(role)
    if name is None:
        return None
    return arm.pose.bones.get(name)


def rot(arm, role: str, roles: dict, frame: int,
        rx=0.0, ry=0.0, rz=0.0):
    """Insert an euler-XYZ rotation keyframe on a semantic role."""
    pb = _bone(arm, role, roles)
    if pb is None:
        return
    pb.rotation_mode = 'XYZ'
    pb.rotation_euler = (math.radians(rx), math.radians(ry), math.radians(rz))
    pb.keyframe_insert('rotation_euler', frame=frame)


def loc(arm, role: str, roles: dict, frame: int,
        x=0.0, y=0.0, z=0.0):
    """Insert a location keyframe on a semantic role."""
    pb = _bone(arm, role, roles)
    if pb is None:
        return
    pb.location = (x, y, z)
    pb.keyframe_insert('location', frame=frame)


# ---------------------------------------------------------------------------
# Animation builders  (all use semantic roles, not bone names)
# ---------------------------------------------------------------------------

def build_idle(arm, roles: dict, fs=0, fe=47):
    """
    Subtle breathing + gentle head sway + pelvis micro-bob.
    Period = 48 frames @ 24 fps = 2 s.
    """
    half = (fe - fs) // 2

    # Breathing — spine_mid pitches forward/back
    for f, rx in ((fs, 0), (fs + half, 2), (fe, 0)):
        rot(arm, 'spine_mid', roles, f, rx=rx)

    # Head sway — neck tilts slightly left/right
    for f, rz in ((fs, 0), (fs + half, 1.5), (fe, 0)):
        rot(arm, 'head', roles, f, rz=rz)

    # Pelvis micro-bob
    for f, y in ((fs, 0), (fs + half, -0.01), (fe, 0)):
        loc(arm, 'pelvis', roles, f, y=y)

    # Arms hang naturally (slight forearm angle)
    for side, sign in (('left', -1), ('right', 1)):
        for f in (fs, fe):
            rot(arm, f'{side}_forearm', roles, f, rx=5 * sign)


def build_walk(arm, roles: dict, fs=48, fe=95):
    """
    Walk cycle: 4-key biped with opposing arm-leg swing.
    """
    q = (fe - fs) // 4
    t = [fs + i * q for i in range(5)]

    # Pelvis vertical bob
    for ti, y in zip(t, [0, -0.03, 0, -0.03, 0]):
        loc(arm, 'pelvis', roles, ti, y=y)

    # Spine counter-rotation
    for ti, rz in zip(t, [0, 3, 0, -3, 0]):
        rot(arm, 'spine_mid', roles, ti, rz=rz)

    # Legs
    leg_data = {
        ('left',  'thigh'): [0, 30, 0, -30, 0],
        ('left',  'shin'):  [0, -45, 0, 0, 0],
        ('right', 'thigh'): [0, -30, 0, 30, 0],
        ('right', 'shin'):  [0, 0, 0, -45, 0],
    }
    for (side, part), vals in leg_data.items():
        for ti, rx in zip(t, vals):
            rot(arm, f'{side}_{part}', roles, ti, rx=rx)

    # Arms opposing legs
    arm_data = {
        'left_upper_arm':  [0, -25, 0, 25, 0],
        'right_upper_arm': [0,  25, 0, -25, 0],
    }
    for role, vals in arm_data.items():
        for ti, rx in zip(t, vals):
            rot(arm, role, roles, ti, rx=rx)


def build_attack_staff_summon(arm, roles: dict, fs=96, fe=143):
    """Staff/wand summon: wind-up → raise overhead → slam → recover."""
    q = (fe - fs) // 4
    t = [fs + i * q for i in range(5)]

    for ti, rx in zip(t, [0, -10, -15, 10, 0]):
        rot(arm, 'spine_mid', roles, ti, rx=rx)

    for ti, rx in zip(t, [0, -60, -120, -90, 0]):    # weapon arm rises
        rot(arm, 'right_upper_arm', roles, ti, rx=rx)

    for ti, rx in zip(t, [0, 30, 60, 20, 0]):         # off-hand extends
        rot(arm, 'left_upper_arm', roles, ti, rx=rx)

    for ti, rx in zip(t, [0, -15, -30, 15, 0]):
        rot(arm, 'head', roles, ti, rx=rx)

    for ti, rx in zip(t, [0, -30, -60, -20, 0]):
        rot(arm, 'right_forearm', roles, ti, rx=rx)


def build_attack_sword_slash(arm, roles: dict, fs=96, fe=143):
    """Overhead sword/axe slash."""
    q = (fe - fs) // 4
    t = [fs + i * q for i in range(5)]

    for ti, rz in zip(t, [0, 15, 0, -20, 0]):
        rot(arm, 'spine_mid', roles, ti, rz=rz)

    for ti, rx in zip(t, [0, -100, -140, 30, 0]):
        rot(arm, 'right_upper_arm', roles, ti, rx=rx)

    for ti, rx in zip(t, [0, 20, 40, -10, 0]):
        rot(arm, 'left_upper_arm', roles, ti, rx=rx)

    for ti, rx in zip(t, [0, -45, -90, 0, 0]):
        rot(arm, 'right_forearm', roles, ti, rx=rx)

    rot(arm, 'right_thigh', roles, t[1], rx=20)
    rot(arm, 'right_thigh', roles, t[2], rx=0)


def build_attack_bow_fire(arm, roles: dict, fs=96, fe=143):
    """Draw bow and loose arrow."""
    q = (fe - fs) // 4
    t = [fs + i * q for i in range(5)]

    for ti, rz in zip(t, [0, 30, 30, 10, 0]):
        rot(arm, 'spine_mid', roles, ti, rz=rz)

    for ti, rx in zip(t, [0, -60, -70, -60, 0]):
        rot(arm, 'left_upper_arm', roles, ti, rx=rx)

    for ti, rx in zip(t, [0, -30, -80, -10, 0]):
        rot(arm, 'right_upper_arm', roles, ti, rx=rx)

    for ti, rx in zip(t, [0, -30, -90, -10, 0]):
        rot(arm, 'right_forearm', roles, ti, rx=rx)

    for ti, rz in zip(t, [0, 30, 30, 20, 0]):
        rot(arm, 'head', roles, ti, rz=rz)


def build_attack_claw_swipe(arm, roles: dict, fs=96, fe=143):
    """Lunge + claw swipe."""
    q = (fe - fs) // 4
    t = [fs + i * q for i in range(5)]

    for ti, rx in zip(t, [0, -20, -5, 5, 0]):
        rot(arm, 'spine_mid', roles, ti, rx=rx)

    for ti, rz in zip(t, [0, -30, 60, 10, 0]):
        rot(arm, 'right_upper_arm', roles, ti, rz=rz)

    for ti, rz in zip(t, [0, 20, -40, -10, 0]):
        rot(arm, 'left_upper_arm', roles, ti, rz=rz)

    for ti, rx in zip(t, [0, 15, 30, 5, 0]):
        rot(arm, 'left_thigh', roles, ti, rx=rx)

    for ti, rx in zip(t, [0, -10, -20, -5, 0]):
        rot(arm, 'right_thigh', roles, ti, rx=rx)


def build_attack_generic_strike(arm, roles: dict, fs=96, fe=143):
    """Generic forward punch."""
    q = (fe - fs) // 4
    t = [fs + i * q for i in range(5)]

    for ti, rx in zip(t, [0, -10, 5, 5, 0]):
        rot(arm, 'spine_mid', roles, ti, rx=rx)

    for ti, rx in zip(t, [0, 10, -80, -30, 0]):
        rot(arm, 'right_upper_arm', roles, ti, rx=rx)

    for ti, rx in zip(t, [0, -90, 0, -20, 0]):
        rot(arm, 'right_forearm', roles, ti, rx=rx)

    for ti, rx in zip(t, [0, -30, -20, -30, 0]):
        rot(arm, 'left_upper_arm', roles, ti, rx=rx)


_ATTACK_BUILDERS = {
    "staff_summon":   build_attack_staff_summon,
    "sword_slash":    build_attack_sword_slash,
    "bow_fire":       build_attack_bow_fire,
    "claw_swipe":     build_attack_claw_swipe,
    "generic_strike": build_attack_generic_strike,
}


# ---------------------------------------------------------------------------
# NLA helper
# ---------------------------------------------------------------------------

def bake_action_to_nla(armature_obj, action, track_name: str, start_frame: int):
    if armature_obj.animation_data is None:
        armature_obj.animation_data_create()
    track = armature_obj.animation_data.nla_tracks.new()
    track.name = track_name
    strip = track.strips.new(name=action.name, start=start_frame, action=action)
    strip.name = track_name
    return strip


# ---------------------------------------------------------------------------
# Mesh / material fixup  (same three fixes as fbx_to_glb.py)
# ---------------------------------------------------------------------------

def _fix_scene_meshes_and_materials():
    """Shade-smooth and material properties before export.

    IMPORTANT — transform_apply is intentionally NOT called here.
    When Blender imports a GLTF it adds a rotation object transform to convert
    from GLTF Y-up to Blender Z-up.  Calling transform_apply(rotation=True) on
    a skinned mesh bakes that rotation into vertex positions while the bone
    positions remain unchanged → vertices shift relative to their bones →
    the animated mesh warps.  The GLTF exporter handles the Y-up ↔ Z-up
    conversion automatically at export time; no manual baking is needed.

    normals_make_consistent is also omitted here: the GLB exported by
    puppeteer_blend_export.py already has correct outward normals and
    doubleSided:true set on all materials.  Re-running it on a posed mesh
    (after animation keyframes have been inserted) can misclassify front/back
    on concave surfaces.
    """
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        # Shade smooth (visual quality — safe on skinned meshes)
        obj.data.polygons.foreach_set("use_smooth", [True] * len(obj.data.polygons))
        obj.data.update()

    for mat in bpy.data.materials:
        mat.use_backface_culling = False   # doubleSided:true in GLTF
        if not mat.use_nodes:
            continue
        principled = next(
            (n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None
        )
        if principled is None:
            continue
        if 'Roughness' in principled.inputs:
            principled.inputs['Roughness'].default_value = 0.8
        if 'Metallic' in principled.inputs:
            principled.inputs['Metallic'].default_value = 0.0
        if 'Specular IOR Level' in principled.inputs:
            principled.inputs['Specular IOR Level'].default_value = 0.0
        elif 'Specular' in principled.inputs:
            principled.inputs['Specular'].default_value = 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"[blender_animate] Input GLB:  {args.input}")
    print(f"[blender_animate] Joints:     {args.joints}")
    print(f"[blender_animate] Output GLB: {args.output}")
    print(f"[blender_animate] Attack:     {args.attack_type}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=args.input)

    # Identify armature
    armature_obj = next(
        (o for o in bpy.data.objects if o.type == 'ARMATURE'), None
    )
    if armature_obj is None:
        print("[blender_animate] WARNING: no armature found in GLB; aborting.")
        return

    # ── Semantic skeleton classification ─────────────────────────────────────
    joints = json.loads(open(args.joints).read())
    roles  = classify_skeleton(joints)

    # Verify roles against actual pose bones
    missing = [r for r, name in roles.items()
               if name not in armature_obj.pose.bones]
    if missing:
        print(f"[blender_animate] NOTE: {len(missing)} role(s) not in pose: "
              f"{missing[:8]}{'...' if len(missing) > 8 else ''}")

    found = {r: name for r, name in roles.items()
             if name in armature_obj.pose.bones}
    print(f"[blender_animate] Semantic mapping ({len(found)}/{len(roles)} roles):")
    for role, name in sorted(found.items()):
        print(f"    {role:22s} → {name}")

    bpy.context.view_layer.objects.active = armature_obj

    # ── Build animations ──────────────────────────────────────────────────────
    bpy.ops.object.mode_set(mode='POSE')
    armature_obj.animation_data_create()

    idle_action = bpy.data.actions.new("idle")
    armature_obj.animation_data.action = idle_action
    build_idle(armature_obj, roles)

    walk_action = bpy.data.actions.new("walk")
    armature_obj.animation_data.action = walk_action
    build_walk(armature_obj, roles)

    attack_action = bpy.data.actions.new("attack")
    armature_obj.animation_data.action = attack_action
    attack_fn = _ATTACK_BUILDERS.get(args.attack_type, build_attack_generic_strike)
    attack_fn(armature_obj, roles)

    bpy.ops.object.mode_set(mode='OBJECT')

    bake_action_to_nla(armature_obj, idle_action,   "idle",   0)
    bake_action_to_nla(armature_obj, walk_action,   "walk",  48)
    bake_action_to_nla(armature_obj, attack_action, "attack", 96)
    armature_obj.animation_data.action = None

    # ── Fix mesh/material before export ──────────────────────────────────────
    _fix_scene_meshes_and_materials()

    # ── Export ────────────────────────────────────────────────────────────────
    print(f"[blender_animate] Exporting animated GLB to {args.output}")
    bpy.ops.export_scene.gltf(
        filepath=args.output,
        export_format="GLB",
        export_normals=True,
        export_animations=True,
        export_nla_strips=True,
        export_current_frame=False,
    )
    print("[blender_animate] Done.")


if __name__ == "__main__":
    main()
