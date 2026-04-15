"""
blender_auto_rig.py — Generic humanoid auto-rigger for the mesh-gen-pipeline.

Strategy
--------
1.  Import mesh (OBJ or GLB).
2.  Analyse per-height-slice vertex statistics ("cross-sections") to locate the
    spine centre, shoulder width, hip width, and — crucially — where the arms
    actually are in world space.  This is pose-independent: a combat-stance
    character, a T-pose, or anything in between all work because we follow the
    geometry rather than assuming a fixed pose.
3.  Build a 20-bone humanoid skeleton placed through the actual geometry.
4.  Skin with ARMATURE_AUTO (bone-heat).  If that fails (low-poly mesh or heat
    solver divergence), fall back to ARMATURE_ENVELOPE with per-bone-category
    radii (torso wider than limbs, extremities narrowest).

Testability
-----------
All geometry-analysis functions (_sections_from_verts, _landmarks_from_verts,
_arm_chains_from_verts) accept plain Python lists of (x, y, z) tuples so they
can be unit-tested without a running Blender instance.  The Blender-dependent
wrappers (compute_sections, detect_landmarks) delegate to these pure functions.
"""

import math
import json
import os
import sys
import argparse
from typing import Any

# bpy and mathutils are only available inside Blender.  Guard the import so the
# pure-Python helpers (_sections_from_verts, _landmarks_from_verts, etc.) can
# be imported and unit-tested without a running Blender instance.
try:
    import bpy  # type: ignore[import]
    from mathutils import Vector  # type: ignore[import]
    _HAS_BPY = True
except ImportError:
    _HAS_BPY = False


# ─── Pure-Python geometry helpers (testable without Blender) ──────────────────

def _pct(arr, p):
    """Return the p-th percentile of a sorted list (0 ≤ p ≤ 1)."""
    if not arr:
        return 0.0
    i = max(0, min(len(arr) - 1, int(len(arr) * p)))
    return arr[i]


def _med(values):
    if not values:
        return 0.0
    s = sorted(values)
    return s[len(s) // 2]


def _sections_from_verts(verts, n=40):
    """
    Slice vertices by height (Z axis) and compute per-slice statistics.

    Args:
        verts: list of (x, y, z) tuples in world space.
        n:     number of height slices.

    Returns:
        sections: list of dicts — {z_ctr, frac, p10_x, med_x, p90_x, span_x,
                                    med_y, count}
        z_min, z_max, height
    """
    if not verts:
        return [], 0.0, 1.0, 1.0

    zs = [v[2] for v in verts]
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]

    z_min, z_max = min(zs), max(zs)
    height = z_max - z_min
    if height < 1e-6:
        return [], z_min, z_max, 1e-6

    sections = []
    for i in range(n):
        lo = z_min + height * i / n
        hi = z_min + height * (i + 1) / n
        z_ctr = (lo + hi) * 0.5
        frac = (z_ctr - z_min) / height

        # Last slice: closed upper bound so the maximum-z vertex is always included
        if i == n - 1:
            idx = [j for j, z in enumerate(zs) if lo <= z <= hi]
        else:
            idx = [j for j, z in enumerate(zs) if lo <= z < hi]
        if len(idx) < 3:
            sections.append(dict(z_ctr=z_ctr, frac=frac,
                                 p10_x=0.0, med_x=0.0, p90_x=0.0,
                                 span_x=0.0, med_y=0.0, count=len(idx)))
            continue

        sx = sorted(xs[j] for j in idx)
        sy = sorted(ys[j] for j in idx)

        p10 = _pct(sx, 0.10)
        med = _pct(sx, 0.50)
        p90 = _pct(sx, 0.90)

        sections.append(dict(
            z_ctr=z_ctr, frac=frac,
            p10_x=p10, med_x=med, p90_x=p90,
            span_x=p90 - p10,
            med_y=_pct(sy, 0.50),
            count=len(idx),
        ))

    return sections, z_min, z_max, height


def _zone(sections, frac_lo, frac_hi, min_count=4):
    """Return sections whose frac is in [frac_lo, frac_hi] with enough verts."""
    return [s for s in sections
            if frac_lo <= s['frac'] <= frac_hi and s['count'] >= min_count]


def _arm_chains_from_verts(verts, cx, cy, sh_x, z_min, height):
    """
    Detect left and right arm bone chains from vertex clusters.
    Uses medians and spatial clustering to be robust to combat poses and props.
    """
    lo = z_min + height * 0.55
    hi = z_min + height * 0.92
    thresh = sh_x * 0.8  # slightly tighter than before

    left_verts  = [v for v in verts if lo <= v[2] <= hi and v[0] < cx - thresh]
    right_verts = [v for v in verts if lo <= v[2] <= hi and v[0] > cx + thresh]

    def _default_arm(sign):
        sz = z_min + height * 0.80
        return [
            (cx + sign * sh_x, cy, sz),
            (cx + sign * sh_x, cy, sz - height * 0.20),
            (cx + sign * sh_x, cy, sz - height * 0.40),
        ]

    def _chain(side_verts, sign):
        if len(side_verts) < 15:
            return _default_arm(sign)

        # 1. Find shoulder (highest vertices near the torso attachment)
        sv = sorted(side_verts, key=lambda v: v[2], reverse=True)
        # Shoulder is median of the highest 20% of vertices
        shoulder_verts = sv[:max(1, len(sv) // 5)]
        shoulder_pos = (
            _med([v[0] for v in shoulder_verts]),
            _med([v[1] for v in shoulder_verts]),
            _med([v[2] for v in shoulder_verts]),
        )

        # 2. Find hand (vertices furthest from shoulder on X or Y)
        # Actually, furthest distance in 3D is more robust
        def dist_sq(v1, v2):
            return sum((a - b)**2 for a, b in zip(v1, v2))
        
        # Sort by distance from shoulder
        dv = sorted(side_verts, key=lambda v: dist_sq(v, shoulder_pos), reverse=True)
        # Hand is median of the 10% furthest vertices
        hand_verts = dv[:max(1, len(dv) // 10)]
        hand_pos = (
            _med([v[0] for v in hand_verts]),
            _med([v[1] for v in hand_verts]),
            _med([v[2] for v in hand_verts]),
        )

        # 3. Elbow (vertices in the middle distance range)
        # We look for the median of vertices that are roughly halfway between shoulder and hand
        mid_dist = math.sqrt(dist_sq(shoulder_pos, hand_pos)) * 0.5
        elbow_candidates = [v for v in side_verts 
                           if abs(math.sqrt(dist_sq(v, shoulder_pos)) - mid_dist) < mid_dist * 0.3]
        if not elbow_candidates:
            elbow_candidates = side_verts
        
        elbow_pos = (
            _med([v[0] for v in elbow_candidates]),
            _med([v[1] for v in elbow_candidates]),
            _med([v[2] for v in elbow_candidates]),
        )

        # Clamp shoulder Z into plausible zone
        sh_lo, sh_hi = z_min + height * 0.72, z_min + height * 0.88
        sz = max(sh_lo, min(sh_hi, shoulder_pos[2]))
        shoulder_pos = (shoulder_pos[0], shoulder_pos[1], sz)

        return [shoulder_pos, elbow_pos, hand_pos]

    return _chain(left_verts, -1), _chain(right_verts, 1)


def _landmarks_from_verts(verts, sections, z_min, height):
    """
    Derive all bone-placement landmarks from vertex list and cross-sections.

    Returns a dict with every field needed by create_armature().
    Pure Python — no bpy dependency.
    """
    def z_at(frac):
        return z_min + height * frac

    # ── Spine centre X and Y ─────────────────────────────────────────────────────
    # Median of per-slice medians over the reliable torso zone (50–82 %).
    torso_secs = _zone(sections, 0.50, 0.82)
    cx = _med([s['med_x'] for s in torso_secs]) if torso_secs else 0.0
    cy = _med([s['med_y'] for s in torso_secs]) if torso_secs else 0.0

    # ── Shoulder half-width ──────────────────────────────────────────────────────
    # p90–p10 X span at 74–86 % height.  The shoulder ATTACHMENT is near the top
    # of the torso regardless of arm pose (combat, T-pose, raised, etc.).
    sh_secs = _zone(sections, 0.74, 0.86)
    sh_spans = [s['span_x'] for s in sh_secs]
    sh_span = _med(sh_spans) if sh_spans else height * 0.40
    sh_x = max(sh_span * 0.45, 0.06)
    sh_x = min(sh_x, height * 0.38)   # guard against prop-inflated bbox

    # ── Hip half-width ───────────────────────────────────────────────────────────
    # Use the 25th-percentile of X spans at 44–58 % height (conservative —
    # excludes combat-stance arm protrusions at waist level).
    hip_secs = _zone(sections, 0.44, 0.58)
    hip_spans_sorted = sorted(s['span_x'] for s in hip_secs)
    if hip_spans_sorted:
        hip_span = hip_spans_sorted[max(0, int(len(hip_spans_sorted) * 0.25))]
    else:
        hip_span = sh_span * 0.60
    hip_x = max(hip_span * 0.28, 0.04)
    hip_x = min(hip_x, height * 0.22)

    # ── Arm chain positions ──────────────────────────────────────────────────────
    arm_l, arm_r = _arm_chains_from_verts(verts, cx, cy, sh_x, z_min, height)

    # ── PROP/ASYMMETRY CORRECTION ────────────────────────────────────────────────
    # If one arm is significantly further or shaped differently (due to a staff/shield),
    # and the other side looks more "natural", we can improve the symmetry of the
    # root/spine/legs.
    # We already used medians for cx, cy which is robust.
    # For legs, we'll keep them symmetrical around cx.

    return dict(
        cx=cx, cy=cy,
        sh_x=sh_x, hip_x=hip_x,
        arm_l=arm_l, arm_r=arm_r,
        z_base=z_min,
        z_pelvis=z_at(0.52),
        z_spine1=z_at(0.62),
        z_spine2=z_at(0.72),
        z_neck=z_at(0.84),
        z_head=z_at(0.92),
        z_knee=z_at(0.27),
        z_ankle=z_at(0.06),
        height=height,
    )


# ─── Blender-dependent helpers ─────────────────────────────────────────────────

def compute_sections(obj, n=40):
    """Extract world-space vertices from a Blender object and call _sections_from_verts."""
    mat = obj.matrix_world
    verts = [(mat @ v.co).to_tuple() for v in obj.data.vertices]
    return _sections_from_verts(verts, n=n)


def detect_landmarks(obj, n=40):
    """Full landmark detection for a Blender mesh object."""
    sections, z_min, z_max, height = compute_sections(obj, n=n)
    mat = obj.matrix_world
    verts = [(mat @ v.co).to_tuple() for v in obj.data.vertices]
    return _landmarks_from_verts(verts, sections, z_min, height)


# ─── Armature creation ─────────────────────────────────────────────────────────

def create_armature(lm):
    """
    Build a 20-bone humanoid armature from the detected landmarks dict.

    Bone categories (used for ARMATURE_ENVELOPE radius sizing):
        torso, head, limb_upper, limb_lower, extremity
    """
    cx     = lm['cx'];    cy     = lm['cy']
    sh_x   = lm['sh_x']; hip_x  = lm['hip_x']
    height = lm['height']
    z0     = lm['z_base']

    pz   = lm['z_pelvis'];  sp1z = lm['z_spine1']
    sp2z = lm['z_spine2'];  nkz  = lm['z_neck']
    hdz  = lm['z_head'];    knz  = lm['z_knee']
    anz  = lm['z_ankle']

    al = list(lm['arm_l'])   # [(x,y,z), ...] × 3
    ar = list(lm['arm_r'])

    # Guarantee exactly 3 control points per arm
    while len(al) < 3:
        al.append(al[-1])
    while len(ar) < 3:
        ar.append(ar[-1])

    def _hand_tip(chain):
        """Extrapolate hand-tip bone tail: continue lower-arm direction by ~8 % height."""
        dx = chain[2][0] - chain[1][0]
        dy = chain[2][1] - chain[1][1]
        dz = chain[2][2] - chain[1][2]
        seg = math.sqrt(dx**2 + dy**2 + dz**2) or (height * 0.10)
        s = min(height * 0.08, seg) / seg
        return (chain[2][0] + dx * s,
                chain[2][1] + dy * s,
                chain[2][2] + dz * s)

    # Toes point forward (−Y in Blender = out of screen for a front-facing char)
    toe_y = cy - height * 0.08

    # ─ Bone definitions ─────────────────────────────────────────────────────────
    # Keys: h=head position, t=tail position, p=parent name, cat=envelope category
    bones_def = {
        # Spine
        "root":     {"h": (cx, cy, z0),       "t": (cx, cy, pz),               "p": None,       "cat": "torso"},
        "pelvis":   {"h": (cx, cy, pz),        "t": (cx, cy, sp1z),             "p": "root",     "cat": "torso"},
        "spine_01": {"h": (cx, cy, sp1z),      "t": (cx, cy, sp2z),             "p": "pelvis",   "cat": "torso"},
        "spine_02": {"h": (cx, cy, sp2z),      "t": (cx, cy, nkz),              "p": "spine_01", "cat": "torso"},
        "neck":     {"h": (cx, cy, nkz),       "t": (cx, cy, hdz),              "p": "spine_02", "cat": "head"},
        "head":     {"h": (cx, cy, hdz),       "t": (cx, cy, hdz + height*0.12),"p": "neck",     "cat": "head"},

        # Left arm
        "shoulder_l":  {"h": (cx - sh_x*0.3, cy, sp2z),      "t": al[0],    "p": "spine_02",   "cat": "limb_upper"},
        "upper_arm_l": {"h": al[0],                           "t": al[1],    "p": "shoulder_l", "cat": "limb_upper"},
        "lower_arm_l": {"h": al[1],                           "t": al[2],    "p": "upper_arm_l","cat": "limb_lower"},
        "hand_l":      {"h": al[2],                           "t": _hand_tip(al), "p": "lower_arm_l","cat": "extremity"},

        # Right arm
        "shoulder_r":  {"h": (cx + sh_x*0.3, cy, sp2z),      "t": ar[0],    "p": "spine_02",   "cat": "limb_upper"},
        "upper_arm_r": {"h": ar[0],                           "t": ar[1],    "p": "shoulder_r", "cat": "limb_upper"},
        "lower_arm_r": {"h": ar[1],                           "t": ar[2],    "p": "upper_arm_r","cat": "limb_lower"},
        "hand_r":      {"h": ar[2],                           "t": _hand_tip(ar), "p": "lower_arm_r","cat": "extremity"},

        # Left leg
        "thigh_l": {"h": (cx - hip_x, cy, pz),  "t": (cx - hip_x, cy, knz), "p": "pelvis",  "cat": "limb_upper"},
        "shin_l":  {"h": (cx - hip_x, cy, knz), "t": (cx - hip_x, cy, anz), "p": "thigh_l", "cat": "limb_lower"},
        "foot_l":  {"h": (cx - hip_x, cy, anz), "t": (cx - hip_x, cy - height*0.1, z0),"p": "shin_l", "cat": "extremity"},

        # Right leg
        "thigh_r": {"h": (cx + hip_x, cy, pz),  "t": (cx + hip_x, cy, knz), "p": "pelvis",  "cat": "limb_upper"},
        "shin_r":  {"h": (cx + hip_x, cy, knz), "t": (cx + hip_x, cy, anz), "p": "thigh_r", "cat": "limb_lower"},
        "foot_r":  {"h": (cx + hip_x, cy, anz), "t": (cx + hip_x, cy - height*0.1, z0),"p": "shin_r", "cat": "extremity"},
    }

    # ─ Create Blender armature ───────────────────────────────────────────────────
    bpy.ops.object.armature_add(enter_editmode=True)
    arm_obj = bpy.context.object
    arm_obj.name = "CharacterRig"
    arm_obj.data.name = "CharacterRigData"

    eb = arm_obj.data.edit_bones
    for b in list(eb):
        eb.remove(b)

    created = {}
    for name, d in bones_def.items():
        b = eb.new(name)
        b.head = d['h']
        b.tail = d['t']
        created[name] = b

    for name, d in bones_def.items():
        if d['p']:
            created[name].parent = created[d['p']]
            created[name].use_connect = False

    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"[blender_auto_rig] Created {len(created)} bones.")
    return arm_obj, bones_def


# ─── Skinning ──────────────────────────────────────────────────────────────────

def skin_mesh(mesh_obj, arm_obj, bones_def, height):
    """
    Parent mesh to armature with automatic weights.

    Attempts ARMATURE_AUTO (bone-heat weighting) first.  If the heat solver
    produces empty vertex groups (common on very low-poly meshes or characters
    in extreme poses where arm bones miss the geometry), falls back to
    ARMATURE_ENVELOPE with per-bone-category radii.

    Per-category radii (as fraction of total character height):
        torso      0.16  — wide enough to cover the trunk
        head       0.09
        limb_upper 0.08
        limb_lower 0.07
        extremity  0.05  — hands and feet get a tighter zone
    """

    # ── ARMATURE_AUTO (bone heat) ────────────────────────────────────────────────
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')

    # Remove empty vertex groups — empty groups crash the GLTF2 exporter
    # ("NoneType object has no attribute joints" in add_neutral_bones).
    def _remove_empty_vgroups(obj):
        empties = [
            vg for vg in obj.vertex_groups
            if not any(
                any(g.group == vg.index and g.weight > 0.001 for g in v.groups)
                for v in obj.data.vertices
            )
        ]
        for vg in empties:
            obj.vertex_groups.remove(vg)
        return len(empties)

    n_removed = _remove_empty_vgroups(mesh_obj)
    if n_removed:
        print(f"[blender_auto_rig] Removed {n_removed} empty vertex groups after AUTO")

    if mesh_obj.vertex_groups:
        print("[blender_auto_rig] Skinning: ARMATURE_AUTO succeeded.")
        return

    # ── ARMATURE_ENVELOPE fallback ───────────────────────────────────────────────
    print("[blender_auto_rig] Bone heat produced no weights → ARMATURE_ENVELOPE fallback")

    _RADII_FRACS = {
        "torso":      0.16,
        "head":       0.09,
        "limb_upper": 0.08,
        "limb_lower": 0.07,
        "extremity":  0.05,
    }
    _MIN_RADIUS = 0.04   # absolute floor for tiny characters

    bpy.ops.object.select_all(action='DESELECT')
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')

    for ebone in arm_obj.data.edit_bones:
        cat = bones_def.get(ebone.name, {}).get('cat', 'limb_lower')
        r = max(height * _RADII_FRACS.get(cat, 0.07), _MIN_RADIUS)
        ebone.head_radius = r
        ebone.tail_radius = r
        ebone.envelope_distance = r * 0.3

    bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.parent_set(type='ARMATURE_ENVELOPE')
    print("[blender_auto_rig] ARMATURE_ENVELOPE applied (per-bone-category radii).")


# ─── Scene helpers ─────────────────────────────────────────────────────────────

def clean_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for col in (bpy.data.meshes, bpy.data.materials, bpy.data.armatures):
        for item in col:
            col.remove(item)


# ─── Entry point ───────────────────────────────────────────────────────────────

def main():
    argv = sys.argv
    if "--" not in argv:
        return
    args = argv[argv.index("--") + 1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       required=True)
    parser.add_argument("--output-fbx",  required=True)
    parser.add_argument("--output-glb",  required=True)
    parser.add_argument("--joints",      required=True)
    parsed = parser.parse_args(args)

    clean_scene()

    # Import mesh
    ext = os.path.splitext(parsed.input)[1].lower()
    if ext == ".obj":
        # OBJ from trimesh/PyMeshLab is usually Y-up. 
        # Blender's default for obj_import (Forward: -Z, Up: Y) matches this.
        bpy.ops.wm.obj_import(
            filepath=parsed.input,
            forward_axis='NEGATIVE_Z',
            up_axis='Y'
        )
    else:
        # GLB is strictly Y-up per spec. Blender's importer handles it.
        bpy.ops.import_scene.gltf(filepath=parsed.input)

    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not meshes:
        raise ValueError("[blender_auto_rig] No mesh found in input file.")

    # Join multiple submeshes into one object
    bpy.ops.object.select_all(action='DESELECT')
    for m in meshes:
        m.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    mesh_obj = bpy.context.active_object

    # ── Auto-Orientation ──────────────────────────────────────────────────
    # If the mesh is "lying down" (horizontal span > vertical span), rotate it.
    # This handles cases where trimesh or PyMeshLab flipped the axes.
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bb = [mesh_obj.matrix_world @ Vector(corner) for corner in mesh_obj.bound_box]
    z_min = min(v.z for v in bb); z_max = max(v.z for v in bb); height = z_max - z_min
    y_min = min(v.y for v in bb); y_max = max(v.y for v in bb); depth  = y_max - y_min
    x_min = min(v.x for v in bb); x_max = max(v.x for v in bb); width  = x_max - x_min

    if depth > height * 1.5 and depth > width:
        print(f"[blender_auto_rig] Mesh detected as lying down (depth={depth:.2f} > height={height:.2f}). Rotating 90deg on X.")
        mesh_obj.rotation_euler[0] = math.radians(90)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    elif width > height * 1.5 and width > depth:
         print(f"[blender_auto_rig] Mesh detected as lying down (width={width:.2f} > height={height:.2f}). Rotating 90deg on Y.")
         mesh_obj.rotation_euler[1] = math.radians(90)
         bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    print(f"[blender_auto_rig] Imported mesh: {mesh_obj.name}, "
          f"{len(mesh_obj.data.vertices)} vertices, "
          f"{len(mesh_obj.data.polygons)} faces")

    # Detect landmarks from actual geometry
    lm = detect_landmarks(mesh_obj)
    height = lm['height']
    print(f"[blender_auto_rig] Landmarks: height={height:.3f} "
          f"cx={lm['cx']:.3f} sh_x={lm['sh_x']:.3f} hip_x={lm['hip_x']:.3f}")
    print(f"[blender_auto_rig] Arm L: shoulder={lm['arm_l'][0]} "
          f"elbow={lm['arm_l'][1]} hand={lm['arm_l'][2]}")
    print(f"[blender_auto_rig] Arm R: shoulder={lm['arm_r'][0]} "
          f"elbow={lm['arm_r'][1]} hand={lm['arm_r'][2]}")

    # Create armature
    arm_obj, bones_def = create_armature(lm)

    # Skin
    skin_mesh(mesh_obj, arm_obj, bones_def, height)

    # ── Remove shininess ──────────────────────────────────────────────────
    # Force high roughness (matte look) to match the "uniform" look.
    for mat in bpy.data.materials:
        if mat.use_nodes:
            nodes = mat.node_tree.nodes
            principled = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
            if principled:
                if 'Roughness' in principled.inputs:
                    principled.inputs['Roughness'].default_value = 0.8
                if 'Metallic' in principled.inputs:
                    principled.inputs['Metallic'].default_value = 0.0
                if 'Specular IOR Level' in principled.inputs: # Blender 4.0+
                    principled.inputs['Specular IOR Level'].default_value = 0.0
                elif 'Specular' in principled.inputs: # Older
                    principled.inputs['Specular'].default_value = 0.0

    # Export joints JSON before FBX/GLB (exporter may fail on edge cases)
    joints_list = [
        {"name": name, "parent": d['p'], "position": list(d['h'])}
        for name, d in bones_def.items()
    ]
    with open(parsed.joints, "w") as f:
        json.dump(joints_list, f, indent=2)
    print(f"[blender_auto_rig] Joints saved: {len(joints_list)} bones → {parsed.joints}")

    # Export FBX
    bpy.ops.export_scene.fbx(
        filepath=parsed.output_fbx,
        use_selection=True,
        object_types={'ARMATURE', 'MESH'},
        add_leaf_bones=False,
        bake_anim=False,
    )
    print(f"[blender_auto_rig] FBX exported → {parsed.output_fbx}")

    # Export GLB
    bpy.ops.export_scene.gltf(
        filepath=parsed.output_glb,
        use_selection=True,
        export_format='GLB',
    )
    print(f"[blender_auto_rig] GLB exported → {parsed.output_glb}")


if __name__ == "__main__":
    main()
