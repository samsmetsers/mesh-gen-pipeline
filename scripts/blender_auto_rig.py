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


def _density_center(verts):
    if not verts: return (0, 0, 0)
    return (_med([v[0] for v in verts]), _med([v[1] for v in verts]), _med([v[2] for v in verts]))

def _arm_chains_from_verts(verts, cx, cy, sh_x, z_min, height):
    """
    Detect left and right arm bone chains using density-based search.
    More robust than simple spatial thresholds.
    """
    def _find_arm_chain(sign):
        # 1. Candidate limb vertices: exclude core torso
        # Limb zone: height 45-90%, and at least 60% of shoulder width away from center on the correct side
        limb_zone = [v for v in verts 
                    if 0.45 * height < (v[2] - z_min) < 0.90 * height 
                    and (sign * (v[0] - cx)) > (sh_x * 0.6)]
        
        if len(limb_zone) < 20:
            # Fallback to T-pose
            sz = z_min + height * 0.78
            return [
                (cx + sign * sh_x, cy, sz),
                (cx + sign * sh_x * 1.5, cy, sz - height * 0.05),
                (cx + sign * sh_x * 2.0, cy, sz - height * 0.10)
            ]

        # 2. Shoulder: highest vertices in the inner limb zone
        # Use sh_x * 1.5 as outer limit for "inner" shoulder zone
        inner_zone = [v for v in limb_zone if (sign * (v[0] - cx)) < (sh_x * 1.5)]
        if not inner_zone: inner_zone = limb_zone
        shoulder_verts = sorted(inner_zone, key=lambda v: v[2], reverse=True)[:max(5, len(inner_zone)//10)]
        shoulder_pos = _density_center(shoulder_verts)
        
        # 3. Hand: vertices furthest from the shoulder in 3D
        def dist_sq(v1, v2):
            return sum((a - b)**2 for a, b in zip(v1, v2))
        
        furthest_verts = sorted(limb_zone, key=lambda v: dist_sq(v, shoulder_pos), reverse=True)[:max(5, len(limb_zone)//15)]
        hand_pos = _density_center(furthest_verts)
        
        # 4. Elbow: search in the middle segment
        # We look for a density cluster roughly between shoulder and hand
        mid_target = [(shoulder_pos[i] + hand_pos[i])*0.5 for i in range(3)]
        dist_sh_hand = math.sqrt(dist_sq(shoulder_pos, hand_pos))
        elbow_candidates = [v for v in limb_zone 
                           if abs(math.sqrt(dist_sq(v, shoulder_pos)) - dist_sh_hand*0.5) < dist_sh_hand*0.2]
        
        if elbow_candidates:
            elbow_pos = _density_center(elbow_candidates)
        else:
            elbow_pos = mid_target

        # 5. Sanity check: Shoulder Z should be near 78-82%
        sh_z_clamped = max(z_min + height * 0.72, min(z_min + height * 0.86, shoulder_pos[2]))
        shoulder_pos = (shoulder_pos[0], shoulder_pos[1], sh_z_clamped)

        return [shoulder_pos, elbow_pos, hand_pos]

    return _find_arm_chain(-1), _find_arm_chain(1)


def _landmarks_from_verts(verts, sections, z_min, height):
    """
    Derive all bone-placement landmarks from vertex list and cross-sections.

    Returns a dict with every field needed by create_armature().
    Pure Python — no bpy dependency.
    """
    def z_at(frac):
        return z_min + height * frac

    # ── Spine centre X and Y ─────────────────────────────────────────────────────
    # Mesh is centered, but we still detect local center.
    torso_secs = _zone(sections, 0.40, 0.82)
    cx = _med([s['med_x'] for s in torso_secs]) if torso_secs else 0.0
    cy = _med([s['med_y'] for s in torso_secs]) if torso_secs else 0.0

    # ── Shoulder half-width ──────────────────────────────────────────────────────
    # p90–p10 X span at 72–84 % height.
    sh_secs = _zone(sections, 0.72, 0.84)
    sh_spans = [s['span_x'] for s in sh_secs if s['count'] > 20]
    sh_span = _med(sh_spans) if sh_spans else height * 0.45
    
    # We want a bone width that represents the character's core width.
    sh_x = sh_span * 0.40  # Represents the width where the shoulders attach
    sh_x = max(sh_x, height * 0.10)
    sh_x = min(sh_x, height * 0.35)

    # ── Hip half-width ───────────────────────────────────────────────────────────
    # We look for a consistent pelvis width around 40–55 % height.
    hip_secs = _zone(sections, 0.40, 0.55)
    hip_spans = sorted(s['span_x'] for s in hip_secs if s['count'] > 20)
    if hip_spans:
        # Use lower-quartile of widths to avoid catching arm protrusions
        hip_span = hip_spans[len(hip_spans) // 4]
    else:
        hip_span = sh_span * 0.70
    
    hip_x = hip_span * 0.40
    hip_x = max(hip_x, height * 0.08) # Slightly wider floor
    hip_x = min(hip_x, height * 0.25)

    # ── Leg landmarks ───────────────────────────────────────────────────────────
    # Detect where the feet actually are. Look for density in the bottom 15%
    foot_secs = _zone(sections, 0.0, 0.15)
    if foot_secs:
        z_ankle = _med([s['z_ctr'] for s in foot_secs])
    else:
        z_ankle = z_at(0.08)

    # ── Arm chain positions ──────────────────────────────────────────────────────
    arm_l, arm_r = _arm_chains_from_verts(verts, cx, cy, sh_x, z_min, height)

    # ── Symmetry ─────────────────────────────────────────────────────────────────
    # For a humanoid skeleton to work well, we force cx=0, cy=0 as the mesh is centered.
    # We still keep the detected values if they are small.
    if abs(cx) < 0.1: cx = 0.0
    if abs(cy) < 0.1: cy = 0.0

    return dict(
        cx=cx, cy=cy,
        sh_x=sh_x, hip_x=hip_x,
        arm_l=arm_l, arm_r=arm_r,
        z_base=z_min,
        z_pelvis=z_at(0.52),
        z_spine1=z_at(0.60),
        z_spine2=z_at(0.72),
        z_neck=z_at(0.85),
        z_head=z_at(0.93),
        z_knee=(z_at(0.52) + z_ankle) * 0.5, # Midpoint between pelvis and detected ankle
        z_ankle=z_ankle,
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

        # Left arm (Shoulder starts closer to spine for better deformation)
        "shoulder_l":  {"h": (cx - sh_x*0.15, cy, sp2z),     "t": al[0],    "p": "spine_02",   "cat": "limb_upper"},
        "upper_arm_l": {"h": al[0],                           "t": al[1],    "p": "shoulder_l", "cat": "limb_upper"},
        "lower_arm_l": {"h": al[1],                           "t": al[2],    "p": "upper_arm_l","cat": "limb_lower"},
        "hand_l":      {"h": al[2],                           "t": _hand_tip(al), "p": "lower_arm_l","cat": "extremity"},

        # Right arm
        "shoulder_r":  {"h": (cx + sh_x*0.15, cy, sp2z),     "t": ar[0],    "p": "spine_02",   "cat": "limb_upper"},
        "upper_arm_r": {"h": ar[0],                           "t": ar[1],    "p": "shoulder_r", "cat": "limb_upper"},
        "lower_arm_r": {"h": ar[1],                           "t": ar[2],    "p": "upper_arm_r","cat": "limb_lower"},
        "hand_r":      {"h": ar[2],                           "t": _hand_tip(ar), "p": "lower_arm_r","cat": "extremity"},

        # Left leg
        "thigh_l": {"h": (cx - hip_x, cy, pz),  "t": (cx - hip_x, cy, knz), "p": "pelvis",  "cat": "limb_upper"},
        "shin_l":  {"h": (cx - hip_x, cy, knz), "t": (cx - hip_x, cy, anz), "p": "thigh_l", "cat": "limb_lower"},
        "foot_l":  {"h": (cx - hip_x, cy, anz), "t": (cx - hip_x, cy - height*0.08, z0),"p": "shin_l", "cat": "extremity"},

        # Right leg
        "thigh_r": {"h": (cx + hip_x, cy, pz),  "t": (cx + hip_x, cy, knz), "p": "pelvis",  "cat": "limb_upper"},
        "shin_r":  {"h": (cx + hip_x, cy, knz), "t": (cx + hip_x, cy, anz), "p": "thigh_r", "cat": "limb_lower"},
        "foot_r":  {"h": (cx + hip_x, cy, anz), "t": (cx + hip_x, cy - height*0.08, z0),"p": "shin_r", "cat": "extremity"},
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

    # ── Center and Orient Mesh ────────────────────────────────────────────
    # 1. Apply all transforms
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    
    # 2. Auto-Orientation (already existing logic, but let's make it robust)
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

    # 3. Centering: Place character exactly at X=0, Y=0 (keep Z as is)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bb = [mesh_obj.matrix_world @ Vector(corner) for corner in mesh_obj.bound_box]
    curr_cx = (min(v.x for v in bb) + max(v.x for v in bb)) * 0.5
    curr_cy = (min(v.y for v in bb) + max(v.y for v in bb)) * 0.5
    print(f"[blender_auto_rig] Centering mesh from ({curr_cx:.3f}, {curr_cy:.3f}) to (0, 0)")
    mesh_obj.location.x -= curr_cx
    mesh_obj.location.y -= curr_cy
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    print(f"[blender_auto_rig] Imported and centered mesh: {mesh_obj.name}, "
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

    # Enable smooth shading on the mesh before export so the character looks
    # smooth in viewers/engines regardless of polygon count.
    for obj in bpy.context.selected_objects:
        if obj.type == 'MESH':
            obj.data.polygons.foreach_set("use_smooth", [True] * len(obj.data.polygons))
            obj.data.update()

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
