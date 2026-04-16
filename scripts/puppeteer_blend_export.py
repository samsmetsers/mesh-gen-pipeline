"""
Puppeteer Textured FBX Export
==============================
Replacement for Puppeteer's export.py that preserves UV maps and textures.

Called from Blender:
    blender --background --python scripts/puppeteer_blend_export.py -- \
        --mesh  <refined.obj>  --rig  <skin.txt>  --output  <out.fbx>

Why this exists instead of upstream export.py
----------------------------------------------
Puppeteer's export.py uses trimesh + bpy.data.meshes.from_pydata(), which
copies only vertex positions and faces — UV coordinates and materials are
silently discarded.  This script uses Blender's own wm.obj_import instead,
so the UV atlas and PBR material created by Stage 3 survive into the FBX.

Vertex-weight mapping
---------------------
skin.txt stores weights by "OBJ position-vertex index" (the 1-based index of
a `v` line in the .obj file).  Blender's OBJ importer may split vertices at
UV seams, creating more mesh vertices than there are `v` lines.  We resolve
the mapping by matching Blender vertex world-positions back to the original
`v` positions using a brute-force nearest-neighbour search (numpy only;
12 k-face mesh → < 0.2 s).
"""

from __future__ import annotations

import argparse
import sys
import math

import bpy          # type: ignore
import numpy as np
from mathutils import Vector  # type: ignore


# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--mesh",       required=True,  help="Path to refined .obj")
    p.add_argument("--rig",        required=True,  help="Path to skin .txt from Puppeteer skinning")
    p.add_argument("--output",     required=False, default="out.fbx", help="Output .fbx path")
    p.add_argument("--output-glb", required=False, default="",        help="Also export GLB directly (bypasses FBX round-trip)")
    return p.parse_args(argv)


# ── skin.txt parser ───────────────────────────────────────────────────────────

def _parse_skin_txt(path: str):
    """
    Returns:
        joint_pos   : dict  name → [x, y, z]   (in OBJ / Y-up space)
        joint_hier  : dict  parent_name → [child_name, ...]
        root_name   : str
        skin_weights: dict  vertex_idx (0-based) → {joint_name: weight}
        joint_order : list[str]   insertion order (matches id_mapping in upstream code)
    """
    joint_pos    = {}
    joint_hier   = {}
    skin_weights = {}
    root_name    = None
    joint_order  = []

    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "joints":
                name = parts[1]
                joint_pos[name] = [float(parts[2]), float(parts[3]), float(parts[4])]
                joint_order.append(name)
            elif parts[0] == "root":
                root_name = parts[1]
            elif parts[0] == "hier":
                parent, child = parts[1], parts[2]
                joint_hier.setdefault(parent, []).append(child)
            elif parts[0] == "skin":
                v_idx = int(parts[1])        # 0-based vertex index
                weights = {}
                for k in range(2, len(parts), 2):
                    jname = parts[k]
                    w     = float(parts[k + 1])
                    weights[jname] = w
                skin_weights[v_idx] = weights

    return joint_pos, joint_hier, root_name, skin_weights, joint_order


# ── coordinate helpers ────────────────────────────────────────────────────────

def _yup_to_zup(pos):
    """Rotate a Y-up position to Z-up: x, y, z → x, -z, y"""
    return [pos[0], -pos[2], pos[1]]


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()

    # ── Clear default scene ───────────────────────────────────────────────────
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # ── Import OBJ with UV/materials ──────────────────────────────────────────
    bpy.ops.wm.obj_import(
        filepath=args.mesh,
        forward_axis='NEGATIVE_Z',
        up_axis='Y',
    )

    # Find the imported mesh object
    mesh_obj = None
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            mesh_obj = obj
            break

    if mesh_obj is None:
        raise RuntimeError(f"No mesh found after importing {args.mesh}")

    # Enable smooth shading.  This makes the low-poly mesh look smooth in any
    # GLB viewer / game engine by interpolating vertex normals across faces.
    mesh_obj.data.polygons.foreach_set("use_smooth", [True] * len(mesh_obj.data.polygons))
    mesh_obj.data.update()

    # ── Parse skin.txt ────────────────────────────────────────────────────────
    joint_pos, joint_hier, root_name, skin_weights, joint_order = _parse_skin_txt(args.rig)

    if not joint_pos:
        raise RuntimeError(f"No joints found in {args.rig}")

    # ── Build vertex-index mapping ────────────────────────────────────────────
    # skin.txt uses 0-based OBJ-position-vertex indices.
    # We need to map each Blender mesh vertex → its OBJ position-vertex index.
    #
    # Strategy: parse `v` lines from the .obj to get the list of original
    # positions (in Y-up space), then for every Blender vertex find the
    # closest original position.
    obj_positions_yup = []
    with open(args.mesh) as f:
        for line in f:
            if line.startswith('v '):
                _, x, y, z = line.split()[:4]
                obj_positions_yup.append([float(x), float(y), float(z)])

    # Convert to Z-up to match Blender's imported positions
    obj_positions_zup = np.array([_yup_to_zup(p) for p in obj_positions_yup], dtype=np.float32)

    # Blender vertex positions (already in Z-up)
    bverts = mesh_obj.data.vertices
    blender_positions = np.array([[v.co.x, v.co.y, v.co.z] for v in bverts], dtype=np.float32)

    # For each Blender vertex find nearest OBJ position vertex (brute-force; fast at 12 k faces)
    # blender_to_obj_idx[i] = OBJ position index for Blender vertex i
    diff   = blender_positions[:, None, :] - obj_positions_zup[None, :, :]   # (B, N, 3)
    dist2  = (diff ** 2).sum(axis=2)                                          # (B, N)
    blender_to_obj_idx = dist2.argmin(axis=1)                                 # (B,)

    # ── Create armature ───────────────────────────────────────────────────────
    arm_data = bpy.data.armatures.new("Armature")
    arm_obj  = bpy.data.objects.new("Armature", arm_data)
    bpy.context.scene.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')

    edit_bones = arm_data.edit_bones

    parent_map = {}  # child_name → parent_name
    for parent, children in joint_hier.items():
        for child in children:
            parent_map[child] = parent

    def _add_bone(name: str, head_zup, tail_zup):
        b = edit_bones.new(name)
        b.head = Vector(head_zup)
        b.tail = Vector(tail_zup)
        return b

    # Compute tail positions (point toward first child, or extrude from parent direction)
    def _tail(name: str):
        pos  = joint_pos[name]           # Y-up
        children = joint_hier.get(name, [])
        if len(children) == 1:
            return _yup_to_zup(joint_pos[children[0]])
        elif len(children) > 1:
            # average child direction
            avg = [sum(joint_pos[c][i] for c in children) / len(children) for i in range(3)]
            d   = [avg[i] - pos[i] for i in range(3)]
            mag = math.sqrt(sum(x*x for x in d)) or 0.05
            return [_yup_to_zup(pos)[i] + _yup_to_zup(d)[i] / mag * 0.05 for i in range(3)]
        elif name in parent_map:
            ppos = joint_pos[parent_map[name]]
            d    = [pos[i] - ppos[i] for i in range(3)]
            mag  = math.sqrt(sum(x*x for x in d)) or 0.05
            ph   = _yup_to_zup(pos)
            dz   = _yup_to_zup(d)
            return [ph[i] + dz[i] / mag * 0.05 for i in range(3)]
        else:
            ph = _yup_to_zup(pos)
            return [ph[0], ph[1], ph[2] + 0.05]

    # Create bones in joint_order for deterministic ordering
    for name in joint_order:
        head = _yup_to_zup(joint_pos[name])
        tail = _tail(name)
        _add_bone(name, head, tail)

    # Set parents
    for name in joint_order:
        if name in parent_map:
            bone   = edit_bones.get(name)
            parent = edit_bones.get(parent_map[name])
            if bone and parent:
                bone.parent       = parent
                bone.use_connect  = False

    bpy.ops.object.mode_set(mode='OBJECT')

    # ── Create vertex groups on mesh ──────────────────────────────────────────
    # Pre-create all groups
    for name in joint_order:
        mesh_obj.vertex_groups.new(name=name)

    # Assign weights
    # skin_weights keys are 0-based OBJ vertex indices
    # We need: for each Blender vertex i, what OBJ index does it map to?
    for blender_idx, obj_idx in enumerate(blender_to_obj_idx):
        weights = skin_weights.get(int(obj_idx), {})
        for joint_name, weight in weights.items():
            if weight > 0 and joint_name in mesh_obj.vertex_groups:
                mesh_obj.vertex_groups[joint_name].add([blender_idx], float(weight), 'REPLACE')

    # ── Parent mesh to armature ───────────────────────────────────────────────
    bpy.context.view_layer.objects.active = arm_obj
    mesh_obj.select_set(True)
    arm_obj.select_set(True)
    bpy.ops.object.parent_set(type='ARMATURE_NAME')

    # ── Fix normals, shading, and materials before any export ────────────────
    # Apply now (before FBX + optional GLB export) so both outputs are clean.
    #
    # Why normals_make_consistent here:
    #   PyMeshLab's meshing_re_orient_faces_coherently() makes normals
    #   consistent but doesn't guarantee they point outward on open meshes.
    #   Recalculating in Blender at rest-pose catches any remaining inward
    #   faces cleanly because the OBJ has no transforms yet.
    #
    # Why NOT transform_apply here:
    #   The OBJ was imported with forward_axis='NEGATIVE_Z', up_axis='Y', which
    #   is a pure rotation — det=+1, no scale flip, winding is preserved.
    #   transform_apply on a mesh that will be bound to an armature can move
    #   vertices relative to bone positions and corrupt skinning.  We leave the
    #   Blender-native Y→Z rotation as an object transform; GLTF export handles
    #   the final Y-up conversion automatically.
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')

    mesh_obj.data.polygons.foreach_set("use_smooth", [True] * len(mesh_obj.data.polygons))
    mesh_obj.data.update()

    for mat in bpy.data.materials:
        mat.use_backface_culling = False
        if not mat.use_nodes:
            continue
        principled = next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
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

    # ── Export FBX ────────────────────────────────────────────────────────────
    bpy.ops.export_scene.fbx(
        filepath=args.output,
        check_existing=False,
        add_leaf_bones=False,
        path_mode='COPY',
        embed_textures=True,
    )
    print(f"[puppeteer_blend_export] FBX exported to {args.output}")

    # ── Export GLB directly (bypasses the FBX → GLB round-trip) ─────────────
    # Exporting GLB from the same Blender session that imported the OBJ gives:
    #   1. Correct PBR materials (no lossy FBX Phong round-trip)
    #   2. Correct winding order (OBJ import is a pure rotation, no scale flip)
    #   3. Correct bone axes (no FBX coordinate transform re-applied)
    # This GLB becomes final.glb (Stage 4 output) and is what Stage 5 animates.
    if args.output_glb:
        bpy.ops.export_scene.gltf(
            filepath=args.output_glb,
            export_format='GLB',
            export_image_format='AUTO',
            export_texcoords=True,
            export_normals=True,
            export_materials='EXPORT',
            export_animations=False,
        )
        print(f"[puppeteer_blend_export] GLB exported to {args.output_glb}")

    print(f"[puppeteer_blend_export] Done.")


if __name__ == "__main__":
    main()
