"""
Blender headless rig standardization.

Takes:
  * --input           : UniRig output GLB (armature + rigged mesh, NO texture
                        because UniRig's merge step drops the PBR atlas).
  * --textured-glb    : The textured mesh from Stage 3 (refined.glb). Its PBR
                        material is copied onto the UniRig-rigged mesh.
  * --masks           : P3-SAM segmentation, used to spawn Grip_* bones.
  * --output-fbx/-glb : Where to save the final rig + mesh + texture.

What we do:
  1. Import UniRig GLB (armature + mesh, flat gray).
  2. Import textured refined GLB → steal its PBR material.
  3. Drop stray loose parts (< STRAY_VERT_THRESHOLD verts) and re-join the
     UniRig mesh into one clean object.
  4. Assign the stolen material to the mesh, rename UV to match.
  5. Rename bones (Mixamo/Unity-style), add Grip_* bones for weapons, add
     twist bones for limbs.
  6. Smooth shading, export FBX + GLB (with embedded textures).
"""
import bpy
import bmesh
import sys
import argparse
import json
import os
from mathutils import Vector

STRAY_VERT_THRESHOLD = 50


def _rename_bones_by_topology(armature):
    """
    Label the UniRig skeleton with Mixamo-style names using topology + rest
    pose positions — UniRig's articulation-XL checkpoint emits bone_0..bone_N.

    Heuristic:
      * Root bone (parent=None)  → Hips
      * From Hips, classify three main branches by rest-pose direction:
          upward (dy>0)       → Spine (then Spine1/2, Neck, Head)
          downward-left (dy<0, dx<0)  → LeftUpperLeg chain
          downward-right (dy<0, dx>0) → RightUpperLeg chain
      * From Spine-top, find two side branches (dx<0 / dx>0) → Left/Right
        Shoulder → Upper/Lower Arm → Hand.

    Bones that don't fit any chain keep their original names (e.g. bone_37).
    The retargeter skips bones it can't address by name, so unknowns are a
    no-op rather than an error.
    """
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    eb = armature.data.edit_bones

    # Find root (no parent).
    roots = [b for b in eb if b.parent is None]
    if not roots:
        bpy.ops.object.mode_set(mode='OBJECT')
        return
    # Determine the "up" axis automatically. After GLTF import Blender uses
    # Z-up; after FBX or OBJ import it may differ. The vertical axis is the
    # one with the widest spread across bone positions (humanoids are ~2 m
    # tall and ~0.5 m wide).
    bone_positions = [b.head for b in eb]
    spans = [max(p[i] for p in bone_positions) - min(p[i] for p in bone_positions)
             for i in range(3)]
    up_axis = spans.index(max(spans))  # 0=x, 1=y, 2=z
    side_axis = (up_axis + 2) % 3 if spans[(up_axis + 2) % 3] > spans[(up_axis + 1) % 3] \
        else (up_axis + 1) % 3

    # Pick the root at the median height (the real pelvis, not the toes).
    def _up(b):
        return b.head[up_axis]
    root = min(roots, key=lambda b: abs(b.head[side_axis]))
    root.name = "Hips"
    root_up = _up(root)
    root_side = root.head[side_axis]

    def _follow_chain(start, names):
        """Walk single-child chain and apply names in order."""
        cur = start
        for name in names:
            if cur is None:
                break
            cur.name = name
            children = [c for c in eb if c.parent == cur]
            cur = children[0] if len(children) == 1 else None
        return cur

    children = [b for b in eb if b.parent == root]

    def _min_subtree_up(bone, depth=0):
        """Minimum up-axis position found in the bone's subtree (head+tail)."""
        if depth > 10:
            return bone.head[up_axis]
        vals = [bone.head[up_axis], bone.tail[up_axis]]
        for c in eb:
            if c.parent is bone:
                vals.append(_min_subtree_up(c, depth + 1))
        return min(vals)

    def _max_subtree_up(bone, depth=0):
        if depth > 10:
            return bone.head[up_axis]
        vals = [bone.head[up_axis], bone.tail[up_axis]]
        for c in eb:
            if c.parent is bone:
                vals.append(_max_subtree_up(c, depth + 1))
        return max(vals)

    # "Up" children: subtree reaches well above the root — the spine chain.
    up = [c for c in children if _max_subtree_up(c) > root_up + 0.1]
    # "Down" children: subtree reaches well below the root — leg chains.
    # Hip bones themselves are often at pelvis height; only the knee/foot
    # descendants are actually below the pelvis, so we inspect the subtree.
    downers = [c for c in children if _min_subtree_up(c) < root_up - 0.1]
    down_left = [c for c in downers if c.head[side_axis] < root_side - 0.01]
    down_right = [c for c in downers if c.head[side_axis] > root_side + 0.01]

    # --- Spine chain ---
    if up:
        # Take the tallest upward child as Spine.
        spine = max(up, key=_up)
        _follow_chain(spine, ["Spine", "Spine1", "Spine2", "Neck", "Head"])

        # From somewhere along the spine there should be two side-branches for
        # shoulders. Walk the spine's children list until we find a bone with
        # a pair of off-centre children.
        def first_child(bone):
            kids = [c for c in eb if c.parent == bone]
            return kids[0] if kids else None

        candidate = spine
        while candidate is not None:
            kids = [c for c in eb if c.parent == candidate]
            sides = [c for c in kids if abs(c.head[side_axis] - root_side) > 0.02]
            if len(sides) >= 2:
                sides.sort(key=lambda b: b.head[side_axis])
                left_sh, right_sh = sides[0], sides[-1]
                left_sh.name = "LeftShoulder"
                right_sh.name = "RightShoulder"
                _follow_chain(first_child(left_sh),
                              ["LeftUpperArm", "LeftLowerArm", "LeftHand"])
                _follow_chain(first_child(right_sh),
                              ["RightUpperArm", "RightLowerArm", "RightHand"])
                # Continue upward via non-side child → Neck, Head.
                up_child = next(
                    (c for c in kids
                     if abs(c.head[side_axis] - root_side) <= 0.02),
                    None,
                )
                if up_child is not None:
                    _follow_chain(up_child, ["Neck", "Head"])
                break
            if len(kids) == 1:
                candidate = kids[0]
            else:
                break

    # --- Leg chains ---
    if down_left:
        lhip = min(down_left, key=lambda b: b.head[side_axis])
        _follow_chain(lhip, ["LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes"])
    if down_right:
        rhip = max(down_right, key=lambda b: b.head[side_axis])
        _follow_chain(rhip, ["RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes"])

    bpy.ops.object.mode_set(mode='OBJECT')


def _import_glb(path):
    """Import a GLB and return (armature|None, meshes list)."""
    pre = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in pre]
    armature = next((o for o in new if o.type == 'ARMATURE'), None)
    meshes = [o for o in new if o.type == 'MESH']
    return armature, meshes


def _drop_small_islands(mesh_obj, threshold):
    """
    Delete connected vertex islands with fewer than `threshold` verts.

    GLTF imports duplicate verts at UV/material boundaries, so two-triangle
    strips that look contiguous to the human eye can share zero edges. We
    treat two verts at the same position (quantised to 5 decimal places) as
    part of the same island — equivalent to an imaginary "zero-length edge"
    weld without actually modifying UVs or materials.
    """
    me = mesh_obj.data
    nv = len(me.vertices)
    if nv == 0:
        return

    # Union-Find over vertex indices.
    parent = list(range(nv))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # 1. Union verts connected by an edge (as imported).
    for e in me.edges:
        union(e.vertices[0], e.vertices[1])

    # 2. Union verts that share a position (welds duplicate seam verts).
    pos_bucket = {}
    for i, v in enumerate(me.vertices):
        key = (round(v.co.x, 5), round(v.co.y, 5), round(v.co.z, 5))
        if key in pos_bucket:
            union(pos_bucket[key], i)
        else:
            pos_bucket[key] = i

    # Collect islands.
    islands = {}
    for i in range(nv):
        r = find(i)
        islands.setdefault(r, []).append(i)
    sizes = sorted(islands.items(), key=lambda kv: len(kv[1]), reverse=True)

    keep_roots = {sizes[0][0]}
    for root, members in sizes[1:]:
        if len(members) >= threshold:
            keep_roots.add(root)

    verts_to_del = {i for i in range(nv) if find(i) not in keep_roots}
    if not verts_to_del:
        print(f"[blender_standardize] islands: total={len(sizes)} kept=all (no tiny)")
        return

    # Execute deletion via bmesh.
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    del_verts = [bm.verts[i] for i in verts_to_del]
    bmesh.ops.delete(bm, geom=del_verts, context='VERTS')
    bm.to_mesh(me)
    bm.free()
    print(f"[blender_standardize] islands: total={len(sizes)} kept={len(keep_roots)} "
          f"dropped={len(sizes) - len(keep_roots)} (verts removed: {len(verts_to_del)})")


def _join_and_clean(meshes):
    """Join meshes, then strip tiny disconnected islands and unused mat slots."""
    if not meshes:
        return None
    meshes.sort(key=lambda o: len(o.data.vertices), reverse=True)
    primary = meshes[0]

    # Delete all secondary meshes (like Unirig joint markers) instead of joining
    if len(meshes) > 1:
        for m in meshes[1:]:
            bpy.data.objects.remove(m, do_unlink=True)
    
    bpy.ops.object.select_all(action='DESELECT')
    primary.select_set(True)
    bpy.context.view_layer.objects.active = primary

    _drop_small_islands(primary, STRAY_VERT_THRESHOLD)
    try:
        bpy.ops.object.material_slot_remove_unused()
    except Exception as exc:
        print(f"[blender_standardize] material_slot_remove_unused failed: {exc}")
    return primary


def _copy_material(src_mesh, dst_mesh):
    """Replace dst_mesh's materials with src_mesh's, keeping UV intact."""
    if not src_mesh or not src_mesh.data.materials:
        print("[blender_standardize] Source has no materials; skipping copy")
        return False

    # Clear destination slots and reassign every face to slot 0, otherwise
    # leftover material_index values on faces would reference slots that no
    # longer exist.
    dst_mesh.data.materials.clear()
    for m in src_mesh.data.materials:
        dst_mesh.data.materials.append(m)

    for poly in dst_mesh.data.polygons:
        poly.material_index = 0

    if dst_mesh.data.uv_layers and src_mesh.data.uv_layers:
        dst_uv = dst_mesh.data.uv_layers.active
        if dst_uv is not None:
            dst_uv.name = src_mesh.data.uv_layers.active.name
    print(f"[blender_standardize] Copied {len(src_mesh.data.materials)} material(s)")
    return True


def main():
    argv = sys.argv
    if "--" not in argv:
        return
    args = argv[argv.index("--") + 1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="UniRig rigged GLB")
    parser.add_argument("--textured-glb", required=False,
                        help="Source textured mesh (Stage 3 refined.glb) — "
                             "its PBR material is copied onto the rigged mesh.")
    parser.add_argument("--output-fbx", required=True)
    parser.add_argument("--output-glb", required=True)
    parser.add_argument("--masks", required=False)
    parsed = parser.parse_args(args)

    # 1. Clean scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # 2. Import UniRig GLB (rig + flat-gray mesh)
    armature, rig_meshes = _import_glb(parsed.input)
    if not armature:
        print("[blender_standardize] WARNING: no armature in UniRig GLB")
    main_mesh = _join_and_clean(rig_meshes)

    # 3. Import textured GLB in a separate link to pull the PBR material.
    if parsed.textured_glb and os.path.exists(parsed.textured_glb) and main_mesh:
        tex_armature, tex_meshes = _import_glb(parsed.textured_glb)
        if tex_meshes:
            tex_meshes.sort(key=lambda o: len(o.data.vertices), reverse=True)
            _copy_material(tex_meshes[0], main_mesh)
            # Remove the scaffolding mesh+armature we imported purely to
            # harvest materials.
            for m in tex_meshes:
                bpy.data.objects.remove(m, do_unlink=True)
            if tex_armature:
                bpy.data.objects.remove(tex_armature, do_unlink=True)

    if armature:
        # 4. Rename bones. UniRig's articulation-XL checkpoint emits anonymous
        #    bone_0..bone_N names, so literal string mapping catches nothing.
        #    We fall back to a topology+position heuristic: walk the skeleton
        #    tree, classify each chain by rest-pose position, and stamp on
        #    Mixamo-style names the retargeter knows how to address.
        bpy.context.view_layer.objects.active = armature
        _rename_bones_by_topology(armature)

        # 5. Grip bones for weapons (from P3-SAM masks).
        if parsed.masks and os.path.exists(parsed.masks):
            with open(parsed.masks, "r") as f:
                mask_data = json.load(f)
            parts = mask_data.get("parts", [])
            if len(parts) > 1:
                # Skip the first part (the main body)
                accessories = parts[1:]
                # Sort accessories by size (number of faces)
                accessories.sort(key=lambda p: len(p.get("face_indices", [])), reverse=True)
                
                left_arm_bones = {"LeftHand", "LeftLowerArm", "LeftUpperArm", "LeftShoulder"}
                right_arm_bones = {"RightHand", "RightLowerArm", "RightUpperArm", "RightShoulder"}
                
                left_weapon_center = None
                right_weapon_center = None
                
                bpy.ops.object.mode_set(mode='EDIT')
                for part in accessories:
                    aabb = part.get("aabb")
                    if not aabb:
                        continue
                    center = Vector([
                        (aabb[0] + aabb[3]) / 2,
                        (aabb[1] + aabb[4]) / 2,
                        (aabb[2] + aabb[5]) / 2,
                    ])
                    
                    # Find closest bone
                    closest_bone = None
                    min_dist = float('inf')
                    for b in armature.data.edit_bones:
                        if b.name.startswith("Grip_"): continue
                        dist = (center - b.head).length
                        if dist < min_dist:
                            min_dist = dist
                            closest_bone = b
                    
                    if closest_bone:
                        if closest_bone.name in left_arm_bones and left_weapon_center is None:
                            left_weapon_center = center
                        elif closest_bone.name in right_arm_bones and right_weapon_center is None:
                            right_weapon_center = center

                if left_weapon_center:
                    l_hand = armature.data.edit_bones.get("LeftHand")
                    if l_hand:
                        grip_l = armature.data.edit_bones.new("Grip_L")
                        grip_l.head = left_weapon_center
                        grip_l.tail = left_weapon_center + Vector((0, 0.1, 0))
                        grip_l.parent = l_hand
                if right_weapon_center:
                    r_hand = armature.data.edit_bones.get("RightHand")
                    if r_hand:
                        grip_r = armature.data.edit_bones.new("Grip_R")
                        grip_r.head = right_weapon_center
                        grip_r.tail = right_weapon_center + Vector((0, 0.1, 0))
                        grip_r.parent = r_hand
                bpy.ops.object.mode_set(mode='OBJECT')

        # 6. Twist bones.
        bpy.ops.object.mode_set(mode='EDIT')
        twist_limbs = [
            ("LeftUpperArm", "LeftLowerArm"),
            ("RightUpperArm", "RightLowerArm"),
            ("LeftLowerArm", "LeftHand"),
            ("RightLowerArm", "RightHand"),
            ("LeftUpperLeg", "LeftLowerLeg"),
            ("RightUpperLeg", "RightLowerLeg"),
            ("LeftLowerLeg", "LeftFoot"),
            ("RightLowerLeg", "RightFoot"),
        ]
        for parent_name, child_name in twist_limbs:
            parent = armature.data.edit_bones.get(parent_name)
            child = armature.data.edit_bones.get(child_name)
            if parent and child:
                tname = f"{parent_name}_Twist"
                if tname not in armature.data.edit_bones:
                    twist = armature.data.edit_bones.new(tname)
                    mid = (parent.head + parent.tail) / 2
                    twist.head = mid
                    twist.tail = (mid + parent.tail) / 2
                    twist.parent = parent
        bpy.ops.object.mode_set(mode='OBJECT')

    # 7. Smooth shading on the final mesh.
    if main_mesh:
        bpy.context.view_layer.objects.active = main_mesh
        bpy.ops.object.mode_set(mode='OBJECT')
        if main_mesh.data.has_custom_normals:
            bpy.ops.mesh.customdata_custom_splitnormals_clear()
        main_mesh.data.polygons.foreach_set(
            "use_smooth", [True] * len(main_mesh.data.polygons)
        )
        main_mesh.data.update()

    # 8. Export FBX + GLB (embed_textures=True via export_image_format='AUTO')
    bpy.ops.export_scene.fbx(
        filepath=parsed.output_fbx,
        use_selection=False,
        object_types={'ARMATURE', 'MESH'},
        add_leaf_bones=True,
        bake_anim=False,
        embed_textures=True,
        path_mode='COPY',
    )
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


if __name__ == "__main__":
    main()

