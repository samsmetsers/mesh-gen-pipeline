"""
OBJ to GLB converter for Stage 3.

Uses Blender (headless) so we can:
  * Import OBJ with MTL + texture sidecar files.
  * Merge multi-object imports into a single mesh (PyMeshLab splits loose
    vertices into separate objects when decimating TRELLIS meshes, which
    bloats the GLB with dozens of 3-vert/1-face stray pieces).
  * Delete degenerate loose parts (< N verts) so the final mesh is clean.
  * Pack the texture atlas back into the GLB (embed_textures=True) so
    downstream stages see real PBR material, not a flat baseColorFactor.
"""
import bpy
import bmesh
import sys
import argparse

# Threshold: loose parts (unconnected vertex islands) with fewer verts than
# this are considered stray/degenerate and removed. PyMeshLab's null-face
# repair routinely produces 3-vert/1-face stubs; the main TRELLIS mesh has
# tens of thousands of verts, so 50 is a safe cut-off.
STRAY_VERT_THRESHOLD = 50


def clean_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def _drop_small_islands(mesh_obj, threshold):
    """
    Remove connected vertex islands with fewer than `threshold` verts.

    Connectivity is computed over edges AND equal positions (rounded to 5
    decimals) so that UV/material seams which produce duplicate verts do not
    fragment a single logical island into thousands of tiny triangles.
    """
    me = mesh_obj.data
    nv = len(me.vertices)
    if nv == 0:
        return

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

    for e in me.edges:
        union(e.vertices[0], e.vertices[1])

    pos_bucket = {}
    for i, v in enumerate(me.vertices):
        key = (round(v.co.x, 5), round(v.co.y, 5), round(v.co.z, 5))
        if key in pos_bucket:
            union(pos_bucket[key], i)
        else:
            pos_bucket[key] = i

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
        print(f"[obj_to_glb] Islands: total={len(sizes)} kept=all")
        return

    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    del_verts = [bm.verts[i] for i in verts_to_del]
    bmesh.ops.delete(bm, geom=del_verts, context='VERTS')
    bm.to_mesh(me)
    bm.free()
    print(f"[obj_to_glb] Islands: total={len(sizes)} kept={len(keep_roots)} "
          f"dropped={len(sizes) - len(keep_roots)}")


def _material_has_texture(mat):
    """True iff the material's node tree references an image texture."""
    if mat is None or not mat.use_nodes or mat.node_tree is None:
        return False
    for node in mat.node_tree.nodes:
        if node.type == 'TEX_IMAGE' and node.image is not None:
            return True
    return False


def _delete_untextured_faces(mesh_obj):
    """
    Keep only faces whose material has an image texture.

    PyMeshLab assigns per-face baseColor-only materials to stray triangles
    produced by its repair pass (each distinct vertex colour becomes its own
    material). The textured PBR atlas we actually care about lives in a single
    material. By deleting every face whose material slot lacks an image texture
    we end up with a single-material, single-primitive mesh.
    """
    me = mesh_obj.data
    textured_slots = {
        i for i, slot in enumerate(mesh_obj.material_slots)
        if _material_has_texture(slot.material)
    }
    if not textured_slots:
        print("[obj_to_glb] No textured material slots found; keeping everything.")
        return
    bm = bmesh.new()
    bm.from_mesh(me)
    to_delete = [f for f in bm.faces if f.material_index not in textured_slots]
    if to_delete:
        bmesh.ops.delete(bm, geom=to_delete, context='FACES')
        # Delete any now-orphan verts.
        orphans = [v for v in bm.verts if not v.link_faces]
        if orphans:
            bmesh.ops.delete(bm, geom=orphans, context='VERTS')
    bm.to_mesh(me)
    bm.free()
    print(f"[obj_to_glb] Kept textured slots={textured_slots}; "
          f"deleted {len(to_delete)} non-textured faces")

    # Now every face uses a textured slot, so the slot cleanup drops the rest.
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode='OBJECT')
    try:
        bpy.ops.object.material_slot_remove_unused()
    except Exception as exc:
        print(f"[obj_to_glb] material_slot_remove_unused failed: {exc}")


def merge_and_clean_meshes():
    """Join all mesh objects into one, then drop stray vertex islands."""
    meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    if not meshes:
        return None

    meshes.sort(key=lambda o: len(o.data.vertices), reverse=True)
    primary = meshes[0]

    bpy.ops.object.select_all(action='DESELECT')
    for m in meshes:
        m.select_set(True)
    bpy.context.view_layer.objects.active = primary
    if len(meshes) > 1:
        bpy.ops.object.join()

    _drop_small_islands(primary, STRAY_VERT_THRESHOLD)
    _delete_untextured_faces(primary)
    return primary


def convert(obj_path, glb_path):
    clean_scene()

    bpy.ops.wm.obj_import(
        filepath=obj_path,
        forward_axis='NEGATIVE_Z',
        up_axis='Y',
    )

    primary = merge_and_clean_meshes()
    if primary is None:
        raise RuntimeError("No mesh produced by OBJ import")

    # Smooth shading + clear custom split normals so Blender-computed smooth
    # normals win in the GLB.
    bpy.context.view_layer.objects.active = primary
    bpy.ops.object.mode_set(mode='OBJECT')
    if primary.data.has_custom_normals:
        bpy.ops.mesh.customdata_custom_splitnormals_clear()
    primary.data.polygons.foreach_set("use_smooth", [True] * len(primary.data.polygons))
    primary.data.update()

    # Export GLB with texture pack-in so the PBR atlas (material_0.png from
    # PyMeshLab save_textures=True) is embedded in the binary.
    bpy.ops.export_scene.gltf(
        filepath=glb_path,
        export_format='GLB',
        use_selection=False,
        export_image_format='AUTO',
        export_materials='EXPORT',
        export_texcoords=True,
        export_normals=True,
    )


if __name__ == "__main__":
    argv = sys.argv
    if "--" in argv:
        args = argv[argv.index("--") + 1:]
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", required=True)
        parser.add_argument("--output", required=True)
        parsed = parser.parse_args(args)
        convert(parsed.input, parsed.output)
