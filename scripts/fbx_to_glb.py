"""
FBX → GLB converter for Stage 4 (rigged character).

Called from Blender:
    blender --background --python scripts/fbx_to_glb.py -- \
        --input  <rigged.fbx>  --output  <rigged.glb>

Why a dedicated script instead of an inline --python-expr
---------------------------------------------------------
The inline one-liner in _convert_fbx_to_glb had four problems:

1. **Inverted winding order due to negative scale**: Blender's FBX importer
   sometimes introduces a negative scale on an axis to handle the Y-up → Z-up
   coordinate flip.  A negative scale flips the winding order of all triangles
   in world space, making every face a back-face → see-through mesh.
   Fix: `transform_apply(scale=True)` bakes the scale into vertex positions,
   correcting winding before any normal work is done.

2. **Residual inverted normals**: Even after step 1, some faces may point
   inward on non-manifold or open meshes.  Fix: `normals_make_consistent
   (inside=False)` in Edit Mode to flip remaining inward faces outward.

3. **doubleSided safety net**: GLTF viewers honour the per-material
   `doubleSided` flag.  When Blender's material has "Backface Culling" ON
   (which can happen after FBX import), the GLB exports with
   `doubleSided:false` — any residual winding error makes part of the mesh
   invisible.  Fix: `use_backface_culling = False` → `doubleSided:true`.

4. **Flat / faceted shading + glistening material**: shade-smooth flag and
   PBR roughness are not preserved across FBX.  Same fix as obj_to_glb.py:
   re-apply shade-smooth and force Roughness=0.8 / Metallic=Specular=0.
"""

from __future__ import annotations

import argparse
import sys

import bpy  # type: ignore


def _parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True, help="Input FBX path")
    p.add_argument("--output", required=True, help="Output GLB path")
    return p.parse_args(argv)


def _fix_mesh(obj) -> None:
    """Apply transforms, fix winding order, and enable shade-smooth on one mesh."""
    # Step 1: Apply scale (bakes any negative-scale axis flip into vertices so
    # winding order is correct in world space before we touch normals).
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

    # Step 2: Recalculate face normals outward (fixes any residual inward faces).
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Step 3: Shade smooth (not preserved across FBX round-trip).
    obj.data.polygons.foreach_set("use_smooth", [True] * len(obj.data.polygons))
    obj.data.update()


def _fix_materials() -> None:
    """Force matte PBR values and double-sided rendering on every material.

    doubleSided=true (use_backface_culling=False) is the belt-and-suspenders
    guard: even if a viewer ignores winding order, both sides are visible.
    Roughness=0.8 / Metallic=0.0 / Specular=0.0 prevents the glistening
    look caused by FBX Phong → Principled BSDF reconstruction with low Ns.
    """
    for mat in bpy.data.materials:
        # Double-sided: GLTF exports this as doubleSided:true
        mat.use_backface_culling = False

        if not mat.use_nodes:
            continue
        principled = next(
            (n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'),
            None,
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


def main() -> None:
    args = _parse_args()

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    print(f"[fbx_to_glb] Importing {args.input}")
    bpy.ops.import_scene.fbx(filepath=args.input)

    for obj in list(bpy.data.objects):
        if obj.type == 'MESH':
            _fix_mesh(obj)

    _fix_materials()

    print(f"[fbx_to_glb] Exporting {args.output}")
    bpy.ops.export_scene.gltf(
        filepath=args.output,
        export_format='GLB',
        export_image_format='AUTO',
        export_texcoords=True,
        export_normals=True,
        export_materials='EXPORT',
        export_animations=True,
    )
    print("[fbx_to_glb] Done.")


if __name__ == "__main__":
    main()
