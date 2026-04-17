"""
Simple OBJ to GLB converter for Stage 3.
Uses Blender to ensure axis-correct and geometry-perfect export.
"""
import bpy
import sys
import argparse
import os

def clean_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def convert(obj_path, glb_path):
    clean_scene()
    
    # Import OBJ (Blender's default handles Y-up correctly)
    bpy.ops.wm.obj_import(
        filepath=obj_path,
        forward_axis='NEGATIVE_Z',
        up_axis='Y'
    )

    # ── Smooth Shading ──────────────────────────────────────────────────
    # Enable smooth shading so the mesh looks smooth in viewers/engines even at
    # low polygon counts. The GLB exporter stores per-vertex normals which
    # implement smooth-shading in any compliant renderer.
    # We also clear any imported custom split normals (from the OBJ) so that
    # the smooth shading can take effect properly.
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode='OBJECT')
            if obj.data.has_custom_normals:
                bpy.ops.mesh.customdata_custom_splitnormals_clear()
            obj.data.polygons.foreach_set("use_smooth", [True] * len(obj.data.polygons))
            obj.data.update()

    # Export GLB
    bpy.ops.export_scene.gltf(
        filepath=glb_path,
        export_format='GLB',
        use_selection=False
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
