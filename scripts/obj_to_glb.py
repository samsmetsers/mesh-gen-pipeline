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

    # ── Remove shininess ──────────────────────────────────────────────────
    # Standard OBJ/MTL from PyMeshLab doesn't define PBR properties (Roughness/Metallic).
    # Blender's importer defaults to Principled BSDF with low roughness (0.5) 
    # and 0 metallic, but it can appear very shiny depending on the environment.
    # We force high roughness (matte look) to match the "uniform" stage 2 look.
    for mat in bpy.data.materials:
        if mat.use_nodes:
            nodes = mat.node_tree.nodes
            principled = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
            if principled:
                # Set Roughness to 0.8 (Matte)
                # In Blender 4.0+, the index for Roughness is usually 9, 
                # but accessing by name or internal data is safer.
                if 'Roughness' in principled.inputs:
                    principled.inputs['Roughness'].default_value = 0.8
                if 'Metallic' in principled.inputs:
                    principled.inputs['Metallic'].default_value = 0.0
                # Set Specular to 0 for extra uniformity
                if 'Specular IOR Level' in principled.inputs: # Blender 4.0+
                    principled.inputs['Specular IOR Level'].default_value = 0.0
                elif 'Specular' in principled.inputs: # Older
                    principled.inputs['Specular'].default_value = 0.0

    # Enable smooth shading so the mesh looks smooth in viewers/engines even at
    # low polygon counts.  The GLB exporter stores per-vertex normals which
    # implement smooth-shading in any compliant renderer.
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
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
