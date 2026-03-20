import bpy
import sys
import argparse
import math

def setup_scene(mesh_path, output_path):
    """
    Imports the mesh, applies continuous rotation, and exports the scene.
    """
    # 1. Clear existing objects
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # 2. Import the mesh (OBJ or GLB/FBX)
    if mesh_path.endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=mesh_path)
    elif mesh_path.endswith(".glb") or mesh_path.endswith(".gltf"):
        bpy.ops.import_scene.gltf(filepath=mesh_path)
    
    # 3. Apply procedural rotation kinematics
    obj = bpy.context.selected_objects[0]
    obj.rotation_mode = 'XYZ'
    
    # Insert keyframes for continuous rotation
    start_frame = 1
    end_frame = 100
    bpy.context.scene.frame_start = start_frame
    bpy.context.scene.frame_end = end_frame
    
    # Initial frame
    obj.rotation_euler = (0, 0, 0)
    obj.keyframe_insert(data_path="rotation_euler", frame=start_frame)
    
    # Final frame (360-degree rotation)
    obj.rotation_euler = (0, 0, 2 * math.pi)
    obj.keyframe_insert(data_path="rotation_euler", frame=end_frame)
    
    # Make rotation linear
    for fcurve in obj.animation_data.action.fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = 'LINEAR'
            
    # 4. Export the final animated scene
    bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLB')
    print(f"Blender: Exported animated scene to {output_path}")

def main():
    # Parse arguments passed after "--"
    args = sys.argv[sys.argv.index("--") + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parsed_args = parser.parse_args(args)
    
    setup_scene(parsed_args.input, parsed_args.output)

if __name__ == "__main__":
    main()
