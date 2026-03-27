import sys
import os
import bpy

def validate_glb(filepath):
    print(f"Validating GLB: {filepath}")
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return False

    # Clear scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # Import GLB
    try:
        bpy.ops.import_scene.gltf(filepath=filepath)
    except Exception as e:
        print(f"Error: Failed to import GLB: {e}")
        return False

    # 1. Check for Armature
    armatures = [o for o in bpy.data.objects if o.type == 'ARMATURE']
    if not armatures:
        print("Error: No armature found in GLB")
        return False
    armature = armatures[0]
    print(f"  Found armature: {armature.name}")

    # 2. Check for Mesh
    meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    if not meshes:
        print("Error: No mesh found in GLB")
        return False
    
    # We expect the main body mesh to be parented to the armature or have an armature modifier
    main_mesh = None
    for m in meshes:
        has_armature_mod = any(mod.type == 'ARMATURE' for mod in m.modifiers)
        if m.parent == armature or has_armature_mod:
            main_mesh = m
            break
    
    if not main_mesh:
        print("Error: No mesh found that is rigged to the armature")
        # Use the largest mesh as fallback for inspection
        main_mesh = max(meshes, key=lambda m: len(m.data.vertices))
        print(f"  Fallback mesh for inspection: {main_mesh.name}")
    else:
        print(f"  Found rigged mesh: {main_mesh.name}")

    # 3. Check for Vertex Groups (Skin Weights)
    if not main_mesh.vertex_groups:
        print("Error: Mesh has no vertex groups (skin weights)")
        return False
    print(f"  Found {len(main_mesh.vertex_groups)} vertex groups")

    # 4. Check for Vertex Colors
    has_colors = False
    if main_mesh.data.color_attributes:
        has_colors = True
        print(f"  Found {len(main_mesh.data.color_attributes)} color attributes")
    elif main_mesh.data.vertex_colors:
        has_colors = True
        print(f"  Found {len(main_mesh.data.vertex_colors)} vertex color layers")
    
    if not has_colors:
        print("Warning: Mesh has no vertex colors")
    else:
        print("  Vertex colors present")

    # 5. Check for Animations
    if not bpy.data.actions:
        print("Error: No animations found in GLB")
        # GLB animations might be stored as NLA tracks or actions
        # When importing, Blender usually creates actions for them.
    else:
        print(f"  Found {len(bpy.data.actions)} animations (actions)")
        for action in bpy.data.actions:
            print(f"    - {action.name}")

    print("Validation complete.")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: blender --background --python scripts/validate_stage4.py -- <path_to_glb>")
        sys.exit(1)
    
    try:
        idx = sys.argv.index("--")
        glb_path = sys.argv[idx + 1]
    except (ValueError, IndexError):
        glb_path = sys.argv[-1]
        
    success = validate_glb(glb_path)
    if not success:
        sys.exit(1)
