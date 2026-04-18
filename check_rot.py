import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath='output/shaman/shaman_animated.glb')
armature = next((o for o in bpy.data.objects if o.type == 'ARMATURE'), None)
if armature:
    action = armature.animation_data.action
    if action:
        print(f"Action: {action.name}")
    hips = armature.pose.bones.get("Hips")
    if hips:
        print(f"Hips rotation: {hips.rotation_quaternion}")
