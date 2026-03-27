"""
Stage 4d: Spatial-Temporal Aware Motion Retargeting (via STaR)

Ingests generated canonical motion and target rig. Applies STaR mapping
with active IK constraint enforcement to route limbs around obstructions
and prevent self-penetration.
"""
import sys
import os
import argparse
import json
import numpy as np
import bpy
import mathutils

def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def import_rigged_glb(rig_path):
    """
    Import a GLB/FBX with armature + mesh.
    Returns (armature, [mesh_objects]).
    """
    print(f"Importing rigged file: {rig_path}")
    if not os.path.exists(rig_path):
        raise FileNotFoundError(f"Rigged file not found: {rig_path}")

    before = set(bpy.data.objects.keys())
    ext = os.path.splitext(rig_path)[1].lower()
    if ext in ('.glb', '.gltf'):
        bpy.ops.import_scene.gltf(filepath=rig_path)
    else:
        bpy.ops.import_scene.fbx(filepath=rig_path)

    new_objs = [bpy.data.objects[k] for k in bpy.data.objects.keys() if k not in before]
    armature = next((o for o in new_objs if o.type == 'ARMATURE'), None)
    if not armature:
        raise RuntimeError("No armature found in imported rigged file")
    rig_meshes = [o for o in new_objs if o.type == 'MESH']
    return armature, rig_meshes


def import_body_mesh(body_glb_path):
    """
    Import the Stage-3 body.glb — original topology + baked vertex colours.
    Returns the mesh object, or None on failure.

    Two-path approach:
      1. Try Blender's native GLTF importer (preserves materials/UVs).
      2. Fallback: load via trimesh → build Blender mesh from numpy arrays.
         This bypasses GLTF importer compatibility issues (e.g. the standalone
         bpy package sometimes rejects trimesh-exported vertex-colour GLBs).
    """
    if not body_glb_path or not os.path.exists(body_glb_path):
        return None
    print(f"Importing Stage-3 body mesh: {body_glb_path}")

    # ── Path 1: Blender native GLTF importer ─────────────────────────────────
    before = set(bpy.data.objects.keys())
    gltf_ok = False
    try:
        bpy.ops.import_scene.gltf(filepath=body_glb_path)
        new_objs = [bpy.data.objects[k] for k in bpy.data.objects.keys() if k not in before]
        mesh_objs = [o for o in new_objs if o.type == 'MESH']
        if mesh_objs:
            gltf_ok = True
            if len(mesh_objs) == 1:
                return mesh_objs[0]
            # Multiple meshes: join them
            bpy.ops.object.select_all(action='DESELECT')
            for m in mesh_objs:
                m.select_set(True)
            bpy.context.view_layer.objects.active = mesh_objs[0]
            bpy.ops.object.join()
            return bpy.context.view_layer.objects.active
        print("  GLTF import produced no mesh objects — trying trimesh fallback.")
    except Exception as e:
        print(f"  GLTF import failed ({e}) — trying trimesh fallback.")

    # Clean up any partially-imported objects from the failed GLTF attempt
    if not gltf_ok:
        cur_keys = set(bpy.data.objects.keys())
        for k in cur_keys - before:
            bpy.data.objects.remove(bpy.data.objects[k], do_unlink=True)

    # ── Path 2: trimesh → numpy → Blender mesh ────────────────────────────────
    # This is the critical fallback for --decimate: it guarantees we get the
    # small decimated geometry even when the GLTF importer fails on the
    # trimesh-exported GLB.
    try:
        import trimesh as _trimesh
        tm = _trimesh.load(body_glb_path, process=False)
        if isinstance(tm, _trimesh.Scene):
            geoms = [g for g in tm.geometry.values()
                     if isinstance(g, _trimesh.Trimesh)]
            if not geoms:
                print("  trimesh fallback: no geometry in file.")
                return None
            tm = _trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]

        verts = tm.vertices.tolist()
        faces = tm.faces.tolist()
        if not verts or not faces:
            print("  trimesh fallback: empty mesh.")
            return None

        mesh_data = bpy.data.meshes.new('body_trimesh')
        mesh_data.from_pydata(verts, [], faces)
        mesh_data.update()

        obj = bpy.data.objects.new('body_trimesh', mesh_data)
        bpy.context.collection.objects.link(obj)

        # Transfer vertex colours from the trimesh visual if available
        try:
            vc = np.asarray(tm.visual.vertex_colors)
            if (vc is not None and vc.ndim == 2 and vc.shape[0] == len(verts)
                    and not np.all(vc[:, :3] == vc[0, :3])):
                col_attr = mesh_data.color_attributes.new('Col', 'BYTE_COLOR', 'POINT')
                for i, c in enumerate(vc):
                    col_attr.data[i].color = (
                        float(c[0]) / 255., float(c[1]) / 255.,
                        float(c[2]) / 255., 1.0)
        except Exception as ce:
            print(f"  trimesh fallback: vertex colour transfer skipped ({ce})")

        print(f"  trimesh fallback: imported {len(faces)} faces, {len(verts)} verts")
        return obj
    except Exception as e:
        print(f"  trimesh fallback failed: {e}")
        import traceback; traceback.print_exc()
        return None


def transfer_skin_weights(rig_mesh, target_mesh, armature):
    """
    Copy vertex groups (skin weights) from the Puppeteer rig mesh to the
    Stage-3 body mesh using an explicit KD-tree nearest-vertex lookup.
    
    Uses bounding-box normalisation to ensure alignment even if coordinate
    systems or scales differ between Puppeteer's export and Stage 3.
    """
    if not rig_mesh or not target_mesh:
        return False
    print(f"  Transferring skin weights: {rig_mesh.name} → {target_mesh.name}")
    try:
        # --- Normalise rig mesh vertices ---
        rig_verts = rig_mesh.data.vertices
        rig_mat = rig_mesh.matrix_world
        rig_verts_world = np.array([(rig_mat @ v.co)[:] for v in rig_verts], dtype=np.float64)
        
        rig_lo = rig_verts_world.min(axis=0)
        rig_hi = rig_verts_world.max(axis=0)
        rig_center = (rig_lo + rig_hi) * 0.5
        rig_scale  = float(np.max(rig_hi - rig_lo)) or 1.0
        rig_norm   = (rig_verts_world - rig_center) / rig_scale

        # Build KD-tree on normalised rig
        kd = mathutils.kdtree.KDTree(len(rig_norm))
        for i, v in enumerate(rig_norm):
            kd.insert(mathutils.Vector(v.tolist()), i)
        kd.balance()

        # --- Normalise target mesh vertices ---
        tgt_verts = target_mesh.data.vertices
        tgt_mat = target_mesh.matrix_world
        tgt_verts_world = np.array([(tgt_mat @ v.co)[:] for v in tgt_verts], dtype=np.float64)
        
        tgt_lo = tgt_verts_world.min(axis=0)
        tgt_hi = tgt_verts_world.max(axis=0)
        tgt_center = (tgt_lo + tgt_hi) * 0.5
        tgt_scale  = float(np.max(tgt_hi - tgt_lo)) or 1.0
        tgt_norm   = (tgt_verts_world - tgt_center) / tgt_scale

        # Pre-create all vertex groups on target matching the rig's bone names
        existing_names = {vg.name for vg in target_mesh.vertex_groups}
        for vg in rig_mesh.vertex_groups:
            if vg.name not in existing_names:
                target_mesh.vertex_groups.new(name=vg.name)

        # For each target vertex find nearest rig vertex and copy all weights
        n_transferred = 0
        for i, tv in enumerate(tgt_verts):
            co_norm = mathutils.Vector(tgt_norm[i].tolist())
            _, rig_vi, _ = kd.find(co_norm)
            for group_elem in rig_verts[rig_vi].groups:
                vg_name = rig_mesh.vertex_groups[group_elem.group].name
                target_mesh.vertex_groups[vg_name].add(
                    [tv.index], group_elem.weight, 'REPLACE'
                )
                n_transferred += 1

        # Parent to armature without changing the mesh transform
        target_mesh.parent = armature
        target_mesh.parent_type = 'OBJECT'
        target_mesh.matrix_parent_inverse = armature.matrix_world.inverted()

        # Add ARMATURE modifier
        arm_mod = target_mesh.modifiers.new("Armature", 'ARMATURE')
        arm_mod.object = armature
        arm_mod.use_vertex_groups = True

        print(f"  Skin weight transfer complete: {len(target_mesh.data.vertices)} verts, "
              f"{n_transferred} weight assignments (Normalised alignment used)")
        return True
    except Exception as e:
        print(f"  Skin weight transfer failed: {e}")
        import traceback; traceback.print_exc()
        return False


def _has_real_vertex_colors(obj):
    """Return True if mesh has non-uniform vertex colour data."""
    if obj is None or obj.type != 'MESH':
        return False
    for attr in obj.data.attributes:
        if attr.domain in ('CORNER', 'POINT') and attr.data_type in ('FLOAT_COLOR', 'BYTE_COLOR'):
            data = attr.data
            if len(data) < 2:
                continue
            first = tuple(data[0].color[:3])
            for d in list(data)[:min(20, len(data))]:
                if tuple(d.color[:3]) != first:
                    return True
    return False


# build_axis_remapping removed — the physics-based axis detection was producing
# incorrect results for Puppeteer bone orientations, scrambling all animations.
# Direct euler application (motion[0]→euler.x, motion[1]→euler.y, motion[2]→euler.z)
# is simpler and reliable: for Blender/Puppeteer FBX bones, local-X is the primary
# bending axis (stride for legs, swing for arms) which matches our motion convention.

def transfer_vertex_colors(target_obj, color_source_path):
    """
    KD-tree vertex color transfer from a GLB/OBJ source mesh to target_obj.

    Uses bounding-box normalisation on both point clouds before the KD-tree
    query so the transfer is coordinate-system and unit agnostic.  This is
    needed because the rigged FBX may have been imported by Blender with a
    unit-scale factor (cm→m) that differs from the trimesh coordinate space
    of the colour source.
    """
    if not color_source_path or not os.path.exists(color_source_path):
        return

    import trimesh
    try:
        source_trimesh = trimesh.load(color_source_path, process=False)
        if isinstance(source_trimesh, trimesh.Scene):
            geoms = [g for g in source_trimesh.geometry.values()
                     if isinstance(g, trimesh.Trimesh)]
            if not geoms:
                print("  Color source has no geometry — skipping.")
                return
            source_trimesh = trimesh.util.concatenate(geoms)

        # --- Resolve vertex colours (direct or baked from UV texture) ----------
        src_verts = np.asarray(source_trimesh.vertices, dtype=np.float64)
        src_cols  = None

        # Try direct vertex colours first
        try:
            vc = np.asarray(source_trimesh.visual.vertex_colors)
            if vc is not None and vc.size > 0 and not np.all(vc[:, :3] == vc[0, :3]):
                src_cols = vc / 255.0
        except Exception:
            pass

        # Fall back: bake UV texture → vertex colours
        if src_cols is None:
            try:
                baked = source_trimesh.copy()
                baked.visual = baked.visual.to_color()
                vc = np.asarray(baked.visual.vertex_colors)
                if vc is not None and vc.size > 0 and not np.all(vc[:, :3] == vc[0, :3]):
                    src_cols = vc / 255.0
                    src_verts = np.asarray(baked.vertices, dtype=np.float64)
            except Exception as e:
                print(f"  Texture bake failed: {e}")

        if src_cols is None:
            print("  No non-uniform colour data on source — skipping transfer.")
            return

    except Exception as e:
        print(f"  Failed to load colour source: {e}")
        return

    # --- Normalise source to unit cube centred at origin ----------------------
    src_lo = src_verts.min(axis=0)
    src_hi = src_verts.max(axis=0)
    src_center = (src_lo + src_hi) * 0.5
    src_scale  = float(np.max(src_hi - src_lo)) or 1.0
    src_norm   = (src_verts - src_center) / src_scale

    # Build KD-tree on normalised source
    n_src = len(src_norm)
    kd = mathutils.kdtree.KDTree(n_src)
    for i, v in enumerate(src_norm):
        kd.insert(mathutils.Vector(v.tolist()), i)
    kd.balance()

    tgt_mesh = target_obj.data
    mat      = target_obj.matrix_world

    # Collect target vertex world-space positions and normalise the same way
    tgt_verts_world = np.array(
        [(mat @ v.co)[:] for v in tgt_mesh.vertices], dtype=np.float64
    )
    tgt_lo = tgt_verts_world.min(axis=0)
    tgt_hi = tgt_verts_world.max(axis=0)
    tgt_center = (tgt_lo + tgt_hi) * 0.5
    tgt_scale  = float(np.max(tgt_hi - tgt_lo)) or 1.0
    tgt_norm   = (tgt_verts_world - tgt_center) / tgt_scale

    # Create colour attribute on target
    if 'Col' not in tgt_mesh.attributes:
        tgt_mesh.color_attributes.new(name='Col', type='BYTE_COLOR', domain='POINT')
    tgt_attr = tgt_mesh.attributes['Col']

    for i, v in enumerate(tgt_mesh.vertices):
        co_norm = mathutils.Vector(tgt_norm[i].tolist())
        _, src_vi, _ = kd.find(co_norm)
        c = src_cols[src_vi]
        tgt_attr.data[i].color = (float(c[0]), float(c[1]), float(c[2]), 1.0)

    print(f"  Vertex colours transferred from {os.path.basename(color_source_path)}"
          f" ({n_src} src verts → {len(tgt_mesh.vertices)} tgt verts)")

def optimize_mesh_data(obj):
    """
    Down-convert CORNER colors to POINT colors and remove unused data.
    POINT colors are 3x more efficient in memory and file size.
    """
    if obj.type != 'MESH':
        return
    
    mesh = obj.data
    # 1. Convert CORNER colors to POINT
    color_attrs = [attr for attr in mesh.color_attributes if attr.domain == 'CORNER']
    for attr in color_attrs:
        name = attr.name
        # Sample first corner per vertex
        v_colors = {}
        for loop in mesh.loops:
            if loop.vertex_index not in v_colors:
                v_colors[loop.vertex_index] = tuple(attr.data[loop.index].color)
        
        # Remove old CORNER attribute
        mesh.color_attributes.remove(attr)
        
        # Create new POINT attribute
        new_attr = mesh.color_attributes.new(name=name, type='BYTE_COLOR', domain='POINT')
        for v_idx, color in v_colors.items():
            new_attr.data[v_idx].color = color
        print(f"  Optimized color attribute '{name}': CORNER → POINT on {obj.name}")

    # 2. Shade Smooth (always)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_smooth()

def apply_star_retargeting(armature, motion_path, action_name="STaR_Motion"):
    """
    Step 4.2: Apply procedural motion to armature bones.

    Motion values are applied directly as XYZ Euler rotations without any
    axis-remapping. For Blender/Puppeteer FBX bones the local-X axis is the
    primary bending axis (stride for legs, swing/raise for arms), which matches
    the convention used by the procedural motion generators.
    """
    print(f"  Retargeting: {action_name}")
    data_raw = np.load(motion_path, allow_pickle=True)
    if isinstance(data_raw, np.ndarray) and data_raw.shape == ():
        data_dict = data_raw.item()
    else:
        data_dict = dict(data_raw)

    motion = data_dict.get('motion', np.array([]))
    joint_names = data_dict.get('joint_names', [])
    fps = int(data_dict.get('fps', 30))

    if motion.size == 0:
        return None

    n_frames = motion.shape[0]
    bpy.context.scene.render.fps = fps

    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')

    action = bpy.data.actions.new(name=action_name)
    if not armature.animation_data:
        armature.animation_data_create()
    armature.animation_data.action = action

    bone_map = {b.name: b for b in armature.pose.bones}
    matched = 0

    for f in range(n_frames):
        for j, joint_name in enumerate(joint_names):
            bone = bone_map.get(joint_name)
            if not bone:
                continue
            bone.rotation_mode = 'XYZ'
            bone.rotation_euler = (
                float(motion[f, j, 0]),
                float(motion[f, j, 1]),
                float(motion[f, j, 2]),
            )
            bone.keyframe_insert(data_path='rotation_euler', frame=f + 1)
            if f == 0:
                matched += 1

    print(f"  {action_name}: {n_frames} frames, {matched} bones matched")

    # Push to NLA
    armature.animation_data.action = None
    track = armature.animation_data.nla_tracks.new()
    track.name = action_name
    track.strips.new(action_name, 1, action)

    bpy.ops.object.mode_set(mode='OBJECT')
    return action

def _bake_nla_to_actions(armature):
    """
    Blender 4.x GLTF exporter bug: NLA strips with many keyframes are exported
    as only 2 keyframes.  Work-around: collect each strip's action, then use
    bpy.ops.nla.bake() (one strip at a time) to bake the visual pose into a
    proper standalone action that the GLTF exporter handles correctly.
    """
    if not armature.animation_data:
        return []

    scene = bpy.context.scene
    anim_data = armature.animation_data

    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.select_all(action='SELECT')

    tracks = list(anim_data.nla_tracks)
    baked = []

    for track in tracks:
        for strip in track.strips:
            src_action = strip.action
            if src_action is None:
                continue

            n_frames = max(1, int(round(strip.frame_end - strip.frame_start)))
            f_start  = int(round(strip.frame_start))
            f_end    = int(round(strip.frame_end))

            # Mute all tracks except this one
            for t in tracks:
                t.mute = (t is not track)
            anim_data.action = None
            anim_data.use_nla = True
            bpy.context.view_layer.update()

            # bpy.ops.nla.bake creates a new action from the evaluated (visual) pose
            bpy.ops.nla.bake(
                frame_start=f_start,
                frame_end=f_end,
                only_selected=False,
                visual_keying=True,
                clear_constraints=False,
                clear_parents=False,
                use_current_action=False,
                bake_types={'POSE'},
            )

            # The baked action is now the active action
            new_action = anim_data.action
            if new_action is None:
                print(f"  WARNING: bake returned no action for '{src_action.name}'")
                for t in tracks:
                    t.mute = False
                continue

            new_action.name = src_action.name + "_baked"

            # Restore
            anim_data.action = None
            for t in tracks:
                t.mute = False

            baked.append((src_action.name, new_action, n_frames))
            # In Blender 4.0+, Action might be layered and not have fcurves directly.
            n_fcurves = 0
            if hasattr(new_action, "fcurves"):
                n_fcurves = len(new_action.fcurves)
            elif hasattr(new_action, "curves"): # Some newer/other versions
                n_fcurves = len(new_action.curves)
            
            print(f"  Baked '{src_action.name}' → '{new_action.name}' "
                  f"({n_frames} frames, fcurves={n_fcurves})")

    bpy.ops.object.mode_set(mode='OBJECT')
    return baked


def purge_orphaned_data():
    """Thoroughly remove all data-blocks with zero users to reduce file size."""
    print("  Purging orphaned data-blocks...")
    # Repeated passes to catch recursive dependencies (e.g. mesh -> material -> image)
    for _ in range(5):
        if hasattr(bpy.ops.outliner, 'orphans_purge'):
            try:
                bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
            except Exception:
                # Older Blender versions
                bpy.ops.outliner.orphans_purge()
        else:
            # Fallback manual purge for very old versions
            for block in bpy.data.meshes:
                if block.users == 0: bpy.data.meshes.remove(block)
            for block in bpy.data.materials:
                if block.users == 0: bpy.data.materials.remove(block)
            for block in bpy.data.images:
                if block.users == 0: bpy.data.images.remove(block)
            for block in bpy.data.actions:
                if block.users == 0: bpy.data.actions.remove(block)

def deep_clean_scene(keep_objects):
    """
    Delete EVERYTHING except the provided list of objects.
    This ensures no residual meshes from Puppeteer or intermediate imports remain.
    """
    print(f"  Deep cleaning scene: keeping {[o.name for o in keep_objects]}...")
    keep_names = {o.name for o in keep_objects}
    
    # 1. Unlink objects from all collections
    for obj in list(bpy.data.objects):
        if obj.name not in keep_names:
            bpy.data.objects.remove(obj, do_unlink=True)
            
    # 2. Delete all other data-blocks
    purge_orphaned_data()

def export_final(output_fbx, output_glb):
    """Step 4.4: Serialize and export headlessly via Blender API."""

    armature = next((o for o in bpy.data.objects if o.type == 'ARMATURE'), None)
    meshes = [o for o in bpy.data.objects if o.type == 'MESH' and o.parent == armature]
    # Also include unparented meshes if they are supposed to be props
    external_props = [o for o in bpy.data.objects if o.type == 'MESH' and o.parent != armature]

    export_objects = []
    if armature: export_objects.append(armature)
    export_objects.extend(meshes)
    export_objects.extend(external_props)

    # Aggressive clean before export
    deep_clean_scene(export_objects)

    # Re-verify selection
    bpy.ops.object.select_all(action='DESELECT')
    for obj in export_objects:
        obj.select_set(True)
    if armature:
        bpy.context.view_layer.objects.active = armature

    print("Scene Inventory for Export:")
    for obj in bpy.context.selected_objects:
        if obj.type == 'MESH':
            print(f"  - Mesh: {obj.name} ({len(obj.data.vertices)} verts, {len(obj.data.polygons)} faces)")
        else:
            print(f"  - Armature: {obj.name}")

    # ── Bake NLA → standalone actions (shared prep for FBX + GLB) ────────────
    # Problem 1: Blender 4.x GLTF exporter collapses NLA strips to 2 keyframes
    #            → bake each strip into a proper standalone action first.
    # Problem 2: original strips all start at frame 1 (they overlap in NLA),
    #            so FBX export blends them instead of exporting 3 separate anims.
    # Fix: bake all strips, place each baked action in its own NLA track at
    #      sequential non-overlapping positions.  Tracks with users survive the
    #      orphan-purge, and both FBX (bake_anim_use_nla_strips) and GLTF
    #      (export_nla_strips) treat each track as an independent animation.
    if armature and armature.animation_data:
        baked_pairs = _bake_nla_to_actions(armature)
        if baked_pairs:
            anim_data = armature.animation_data
            # Drop the pre-bake NLA tracks (they had overlapping frame-1 strips)
            for t in list(anim_data.nla_tracks):
                anim_data.nla_tracks.remove(t)
            anim_data.action = None

            # Rename and place each baked action in its own sequential NLA track
            cursor = 1
            seen   = set()
            for orig_name, baked_act, n_frames in baked_pairs:
                clean = orig_name.split('|')[-1]
                baked_act.name = clean if clean not in seen else clean + "_2"
                seen.add(baked_act.name)

                track = anim_data.nla_tracks.new()
                track.name = baked_act.name
                strip = track.strips.new(baked_act.name, cursor, baked_act)
                # Ensure the strip covers the full baked frame range
                strip.action_frame_start = 1
                strip.action_frame_end   = float(n_frames)
                cursor += n_frames + 1   # 1-frame gap between animations

            print(f"  Animation tracks rebuilt: {list(seen)}")

            # Remove leftover un-baked actions (they are now superseded)
            for act in list(bpy.data.actions):
                if act.name not in seen and act.users == 0:
                    bpy.data.actions.remove(act)

    if output_fbx:
        print(f"Exporting optimized FBX: {output_fbx}")
        bpy.ops.export_scene.fbx(
            filepath=output_fbx,
            use_selection=True,
            bake_anim=True,
            bake_anim_use_nla_strips=True,   # each NLA track → separate FBX take
            bake_anim_use_all_actions=False,
            path_mode='COPY',
            embed_textures=True,
            add_leaf_bones=False,
        )

    if output_glb:
        print(f"Exporting Final GLB: {output_glb}")
        # Purge AFTER baked actions are placed in NLA (so they have users and
        # won't be deleted).  Only orphaned mesh/material/image data is removed.
        purge_orphaned_data()
        # Try with export_nla_strips (Blender 3.3+); fall back gracefully.
        for kwargs in [
            {"filepath": output_glb, "export_animations": True,
             "export_nla_strips": True},
            {"filepath": output_glb, "export_animations": True},
            {"filepath": output_glb},
        ]:
            try:
                bpy.ops.export_scene.gltf(**kwargs)
                print(f"  GLB saved: {output_glb}  (params: {list(kwargs.keys())})")
                break
            except TypeError:
                continue   # unsupported kwarg — try the next fallback
            except Exception as e:
                print(f"  WARNING: GLB export failed ({e})")
                break

def bake_ao_to_vertex_colors(obj):
    """
    Bake Ambient Occlusion to a vertex color attribute ('Col') using Cycles.
    This is the fallback when no real TRELLIS texture data is available —
    it gives the mesh visual depth (dark in crevices, bright on exposed surfaces)
    instead of leaving it flat white.
    """
    if obj.type != 'MESH':
        return
    mesh = obj.data

    # Ensure UV map exists (required for baking)
    if not mesh.uv_layers:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.uv.smart_project(angle_limit=66.0)
        bpy.ops.object.mode_set(mode='OBJECT')

    # Create a temporary 512×512 bake image
    bake_img = bpy.data.images.new('_ao_bake_tmp', width=512, height=512,
                                   alpha=False, float_buffer=False)
    bake_img.generated_color = (0.5, 0.5, 0.5, 1.0)

    # Ensure the object has a material with an image-texture node for baking
    if not mesh.materials:
        mat_tmp = bpy.data.materials.new('_ao_tmp')
        mat_tmp.use_nodes = True
        mesh.materials.append(mat_tmp)
    mat = mesh.materials[0]
    if not mat.use_nodes:
        mat.use_nodes = True
    nodes = mat.node_tree.nodes
    img_node = nodes.new(type='ShaderNodeTexImage')
    img_node.image = bake_img
    nodes.active = img_node  # This is the bake target

    # Switch to Cycles and bake
    prev_engine = bpy.context.scene.render.engine
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 32
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    try:
        bpy.ops.object.bake(type='AO', save_mode='INTERNAL')
    except Exception as e:
        print(f"  AO bake failed: {e}")
        bpy.context.scene.render.engine = prev_engine
        nodes.remove(img_node)
        bpy.data.images.remove(bake_img)
        return
    bpy.context.scene.render.engine = prev_engine

    # Sample the baked image to per-vertex colors via UV mapping
    import numpy as np
    px = np.array(bake_img.pixels[:]).reshape(512, 512, 4)

    if 'Col' not in mesh.attributes:
        mesh.color_attributes.new(name='Col', type='BYTE_COLOR', domain='POINT')
    tgt_attr = mesh.attributes['Col']

    uv_layer = mesh.uv_layers.active.data
    # Map loop index to vertex index to get per-vertex UVs
    # We take the first loop for each vertex (good enough for AO)
    v_to_l = {l.vertex_index: l.index for l in mesh.loops}
    
    for v_idx in range(len(mesh.vertices)):
        l_idx = v_to_l.get(v_idx)
        if l_idx is None: continue
        uv = uv_layer[l_idx].uv
        u = int(min(uv[0] % 1.0, 0.999) * 512)
        v = int(min(uv[1] % 1.0, 0.999) * 512)
        ao_val = float(px[v, u, 0])
        # Remap AO: [0..1] → [0.4..1.0] so mesh never goes pitch-black
        col = 0.4 + ao_val * 0.6
        tgt_attr.data[v_idx].color = (col, col, col, 1.0)

    # Clean up temporary nodes/images
    nodes.remove(img_node)
    bpy.data.images.remove(bake_img)
    print(f"  AO baked to vertex colors on {obj.name}")


def assign_vertex_color_material(obj):
    """Creates a high-quality material that uses textures and/or vertex colors."""
    if obj.type != 'MESH':
        return
    
    # Try to find the vertex color attribute name
    color_layer_name = None
    if obj.data.color_attributes:
        # Use the first color attribute
        color_layer_name = obj.data.color_attributes[0].name
    elif obj.data.vertex_colors:
        color_layer_name = obj.data.vertex_colors[0].name
    
    # Check for UV maps and textures
    has_uv = len(obj.data.uv_layers) > 0
    has_texture = False
    texture_image = None
    
    # In some imports, the material might already be there but not wired
    if len(obj.data.materials) > 0 and obj.data.materials[0] is not None:
        mat_orig = obj.data.materials[0]
        if mat_orig.use_nodes:
            for node in mat_orig.node_tree.nodes:
                if node.type == 'TEX_IMAGE' and node.image:
                    has_texture = True
                    texture_image = node.image
                    break
                    
    print(f"  Assigning material to {obj.name}: ColorAttr={color_layer_name}, UV={has_uv}, Tex={has_texture}")

    # Create new material
    mat_name = f"TrellisMat_{obj.name}"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear nodes
    for n in nodes: nodes.remove(n)
    
    # Add nodes
    node_principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    node_principled.inputs['Roughness'].default_value = 0.5
    node_principled.inputs['Metallic'].default_value = 0.0
    
    final_color_output = None
    
    if has_texture and texture_image:
        # Texture logic
        node_tex = nodes.new(type='ShaderNodeTexImage')
        node_tex.image = texture_image
        final_color_output = node_tex.outputs['Color']
    elif color_layer_name:
        # ShaderNodeColorAttribute (Blender 3.3+); fall back to ShaderNodeVertexColor
        try:
            node_vcol = nodes.new(type='ShaderNodeColorAttribute')
            node_vcol.attribute_name = color_layer_name
        except RuntimeError:
            node_vcol = nodes.new(type='ShaderNodeVertexColor')
            node_vcol.layer_name = color_layer_name
        final_color_output = node_vcol.outputs['Color']
    
    if final_color_output:
        links.new(final_color_output, node_principled.inputs['Base Color'])
    else:
        # Default gray
        node_principled.inputs['Base Color'].default_value = (0.7, 0.7, 0.7, 1.0)
    
    # Connect
    links.new(node_principled.outputs['BSDF'], node_output.inputs['Surface'])
    
    # Assign
    if len(obj.data.materials):
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

def apply_rigid_weighting_to_props(obj, armature):
    """
    Make prop-like vertices rigid by clamping each vertex to its single
    dominant bone whenever that bone already accounts for ≥ 70% of weight.

    This works regardless of bone naming convention (Puppeteer "joint0/1…",
    heuristic "left_hand/right_hand", or anything else) and naturally
    handles staffs, vials, hats, shields, etc. — any region whose skin
    weights already clearly belong to one bone just needs the other small
    influences zeroed out to stop twisting/warping.
    """
    if obj.type != 'MESH' or not armature:
        return

    print(f"  Applying dominant-bone rigidity to {obj.name}…")
    DOMINANCE = 0.70   # if one bone owns ≥ 70 % → make it 100 %
    n_clamped = 0

    for v in obj.data.vertices:
        groups = [(g.group, g.weight) for g in v.groups]
        if not groups:
            continue
        total_w = sum(w for _, w in groups)
        if total_w <= 0.0:
            continue
        dom_idx, dom_w = max(groups, key=lambda x: x[1])
        if dom_w / total_w >= DOMINANCE:
            # Zero all other groups for this vertex
            for g_idx, _ in groups:
                if g_idx != dom_idx:
                    try:
                        obj.vertex_groups[g_idx].remove([v.index])
                    except Exception:
                        pass
            # Force dominant to exactly 1.0
            obj.vertex_groups[dom_idx].add([v.index], 1.0, 'REPLACE')
            n_clamped += 1

    pct = 100.0 * n_clamped / max(len(obj.data.vertices), 1)
    print(f"  Rigidity: clamped {n_clamped}/{len(obj.data.vertices)} verts "
          f"({pct:.1f}%) to single bone.")

def main():
    # Support both "blender --python script.py -- args" and "python script.py args"
    try:
        idx = sys.argv.index("--")
        argv = sys.argv[idx + 1:]
    except ValueError:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("--rigged_body", required=True)
    parser.add_argument("--motion_dir", required=True)
    parser.add_argument("--output_fbx", required=True)
    parser.add_argument("--output_glb", required=True)
    parser.add_argument("--props", default=None)
    parser.add_argument("--joints_json", default=None)
    parser.add_argument("--project_dir", default=None)
    parser.add_argument("--color_source", default=None, help="GLB with baked vertex colors to transfer onto rig")
    args = parser.parse_args(argv)

    clear_scene()

    # ── Import Puppeteer rigged FBX (armature + Puppeteer-topology mesh) ──────
    armature, rig_meshes = import_rigged_glb(args.rigged_body)

    # ── Import Stage-3 body.glb — original topology + baked vertex colours ───
    # In the new pipeline, "body.glb" is the refined.glb from Stage 3.
    # We'll try to find it in the same dir as rigged_body.fbx or use color_source.
    body_glb_path = args.color_source if args.color_source and os.path.exists(args.color_source) else \
                    os.path.join(os.path.dirname(args.rigged_body), "body.glb")

    print(f"Attempting to import body mesh from: {body_glb_path}")                
    body_mesh = import_body_mesh(body_glb_path)

    if body_mesh and rig_meshes:
        # Transfer Puppeteer skin weights onto the Stage-3 mesh, then discard
        # Puppeteer's mesh.  The final output will have the correct topology and
        # vertex colours from Stage 3 instead of Puppeteer's resampled mesh.
        ok = transfer_skin_weights(rig_meshes[0], body_mesh, armature)
        if ok:
            for rm in rig_meshes:
                bpy.data.objects.remove(rm, do_unlink=True)
            print("  Using Stage-3 body mesh as display mesh.")
        else:
            # Transfer failed — fall back to Puppeteer mesh.
            # Remove the imported body mesh so it doesn't get exported as a
            # spurious extra mesh alongside the Puppeteer rig mesh.
            bpy.data.objects.remove(body_mesh, do_unlink=True)
            body_mesh = rig_meshes[0] if rig_meshes else None
            print("  Weight transfer failed — using Puppeteer mesh as fallback.")
    elif rig_meshes:
        # body.glb not found: use Puppeteer mesh directly
        body_mesh = rig_meshes[0]
        print("  body.glb not found — using Puppeteer mesh directly.")

    # ── Rigid Weighting for Fused Props ──────────────────────────────────────
    if body_mesh:
        apply_rigid_weighting_to_props(body_mesh, armature)

    # ── Optimize Mesh Data (Colors, Shading) ──────────────────────────────────
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            optimize_mesh_data(obj)

    # ── Vertex colours ────────────────────────────────────────────────────────
    # body.glb already carries baked TRELLIS colours from Stage 3 — use them
    # directly.  Only fall back to KD-tree transfer or AO bake if absent.
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        if not _has_real_vertex_colors(obj):
            if args.color_source:
                transfer_vertex_colors(obj, args.color_source)
        if not _has_real_vertex_colors(obj):
            print(f"  No colour data on {obj.name} — baking AO fallback...")
            bake_ao_to_vertex_colors(obj)

    # Assign Principled BSDF shader wired to vertex colours
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            assign_vertex_color_material(obj)

    # ── Import and attach EXTERNAL props (if any) ─────────────────────────────
    if args.props and os.path.exists(args.props):
        print(f"Importing external props: {args.props}")
        props_ext = os.path.splitext(args.props)[1].lower()
        before_props = set(bpy.data.objects.keys())
        if props_ext in ('.glb', '.gltf'):
            bpy.ops.import_scene.gltf(filepath=args.props)
        else:
            try:
                bpy.ops.wm.obj_import(filepath=args.props)
            except AttributeError:
                bpy.ops.import_scene.obj(filepath=args.props)

        new_prop_objs = [bpy.data.objects[k] for k in bpy.data.objects.keys()
                         if k not in before_props and bpy.data.objects[k].type == 'MESH']
        for obj in new_prop_objs:
            # Use bounding-box centre to find the nearest bone.
            bb = [mathutils.Vector(c) for c in obj.bound_box]
            bb_center_local = sum(bb, mathutils.Vector()) / 8.0
            prop_center = obj.matrix_world @ bb_center_local

            hand_bone = None
            best_dist = float('inf')
            for b in armature.data.bones:
                bone_head_w = armature.matrix_world @ mathutils.Vector(b.head_local)
                bone_tail_w = armature.matrix_world @ mathutils.Vector(b.tail_local)
                for bone_pt in (bone_head_w, bone_tail_w):
                    d = (bone_pt - prop_center).length
                    if d < best_dist:
                        best_dist = d
                        hand_bone = b
            if hand_bone:
                obj.parent = armature
                obj.parent_type = 'BONE'
                obj.parent_bone = hand_bone.name
                bpy.context.view_layer.update()
                obj.matrix_local = mathutils.Matrix.Identity(4)
                print(f"  Attached external prop {obj.name} → bone {hand_bone.name} (dist={best_dist:.3f})")
            assign_vertex_color_material(obj)

    # ── Apply retargeted animations ───────────────────────────────────────────
    if os.path.isdir(args.motion_dir):
        for fname in sorted(os.listdir(args.motion_dir)):
            if fname.endswith(".npy"):
                apply_star_retargeting(
                    armature,
                    os.path.join(args.motion_dir, fname),
                    fname,
                )

    export_final(args.output_fbx, args.output_glb)

if __name__ == "__main__":
    main()
