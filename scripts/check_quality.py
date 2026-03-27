"""
Quality Checker for Mesh Generation Pipeline
=============================================
Analyzes a pipeline output directory and reports pass/fail for:
  - Mesh quality (face count, normals, bounding box)
  - Material quality (textures, not white/default)
  - Rig quality (armature, bone count, hierarchy)
  - Animation quality (tracks, keyframe count)

Usage:
  python scripts/check_quality.py output/shaman/ [--verbose]
  python scripts/check_quality.py output/shaman/ --verbose
  python scripts/check_quality.py output/shaman/ --run_pipeline

Exit code 0 if all checks pass, 1 if any fail.
"""
import os
import sys
import struct
import json
import argparse
import subprocess


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_glb_json_chunk(glb_path):
    """
    Parse a GLB file's JSON chunk.
    GLB format: 12-byte header (magic 4B, version 4B, length 4B)
                then one or more chunks: 4B chunk_length + 4B chunk_type + data
    JSON chunk type = 0x4E4F534A  ('JSON')
    Returns parsed dict or None on failure.
    """
    try:
        with open(glb_path, 'rb') as f:
            header = f.read(12)
            if len(header) < 12:
                return None
            magic, version, total_length = struct.unpack_from('<4sII', header)
            if magic != b'glTF':
                return None

            # Read first chunk (should be JSON)
            chunk_header = f.read(8)
            if len(chunk_header) < 8:
                return None
            chunk_length, chunk_type = struct.unpack_from('<II', chunk_header)
            if chunk_type != 0x4E4F534A:  # 'JSON'
                return None
            chunk_data = f.read(chunk_length)
            return json.loads(chunk_data.decode('utf-8'))
    except Exception as e:
        return None


def _find_glb(project_dir):
    """Find the final GLB in the project directory (not in intermediate/)."""
    for fname in os.listdir(project_dir):
        if fname.endswith('_final.glb') and os.path.isfile(os.path.join(project_dir, fname)):
            return os.path.join(project_dir, fname)
    # Fall back to any top-level GLB
    for fname in os.listdir(project_dir):
        if fname.endswith('.glb') and os.path.isfile(os.path.join(project_dir, fname)):
            return os.path.join(project_dir, fname)
    return None


# ── Check functions ───────────────────────────────────────────────────────────

class CheckResult:
    def __init__(self, name, passed, detail=""):
        self.name = name
        self.passed = passed
        self.detail = detail

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        suffix = f"  ({self.detail})" if self.detail else ""
        return f"  [{status}] {self.name}{suffix}"


def check_mesh_quality(glb_path, gltf, verbose):
    results = []

    # 1. GLB exists
    results.append(CheckResult("Final GLB exists", True, glb_path))

    # 2. Face count via trimesh
    face_count = None
    try:
        import trimesh
        scene = trimesh.load(glb_path, force='scene')
        if hasattr(scene, 'geometry') and scene.geometry:
            total_faces = sum(g.faces.shape[0] for g in scene.geometry.values() if hasattr(g, 'faces'))
            geoms = list(scene.geometry.values())
        elif hasattr(scene, 'faces'):
            total_faces = scene.faces.shape[0]
            geoms = [scene]
        else:
            total_faces = 0
            geoms = []
        face_count = total_faces

        if face_count < 1000:
            results.append(CheckResult("Face count reasonable", False,
                                        f"{face_count} faces — suspiciously low (<1000)"))
        elif face_count > 500000:
            results.append(CheckResult("Face count reasonable", False,
                                        f"{face_count} faces — too high (>500K, check decimation)"))
        elif face_count > 50000:
            results.append(CheckResult("Face count reasonable", True,
                                        f"{face_count} faces — acceptable but high (target: 10K-50K)"))
        else:
            results.append(CheckResult("Face count reasonable", True,
                                        f"{face_count} faces"))

        # 3. Disconnected components check
        total_components = 0
        for g in geoms:
            if hasattr(g, 'faces'):
                comps = g.split(only_watertight=False)
                total_components += len(comps)

        if total_components > 5:
            results.append(CheckResult("Mesh is clean (few components)", False,
                                        f"{total_components} disconnected components found — might be noisy"))
        else:
            results.append(CheckResult("Mesh is clean (few components)", True,
                                        f"{total_components} component(s)"))

        # 4. Normals
        has_normals = False
        if hasattr(scene, 'geometry') and scene.geometry:
            for g in scene.geometry.values():
                if hasattr(g, 'vertex_normals') and g.vertex_normals is not None and len(g.vertex_normals) > 0:
                    has_normals = True
                    break
        elif hasattr(scene, 'vertex_normals') and scene.vertex_normals is not None:
            has_normals = len(scene.vertex_normals) > 0
        results.append(CheckResult("Mesh has normals", has_normals,
                                    "normals present" if has_normals else "no vertex normals found"))

        # 5. Bounding box not degenerate

        bb_ok = False
        bb_detail = ""
        if hasattr(scene, 'geometry') and scene.geometry:
            all_extents = []
            for g in scene.geometry.values():
                if hasattr(g, 'bounds') and g.bounds is not None:
                    extents = g.bounds[1] - g.bounds[0]
                    all_extents.append(extents)
            if all_extents:
                import numpy as np
                total_extents = np.max(all_extents, axis=0)
                min_extent = float(total_extents.min())
                bb_ok = min_extent > 1e-6
                bb_detail = f"extents {total_extents.tolist()}"
        elif hasattr(scene, 'bounds') and scene.bounds is not None:
            extents = scene.bounds[1] - scene.bounds[0]
            min_extent = float(extents.min())
            bb_ok = min_extent > 1e-6
            bb_detail = f"extents {extents.tolist()}"
        results.append(CheckResult("Bounding box not degenerate", bb_ok, bb_detail))

    except ImportError:
        results.append(CheckResult("Face count reasonable", False,
                                    "trimesh not installed — skipping mesh checks"))
        results.append(CheckResult("Mesh has normals", False, "trimesh not installed"))
        results.append(CheckResult("Bounding box not degenerate", False, "trimesh not installed"))

    return results


def check_material_quality(glb_path, gltf, verbose):
    results = []
    if gltf is None:
        results.append(CheckResult("Materials with textures", False, "could not parse GLB JSON chunk"))
        results.append(CheckResult("No white/default-only materials", False, "could not parse GLB JSON chunk"))
        return results

    images = gltf.get('images', [])
    materials = gltf.get('materials', [])

    has_images = len(images) > 0
    results.append(CheckResult("Has texture images", has_images,
                                f"{len(images)} image(s) found" if has_images else "no images in GLB"))

    # Check if any material references a texture (via pbrMetallicRoughness.baseColorTexture)
    textured_mats = 0
    white_only = True
    for mat in materials:
        pbr = mat.get('pbrMetallicRoughness', {})
        if 'baseColorTexture' in pbr:
            textured_mats += 1
            white_only = False
        base_color = pbr.get('baseColorFactor', [1, 1, 1, 1])
        # Not white if any channel differs noticeably from 1.0
        if any(abs(c - 1.0) > 0.05 for c in base_color[:3]):
            white_only = False

    has_textured = textured_mats > 0
    results.append(CheckResult("Has materials with textures", has_textured,
                                f"{textured_mats}/{len(materials)} materials textured"))

    # Check for brightness
    avg_brightness = 0
    if materials:
        for mat in materials:
            pbr = mat.get('pbrMetallicRoughness', {})
            base_color = pbr.get('baseColorFactor', [1, 1, 1, 1])
            avg_brightness += sum(base_color[:3]) / 3
        avg_brightness /= len(materials)
    
    if avg_brightness < 0.3:
        results.append(CheckResult("Material is not too dark", False,
                                    f"avg brightness {avg_brightness:.2f} — suspiciously dark (<0.3)"))
    else:
        results.append(CheckResult("Material is not too dark", True,
                                    f"avg brightness {avg_brightness:.2f}"))

    if materials and white_only and not has_images:
        results.append(CheckResult("No white/default-only materials", False,
                                    "all materials appear to be white/default"))
    else:
        results.append(CheckResult("No white/default-only materials", True,
                                    f"{len(materials)} material(s) found"))

    return results


def check_rig_quality(gltf, verbose):
    results = []
    if gltf is None:
        results.append(CheckResult("Has armature", False, "could not parse GLB JSON chunk"))
        return results

    skins = gltf.get('skins', [])
    nodes = gltf.get('nodes', [])

    has_armature = len(skins) > 0
    results.append(CheckResult("Has armature/skin", has_armature,
                                f"{len(skins)} skin(s)" if has_armature else "no skins in GLB"))

    if skins:
        # Count bones (joints in first skin)
        first_skin = skins[0]
        joints = first_skin.get('joints', [])
        bone_count = len(joints)

        if bone_count < 10:
            results.append(CheckResult("Bone count reasonable (10-100)", False,
                                        f"{bone_count} bones — too few (<10)"))
        elif bone_count > 100:
            results.append(CheckResult("Bone count reasonable (10-100)", False,
                                        f"{bone_count} bones — too many (>100)"))
        else:
            results.append(CheckResult("Bone count reasonable (10-100)", True,
                                        f"{bone_count} bones"))

        # Check bone hierarchy (nodes have children references)
        joint_set = set(joints)
        has_hierarchy = False
        for j in joints:
            if j < len(nodes):
                if nodes[j].get('children'):
                    has_hierarchy = True
                    break
        results.append(CheckResult("Bones have valid hierarchy", has_hierarchy,
                                    "parent-child relationships found" if has_hierarchy
                                    else "no hierarchy detected"))
    else:
        results.append(CheckResult("Bone count reasonable (10-100)", False, "no skins"))
        results.append(CheckResult("Bones have valid hierarchy", False, "no skins"))

    return results


def check_animation_quality(gltf, verbose):
    results = []
    if gltf is None:
        results.append(CheckResult("Has animation tracks", False, "could not parse GLB JSON chunk"))
        return results

    animations = gltf.get('animations', [])
    accessors = gltf.get('accessors', [])

    has_anims = len(animations) > 0
    results.append(CheckResult("Has animation tracks", has_anims,
                                f"{len(animations)} animation(s)" if has_anims else "no animations"))

    if animations:
        # Check keyframe counts across all animations
        min_keyframes = float('inf')
        max_keyframes = 0
        for anim in animations:
            for sampler in anim.get('samplers', []):
                input_acc = sampler.get('input')
                if input_acc is not None and input_acc < len(accessors):
                    count = accessors[input_acc].get('count', 0)
                    min_keyframes = min(min_keyframes, count)
                    max_keyframes = max(max_keyframes, count)

        if min_keyframes == float('inf'):
            min_keyframes = 0

        has_enough_kf = max_keyframes >= 30
        results.append(CheckResult("Animations have enough keyframes (>30)", has_enough_kf,
                                    f"max {max_keyframes} keyframes across {len(animations)} animation(s)"))
    else:
        results.append(CheckResult("Animations have enough keyframes (>30)", False, "no animations"))

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def run_checks(project_dir, verbose=False, glb_override=None):
    print(f"\nQuality Check: {project_dir}")
    print("=" * 60)

    all_results = []
    any_fail = False

    # Find GLB
    glb_path = glb_override if glb_override else _find_glb(project_dir)
    if glb_path is None:
        print("  [FAIL] No GLB file found in project directory")
        return 1

    print(f"  Analyzing: {glb_path}")
    gltf = _read_glb_json_chunk(glb_path)
    if verbose and gltf:
        print(f"  GLB JSON keys: {list(gltf.keys())}")

    # Run all check groups
    print("\nMesh Quality:")
    mesh_results = check_mesh_quality(glb_path, gltf, verbose)
    for r in mesh_results:
        print(repr(r))
    all_results.extend(mesh_results)

    print("\nMaterial Quality:")
    mat_results = check_material_quality(glb_path, gltf, verbose)
    for r in mat_results:
        print(repr(r))
    all_results.extend(mat_results)

    print("\nRig Quality:")
    rig_results = check_rig_quality(gltf, verbose)
    for r in rig_results:
        print(repr(r))
    all_results.extend(rig_results)

    print("\nAnimation Quality:")
    anim_results = check_animation_quality(gltf, verbose)
    for r in anim_results:
        print(repr(r))
    all_results.extend(anim_results)

    # Summary
    passed = sum(1 for r in all_results if r.passed)
    failed = sum(1 for r in all_results if not r.passed)
    print("\n" + "=" * 60)
    print(f"  SUMMARY: {passed} passed, {failed} failed out of {len(all_results)} checks")
    if failed > 0:
        print("  STATUS: FAIL")
        print("\n  Failed checks:")
        for r in all_results:
            if not r.passed:
                print(f"    - {r.name}: {r.detail}")
    else:
        print("  STATUS: PASS")
    print()

    return 1 if failed > 0 else 0


def main():
    parser = argparse.ArgumentParser(
        description="Quality checker for mesh-gen-pipeline output directories"
    )
    parser.add_argument("project_dir", help="Path to output/<name>/ directory")
    parser.add_argument("--verbose", action="store_true", help="Show extra debug info")
    parser.add_argument("--glb", default=None, help="Specific GLB file to check instead of auto-detect")
    parser.add_argument("--run_pipeline", action="store_true",
                        help="Re-run the shaman test case through the pipeline before checking")
    args = parser.parse_args()

    if args.run_pipeline:
        print("Re-running shaman test case through pipeline...")
        pipeline_cmd = [
            sys.executable, "main.py",
            "--prompt", "A shaman with a staff",
            "--output_name", "shaman",
        ]
        print(f"  Running: {' '.join(pipeline_cmd)}")
        result = subprocess.run(pipeline_cmd)
        if result.returncode != 0:
            print("  Pipeline run failed (non-zero exit). Proceeding with quality check anyway.")

    if not os.path.isdir(args.project_dir):
        print(f"Error: '{args.project_dir}' is not a directory")
        sys.exit(1)

    exit_code = run_checks(args.project_dir, verbose=args.verbose, glb_override=args.glb)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
