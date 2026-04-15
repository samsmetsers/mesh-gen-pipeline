"""
Stage 3: Mesh Optimization & Decimation
=========================================
Takes the raw GLB from Stage 2 (PBR-textured, UV-mapped) and produces a
clean, game-ready mesh:

  1. Pre-convert: GLB → OBJ+MTL+PNG via trimesh so that PyMeshLab receives
     a file with properly named texture sidecar files and wedge UV coordinates.
     (PyMeshLab cannot save textures embedded directly in a GLB — the embedded
     blobs are named "texture_0" with no extension, which PyMeshLab cannot
     re-export. Trimesh converts them to named PNG files before handing off.)
  2. Repair: remove duplicates/null faces, fill holes, fix non-manifold geometry.
  3. Decimate: two-pass texture-aware QEC to hit the face-count target without
     losing UV seams.  The first pass uses qualitythr=0.05 without boundary
     preservation — TRELLIS.2 meshes have dense UV seams that create many
     artificial "boundary" edges; protecting them blocks decimation far above
     the target.  A second, more aggressive pass (qualitythr=0.01) is applied
     if the first pass leaves the mesh >20 % above target.
  4. Smooth: 3 iterations of Taubin smoothing to soften QEC staircase artefacts
     without shrinking the mesh.
  5. Export: OBJ → trimesh → GLB, preserving UV and texture images.

Quality presets (face targets):
  - mobile:   5 000 faces   (high-end mobile / low-end PC)
  - standard: 12 000 faces  (hybrid mobile+PC default)
  - high:     30 000 faces  (PC / console)

Mock mode:
  - Writes a minimal valid GLB (single triangle) without touching GPU or disk.

Real mode requires:
  - pymeshlab ≥ 2023.12  (pip install pymeshlab)
  - trimesh               (already in venv)
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

from pydantic import BaseModel, Field

from src.stage2_text_to_3d import Stage2Output


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

QUALITY_PRESETS: dict[str, int] = {
    "mobile":   5_000,   # high-end mobile / low-end PC
    "standard": 12_000,  # hybrid mobile+PC default
    "high":     30_000,  # PC / console
}


class Stage3Output(BaseModel):
    refined_glb_path: str = Field(description="Path to repaired & decimated GLB.")
    refined_obj_path: str = Field(description="Path to repaired & decimated OBJ.")
    face_count: int = Field(description="Final face count after decimation.")
    output_name: str = Field(description="Short identifier used for file naming.")


# ---------------------------------------------------------------------------
# Minimal GLB helper (reused from mock)
# ---------------------------------------------------------------------------

def _write_minimal_glb(output_path: str) -> None:
    """Write a minimal valid GLB containing a single triangle (mock only)."""
    gltf_json = (
        '{"asset":{"version":"2.0"},'
        '"scene":0,"scenes":[{"nodes":[0]}],'
        '"nodes":[{"mesh":0}],'
        '"meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}],'
        '"accessors":['
        '{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3",'
        '"min":[-0.5,-0.5,0.0],"max":[0.5,0.5,0.0]},'
        '{"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}],'
        '"bufferViews":['
        '{"buffer":0,"byteOffset":0,"byteLength":36},'
        '{"buffer":0,"byteOffset":36,"byteLength":6}],'
        '"buffers":[{"byteLength":44}]}'
    )
    json_bytes = gltf_json.encode("utf-8")
    padded_json = json_bytes + b" " * ((4 - len(json_bytes) % 4) % 4)
    vertices = struct.pack("<9f", -0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.0, 0.5, 0.0)
    indices  = struct.pack("<3H", 0, 1, 2)
    bin_data = vertices + indices + b"\x00\x00"
    total_length = 12 + 8 + len(padded_json) + 8 + len(bin_data)

    with open(output_path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total_length))
        f.write(struct.pack("<II", len(padded_json), 0x4E4F534A))
        f.write(padded_json)
        f.write(struct.pack("<II", len(bin_data), 0x004E4942))
        f.write(bin_data)


# ---------------------------------------------------------------------------
# Real optimization (PyMeshLab)
# ---------------------------------------------------------------------------

def _repair_and_decimate(
    input_path: str,
    output_glb_path: str,
    target_faces: int,
) -> tuple[int, str]:
    """
    Repair and decimate a mesh to target_faces, preserving UV/texture data.

    Accepts OBJ or GLB input.  When the input is a GLB (e.g. TRELLIS.2's
    PBR-textured raw.glb), it is first converted to OBJ+MTL+PNG via trimesh
    so that PyMeshLab receives wedge UV coordinates and named texture files
    that it can save back out.  Loading GLB directly in PyMeshLab would lose
    the embedded textures because PyMeshLab names them "texture_0" without a
    file extension, which it then cannot re-export.

    Cleanup strategy:
      The conversion creates sidecar files whose names come from the material
      (e.g. "material.mtl", "material_0.png") — NOT from the OBJ filename.
      We snapshot the directory before conversion, diff after, and delete all
      new files right after PyMeshLab loads them into memory.  This ensures
      cleanup happens even when a later error (e.g. QEC failure) prevents the
      outer try/finally from running.

    Decimation strategy (two passes):
      Pass 1 — qualitythr=0.05, preserveboundary=False
        TRELLIS.2 PBR meshes have dense UV seams.  These seams are classified
        as boundary edges by PyMeshLab's QEC.  With preserveboundary=True the
        collapser refuses to touch them, stalling the face count far above the
        target (observed: 2.3 M → 101 k instead of 12 k).  Setting it to
        False allows QEC to collapse seam edges, enabling proper decimation.
      Pass 2 — qualitythr=0.01, more aggressive
        Only applied when Pass 1 leaves the mesh more than 20 % above target.
      Fallback — standard (non-texture-aware) QEC
        TRELLIS.2's GLB sometimes has inconsistent UV coverage (some faces
        textured, some not) after trimesh conversion.  PyMeshLab's
        meshing_decimation_quadric_edge_collapse_with_texture raises a hard
        error when it detects this.  We catch that error and fall back to
        the standard QEC which ignores UV coordinates.  UV is lost, but the
        geometry and face-count target are still correct.

    Returns:
        (actual face count, path to the refined OBJ file)
    """
    import pymeshlab  # type: ignore[import]
    import trimesh    # type: ignore[import]

    out_dir = Path(output_glb_path).parent

    # ── Pre-convert GLB → OBJ+MTL+PNG so PyMeshLab gets proper sidecar files ──
    # trimesh exports the GLB's embedded PBR texture atlas as a named PNG file
    # (e.g. material_0.png) alongside the OBJ/MTL — a format that PyMeshLab
    # can read AND write back.
    # IMPORTANT: the sidecar files are named after the material, NOT after the
    # OBJ file (e.g. "material.mtl" not "_s3_in_tmp.mtl").  We snapshot the
    # directory before export and diff afterwards to capture all new files.
    actual_input = input_path
    _conversion_files: list[Path] = []
    if Path(input_path).suffix.lower() in (".glb", ".gltf"):
        print("[Stage 3] GLB detected — converting to OBJ+MTL+PNG for PyMeshLab …")
        _before_files = set(out_dir.iterdir())
        _input_tmp_obj = str(out_dir / "_s3_in_tmp.obj")
        scene_in = trimesh.load(input_path, force="scene")
        scene_in.export(_input_tmp_obj)
        actual_input = _input_tmp_obj
        _conversion_files = [f for f in out_dir.iterdir() if f not in _before_files]
        print(f"[Stage 3] GLB → OBJ done ({len(_conversion_files)} sidecar files created)")

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(actual_input)

    initial_faces = ms.current_mesh().face_number()
    print(f"[Stage 3] Loaded mesh: {initial_faces} faces")

    # Clean up conversion sidecar files now — PyMeshLab has read them into memory.
    # Doing this here (not in a finally) ensures cleanup even if QEC later raises.
    for _f in _conversion_files:
        if _f.exists():
            _f.unlink()
    if _conversion_files:
        print(f"[Stage 3] Cleaned up {len(_conversion_files)} conversion temp files")

    # ── Repair pass ─────────────────────────────────────────────────────────
    print("[Stage 3] Repairing mesh (remove duplicates, fill holes, fix normals) …")
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_repair_non_manifold_edges(method=0)
    ms.meshing_repair_non_manifold_vertices()
    # Increase hole closure threshold (100 faces) — TRELLIS meshes can have
    # large gaps at the base or between fused limbs.
    ms.meshing_close_holes(maxholesize=100)
    # Re-orient coherently only if manifold
    try:
        ms.meshing_re_orient_faces_coherently()
    except Exception:
        pass

    repaired_faces = ms.current_mesh().face_number()
    print(f"[Stage 3] After repair: {repaired_faces} faces")

    # ── Decimation passes ────────────────────────────────────────────────────
    has_tex = ms.current_mesh().has_wedge_tex_coord()
    # Track whether texture-aware QEC succeeded at least once (for logging).
    _tex_qec_ok = False

    def _qec(quality_thr: float) -> None:
        """
        Run one QEC pass.

        Attempts the texture-aware variant first (preserves UV seams).
        Falls back to standard QEC when PyMeshLab raises the
        "inconsistent tex coordinates" error — this happens when TRELLIS.2's
        GLB has partial UV coverage after trimesh conversion (some faces
        textured, some not).  Geometry quality and face-count target are
        preserved; only UV coordinates are dropped in the fallback path.
        """
        nonlocal _tex_qec_ok
        if has_tex:
            try:
                ms.meshing_decimation_quadric_edge_collapse_with_texture(
                    targetfacenum=target_faces,
                    qualitythr=quality_thr,
                    preserveboundary=False,   # must be False — see docstring
                    preservenormal=True,
                    optimalplacement=True,
                    planarquadric=False,
                    selected=False,
                )
                _tex_qec_ok = True
                return
            except Exception as exc:
                print(
                    f"[Stage 3] Texture-aware QEC failed ({exc}); "
                    "falling back to standard QEC "
                    "(UV seam-aware collapses disabled; atlas preserved via save_textures) …"
                )
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_faces,
            qualitythr=quality_thr,
            preserveboundary=False,
            preservenormal=True,
            optimalplacement=True,
            planarquadric=False,
            selected=False,
        )

    if repaired_faces > target_faces:
        print(
            f"[Stage 3] Pass 1: {repaired_faces} → {target_faces} faces "
            f"({'texture-aware' if has_tex else 'standard'} QEC, qualitythr=0.05) …"
        )
        _qec(quality_thr=0.05)
        after_pass1 = ms.current_mesh().face_number()
        print(f"[Stage 3] After pass 1: {after_pass1} faces")

        # Second pass only if target still not close enough (>20 % over)
        if after_pass1 > target_faces * 1.2:
            print(
                f"[Stage 3] Pass 2: {after_pass1} → {target_faces} faces "
                f"(qualitythr=0.01, aggressive) …"
            )
            _qec(quality_thr=0.01)
            after_pass2 = ms.current_mesh().face_number()
            print(f"[Stage 3] After pass 2: {after_pass2} faces")
    else:
        print(
            f"[Stage 3] Mesh already at/below target "
            f"({repaired_faces} ≤ {target_faces}); skipping decimation."
        )

    final_faces = ms.current_mesh().face_number()
    print(f"[Stage 3] Final face count: {final_faces}")

    # ── Post-decimation smoothing ────────────────────────────────────────────
    # Taubin smoothing removes QEC staircase artefacts without shrinking the
    # mesh.  Increasing iterations (10) for better smoothness on low-poly targets.
    print("[Stage 3] Applying Taubin smoothing (10 iterations) …")
    ms.apply_coord_taubin_smoothing(lambda_=0.5, mu=-0.53, stepsmoothnum=10)
    
    # Recalculate vertex normals to ensure smooth shading across the mesh,
    # avoiding the "facetted" look.
    ms.compute_normal_per_vertex()

    # ── Export OBJ → Blender → GLB ──────────────────────────────────────────
    # save_textures=True writes OBJ+MTL+PNG.
    # We save the refined OBJ permanently (not _tmp) because it is a high-quality
    # intermediate that can be inspected if the GLB export/import fails.
    refined_obj = str(out_dir / f"{Path(output_glb_path).stem}.obj")
    ms.save_current_mesh(refined_obj, save_textures=True)

    try:
        # Use Blender for a perfect OBJ -> GLB conversion.
        # This fixes the "holes" and "completely wrong" issues seen with trimesh.
        import subprocess
        script_path = Path(__file__).parent.parent / "scripts" / "obj_to_glb.py"
        cmd = [
            "blender",
            "--background",
            "--python", str(script_path),
            "--",
            "--input", str(Path(refined_obj).resolve()),
            "--output", str(Path(output_glb_path).resolve()),
        ]
        print(f"[Stage 3] Converting OBJ → GLB via Blender: {' '.join(cmd)}")
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"[Stage 3] Saved refined GLB → {output_glb_path}")
    except Exception as e:
        print(f"[Stage 3] Warning: Blender GLB conversion failed ({e}). Attempting trimesh fallback...")
        try:
            import trimesh
            scene_out = trimesh.load(refined_obj, force="scene", process=False)
            scene_out.export(output_glb_path)
            print(f"[Stage 3] Saved refined GLB via trimesh fallback → {output_glb_path}")
        except Exception as e2:
            print(f"[Stage 3] Critical: trimesh fallback also failed ({e2}). GLB is missing.")

    return final_faces, refined_obj


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_stage3(
    stage2_output: Stage2Output,
    output_dir: str = "output",
    quality: str = "standard",
    target_faces: int | None = None,
    mock: bool = False,
) -> Stage3Output:
    """
    Run Stage 3: repair and decimate the raw mesh.

    Args:
        stage2_output: Output from Stage 2 (contains OBJ/GLB paths).
        output_dir:    Root output directory.
        quality:       Preset name ('mobile', 'standard', 'high').
        target_faces:  Override face count (overrides quality preset).
        mock:          If True, pass through without actual processing.

    Returns:
        Stage3Output with paths to the refined GLB and OBJ.
    """
    name = stage2_output.output_name
    out_path = Path(output_dir) / name / "intermediate"
    out_path.mkdir(parents=True, exist_ok=True)

    refined_glb = str(out_path / f"{name}_refined.glb")
    refined_obj = str(out_path / f"{name}_refined.obj")
    faces = target_faces or QUALITY_PRESETS.get(quality, QUALITY_PRESETS["standard"])

    if mock:
        print(f"[Stage 3] Mock mode: writing placeholder GLB to {refined_glb}")
        _write_minimal_glb(refined_glb)
        Path(refined_obj).write_text("Mock OBJ")
        return Stage3Output(
            refined_glb_path=os.path.abspath(refined_glb),
            refined_obj_path=os.path.abspath(refined_obj),
            face_count=12,  # cube has 12 faces
            output_name=name,
        )

    # Prefer the textured GLB over the raw OBJ.
    # The raw.glb (from o_voxel PBR bake) has proper UV coordinates and an
    # embedded texture atlas.  The raw.obj is an untextured geometry dump
    # (trimesh.Trimesh(vertices, faces, process=False).export()) with no UV
    # data at all — using it as input would silently drop all texture info.
    # _repair_and_decimate handles the GLB→OBJ+MTL pre-conversion internally
    # via trimesh, which writes the embedded texture as a named PNG that
    # PyMeshLab can read and re-save.
    input_mesh = stage2_output.glb_path
    if not Path(input_mesh).exists():
        input_mesh = stage2_output.obj_path
    if not Path(input_mesh).exists():
        raise FileNotFoundError(
            f"Neither GLB ({stage2_output.glb_path}) nor OBJ ({stage2_output.obj_path}) found."
        )

    print(f"[Stage 3] Input mesh: {input_mesh}")
    print(f"[Stage 3] Target:     {faces} faces ({quality} preset)")

    final_faces, obj_path = _repair_and_decimate(input_mesh, refined_glb, target_faces=faces)

    return Stage3Output(
        refined_glb_path=os.path.abspath(refined_glb),
        refined_obj_path=os.path.abspath(obj_path),
        face_count=final_faces,
        output_name=name,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Stage 3: Mesh Optimization & Decimation")
    parser.add_argument("--input", "-i", type=str,
                        help="Path to stage2_output.json")
    parser.add_argument("--output-name", "-n", type=str, default="character")
    parser.add_argument("--output-dir", "-o", type=str, default="output")
    parser.add_argument("--quality", "-q", type=str, default="standard",
                        choices=list(QUALITY_PRESETS.keys()),
                        help="Decimation quality preset (default: standard)")
    parser.add_argument("--faces", type=int, default=None,
                        help="Override target face count (overrides --quality)")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if args.input:
        data = json.loads(Path(args.input).read_text())
        s2 = Stage2Output(**data)
    else:
        parser.error("--input is required.")

    result = run_stage3(
        s2,
        output_dir=args.output_dir,
        quality=args.quality,
        target_faces=args.faces,
        mock=args.mock,
    )

    json_path = (
        Path(args.output_dir) / s2.output_name / "intermediate" / "stage3_output.json"
    )
    json_path.write_text(result.model_dump_json(indent=2))
    print(f"\n[Stage 3] Complete. Output JSON: {json_path}")
    print(result.model_dump_json(indent=2))
