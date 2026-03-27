"""
Stage 3: Modular Mesh Optimization

Designed specifically for Trellis AI-generated meshes which are typically:
  - Already geometrically sound (0–2% non-manifold edges)
  - Just too dense (500K–1M faces) and potentially with small holes/fragments
  - Needing decimation, NOT voxel-based reconstruction

Pipeline:
  Phase 1 — In-place repair (PyMeshLab): merge duplicates, close small holes,
             remove fragments, fix non-manifold issues without touching geometry.
  Phase 2 — Adaptive reconstruction (Open3D Poisson): ONLY if non-manifold ratio
             exceeds threshold after repair.  Skipped for clean meshes.
  Phase 3 — Feature-preserving decimation (PyMeshLab QEC primary, CuMesh fallback).
  Final   — Vertex colour transfer from original mesh via KD-tree.
"""
import os
import sys
import argparse
import logging
import shutil
import tempfile
from typing import Optional, Tuple

import numpy as np
import trimesh

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Non-manifold edge ratio above which we fall back to reconstruction
_RECON_THRESHOLD = 0.05   # 5 %


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _non_manifold_ratio(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Fraction of unique edges shared by more than 2 faces."""
    try:
        m = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        edges = m.edges_sorted
        _, counts = np.unique(edges, axis=0, return_counts=True)
        return float((counts > 2).sum()) / max(len(counts), 1)
    except Exception:
        return 0.0


def _pymeshlab_mesh(vertices: np.ndarray, faces: np.ndarray):
    import pymeshlab
    return pymeshlab.Mesh(
        vertex_matrix=vertices.astype(np.float64),
        face_matrix=faces.astype(np.int32),
    )


def _bbox_diagonal(vertices: np.ndarray) -> float:
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    return float(np.linalg.norm(hi - lo))


# ─── Phase 1: In-place repair (PyMeshLab) ────────────────────────────────────

def _merge_close_vertices(ms, diag: float) -> None:
    """
    Merge seam-duplicate vertices produced by UV-unwrapped OBJ exports.
    Trellis meshes have 20K+ disconnected UV-island components that share 3D
    positions at seams but have separate vertex entries.  Merging reconnects them.
    Threshold: 0.01% of bounding box diagonal — tight enough to only merge true
    seam duplicates, not distinct nearby vertices.
    """
    import pymeshlab

    thr = diag * 0.0001   # 0.01% of diagonal
    # Try AbsoluteValue (newer pymeshlab), then Percentage (older), then skip.
    for make_val in (
        lambda t: pymeshlab.AbsoluteValue(t),
        lambda t: pymeshlab.PercentageValue(t / diag * 100),
        lambda t: pymeshlab.Percentage(t / diag * 100),
    ):
        try:
            ms.meshing_merge_close_vertices(threshold=make_val(thr))
            return
        except Exception:
            continue
    logger.warning("    merge_close_vertices: all API variants failed, skipping.")


def phase1_repair(
    vertices: np.ndarray, faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    In-place topology repair that preserves original geometry.
    Does NOT reconstruct or voxelise.  Steps:
      1. Remove unreferenced vertices, duplicate/null faces
      2. Merge seam-duplicate vertices (reconnects UV-island components)
      3. Repair non-manifold edges and vertices
      4. Close small holes (≤ 30-edge boundary loops)
      5. Remove only true micro-fragments (< 10 faces absolute) — NOT relative
         threshold, which would destroy the many small UV-island components that
         make up a Trellis character mesh.
    """
    import pymeshlab

    logger.info("  Phase 1: PyMeshLab in-place repair...")
    n_in = len(faces)

    ms = pymeshlab.MeshSet()
    ms.add_mesh(_pymeshlab_mesh(vertices, faces))

    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()

    diag = _bbox_diagonal(np.asarray(ms.current_mesh().vertex_matrix()))
    _merge_close_vertices(ms, diag)

    try:
        ms.meshing_repair_non_manifold_edges(method=0)
    except Exception as e:
        logger.warning(f"    repair_non_manifold_edges failed: {e}")
    try:
        ms.meshing_repair_non_manifold_vertices()
    except Exception as e:
        logger.warning(f"    repair_non_manifold_vertices failed: {e}")

    try:
        ms.meshing_close_holes(maxholesize=30)
    except Exception as e:
        logger.warning(f"    close_holes failed: {e}")

    # Absolute minimum — only removes truly degenerate floating triangles.
    # Do NOT use a relative threshold here: Trellis meshes have 20K+ small
    # UV-island components, all of which are valid character geometry.
    try:
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=10)
    except Exception as e:
        logger.warning(f"    remove_micro_fragments failed: {e}")

    out = ms.current_mesh()
    v = np.asarray(out.vertex_matrix(), dtype=np.float32)
    f = np.asarray(out.face_matrix(), dtype=np.int32)
    logger.info(f"  Phase 1 done: {n_in} → {len(f)} faces")
    return v, f


# ─── Phase 2: Reconstruction (only if severely non-manifold) ─────────────────

def phase2_open3d_poisson(
    vertices: np.ndarray, faces: np.ndarray, depth: int = 9
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Open3D Screened Poisson Surface Reconstruction.
    Used ONLY as fallback when in-place repair leaves > 5% non-manifold edges.
    Produces smooth, organic results better suited to characters than ManifoldPlus.
    """
    import open3d as o3d

    logger.info(f"  Phase 2: Open3D Poisson reconstruction (depth={depth})...")
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

    n_pts = min(300_000, max(len(faces) * 6, 50_000))
    logger.info(f"    Sampling {n_pts} points...")
    pcd = o3d_mesh.sample_points_poisson_disk(number_of_points=n_pts)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=60)
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)

    recon, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )

    # Trim low-density fringe (boundary noise)
    d = np.asarray(densities)
    recon.remove_vertices_by_mask(d < np.quantile(d, 0.02))
    recon.remove_degenerate_triangles()
    recon.remove_unreferenced_vertices()

    v = np.asarray(recon.vertices, dtype=np.float32)
    f = np.asarray(recon.triangles, dtype=np.int32)
    logger.info(f"  Phase 2 done: {len(v)} verts, {len(f)} faces")
    return v, f


def phase2_maybe_reconstruct(
    vertices: np.ndarray, faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Check non-manifold ratio.  If below threshold, skip reconstruction entirely
    (preserving original geometry).  Only reconstruct if mesh is severely broken.
    """
    ratio = _non_manifold_ratio(vertices, faces)
    logger.info(f"  Phase 2 check: non-manifold edge ratio = {ratio:.2%}")

    if ratio <= _RECON_THRESHOLD:
        logger.info("  Phase 2: mesh is clean enough — skipping reconstruction.")
        return vertices, faces

    logger.warning(
        f"  Phase 2: non-manifold ratio {ratio:.2%} > {_RECON_THRESHOLD:.0%} threshold — "
        "falling back to Open3D Poisson reconstruction."
    )
    return phase2_open3d_poisson(vertices, faces)


# ─── Phase 3: Feature-Preserving Decimation ──────────────────────────────────

def _pymeshlab_qec(
    vertices: np.ndarray, faces: np.ndarray, target_faces: int, has_uv: bool = False
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Two-pass PyMeshLab Quadric Edge Collapse Decimation with Texture Preservation.
    Returns (vertices, faces, uv_coords)
    """
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.add_mesh(_pymeshlab_mesh(vertices, faces))

    # Cleanup before decimation (crucial for PyMeshLab stability)
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    ms.meshing_remove_unreferenced_vertices()

    # Determine filter name based on UV presence
    # Older versions used preservetex=True, but modern PyMeshLab often has a separate filter.
    filter_name = "meshing_decimation_quadric_edge_collapse"
    if has_uv:
        # Check if with_texture variant exists in this version
        if hasattr(ms, "meshing_decimation_quadric_edge_collapse_with_texture"):
            filter_name = "meshing_decimation_quadric_edge_collapse_with_texture"

    def apply_qec(target, preserve_boundary):
        kwargs = {
            "targetfacenum": target,
            "preserveboundary": preserve_boundary,
            "planarquadric": True,
            "qualitythr": 0.3,
            "preservenormal": True,
            "optimalplacement": True,
        }
        # Only add weight/preserve if using the standard filter (some versions support it, some don't)
        # Using getattr and try/except is safer than hardcoding parameters that change across versions.
        getattr(ms, filter_name)(**kwargs)

    # Pass 1: boundary-safe reduction
    intermediate = min(target_faces * 4, len(faces) - 1)
    if intermediate > target_faces:
        try:
            apply_qec(intermediate, True)
        except Exception as e:
            logger.warning(f"    Pass 1 failed: {e}. Retrying without advanced params...")
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=intermediate)

    # Pass 2: final reduction
    try:
        apply_qec(target_faces, False)
    except Exception as e:
        logger.warning(f"    Pass 2 failed: {e}. Retrying with basic params...")
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)

    out = ms.current_mesh()
    v_out = np.asarray(out.vertex_matrix(), dtype=np.float32)
    f_out = np.asarray(out.face_matrix(), dtype=np.int32)
    vt_out = None
    if has_uv:
        try:
            # Try to get vertex texture coordinates
            vt_out = np.asarray(out.vertex_tex_coords_matrix(), dtype=np.float32)
        except Exception as e:
            logger.warning(f"    Failed to extract UVs from PyMeshLab: {e}")

    return v_out, f_out, vt_out


def _trimesh_decimate(
    vertices: np.ndarray, faces: np.ndarray, target_faces: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback decimation using trimesh (fast but less feature-preserving)."""
    import trimesh
    m = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    
    # Calculate reduction ratio if face_count is not accepted directly
    # In some trimesh versions, simplify_quadric_decimation(face_count) works.
    # In others, it wants a ratio.
    try:
        ratio = target_faces / float(len(faces))
        # Ensure ratio is valid (between 0 and 1)
        ratio = max(0.001, min(0.999, ratio))
        out = m.simplify_quadric_decimation(ratio)
    except Exception:
        logger.warning("Trimesh decimation failed with ratio, trying face_count...")

    return np.asarray(out.vertices, dtype=np.float32), np.asarray(out.faces, dtype=np.int32)


def phase3_decimate(
    vertices: np.ndarray, faces: np.ndarray, target_faces: int, has_uv: bool = False
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Phase 3: PyMeshLab QEC primary (reliable, feature-preserving), trimesh fallback."""
    if len(faces) <= target_faces:
        logger.info(f"  Phase 3: already at {len(faces)} faces, skipping.")
        return vertices, faces, None

    logger.info(f"  Phase 3: decimating {len(faces)} → {target_faces} faces (preserve_tex={has_uv})...")
    
    # Attempt PyMeshLab QEC
    try:
        v, f, vt = _pymeshlab_qec(vertices, faces, target_faces, has_uv=has_uv)
        logger.info(f"  Phase 3 (PyMeshLab QEC): {len(f)} faces")
        return v, f, vt
    except Exception as e:
        logger.warning(f"  PyMeshLab QEC failed ({e}), trying trimesh fallback...")

    # Fallback to trimesh decimation
    try:
        v, f = _trimesh_decimate(vertices, faces, target_faces)
        logger.info(f"  Phase 3 (trimesh fallback): {len(f)} faces")
        return v, f, None
    except Exception as e:
        logger.warning(f"  Trimesh decimation failed ({e}). Returning repaired mesh as-is.")
        return vertices, faces, None


# ─── Colour / Texture Transfer ────────────────────────────────────────────────

def _bake_vertex_colors(source: trimesh.Trimesh) -> Optional[np.ndarray]:
    # Case 1: already has per-vertex colours (GLB COLOR_0 attribute)
    if hasattr(source.visual, 'vertex_colors') and source.visual.vertex_colors is not None:
        vc = np.asarray(source.visual.vertex_colors)
        if vc is not None and len(vc) == len(source.vertices):
            # Reject uniform fallback colour (all identical = failed/default bake)
            if not np.all(vc[:, :3] == vc[0, :3]):
                logger.info("  Using existing vertex colours from source mesh.")
                return vc

    # Case 2: UV-textured mesh — bake texture → per-vertex colours
    try:
        # Check if it has a texture
        if hasattr(source.visual, 'material') and source.visual.material is not None:
            src = source.copy()
            src.visual = src.visual.to_color()
            vc = np.asarray(src.visual.vertex_colors)
            if vc is not None and len(vc) == len(src.vertices):
                if not np.all(vc[:, :3] == vc[0, :3]):
                    logger.info("  UV texture baked to vertex colours.")
                    return vc
                else:
                    logger.warning("  to_color() returned uniform colour — texture bake failed.")
    except Exception as e:
        logger.warning(f"  Visual baking failed: {e}")
    return None


def transfer_colors(
    source: trimesh.Trimesh, target_verts: np.ndarray
) -> Optional[np.ndarray]:
    """KD-tree nearest-neighbour colour transfer."""
    colors = _bake_vertex_colors(source)
    if colors is None:
        return None
    from scipy.spatial import cKDTree
    # Use float64 for KDTree stability
    _, idx = cKDTree(source.vertices.astype(np.float64)).query(target_verts.astype(np.float64), workers=-1)
    logger.info(f"  Vertex colours transferred ({len(target_verts)} verts).")
    return colors[idx]


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def optimize_mesh(
    input_path: str,
    output_path: str,
    target_faces: int = 10_000,
    decimate: bool = False
) -> None:
    """
    Main Stage 3 entry point.
    ...
    """
    logger.info(f"Loading: {input_path}")
    # ... rest of function ...
    raw = trimesh.load(input_path, process=False)
    if isinstance(raw, trimesh.Scene):
        geoms = [g for g in raw.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"No Trimesh geometry found in scene: {input_path}")
        raw = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]

    vertices = np.array(raw.vertices, dtype=np.float32)
    faces    = np.array(raw.faces,    dtype=np.int32)

    has_uv = (
        (hasattr(raw.visual, 'uv') and raw.visual.uv is not None)
        or (hasattr(raw.visual, 'material')
            and getattr(raw.visual.material, 'image', None) is not None)
    )

    logger.info(f"  Input: {len(vertices)} verts, {len(faces)} faces  "
                f"| watertight={raw.is_watertight} "
                f"| NM-edges={_non_manifold_ratio(vertices, faces):.2%} "
                f"| has_uv={has_uv} | decimate={decimate}")

    try:
        # ── Phase 1: repair ────────────────────────────────────────────────
        v1, f1 = phase1_repair(vertices, faces)

        # ── Phase 2: optional reconstruction ──────────────────────────────
        v2, f2 = phase2_maybe_reconstruct(v1, f1)
        reconstructed = (v2 is not v1)

        # ── Phase 3: optional decimation ──────────────────────────────────
        if decimate:
            v_final, f_final, _ = phase3_decimate(v2, f2, target_faces, has_uv=has_uv and not reconstructed)
            logger.info(f"  Decimated: {len(f_final)} faces")
        else:
            logger.info("  Phase 3: decimation skipped (pass --decimate to enable).")
            v_final, f_final = v2, f2

        # ── Build output mesh ──────────────────────────────────────────────
        # After decimation the vertex order and count change, so we always
        # KD-tree transfer colors from the original.  For the no-decimation
        # path we additionally try to keep the original UV visual (cheaper and
        # avoids re-baking the texture).
        topology_unchanged = (not decimate) and (not reconstructed)

        if topology_unchanged and has_uv:
            # Geometry is identical to repaired Phase-1 output; UV mapping
            # may still be valid. Carry the original visual as-is.
            out_mesh = trimesh.Trimesh(vertices=v_final, faces=f_final, process=True)
            try:
                out_mesh.visual = raw.visual
                logger.info("  UV visual copied from original mesh.")
            except Exception as e:
                logger.warning(f"  Could not copy UV visual: {e}")
        else:
            out_mesh = trimesh.Trimesh(vertices=v_final, faces=f_final, process=True)

        # Always transfer vertex colors from the original via KD-tree.
        # This is the primary color source for decimated meshes and a useful
        # fallback for reconstructed ones.
        colors = transfer_colors(raw, np.array(out_mesh.vertices))
        if colors is not None:
            out_mesh.visual = trimesh.visual.ColorVisuals(
                mesh=out_mesh, vertex_colors=colors
            )
            logger.info(f"  Vertex colors transferred ({len(colors)} verts).")
        elif not has_uv:
            logger.warning("  No colour source found — mesh will be uncoloured.")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        out_mesh.export(output_path)
        logger.info(
            f"Stage 3 complete: {output_path} "
            f"({len(out_mesh.vertices)} verts, {len(out_mesh.faces)} faces)"
        )

    except Exception as e:
        logger.error(f"Stage 3 failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: Mesh Optimization")
    parser.add_argument("--input",        required=True)
    parser.add_argument("--output",       required=True)
    parser.add_argument("--target_faces", type=int, default=10_000)
    parser.add_argument("--decimate",     action="store_true",
                        help="Enable Phase 3 QEC decimation to --target_faces")
    args = parser.parse_args()
    optimize_mesh(args.input, args.output, args.target_faces, decimate=args.decimate)
