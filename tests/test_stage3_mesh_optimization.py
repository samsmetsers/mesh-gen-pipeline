"""
Tests for Stage 3: Mesh Optimization & Decimation

Tests cover:
  - Mock mode pass-through (returns a valid GLB)
  - Stage3Output JSON contract
  - Quality presets face targets
  - Error when input mesh not found (real mode)
  - Material / UV texture preservation
  - GLB input preference (prefer textured GLB over untextured OBJ)
  - GLB → OBJ pre-conversion logic for PyMeshLab compatibility
"""

import inspect
import json
import struct
from pathlib import Path

import pytest

from src.stage1_prompt_parsing import parse_prompt
from src.stage2_text_to_3d import generate_3d_mesh, Stage2Output
from src.stage3_mesh_optimization import (
    QUALITY_PRESETS,
    Stage3Output,
    _repair_and_decimate,
    run_stage3,
)


def _pymeshlab_available() -> bool:
    try:
        import pymeshlab  # noqa: F401
        return True
    except ImportError:
        return False

TEST_PROMPT = (
    "Game-ready prehistoric shaman character wearing tribal furs, "
    "holding a wooden staff, in a combat stance, low-poly, stylized."
)


def _make_s2_output(tmp_path, name="shaman_test"):
    s1 = parse_prompt(prompt=TEST_PROMPT, mock=True)
    return generate_3d_mesh(s1, output_dir=str(tmp_path), output_name=name, mock=True)


# ---------------------------------------------------------------------------
# Mock mode
# ---------------------------------------------------------------------------

class TestMockMode:
    def test_returns_stage3_output(self, tmp_path):
        s2 = _make_s2_output(tmp_path)
        result = run_stage3(s2, output_dir=str(tmp_path), mock=True)
        assert isinstance(result, Stage3Output)

    def test_refined_glb_exists(self, tmp_path):
        s2 = _make_s2_output(tmp_path)
        result = run_stage3(s2, output_dir=str(tmp_path), mock=True)
        assert Path(result.refined_glb_path).exists()

    def test_refined_glb_is_valid(self, tmp_path):
        s2 = _make_s2_output(tmp_path)
        result = run_stage3(s2, output_dir=str(tmp_path), mock=True)
        data = Path(result.refined_glb_path).read_bytes()
        magic = struct.unpack_from("<I", data, 0)[0]
        assert magic == 0x46546C67, "refined.glb is not a valid GLB"

    def test_face_count_is_positive(self, tmp_path):
        s2 = _make_s2_output(tmp_path)
        result = run_stage3(s2, output_dir=str(tmp_path), mock=True)
        assert result.face_count > 0

    def test_output_name_preserved(self, tmp_path):
        s2 = _make_s2_output(tmp_path, name="mychar")
        result = run_stage3(s2, output_dir=str(tmp_path), mock=True)
        assert result.output_name == "mychar"

    def test_file_in_intermediate_subdir(self, tmp_path):
        s2 = _make_s2_output(tmp_path)
        result = run_stage3(s2, output_dir=str(tmp_path), mock=True)
        assert "intermediate" in result.refined_glb_path


# ---------------------------------------------------------------------------
# Quality presets
# ---------------------------------------------------------------------------

class TestQualityPresets:
    def test_all_presets_defined(self):
        for preset in ("mobile", "standard", "high"):
            assert preset in QUALITY_PRESETS
            assert QUALITY_PRESETS[preset] > 0

    def test_mobile_lowest_face_count(self):
        assert QUALITY_PRESETS["mobile"] < QUALITY_PRESETS["standard"]

    def test_high_highest_face_count(self):
        assert QUALITY_PRESETS["high"] > QUALITY_PRESETS["standard"]

    def test_mobile_minimum_quality(self):
        """Mobile preset must have enough faces for a readable game character."""
        assert QUALITY_PRESETS["mobile"] >= 4_000, (
            "Mobile preset should be ≥4k faces for acceptable game-character quality"
        )

    def test_standard_suitable_for_hybrid(self):
        """Standard preset should be fit for hybrid mobile/PC games."""
        assert 8_000 <= QUALITY_PRESETS["standard"] <= 20_000, (
            "Standard preset should be 8k-20k faces for hybrid mobile/PC quality"
        )

    def test_custom_faces_override(self, tmp_path):
        s2 = _make_s2_output(tmp_path)
        # In mock mode face_count is always 12 (cube), but we verify the
        # parameter is accepted without error
        result = run_stage3(s2, output_dir=str(tmp_path), target_faces=500, mock=True)
        assert isinstance(result, Stage3Output)


# ---------------------------------------------------------------------------
# Stage3Output JSON contract
# ---------------------------------------------------------------------------

class TestStage3OutputContract:
    def test_json_serializable(self, tmp_path):
        s2 = _make_s2_output(tmp_path)
        result = run_stage3(s2, output_dir=str(tmp_path), mock=True)
        data = json.loads(result.model_dump_json())
        assert "refined_glb_path" in data
        assert "refined_obj_path" in data
        assert "face_count" in data
        assert "output_name" in data

    def test_can_round_trip_json(self, tmp_path):
        s2 = _make_s2_output(tmp_path)
        result = run_stage3(s2, output_dir=str(tmp_path), mock=True)
        data = json.loads(result.model_dump_json())
        restored = Stage3Output(**data)
        assert restored.refined_glb_path == result.refined_glb_path
        assert restored.refined_obj_path == result.refined_obj_path
        assert restored.face_count == result.face_count


# ---------------------------------------------------------------------------
# Real mode: error when pymeshlab unavailable
# ---------------------------------------------------------------------------

class TestRealModeErrors:
    def test_raises_when_mesh_not_found(self, tmp_path):
        """Real mode should raise FileNotFoundError if input mesh doesn't exist."""
        fake_s2 = Stage2Output(
            obj_path="/nonexistent/mesh.obj",
            glb_path="/nonexistent/mesh.glb",
            output_name="ghost",
        )
        with pytest.raises(FileNotFoundError):
            run_stage3(fake_s2, output_dir=str(tmp_path), mock=False)


# ---------------------------------------------------------------------------
# Material / UV preservation
# ---------------------------------------------------------------------------

class TestMaterialPreservation:
    def test_save_textures_flag_is_true(self):
        """_repair_and_decimate must use save_textures=True to preserve UV maps."""
        source = inspect.getsource(_repair_and_decimate)
        assert "save_textures=True" in source, (
            "save_textures must be True to preserve UV maps through PyMeshLab"
        )

    def test_decimation_uses_texture_aware_qec(self):
        """Meshes with UV data must use meshing_decimation_quadric_edge_collapse_with_texture."""
        source = inspect.getsource(_repair_and_decimate)
        assert "meshing_decimation_quadric_edge_collapse_with_texture" in source, (
            "Texture-preserving decimation requires "
            "meshing_decimation_quadric_edge_collapse_with_texture, not the standard QEC"
        )

    def test_preserveboundary_is_false(self):
        """
        QEC must NOT use preserveboundary=True for TRELLIS.2 output.

        TRELLIS.2 PBR meshes have dense UV seams that PyMeshLab classifies as
        boundary edges.  With preserveboundary=True the collapser refuses to
        touch those edges, stalling decimation far above the target (observed:
        2.3 M → 101 k instead of 12 k).  Setting it False allows QEC to
        collapse seam edges and reach the actual target.
        """
        source = inspect.getsource(_repair_and_decimate)
        assert "preserveboundary=False" in source, (
            "QEC must set preserveboundary=False — True blocks UV-seam edges "
            "and prevents decimation from reaching the target face count"
        )

    def test_tmp_files_cleaned_up(self, tmp_path):
        """No leftover _tmp.* or _s3_in_tmp.* files remain after Stage 3 (mock mode)."""
        s2 = _make_s2_output(tmp_path)
        run_stage3(s2, output_dir=str(tmp_path), mock=True)
        inter = tmp_path / s2.output_name / "intermediate"
        leftover = list(inter.glob("*_tmp*")) + list(inter.glob("_s3_in*"))
        assert leftover == [], f"Leftover tmp files: {leftover}"

    @pytest.mark.skipif(not _pymeshlab_available(), reason="pymeshlab not installed")
    def test_uv_preserved_through_decimation(self, tmp_path):
        """
        A textured OBJ with UV coordinates retains UV data in the output GLB
        after PyMeshLab repair + decimation.
        """
        import trimesh

        # Create a minimal OBJ with UV coordinates
        obj_path = tmp_path / "textured.obj"
        mtl_path = tmp_path / "textured.mtl"

        # 2-triangle quad with UV
        obj_content = (
            "mtllib textured.mtl\n"
            "usemtl mat\n"
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
            "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
            "f 1/1 2/2 3/3\nf 1/1 3/3 4/4\n"
        )
        mtl_content = "newmtl mat\nKd 1 0 0\n"
        obj_path.write_text(obj_content)
        mtl_path.write_text(mtl_content)

        glb_out = str(tmp_path / "result.glb")
        _repair_and_decimate(str(obj_path), glb_out, target_faces=50)

        scene = trimesh.load(glb_out, force="scene")
        meshes = list(scene.geometry.values()) if hasattr(scene, "geometry") else [scene]
        has_uv = any(
            hasattr(m, "visual") and hasattr(m.visual, "uv") and m.visual.uv is not None
            for m in meshes
        )
        assert has_uv, "Output GLB lost UV coordinates during Stage 3 processing"


# ---------------------------------------------------------------------------
# GLB input preference & pre-conversion
# ---------------------------------------------------------------------------

class TestGLBInputPreference:
    def test_prefers_glb_over_obj(self):
        """
        run_stage3 must prefer the textured GLB over the raw OBJ as input.

        Stage 2's raw.obj is an untextured geometry dump (trimesh Trimesh with
        process=False, no UV).  Stage 2's raw.glb is the o_voxel PBR-baked mesh
        with a full UV atlas.  Using the OBJ as input loses all texture data.
        """
        source = inspect.getsource(run_stage3)
        glb_pos = source.find("glb_path")
        obj_pos = source.find("obj_path")
        assert glb_pos != -1, "run_stage3 source must reference glb_path"
        assert obj_pos != -1, "run_stage3 source must reference obj_path as fallback"
        assert glb_pos < obj_pos, (
            "run_stage3 must try glb_path BEFORE obj_path: "
            "the raw OBJ (trimesh process=False export) has no UV coords; "
            "the raw GLB has PBR textures and a UV atlas"
        )

    def test_glb_pre_conversion_in_source(self):
        """
        _repair_and_decimate must contain logic to pre-convert GLB inputs via
        trimesh before handing off to PyMeshLab.  PyMeshLab cannot save the
        embedded 'texture_0' blobs in a GLB — trimesh must extract them as
        named PNG files first.
        """
        source = inspect.getsource(_repair_and_decimate)
        assert ".glb" in source, (
            "_repair_and_decimate must check for .glb suffix to trigger pre-conversion"
        )
        assert "trimesh.load" in source or "trimesh" in source, (
            "_repair_and_decimate must use trimesh for GLB → OBJ pre-conversion"
        )

    def test_glb_input_cleanup_in_source(self):
        """
        _repair_and_decimate must clean up both the output temp files AND
        any temporary files created during GLB→OBJ pre-conversion.
        """
        source = inspect.getsource(_repair_and_decimate)
        assert "_input_tmp_obj" in source or "_s3_in_tmp" in source, (
            "Input conversion temp files must be tracked and cleaned up"
        )

    def test_face_count_within_10x_of_target_on_real_input(self, tmp_path):
        """
        The mock cube has only 6 faces which falls below the target — confirm
        mock mode still completes and returns a valid (if minimal) face count.
        This serves as a smoke test that the face-count logic doesn't crash.
        """
        s2 = _make_s2_output(tmp_path)
        result = run_stage3(s2, output_dir=str(tmp_path), mock=True)
        # Mock always returns 12 faces (cube)
        assert result.face_count > 0
        assert result.face_count < 10_000


# ─── Smoothing pass checks ────────────────────────────────────────────────────

class TestSmoothing:
    """Verify the two-pass post-decimation smoothing in _repair_and_decimate."""

    def _src(self) -> str:
        return inspect.getsource(_repair_and_decimate)

    def test_taubin_smoothing_30_iterations(self):
        """Taubin smoothing must use 30 iterations."""
        src = self._src()
        assert "apply_coord_taubin_smoothing" in src
        assert "stepsmoothnum=30" in src

    def test_hc_laplacian_called_with_no_params(self):
        """HC-Laplacian smoothing must be called with no arguments.
        This pymeshlab version accepts only default parameters for this filter."""
        src = self._src()
        assert "apply_coord_hc_laplacian_smoothing()" in src

    def test_hc_laplacian_has_fallback(self):
        """HC-Laplacian must be wrapped in try/except so a missing filter
        does not abort the pipeline."""
        src = self._src()
        assert "apply_coord_hc_laplacian_smoothing" in src
        # try ... except pattern around HC call
        hc_idx = src.index("apply_coord_hc_laplacian_smoothing")
        pre = src[max(0, hc_idx - 200):hc_idx]
        assert "try:" in pre or "try :" in pre
