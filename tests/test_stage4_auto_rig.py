"""
Tests for Stage 4: Auto-Rigging (Blender)

Coverage:
  - Mock mode and Stage4Output JSON contract
  - Material / UV preservation in GLB → OBJ conversion
  - Cross-section analysis (_sections_from_verts) — testable without Blender
  - Landmark detection (_landmarks_from_verts) — testable without Blender
  - Arm chain detection (_arm_chains_from_verts) for T-pose & combat-stance
  - Blender script structural checks (ARMATURE_AUTO, ENVELOPE fallback, etc.)
"""

import json
import math
import struct
from pathlib import Path

import pytest

from src.stage3_mesh_optimization import Stage3Output
from src.stage4_auto_rig import (
    Stage4Output,
    _glb_to_obj,
    run_stage4,
)

TEST_PROMPT = (
    "Game-ready prehistoric shaman character wearing tribal furs, "
    "holding a wooden staff, in a combat stance, low-poly, stylized."
)


def _make_s3_output(tmp_path, name="shaman_test"):
    from src.stage1_prompt_parsing import parse_prompt
    from src.stage2_text_to_3d import generate_3d_mesh
    from src.stage3_mesh_optimization import run_stage3

    s1 = parse_prompt(prompt=TEST_PROMPT, mock=True)
    s2 = generate_3d_mesh(s1, output_dir=str(tmp_path), output_name=name, mock=True)
    return run_stage3(s2, output_dir=str(tmp_path), mock=True)


# ─── Fixtures: simple synthetic meshes ────────────────────────────────────────

def _cube_verts(cx=0.0, cy=0.0, cz=0.5, half=0.5):
    """8 corners of an axis-aligned cube centred at (cx, cy, cz)."""
    pts = []
    for dx in (-half, half):
        for dy in (-half, half):
            for dz in (-half, half):
                pts.append((cx + dx, cy + dy, cz + dz))
    return pts


def _humanoid_verts(height=2.0, n_per_section=20):
    """
    Synthetic T-pose humanoid vertex cloud.

    Spine (centre column) + two arms hanging at ±0.35*height from cx + two legs.
    Useful for testing landmark and arm detection without Blender.
    """
    import random
    rng = random.Random(42)

    def spread(cx, cy, cz, rx, ry, rz, n):
        return [
            (cx + rng.uniform(-rx, rx),
             cy + rng.uniform(-ry, ry),
             cz + rng.uniform(-rz, rz))
            for _ in range(n)
        ]

    sh_x = height * 0.32
    hip_x = height * 0.12
    verts = []

    # Torso (50–90 % height)
    for frac in (0.55, 0.62, 0.70, 0.78, 0.85):
        verts += spread(0, 0, frac * height, 0.10, 0.08, 0.02, n_per_section)

    # Head (90–100 %)
    verts += spread(0, 0, 0.95 * height, 0.08, 0.08, 0.05, n_per_section)

    # Left arm (hang straight down from shoulder at ~80 %)
    for frac in (0.80, 0.70, 0.60):
        verts += spread(-sh_x, 0, frac * height, 0.05, 0.05, 0.02, n_per_section)

    # Right arm
    for frac in (0.80, 0.70, 0.60):
        verts += spread(+sh_x, 0, frac * height, 0.05, 0.05, 0.02, n_per_section)

    # Left leg
    for frac in (0.45, 0.28, 0.10):
        verts += spread(-hip_x, 0, frac * height, 0.06, 0.06, 0.02, n_per_section)

    # Right leg
    for frac in (0.45, 0.28, 0.10):
        verts += spread(+hip_x, 0, frac * height, 0.06, 0.06, 0.02, n_per_section)

    return verts, height, sh_x, hip_x


def _combat_stance_verts(height=2.0, n_per_section=20):
    """
    Synthetic combat-stance humanoid: arms lowered to ~50–70 % height
    and extended to the sides (still ±sh_x but at waist/chest level).
    """
    import random
    rng = random.Random(7)

    def spread(cx, cy, cz, rx, ry, rz, n):
        return [
            (cx + rng.uniform(-rx, rx),
             cy + rng.uniform(-ry, ry),
             cz + rng.uniform(-rz, rz))
            for _ in range(n)
        ]

    sh_x = height * 0.30
    hip_x = height * 0.12
    verts = []

    # Torso
    for frac in (0.55, 0.65, 0.75, 0.83):
        verts += spread(0, 0, frac * height, 0.10, 0.08, 0.02, n_per_section)

    # Head
    verts += spread(0, 0, 0.94 * height, 0.08, 0.08, 0.05, n_per_section)

    # Left arm at waist/chest level (55–72 % height)
    for frac in (0.72, 0.64, 0.56):
        verts += spread(-sh_x, 0, frac * height, 0.05, 0.05, 0.02, n_per_section)

    # Right arm
    for frac in (0.72, 0.64, 0.56):
        verts += spread(+sh_x, 0, frac * height, 0.05, 0.05, 0.02, n_per_section)

    # Legs
    for frac in (0.45, 0.28, 0.10):
        verts += spread(-hip_x, 0, frac * height, 0.06, 0.06, 0.02, n_per_section)
    for frac in (0.45, 0.28, 0.10):
        verts += spread(+hip_x, 0, frac * height, 0.06, 0.06, 0.02, n_per_section)

    return verts, height, sh_x, hip_x


# ─── Mock mode ────────────────────────────────────────────────────────────────

class TestMockMode:
    def test_returns_stage4_output(self, tmp_path):
        s3 = _make_s3_output(tmp_path)
        result = run_stage4(s3, output_dir=str(tmp_path), mock=True)
        assert isinstance(result, Stage4Output)

    def test_fbx_file_exists(self, tmp_path):
        s3 = _make_s3_output(tmp_path)
        result = run_stage4(s3, output_dir=str(tmp_path), mock=True)
        assert Path(result.fbx_path).exists()

    def test_glb_file_exists(self, tmp_path):
        s3 = _make_s3_output(tmp_path)
        result = run_stage4(s3, output_dir=str(tmp_path), mock=True)
        assert Path(result.glb_path).exists()

    def test_joints_json_exists(self, tmp_path):
        s3 = _make_s3_output(tmp_path)
        result = run_stage4(s3, output_dir=str(tmp_path), mock=True)
        assert Path(result.joints_path).exists()

    def test_joints_json_valid(self, tmp_path):
        s3 = _make_s3_output(tmp_path)
        result = run_stage4(s3, output_dir=str(tmp_path), mock=True)
        joints = json.loads(Path(result.joints_path).read_text())
        assert isinstance(joints, list)
        assert len(joints) > 0

    def test_rigging_method_is_mock(self, tmp_path):
        s3 = _make_s3_output(tmp_path)
        result = run_stage4(s3, output_dir=str(tmp_path), mock=True)
        assert result.rigging_method == "mock"

    def test_output_name_preserved(self, tmp_path):
        s3 = _make_s3_output(tmp_path, name="mychar")
        result = run_stage4(s3, output_dir=str(tmp_path), mock=True)
        assert result.output_name == "mychar"


# ─── Stage4Output JSON contract ───────────────────────────────────────────────

class TestStage4OutputContract:
    def test_json_serializable(self, tmp_path):
        s3 = _make_s3_output(tmp_path)
        result = run_stage4(s3, output_dir=str(tmp_path), mock=True)
        data = json.loads(result.model_dump_json())
        assert "fbx_path" in data
        assert "glb_path" in data
        assert "joints_path" in data
        assert "joint_count" in data
        assert "rigging_method" in data

    def test_can_round_trip_json(self, tmp_path):
        s3 = _make_s3_output(tmp_path)
        result = run_stage4(s3, output_dir=str(tmp_path), mock=True)
        data = json.loads(result.model_dump_json())
        restored = Stage4Output(**data)
        assert restored.fbx_path == result.fbx_path
        assert restored.joint_count == result.joint_count


# ─── GLB → OBJ material preservation ─────────────────────────────────────────

class TestGlbToObjMaterialPreservation:
    def test_glb_to_obj_preserves_uv_coordinates(self, tmp_path):
        """_glb_to_obj must write UV (vt) lines in the OBJ."""
        import trimesh
        import numpy as np

        vertices = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float32)
        faces    = np.array([[0, 1, 2]], dtype=np.int64)
        uvs      = np.array([[0, 0],    [1, 0],    [0.5, 1]], dtype=np.float32)
        visuals  = trimesh.visual.TextureVisuals(uv=uvs)
        mesh     = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visuals)

        glb_path = str(tmp_path / "textured.glb")
        obj_path = str(tmp_path / "textured.obj")
        mesh.export(glb_path)

        _glb_to_obj(glb_path, obj_path)

        obj_text = Path(obj_path).read_text()
        assert "vt " in obj_text, "OBJ must contain UV (vt) texture coordinates"

    def test_glb_to_obj_uses_direct_scene_export(self):
        """_glb_to_obj must use scene.export(), not trimesh.util.concatenate()."""
        import inspect
        from src.stage4_auto_rig import _glb_to_obj as fn
        source = inspect.getsource(fn)
        assert "scene_or_mesh.export" in source
        assert "trimesh.util.concatenate" not in source


# ─── Cross-section analysis ───────────────────────────────────────────────────

class TestSectionsFromVerts:
    """Unit tests for _sections_from_verts — pure Python, no Blender needed."""

    def _import(self):
        from scripts.blender_auto_rig import _sections_from_verts
        return _sections_from_verts

    def test_empty_returns_empty(self):
        fn = self._import()
        secs, z_min, z_max, h = fn([])
        assert secs == []

    def test_section_count_equals_n(self):
        fn = self._import()
        verts = _cube_verts()
        secs, *_ = fn(verts, n=10)
        assert len(secs) == 10

    def test_fracs_span_zero_to_one(self):
        fn = self._import()
        verts = _cube_verts()
        secs, *_ = fn(verts, n=20)
        populated = [s for s in secs if s['count'] >= 3]
        assert any(s['frac'] < 0.1 for s in populated)
        assert any(s['frac'] > 0.9 for s in populated)

    def test_span_x_is_nonnegative(self):
        fn = self._import()
        verts = _cube_verts()
        secs, *_ = fn(verts, n=20)
        for s in secs:
            assert s['span_x'] >= 0.0

    def test_height_is_correct(self):
        fn = self._import()
        verts = _cube_verts(cz=1.0, half=1.0)  # z in [0, 2]
        _, z_min, z_max, height = fn(verts, n=10)
        assert abs(height - 2.0) < 1e-6

    def test_symmetric_mesh_has_near_zero_med_x(self):
        """Perfectly symmetric character should have spine median near 0."""
        fn = self._import()
        verts, height, sh_x, hip_x = _humanoid_verts()
        secs, z_min, z_max, h = fn(verts, n=40)
        from scripts.blender_auto_rig import _zone, _med
        torso_meds = [s['med_x'] for s in _zone(secs, 0.50, 0.82)]
        cx = _med(torso_meds)
        assert abs(cx) < 0.05, f"Symmetric mesh cx should be ~0, got {cx}"


# ─── Landmark detection ───────────────────────────────────────────────────────

class TestLandmarksFromVerts:
    """Unit tests for _landmarks_from_verts — pure Python, no Blender needed."""

    def _compute(self, verts, n=40):
        from scripts.blender_auto_rig import _sections_from_verts, _landmarks_from_verts
        secs, z_min, z_max, height = _sections_from_verts(verts, n=n)
        return _landmarks_from_verts(verts, secs, z_min, height)

    def test_all_required_keys_present(self):
        verts, *_ = _humanoid_verts()
        lm = self._compute(verts)
        required = {'cx', 'cy', 'sh_x', 'hip_x', 'arm_l', 'arm_r',
                    'z_base', 'z_pelvis', 'z_spine1', 'z_spine2',
                    'z_neck', 'z_head', 'z_knee', 'z_ankle', 'height'}
        assert required <= lm.keys()

    def test_pelvis_above_feet(self):
        verts, *_ = _humanoid_verts()
        lm = self._compute(verts)
        assert lm['z_pelvis'] > lm['z_base']

    def test_head_above_neck_above_pelvis(self):
        verts, *_ = _humanoid_verts()
        lm = self._compute(verts)
        assert lm['z_head'] > lm['z_neck'] > lm['z_pelvis']

    def test_ankle_below_knee_below_pelvis(self):
        verts, *_ = _humanoid_verts()
        lm = self._compute(verts)
        assert lm['z_ankle'] < lm['z_knee'] < lm['z_pelvis']

    def test_sh_x_reasonable_fraction_of_height(self):
        """Shoulder half-width should be 6–38 % of total height."""
        verts, height, *_ = _humanoid_verts()
        lm = self._compute(verts)
        assert 0.06 <= lm['sh_x'] / height <= 0.38

    def test_hip_x_less_than_sh_x(self):
        """Hips are narrower than shoulders for typical humanoids."""
        verts, *_ = _humanoid_verts()
        lm = self._compute(verts)
        assert lm['hip_x'] <= lm['sh_x']

    def test_sh_x_prop_inflation_clamped(self):
        """A very wide prop (e.g. staff) should not inflate sh_x beyond 38 % height."""
        import random
        rng = random.Random(99)
        # Base humanoid
        verts, height, sh_x, _ = _humanoid_verts(height=2.0)
        # Add a wide staff at shoulder height that reaches far left
        for _ in range(50):
            verts.append((-height * 0.8 + rng.uniform(-0.05, 0.05),
                          rng.uniform(-0.05, 0.05),
                          height * 0.80 + rng.uniform(-0.05, 0.05)))
        lm = self._compute(verts)
        assert lm['sh_x'] <= height * 0.38, (
            f"sh_x={lm['sh_x']:.3f} should be ≤ {height * 0.38:.3f} (38 % height)"
        )

    def test_pelvis_at_roughly_52_percent(self):
        verts, *_ = _humanoid_verts()
        lm = self._compute(verts)
        # Use the bounding-box height stored in lm, not the synthetic input height
        frac = (lm['z_pelvis'] - lm['z_base']) / lm['height']
        assert abs(frac - 0.52) < 0.01, f"Pelvis should be at ~52% height, got {frac:.3f}"

    def test_arm_l_has_three_points(self):
        verts, *_ = _humanoid_verts()
        lm = self._compute(verts)
        assert len(lm['arm_l']) == 3

    def test_arm_r_has_three_points(self):
        verts, *_ = _humanoid_verts()
        lm = self._compute(verts)
        assert len(lm['arm_r']) == 3

    def test_arm_l_is_left_of_cx(self):
        """Shoulder of left arm must be to the left (negative X) of spine centre."""
        verts, *_ = _humanoid_verts()
        lm = self._compute(verts)
        cx = lm['cx']
        arm_shoulder_x = lm['arm_l'][0][0]
        assert arm_shoulder_x < cx, (
            f"Left arm shoulder x={arm_shoulder_x:.3f} should be left of cx={cx:.3f}"
        )

    def test_arm_r_is_right_of_cx(self):
        """Shoulder of right arm must be to the right of spine centre."""
        verts, *_ = _humanoid_verts()
        lm = self._compute(verts)
        cx = lm['cx']
        arm_shoulder_x = lm['arm_r'][0][0]
        assert arm_shoulder_x > cx, (
            f"Right arm shoulder x={arm_shoulder_x:.3f} should be right of cx={cx:.3f}"
        )

    def test_arm_chain_sorted_top_to_bottom(self):
        """shoulder.z ≥ elbow.z ≥ hand.z for T-pose arms."""
        verts, *_ = _humanoid_verts()
        lm = self._compute(verts)
        for arm_key in ('arm_l', 'arm_r'):
            chain = lm[arm_key]
            assert chain[0][2] >= chain[1][2] >= chain[2][2], (
                f"{arm_key} chain not sorted top-to-bottom: {chain}"
            )


# ─── Arm chain detection for different poses ─────────────────────────────────

class TestArmChainDetection:
    """
    _arm_chains_from_verts must detect arm positions for both T-pose and
    combat-stance characters without Blender.
    """

    def _detect(self, verts, height, sh_x):
        from scripts.blender_auto_rig import _arm_chains_from_verts
        return _arm_chains_from_verts(verts, cx=0.0, cy=0.0,
                                      sh_x=sh_x, z_min=0.0, height=height)

    def test_tpose_left_arm_detected_left(self):
        verts, height, sh_x, _ = _humanoid_verts()
        arm_l, arm_r = self._detect(verts, height, sh_x)
        assert arm_l[0][0] < 0, "T-pose left arm shoulder must be at negative X"

    def test_tpose_right_arm_detected_right(self):
        verts, height, sh_x, _ = _humanoid_verts()
        arm_l, arm_r = self._detect(verts, height, sh_x)
        assert arm_r[0][0] > 0, "T-pose right arm shoulder must be at positive X"

    def test_combat_stance_arm_detected(self):
        """Arms at waist level in combat stance should still be detected."""
        verts, height, sh_x, _ = _combat_stance_verts()
        arm_l, arm_r = self._detect(verts, height, sh_x)
        assert arm_l[0][0] < 0
        assert arm_r[0][0] > 0

    def test_fallback_when_no_arm_verts(self):
        """When insufficient off-torso vertices exist, fall back to T-pose default."""
        from scripts.blender_auto_rig import _arm_chains_from_verts
        # Torso-only verts — no arm candidates
        import random
        rng = random.Random(5)
        verts = [(rng.uniform(-0.1, 0.1), rng.uniform(-0.05, 0.05),
                  rng.uniform(0.5, 1.8)) for _ in range(200)]
        height = 2.0
        sh_x   = 0.30
        arm_l, arm_r = _arm_chains_from_verts(verts, cx=0.0, cy=0.0,
                                               sh_x=sh_x, z_min=0.0, height=height)
        # Fallback: arm shoulder is at ±sh_x, straight down
        assert arm_l[0][0] < 0
        assert arm_r[0][0] > 0
        assert len(arm_l) == 3
        assert len(arm_r) == 3

    def test_arm_shoulder_z_clamped_to_shoulder_zone(self):
        """Arm shoulder Z must be in the 70–90 % height zone even for combat stance."""
        from scripts.blender_auto_rig import _arm_chains_from_verts
        verts, height, sh_x, _ = _combat_stance_verts()
        arm_l, arm_r = _arm_chains_from_verts(verts, cx=0.0, cy=0.0,
                                               sh_x=sh_x, z_min=0.0, height=height)
        for arm in (arm_l, arm_r):
            sh_frac = arm[0][2] / height
            assert 0.70 <= sh_frac <= 0.90, (
                f"Shoulder z fraction {sh_frac:.3f} outside [0.70, 0.90]"
            )

    def test_each_arm_chain_has_three_points(self):
        verts, height, sh_x, _ = _humanoid_verts()
        arm_l, arm_r = self._detect(verts, height, sh_x)
        assert len(arm_l) == 3
        assert len(arm_r) == 3


# ─── Blender script structural checks ────────────────────────────────────────

class TestRiggingScript:
    def _src(self):
        p = Path(__file__).parent.parent / "scripts" / "blender_auto_rig.py"
        return p.read_text()

    def test_armature_auto_present(self):
        assert "ARMATURE_AUTO" in self._src()

    def test_armature_auto_before_envelope(self):
        src = self._src()
        assert src.index("ARMATURE_AUTO") < src.index("ARMATURE_ENVELOPE")

    def test_armature_envelope_fallback_present(self):
        assert "ARMATURE_ENVELOPE" in self._src()

    def test_removes_empty_vertex_groups(self):
        assert "vertex_groups.remove" in self._src()

    def test_cross_section_analysis_present(self):
        """Script must use per-height-slice analysis (compute_sections)."""
        assert "compute_sections" in self._src() or "_sections_from_verts" in self._src()

    def test_detect_landmarks_called(self):
        """Main must call detect_landmarks, not hard-coded bbox heuristics."""
        assert "detect_landmarks" in self._src()

    def test_arm_chain_detection_present(self):
        """Script must detect actual arm geometry for pose-independent rigging."""
        assert "_arm_chains_from_verts" in self._src() or "arm_chains" in self._src()

    def test_per_bone_envelope_radii(self):
        """ENVELOPE fallback must use per-category radii, not a single global radius."""
        src = self._src()
        assert "_RADII_FRACS" in src or "radii" in src.lower(), (
            "ENVELOPE fallback must use per-bone-category radii"
        )

    def test_joints_saved_before_exports(self):
        """joints.json must be written before FBX/GLB export."""
        src = self._src()
        joints_pos = src.index("joints")
        fbx_pos    = src.index("export_scene.fbx")
        assert joints_pos < fbx_pos, "joints.json must be written before FBX export"

    def test_spine_centre_detection(self):
        """Script must derive spine centre X from geometry, not assume 0."""
        src = self._src()
        assert "cx" in src, "Script must compute spine centre X (cx)"

    def test_shoulder_width_from_cross_sections(self):
        """Shoulder width must come from cross-section percentiles, not bbox * const."""
        src = self._src()
        # Must NOT use the old bbox * 0.35 heuristic
        assert "width * 0.35" not in src, (
            "Shoulder width must come from cross-section analysis, not 'width * 0.35'"
        )
        # Must use sh_x derived from spans
        assert "sh_x" in src

    def test_pelvis_at_52_percent(self):
        """Pelvis fraction must be 0.52 (not old 0.50)."""
        src = self._src()
        assert "0.52" in src, "Pelvis must be placed at 52% of height (not 50%)"

    def test_blender_timeout_present_in_stage4(self):
        """Stage 4 must have a subprocess timeout to prevent WSL OOM."""
        stage4_src = (
            Path(__file__).parent.parent / "src" / "stage4_auto_rig.py"
        ).read_text()
        assert "timeout" in stage4_src
        assert "TimeoutExpired" in stage4_src
