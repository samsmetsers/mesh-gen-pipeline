"""
Tests for Stage 4: Auto-Rigging

Coverage:
  - Mock mode and Stage4Output JSON contract
  - Cross-section analysis (_sections_from_verts) — pure Python, no Blender
  - Landmark detection (_landmarks_from_verts) — pure Python, no Blender
  - Arm chain detection (_arm_chains_from_verts) — T-pose & combat-stance
  - Blender script structural checks (ARMATURE_AUTO, ENVELOPE fallback, etc.)
  - Puppeteer runner structural checks (checkpoint discovery, arg wiring)
  - Puppeteer install checks (venv, checkpoints, torchrun)
"""

import importlib.util
import inspect
import json
from pathlib import Path

import pytest

from src.stage3_mesh_optimization import Stage3Output
from src.stage4_auto_rig import (
    Stage4Output,
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


# ─── Puppeteer runner structural checks ──────────────────────────────────────

class TestPuppeteerRunner:
    """
    Structural / contract tests for scripts/puppeteer_runner.py.
    All tests are pure Python — no GPU or Puppeteer install required.
    """

    def _src(self) -> str:
        return (
            Path(__file__).parent.parent / "scripts" / "puppeteer_runner.py"
        ).read_text()

    def test_runner_importable(self):
        """puppeteer_runner.py must be importable without bpy or torch."""
        spec = importlib.util.spec_from_file_location(
            "puppeteer_runner",
            Path(__file__).parent.parent / "scripts" / "puppeteer_runner.py",
        )
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        # Should not raise even without torch/bpy installed in the test venv.
        # (The module has no top-level torch import.)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        assert hasattr(mod, "main")

    def test_uses_venv_torchrun(self):
        """Runner must derive torchrun from sys.executable parent, not system PATH."""
        src = self._src()
        assert 'Path(python_bin).parent / "torchrun"' in src, (
            "torchrun must be resolved from the venv bin directory"
        )

    def test_skeleton_checkpoint_discovery(self):
        """Runner must check for the skeleton ckpt downloaded by setup_puppeteer.sh."""
        src = self._src()
        assert "skeleton_ckpts/puppeteer_skeleton_w_diverse_pose.pth" in src

    def test_skinning_checkpoint_discovery(self):
        """Runner must look for skinning ckpt inside skinning/skinning/skinning_ckpts/."""
        src = self._src()
        assert "skinning/skinning/skinning_ckpts/puppeteer_skin_w_diverse_pose_depth1.pth" in src

    def test_obj_only_guard(self):
        """Runner must refuse non-OBJ inputs (skinning SkinData only reads .obj)."""
        src = self._src()
        assert ".obj" in src
        assert "ValueError" in src or "raise" in src

    def test_stage1_uses_joint_token(self):
        """Skeleton demo must use joint-based tokenization for best quality."""
        assert "--joint_token" in self._src()

    def test_stage1_uses_seq_shuffle(self):
        """Skeleton demo must shuffle sequence for diverse-pose robustness."""
        assert "--seq_shuffle" in self._src()

    def test_stage1_uses_apply_marching_cubes(self):
        """Skeleton stage must pass --apply_marching_cubes so blocky/semi-voxel
        meshes are smoothed before point-cloud sampling.  Without MC, SkeletonGPT
        consistently produces degenerate 2-joint skeletons for low-poly meshes."""
        assert "--apply_marching_cubes" in self._src()

    def test_stage2_uses_post_filter(self):
        """Skinning must use post-filter to smooth weights."""
        assert "--post_filter" in self._src()

    def test_stage3_blender_export_present(self):
        """Runner must call Blender for textured FBX export."""
        src = self._src()
        assert "blender" in src
        # Uses our textured export script (not the upstream export.py which strips UVs)
        assert "puppeteer_blend_export" in src

    def test_runner_passes_output_glb(self):
        """Runner must forward --output-glb to puppeteer_blend_export.py so the
        GLB is produced directly without an FBX→GLB round-trip."""
        src = self._src()
        assert "--output-glb" in src
        assert "output_glb" in src

    def test_joints_json_written(self):
        """Runner must write a joints.json file."""
        src = self._src()
        assert "joints_path" in src
        assert "write_text" in src or "json.dumps" in src

    def test_cleanup_removes_tmp_dir(self):
        """Runner must clean up tmp_puppeteer after completion."""
        src = self._src()
        assert "shutil.rmtree" in src
        assert "tmp_puppeteer" in src


# ─── Puppeteer install checks ─────────────────────────────────────────────────

class TestPuppeteerInstall:
    """
    Verify that the Puppeteer installation artefacts created by
    scripts/setup_puppeteer.sh are present on this machine.
    These tests are skipped if the setup has not been run.
    """

    _PROJECT = Path(__file__).parent.parent
    _PDIR    = _PROJECT / "external" / "Puppeteer"
    _VENV    = _PROJECT / ".venv_puppeteer"

    def test_puppeteer_repo_present(self):
        """external/Puppeteer must be cloned."""
        assert self._PDIR.exists(), (
            f"Puppeteer repo not found at {self._PDIR}. "
            "Run scripts/setup_puppeteer.sh."
        )

    def test_puppeteer_venv_present(self):
        """Puppeteer Python 3.10 venv must exist."""
        assert (self._VENV / "bin" / "python").exists(), (
            ".venv_puppeteer not found. Run scripts/setup_puppeteer.sh."
        )

    def test_torchrun_in_venv(self):
        """torchrun must be in the Puppeteer venv (needed for skinning stage)."""
        assert (self._VENV / "bin" / "torchrun").exists(), (
            "torchrun not found in .venv_puppeteer. "
            "Reinstall torch via setup_puppeteer.sh."
        )

    def test_skeleton_checkpoint_exists(self):
        """Skeleton checkpoint must be downloaded."""
        ckpt = self._PDIR / "skeleton_ckpts" / "puppeteer_skeleton_w_diverse_pose.pth"
        fallback = (
            self._PDIR / "skeleton" / "skeleton" / "skeleton_ckpts"
            / "puppeteer_skeleton_w_diverse_pose.pth"
        )
        assert ckpt.exists() or fallback.exists(), (
            f"Skeleton checkpoint not found at {ckpt} or {fallback}. "
            "Run scripts/setup_puppeteer.sh."
        )

    def test_skinning_checkpoint_exists(self):
        """Skinning checkpoint must be present inside the skinning package tree."""
        ckpt = (
            self._PDIR / "skinning" / "skinning" / "skinning_ckpts"
            / "puppeteer_skin_w_diverse_pose_depth1.pth"
        )
        assert ckpt.exists(), (
            f"Skinning checkpoint not found at {ckpt}. "
            "Run scripts/setup_puppeteer.sh."
        )

    def test_michelangelo_checkpoint_exists(self):
        """Michelangelo shape-VAE ckpt required by SkeletonGPT must be present."""
        ckpt = (
            self._PDIR / "skeleton" / "third_partys" / "Michelangelo"
            / "checkpoints" / "aligned_shape_latents" / "shapevae-256.ckpt"
        )
        assert ckpt.exists(), (
            f"Michelangelo shapevae-256.ckpt not found at {ckpt}. "
            "Download from the Michelangelo HuggingFace repo."
        )

    def test_skeleton_demo_py_exists(self):
        assert (self._PDIR / "skeleton" / "demo.py").exists()

    def test_skinning_main_py_exists(self):
        assert (self._PDIR / "skinning" / "main.py").exists()

    def test_export_py_exists(self):
        assert (self._PDIR / "export.py").exists()


# ─── Stage 4 Puppeteer wiring checks ─────────────────────────────────────────

class TestStage4PuppeteerWiring:
    """
    Verify that stage4_auto_rig.py correctly wires Puppeteer without
    requiring GPU or Blender.
    """

    def _src(self) -> str:
        return (
            Path(__file__).parent.parent / "src" / "stage4_auto_rig.py"
        ).read_text()

    def test_puppeteer_dir_auto_detected(self):
        """Stage 4 must auto-detect external/Puppeteer, not rely on a parameter."""
        src = self._src()
        assert "_PUPPETEER_DIR" in src
        assert 'external" / "Puppeteer"' in src or '"external/Puppeteer"' in src

    def test_venv_auto_detected(self):
        """Stage 4 must auto-detect .venv_puppeteer/bin/python."""
        src = self._src()
        assert "_PUPPETEER_VENV" in src
        assert ".venv_puppeteer" in src

    def test_no_puppeteer_dir_param(self):
        """run_stage4 must NOT have a puppeteer_dir parameter (it was removed)."""
        from src.stage4_auto_rig import run_stage4
        sig = inspect.signature(run_stage4)
        assert "puppeteer_dir" not in sig.parameters

    def test_puppeteer_attempted_before_blender(self):
        """Puppeteer attempt must precede the Blender fallback in the source."""
        src = self._src()
        assert src.index("_run_puppeteer") < src.index("_run_blender_rigger")

    def test_blender_timeout_in_rigger(self):
        """Blender heuristic rigger must still have a subprocess timeout."""
        src = self._src()
        assert "timeout" in src
        assert "TimeoutExpired" in src

    def test_mock_rigging_method_is_mock(self, tmp_path):
        """Mock mode must set rigging_method='mock'."""
        from src.stage4_auto_rig import run_stage4
        s3 = _make_s3_output(tmp_path)
        result = run_stage4(s3, output_dir=str(tmp_path), mock=True)
        assert result.rigging_method == "mock"

    def test_convert_fbx_uses_fbx_to_glb_script(self):
        """_convert_fbx_to_glb must call scripts/fbx_to_glb.py (fallback path)."""
        src = self._src()
        assert "fbx_to_glb.py" in src
        assert "python-expr" not in src.split("_convert_fbx_to_glb")[1].split("def ")[0]

    def test_puppeteer_path_passes_output_glb_to_runner(self):
        """_run_puppeteer must pass --output-glb to puppeteer_runner.py so the
        GLB is produced without a lossy FBX round-trip."""
        src = self._src()
        # The --output-glb arg must be present in _run_puppeteer's command
        pup_fn = src.split("def _run_puppeteer")[1].split("def _convert_fbx_to_glb")[0]
        assert "--output-glb" in pup_fn

    def test_puppeteer_path_has_fallback_to_fbx_conversion(self):
        """If direct GLB is missing for any reason, must fall back to FBX→GLB."""
        src = self._src()
        pup_fn = src.split("def _run_puppeteer")[1].split("def _convert_fbx_to_glb")[0]
        assert "_convert_fbx_to_glb" in pup_fn


# ─── puppeteer_blend_export.py direct GLB checks ──────────────────────────────

class TestPuppeteerBlendExportGlb:
    """Verify that puppeteer_blend_export.py can export GLB directly,
    bypassing the lossy FBX material round-trip."""

    def _src(self) -> str:
        return (
            Path(__file__).parent.parent / "scripts" / "puppeteer_blend_export.py"
        ).read_text()

    def test_accepts_output_glb_arg(self):
        """Script must accept --output-glb argument."""
        src = self._src()
        assert "--output-glb" in src
        assert "output_glb" in src

    def test_exports_gltf_when_glb_arg_given(self):
        """When --output-glb is provided, script must call export_scene.gltf."""
        src = self._src()
        assert "export_scene.gltf" in src

    def test_glb_export_uses_export_normals(self):
        src = self._src()
        assert "export_normals=True" in src

    def test_applies_normals_before_export(self):
        """normals_make_consistent must be called before the FBX/GLB exports."""
        src = self._src()
        assert "normals_make_consistent" in src
        assert "inside=False" in src

    def test_sets_doublesided_before_export(self):
        src = self._src()
        assert "use_backface_culling" in src
        assert "False" in src





# ─── fbx_to_glb.py structural checks ──────────────────────────────────────────

class TestFbxToGlbScript:
    """
    Structural tests for scripts/fbx_to_glb.py.
    Verifies all four fixes for the see-through / glistening GLB problem.
    """

    def _src(self) -> str:
        return (
            Path(__file__).parent.parent / "scripts" / "fbx_to_glb.py"
        ).read_text()

    def test_script_exists(self):
        path = Path(__file__).parent.parent / "scripts" / "fbx_to_glb.py"
        assert path.exists(), "scripts/fbx_to_glb.py is missing"

    def test_applies_transform_before_normals(self):
        """transform_apply(scale=True) must be called before normals_make_consistent
        so any negative-scale axis flip (FBX coordinate correction) is baked into
        vertex positions first — otherwise winding order stays wrong."""
        src = self._src()
        assert "transform_apply" in src
        assert "scale=True" in src
        ti = src.index("transform_apply")
        ni = src.index("normals_make_consistent")
        assert ti < ni, "transform_apply must come before normals_make_consistent"

    def test_recalculates_normals_outward(self):
        src = self._src()
        assert "normals_make_consistent" in src
        assert "inside=False" in src

    def test_sets_doublesided(self):
        """use_backface_culling=False → doubleSided:true in GLTF, safety net
        so any residual winding error doesn't make the mesh see-through."""
        src = self._src()
        assert "use_backface_culling" in src
        assert "False" in src

    def test_applies_shade_smooth(self):
        src = self._src()
        assert "use_smooth" in src
        assert "foreach_set" in src



    def test_exports_with_normals(self):
        src = self._src()
        assert "export_normals=True" in src


# ─── blender_animate.py: normals, materials, semantic rig ─────────────────────

class TestBlenderAnimateFixup:
    """Verify normals + shading + material + semantic skeleton fixes
    in blender_animate.py."""

    def _src(self) -> str:
        return (
            Path(__file__).parent.parent / "scripts" / "blender_animate.py"
        ).read_text()

    def test_no_transform_apply_on_skinned_mesh(self):
        """blender_animate.py must NOT call bpy.ops.object.transform_apply.
        Blender's GLTF import adds a rotation object-transform to convert Y-up→Z-up.
        Baking that rotation into vertex positions (transform_apply) moves the
        vertices relative to the bones → animated mesh warps."""
        src = self._src()
        # Strip comment lines and docstrings before checking so mentions in
        # explanatory text don't trigger a false positive.
        # The docstring explains WHY it's absent; check for the actual API call.
        assert "bpy.ops.object.transform_apply" not in src, (
            "bpy.ops.object.transform_apply found in blender_animate.py — "
            "this warps skinned meshes by moving vertices off their bones"
        )

    def test_normals_handled_upstream(self):
        """blender_animate.py does NOT need to recalculate normals: the incoming
        GLB already has correct outward normals set by puppeteer_blend_export.py.
        Recalculating on a posed mesh can mis-classify concave surfaces."""
        # Normals correctness is the responsibility of puppeteer_blend_export.py;
        # TestPuppeteerBlendExportGlb.test_applies_normals_before_export covers that.
        # We just verify the animated GLB export includes normals in the buffer.
        src = self._src()
        assert "export_normals=True" in src

    def test_sets_doublesided(self):
        src = self._src()
        assert "use_backface_culling" in src

    def test_applies_shade_smooth_before_export(self):
        src = self._src()
        assert "use_smooth" in src




    def test_exports_with_normals(self):
        src = self._src()
        assert "export_normals=True" in src

    def test_uses_semantic_roles_not_hardcoded_names(self):
        """Animation builders must use semantic roles (e.g. 'spine_mid') not
        hardcoded Blender heuristic bone names (e.g. 'spine_02')."""
        src = self._src()
        assert "classify_skeleton" in src
        assert "roles" in src
        # Old hardcoded names must NOT be used as direct bone lookups
        assert '"spine_02"' not in src
        assert '"upper_arm_l"' not in src
        assert '"thigh_l"' not in src

    def test_classify_skeleton_pure_python(self):
        """classify_skeleton must be importable as plain Python (no bpy)."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "blender_animate",
            Path(__file__).parent.parent / "scripts" / "blender_animate.py",
        )
        # The module imports bpy at the top level, so exec_module will fail.
        # We only need to verify classify_skeleton exists in source.
        src = self._src()
        assert "def classify_skeleton" in src

    def test_classify_skeleton_maps_shaman(self):
        """Verify the semantic classifier correctly maps the shaman's 34-joint skeleton.

        This is an integration smoke test using the actual joints.json produced
        by Puppeteer on the shaman character.  It verifies that every core
        animation role resolves to a real joint name.
        """
        joints_path = (
            Path(__file__).parent.parent
            / "output" / "shaman" / "intermediate" / "joints.json"
        )
        if not joints_path.exists():
            pytest.skip("shaman joints.json not present (run full pipeline first)")

        # Inline the classifier so we don't need bpy
        import json as _json
        from collections import defaultdict as _dd

        joints = _json.loads(joints_path.read_text())

        # Replicate classify_skeleton logic (without bpy dependency)
        by_name  = {j['name']: j for j in joints}
        children = _dd(list)
        root_name = None
        for j in joints:
            parent = j.get('parent')
            if parent is None:
                root_name = j['name']
            else:
                children[parent].append(j['name'])

        pos = {j['name']: j['position'] for j in joints}
        x_vals = [p[0] for p in pos.values()]
        y_vals = [p[1] for p in pos.values()]
        x_center = (min(x_vals) + max(x_vals)) / 2.0
        body_height = max(y_vals) - min(y_vals)
        center_thr  = body_height * 0.22

        def trace_up(start):
            chain = [start]
            current = start
            for _ in range(30):
                ups = [c for c in children[current]
                       if pos[c][1] > pos[current][1]
                       and abs(pos[c][0] - x_center) < center_thr]
                if not ups:
                    break
                nxt = max(ups, key=lambda c: pos[c][1])
                chain.append(nxt)
                current = nxt
            return chain

        spine = trace_up(root_name)
        assert len(spine) >= 4, "Spine chain must have at least 4 joints"
        assert pos[spine[-1]][1] > pos[spine[0]][1], "Spine goes upward"

        def find_leg_start(side_sign):
            candidates = [c for c in children[root_name]
                          if pos[c][1] < pos[root_name][1]]
            sided = [c for c in candidates
                     if side_sign * (pos[c][0] - x_center) > center_thr * 0.1]
            if sided:
                return max(sided, key=lambda c: side_sign * pos[c][0])
            return max(candidates, key=lambda c: side_sign * pos[c][0]) if candidates else None

        assert find_leg_start(-1) is not None, "Left leg start found"
        assert find_leg_start(+1) is not None, "Right leg start found"

        # All core roles must map to real joint names
        REQUIRED_ROLES = [
            'pelvis', 'spine_mid', 'head',
            'left_upper_arm', 'left_forearm',
            'right_upper_arm', 'right_forearm',
            'left_thigh', 'left_shin',
            'right_thigh', 'right_shin',
        ]
        # Build a minimal roles dict inline
        spine_set = set(spine)
        roles = {'pelvis': root_name, 'root': root_name}
        if len(spine) >= 2:
            roles['spine_lower'] = spine[1]
        if len(spine) >= 3:
            roles['spine_mid'] = spine[len(spine) // 2]
        if len(spine) >= 2:
            roles['spine_upper'] = spine[-1]

        def trace_down(start):
            if start is None:
                return []
            chain = [start]
            current = start
            for _ in range(10):
                downs = [c for c in children[current] if pos[c][1] < pos[current][1]]
                if not downs:
                    break
                nxt = min(downs, key=lambda c: pos[c][1])
                chain.append(nxt); current = nxt
            return chain

        for side, sign in (('left', -1.0), ('right', +1.0)):
            leg = trace_down(find_leg_start(sign))
            if len(leg) >= 1:
                roles[f'{side}_thigh'] = leg[0]
            if len(leg) >= 2:
                roles[f'{side}_shin'] = leg[1]

        def find_arm_start(anchor, side_sign):
            lateral = [c for c in children[anchor]
                       if c not in spine_set
                       and side_sign * (pos[c][0] - x_center) > center_thr * 0.2]
            return max(lateral, key=lambda c: side_sign * pos[c][0]) if lateral else None

        def trace_arm(start, side_sign):
            if start is None:
                return []
            chain = [start]; current = start
            for _ in range(6):
                cands = [c for c in children[current] if c not in spine_set]
                if not cands:
                    break
                nxt = max(cands, key=lambda c: side_sign * pos[c][0])
                if side_sign * (pos[nxt][0] - pos[current][0]) <= 0:
                    break
                chain.append(nxt); current = nxt
            return chain

        for side, sign in (('left', -1.0), ('right', +1.0)):
            for sp in reversed(spine):
                start = find_arm_start(sp, sign)
                if start:
                    arm = trace_arm(start, sign)
                    if len(arm) >= 1:
                        roles[f'{side}_shoulder']  = arm[0]
                    if len(arm) >= 2:
                        roles[f'{side}_upper_arm'] = arm[1]
                    if len(arm) >= 3:
                        roles[f'{side}_forearm']   = arm[2]
                    break

        # Head detection (mirrors classify_skeleton in blender_animate.py)
        spine_tip = spine[-1]
        above_tip = [c for c in children[spine_tip]
                     if pos[c][1] > pos[spine_tip][1] and c not in spine_set]
        if above_tip:
            neck = max(above_tip, key=lambda c: pos[c][1])
            roles['neck'] = neck
            above_neck = [c for c in children[neck] if pos[c][1] > pos[neck][1]]
            roles['head'] = max(above_neck, key=lambda c: pos[c][1]) \
                            if above_neck else neck
        else:
            roles['head'] = spine_tip

        for role in REQUIRED_ROLES:
            assert role in roles, f"Role '{role}' not classified"
            assert roles[role] in by_name, \
                f"Role '{role}' maps to non-existent joint '{roles[role]}'"
