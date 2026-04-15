"""
Integration tests for the full pipeline (mock mode).

Tests the complete prompt → rigged-character flow end-to-end,
verifying that stage outputs chain correctly.
"""

import json
from pathlib import Path

from main import run_pipeline

TEST_PROMPT = (
    "Game-ready prehistoric shaman character wearing tribal furs and a dinosaur skull mask, "
    "holding a wooden staff with glowing mushrooms, in a combat stance, "
    "low-poly, stylized, mobile-optimized, clean quad-based topology."
)


class TestFullPipelineMock:
    def test_pipeline_completes_without_error(self, tmp_path):
        results = run_pipeline(
            prompt=TEST_PROMPT,
            output_name="shaman",
            output_dir=str(tmp_path),
            mock=True,
        )
        assert results  # non-empty

    def test_all_five_stages_in_results(self, tmp_path):
        results = run_pipeline(
            prompt=TEST_PROMPT,
            output_name="shaman",
            output_dir=str(tmp_path),
            mock=True,
        )
        for stage_key in ("stage1", "stage2", "stage3", "stage4", "stage5"):
            assert stage_key in results, f"Missing {stage_key} in results"

    def test_all_output_files_exist(self, tmp_path):
        results = run_pipeline(
            prompt=TEST_PROMPT,
            output_name="shaman",
            output_dir=str(tmp_path),
            mock=True,
        )
        assert Path(results["stage2"]["obj_path"]).exists()
        assert Path(results["stage2"]["glb_path"]).exists()
        assert Path(results["stage3"]["refined_glb_path"]).exists()
        assert Path(results["stage4"]["fbx_path"]).exists()
        assert Path(results["stage4"]["glb_path"]).exists()
        assert Path(results["stage4"]["joints_path"]).exists()
        assert Path(results["stage5"]["animated_glb_path"]).exists()
        assert Path(results["stage5"]["animations_json_path"]).exists()

    def test_stage_json_files_written_to_disk(self, tmp_path):
        run_pipeline(
            prompt=TEST_PROMPT,
            output_name="shaman",
            output_dir=str(tmp_path),
            mock=True,
        )
        intermediate = tmp_path / "shaman" / "intermediate"
        for i in range(1, 6):
            assert (intermediate / f"stage{i}_output.json").exists()

    def test_output_name_in_directory_structure(self, tmp_path):
        run_pipeline(
            prompt=TEST_PROMPT,
            output_name="mychar",
            output_dir=str(tmp_path),
            mock=True,
        )
        assert (tmp_path / "mychar").is_dir()
        assert (tmp_path / "mychar" / "intermediate").is_dir()

    def test_animation_type_attack_for_combat_prompt(self, tmp_path):
        results = run_pipeline(
            prompt=TEST_PROMPT,
            output_name="shaman",
            output_dir=str(tmp_path),
            mock=True,
        )
        assert results["stage1"]["animation_type"] == "attack"

    def test_final_fbx_named_correctly(self, tmp_path):
        results = run_pipeline(
            prompt=TEST_PROMPT,
            output_name="shaman",
            output_dir=str(tmp_path),
            mock=True,
        )
        fbx_path = Path(results["stage4"]["fbx_path"])
        assert fbx_path.stem.startswith("shaman")

    def test_stage_selective_run(self, tmp_path):
        """Running only stages 1 and 2 should not produce stage 3/4 outputs."""
        results = run_pipeline(
            prompt=TEST_PROMPT,
            output_name="shaman",
            output_dir=str(tmp_path),
            stages=[1, 2],
            mock=True,
        )
        assert "stage1" in results
        assert "stage2" in results
        assert "stage3" not in results
        assert "stage4" not in results
        assert "stage5" not in results

    def test_pipeline_with_walk_prompt(self, tmp_path):
        results = run_pipeline(
            prompt="A soldier marching through snow, game-ready, low-poly.",
            output_name="soldier",
            output_dir=str(tmp_path),
            mock=True,
        )
        assert results["stage1"]["animation_type"] == "walk"

    def test_joints_json_valid_structure(self, tmp_path):
        results = run_pipeline(
            prompt=TEST_PROMPT,
            output_name="shaman",
            output_dir=str(tmp_path),
            mock=True,
        )
        joints = json.loads(Path(results["stage4"]["joints_path"]).read_text())
        assert isinstance(joints, list)
        assert len(joints) > 0
        # Each joint must have name, parent, position
        for joint in joints:
            assert "name" in joint
            assert "parent" in joint
            assert "position" in joint
            assert len(joint["position"]) == 3
