import json
import os
from pathlib import Path

import pytest

from src.stage4_auto_rig import Stage4Output
from src.stage5_animation import (
    AnimationClip,
    Stage5Output,
    infer_attack_type,
    run_stage5,
)

def _make_s4_output(tmp_path: Path) -> Stage4Output:
    """Helper to create a dummy Stage 4 output for testing."""
    glb_path = tmp_path / "character_rigged.glb"
    fbx_path = tmp_path / "character_rigged.fbx"
    joints_path = tmp_path / "joints.json"

    # Create dummy files
    glb_path.write_text("dummy glb")
    fbx_path.write_text("dummy fbx")
    joints_path.write_text("{}")

    return Stage4Output(
        fbx_path=str(fbx_path),
        glb_path=str(glb_path),
        joints_path=str(joints_path),
        rigging_method="mock",
        joint_count=21,
        output_name="character"
    )

class TestInferAttackType:
    def test_staff_summon(self):
        assert infer_attack_type("shaman holding a magical glowing staff") == "staff_summon"
        assert infer_attack_type("wizard with a spellbook") == "staff_summon"

    def test_sword_slash(self):
        assert infer_attack_type("knight with a greatsword") == "sword_slash"
        assert infer_attack_type("barbarian wielding an axe") == "sword_slash"

    def test_bow_fire(self):
        assert infer_attack_type("elven ranger with a bow and arrow") == "bow_fire"

    def test_claw_swipe(self):
        assert infer_attack_type("unarmed beast with sharp claws") == "claw_swipe"

    def test_gun_fire(self):
        assert infer_attack_type("soldier with a rifle") == "gun_fire"
        assert infer_attack_type("sniper with a blaster") == "gun_fire"

class TestMockMode:
    def test_returns_stage5_output(self, tmp_path):
        s4 = _make_s4_output(tmp_path)
        result = run_stage5(
            stage4_output=s4,
            original_prompt="shaman holding a magical glowing staff",
            output_dir=str(tmp_path),
            mock=True,
        )
        assert isinstance(result, Stage5Output)
        assert result.output_name == "character"
        assert os.path.exists(result.animated_glb_path)
        assert os.path.exists(result.animations_json_path)

    def test_animations_json_valid(self, tmp_path):
        s4 = _make_s4_output(tmp_path)
        result = run_stage5(
            stage4_output=s4,
            original_prompt="shaman with a staff",
            output_dir=str(tmp_path),
            mock=True,
        )
        
        with open(result.animations_json_path, "r") as f:
            data = json.load(f)
            
        assert len(data) == 3
        names = [clip["name"] for clip in data]
        assert "idle" in names
        assert "walk" in names
        assert "attack" in names
        
        attack_clip = next(c for c in data if c["name"] == "attack")
        assert attack_clip["attack_type"] == "staff_summon"

    def test_mock_glb_is_valid_magic(self, tmp_path):
        s4 = _make_s4_output(tmp_path)
        result = run_stage5(
            stage4_output=s4,
            original_prompt="knight with sword",
            output_dir=str(tmp_path),
            mock=True,
        )
        
        with open(result.animated_glb_path, "rb") as f:
            magic = f.read(4)
        assert magic == b"glTF"
