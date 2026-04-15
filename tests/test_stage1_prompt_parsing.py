import os
import json
import pytest
from src.stage1_prompt_parsing import parse_prompt, ParsedPrompt, wrap_prompt

def test_parse_prompt_mock():
    """Test the mock parser logic with the user's test prompt."""
    prompt = "Game-ready prehistoric shaman character wearing tribal furs and a dinosaur skull mask, holding a wooden staff with glowing mushrooms, in a combat stance, low-poly, stylized, mobile-optimized, clean quad-based topology."
    
    # Run in mock mode to avoid requiring an active LLM for unit tests
    result = parse_prompt(prompt, mock=True)
    
    # Validate the data structure
    assert isinstance(result, ParsedPrompt)
    
    # Validate fields
    assert result.animation_type == "attack", "Combat stance should infer 'attack' animation."
    assert "wooden staff with glowing mushrooms" in result.rigid_object, "Should extract the rigid object."
    assert "low-poly" in result.style_tags
    assert "game-ready" in result.style_tags

def test_parsed_prompt_schema():
    """Ensure the pydantic schema serializes correctly for the next stage."""
    data = {
        "character_description": "A fierce warrior",
        "rigid_object": "A giant sword",
        "animation_type": "walk",
        "style_tags": ["high-res"]
    }
    
    parsed = ParsedPrompt(**data)
    json_str = parsed.model_dump_json()
    reloaded = json.loads(json_str)
    
    assert reloaded["animation_type"] == "walk"
    assert reloaded["rigid_object"] == "A giant sword"

def test_invalid_animation_type():
    """Ensure pydantic catches invalid animation types."""
    data = {
        "character_description": "A fierce warrior",
        "rigid_object": "A giant sword",
        "animation_type": "flying", # Invalid literal
        "style_tags": ["high-res"]
    }

    with pytest.raises(ValueError):
        ParsedPrompt(**data)


# ---------------------------------------------------------------------------
# Semi-voxel prompt wrapper
# ---------------------------------------------------------------------------

class TestWrapPrompt:
    def test_wrap_prompt_contains_semi_voxel(self):
        result = wrap_prompt("a warrior with a sword")
        assert "semi-voxel" in result.lower()

    def test_wrap_prompt_contains_original_text(self):
        original = "a wizard wearing robes"
        result = wrap_prompt(original)
        assert original in result

    def test_wrap_prompt_mentions_game_ready(self):
        result = wrap_prompt("a knight")
        assert "game" in result.lower()

    def test_wrap_prompt_mentions_blocky_proportions(self):
        result = wrap_prompt("a goblin")
        assert "block" in result.lower() or "chunk" in result.lower()

    def test_wrap_prompt_mentions_hybrid_target(self):
        """Wrapper should reference hybrid mobile/PC game context."""
        result = wrap_prompt("a knight")
        assert "mobile" in result.lower() or "pc" in result.lower() or "game" in result.lower()

    def test_mock_parse_always_includes_semi_voxel_tag(self):
        result = parse_prompt("A simple character.", mock=True)
        assert "semi-voxel" in result.style_tags

    def test_mock_parse_semi_voxel_combined_with_other_tags(self):
        result = parse_prompt("A game-ready low-poly knight.", mock=True)
        assert "semi-voxel" in result.style_tags
        assert "game-ready" in result.style_tags
        assert "low-poly" in result.style_tags


class TestImagePromptQuality:
    """Verify the SDXL image prompt is optimised for Hunyuan3D input."""

    def _get_prompt(self, desc="a warrior", rigid=None, tags=None):
        from src.stage1_prompt_parsing import ParsedPrompt
        from src.stage2_text_to_3d import _build_image_prompt
        p = ParsedPrompt(
            character_description=desc,
            rigid_object=rigid,
            animation_type="idle",
            style_tags=tags or ["semi-voxel", "game-ready"],
        )
        return _build_image_prompt(p)

    def test_positive_contains_t_pose(self):
        pos, _, _ = self._get_prompt()
        assert "t-pose" in pos.lower() or "T-pose" in pos

    def test_positive_contains_front_view(self):
        pos, _, _ = self._get_prompt()
        assert "front" in pos.lower() or "facing camera" in pos.lower()

    def test_positive_contains_white_background(self):
        pos, _, _ = self._get_prompt()
        assert "white background" in pos.lower()

    def test_negative_excludes_grey_background(self):
        _, _, neg = self._get_prompt()
        assert "grey background" in neg.lower()

    def test_negative_excludes_action_poses(self):
        _, _, neg = self._get_prompt()
        assert "action pose" in neg.lower() or "dynamic" in neg.lower()

    def test_negative_excludes_perspective_distortion(self):
        _, _, neg = self._get_prompt()
        assert "perspective" in neg.lower() or "3/4 view" in neg.lower()

    def test_negative_excludes_design_sheet(self):
        """'design sheet' / 'turnaround' cause SDXL to generate multi-view reference sheets."""
        _, _, neg = self._get_prompt()
        assert "design sheet" in neg.lower() or "character sheet" in neg.lower()

    def test_negative_excludes_sketch_and_monochrome(self):
        """Sketch/line-art style produces uncoloured concept images that confuse Hunyuan3D."""
        _, _, neg = self._get_prompt()
        assert "sketch" in neg.lower() or "line art" in neg.lower()
        assert "monochrome" in neg.lower() or "black and white" in neg.lower()

    def test_prompt2_does_not_contain_design_sheet(self):
        """prompt_2 must NOT use 'design sheet' — it causes multi-view reference sheet output."""
        _, p2, _ = self._get_prompt()
        assert "design sheet" not in p2.lower()
        assert "turnaround" not in p2.lower()
        assert "multiple views" not in p2.lower()

    def test_topology_tags_excluded_from_sdxl_prompt(self):
        """Tags like 'mobile-optimized' and 'clean quad-based topology' are not visual — must not appear in prompt."""
        pos, p2, _ = self._get_prompt(tags=["semi-voxel", "mobile-optimized", "clean quad-based topology"])
        assert "mobile-optimized" not in pos
        assert "quad-based" not in pos
        assert "mobile-optimized" not in p2
        assert "quad-based" not in p2

    def test_semi_voxel_tag_expands_to_style_keywords(self):
        pos, _, _ = self._get_prompt(tags=["semi-voxel"])
        assert "semi-voxel" in pos.lower() or "stylized" in pos.lower() or "voxel" in pos.lower()

    def test_character_description_before_style(self):
        """Character identity must appear before style keywords in the primary prompt.
        'Clash-of-Clans style' / generic style tokens before the character description
        cause SDXL to produce generic barbarians that ignore the specific costume."""
        pos, _, _ = self._get_prompt(desc="shaman with a skull mask", tags=["semi-voxel"])
        char_pos = pos.lower().find("shaman")
        style_pos = pos.lower().find("semi-voxel")
        assert char_pos != -1, "Character description missing from prompt"
        assert style_pos != -1, "Style missing from prompt"
        assert char_pos < style_pos, (
            "Character description must appear BEFORE style terms in the primary prompt "
            f"(char at {char_pos}, style at {style_pos})"
        )

    def test_rigid_object_included_in_prompt(self):
        pos, _, _ = self._get_prompt(rigid="a magic staff")
        assert "magic staff" in pos.lower()
