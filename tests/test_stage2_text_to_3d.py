import inspect
import os
import json
import pytest
from src.stage1_prompt_parsing import ParsedPrompt
from src.stage2_text_to_3d import (
    generate_3d_mesh,
    Stage2Output,
    MOCK_OBJ_CONTENT,
    _generate_concept_image,
    _generate_concept_image_sdxl_lightning,
)


# ---------------------------------------------------------------------------
# Mock mode
# ---------------------------------------------------------------------------

def test_generate_3d_mesh_mock(tmp_path):
    """Test generating a 3D mesh in mock mode."""
    parsed = ParsedPrompt(
        character_description="A cool wizard",
        rigid_object="staff",
        animation_type="idle",
        style_tags=["low-poly", "game-ready"]
    )
    result = generate_3d_mesh(parsed, output_dir=str(tmp_path / "output"), mock=True)
    assert isinstance(result, Stage2Output)
    assert os.path.exists(result.obj_path)
    assert os.path.exists(result.glb_path)
    with open(result.obj_path, "r") as f:
        assert f.read() == MOCK_OBJ_CONTENT


# ---------------------------------------------------------------------------
# Image generation model selection (source-level)
# ---------------------------------------------------------------------------

class TestImageGenerationModel:
    """Verify that FLUX.2-klein-4B is the active image model."""

    def test_default_uses_flux2_klein(self):
        """_generate_concept_image must use black-forest-labs/FLUX.2-klein-4B."""
        src = inspect.getsource(_generate_concept_image)
        assert "FLUX.2-klein-4B" in src, (
            "_generate_concept_image must use FLUX.2-klein-4B for high adherence within 10 GB VRAM."
        )

    def test_default_uses_flux_pipeline(self):
        """FLUX.2-klein-4B uses DiffusionPipeline to automatically resolve the correct pipeline class."""
        src = inspect.getsource(_generate_concept_image)
        assert "DiffusionPipeline" in src
        assert "FluxPipeline" not in src
        assert "StableDiffusionXLPipeline" not in src

    def test_default_uses_cpu_offload(self):
        """CPU offload is required to keep VRAM strictly under 10 GB."""
        src = inspect.getsource(_generate_concept_image)
        assert "enable_model_cpu_offload" in src, (
            "FLUX.2-klein-4B must use enable_model_cpu_offload() to fit nicely in 10 GB VRAM"
        )

    def test_default_uses_bfloat16(self):
        """FLUX models are tuned for bfloat16."""
        src = inspect.getsource(_generate_concept_image)
        assert "bfloat16" in src

    def test_num_inference_steps_set(self):
        """
        FLUX.2-klein-4B is a 4-step distilled model; real generation should use
        ≥4 steps.  guidance_scale is intentionally omitted — FLUX.2-klein is
        guidance-free (distilled), so the model default (0.0) is correct.
        """
        src = inspect.getsource(_generate_concept_image)
        assert "num_inference_steps" in src, (
            "num_inference_steps must be passed to the FLUX.2-klein-4B pipeline call"
        )

    def test_sdxl_lightning_preserved_as_fallback(self):
        """SDXL Lightning kept as _generate_concept_image_sdxl_lightning for emergency fallback."""
        src = inspect.getsource(_generate_concept_image_sdxl_lightning)
        assert "SDXL-Lightning" in src or "sdxl_lightning" in src.lower()
