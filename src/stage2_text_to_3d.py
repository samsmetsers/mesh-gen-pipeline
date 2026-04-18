"""
Stage 2: Text-to-3D
====================
Converts a ParsedPrompt into a UV-textured 3D mesh via a two-sub-step process:

  Sub-step A — Text → Image (FLUX.2-klein-4B, 4-step distilled FLUX.1-dev)
    - Saves intermediate PNG for inspection

    Why not SDXL Lightning:
      SDXL Lightning (4-8 step distilled SDXL) uses CLIP-L + CLIP-G only.
      CLIP's 77-token limit and shallow semantic understanding causes complex
      costume descriptions to render as generic warriors. FLUX.2's T5-based
      encoder removes this constraint entirely.

    Why not FLUX.1:
      FLUX.1 requires ~24GB fp16 or complex bitsandbytes nf4 quantization.

    FLUX.2-klein-4B matches FLUX's legendary prompt adherence with a 4B
    parameter footprint that fits comfortably on a 10 GB RTX 3080.

  Sub-step A.5 — Background removal (BiRefNet via TRELLIS.2 rembg module)
    - Strips the white background so TRELLIS.2 receives a clean RGBA image

  Sub-step B — Image → 3D (TRELLIS.2-4B, Microsoft)
    - `Trellis2ImageTo3DPipeline` (pipeline_512_only=True) for 10 GB VRAM
    - `o_voxel.postprocess.to_glb()` for PBR texture baking + GLB export
    - OBJ exported from the raw mesh vertices/faces before texture baking

  Why TRELLIS.2 over Hunyuan3D 2.0:
    - TRELLIS.2 produces MeshWithVoxel outputs: geometry + volumetric PBR
      textures in one pass, no separate paint pipeline needed.
    - Better topology quality and watertight meshes out of the box.
    - CuMesh simplify() (16 M triangle limit for nvdiffrast) built into export.

Mock mode:
  - Writes a minimal cube OBJ + dummy GLB without touching GPU.

Output:
  - <name>_concept.png       ← generated character image (for inspection)
  - <name>_raw.obj           ← untextured geometry
  - <name>_raw.glb           ← PBR UV-textured mesh
  - stage2_output.json
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

from PIL import Image
from pydantic import BaseModel, Field

from src.stage1_prompt_parsing import ParsedPrompt


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class Stage2Output(BaseModel):
    obj_path: str = Field(description="Path to the generated 3D mesh (.obj).")
    glb_path: str = Field(description="Path to the UV-textured 3D mesh (.glb).")
    concept_image_path: str = Field(default="", description="Path to the generated concept image (.png).")
    output_name: str = Field(description="Short identifier for file naming.")


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

MOCK_OBJ_CONTENT = """# Mock 3D Mesh (Cube)
v -0.5 -0.5 0.5
v 0.5 -0.5 0.5
v -0.5 0.5 0.5
v 0.5 0.5 0.5
v -0.5 0.5 -0.5
v 0.5 0.5 -0.5
v -0.5 -0.5 -0.5
v 0.5 -0.5 -0.5
f 1 2 4 3
f 3 4 6 5
f 5 6 8 7
f 7 8 2 1
f 1 3 5 7
f 2 8 6 4
"""


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_image_prompt(parsed: ParsedPrompt) -> tuple[str, str, str]:
    """
    Build SDXL dual-encoder prompts + negative from a ParsedPrompt.

    SDXL Lightning uses two CLIP encoders each capped at 77 tokens.
    Strategy:
      • prompt  (CLIP-L + CLIP-G) — ordering: pose/bg > CHARACTER DESCRIPTION > style.
        Character identity must come before style terms so the model renders the correct
        character archetype rather than a generic "blocky warrior" (Clash-of-Clans style
        tokens dominate when placed first and wash out costume/prop detail).
      • prompt_2 (CLIP-G only)   — single-character framing (NOT "design sheet"
        or "turnaround" — those cause multi-view reference sheet output).

    Optimised for TRELLIS.2 input:
      - T-pose front view → geometry model reconstructs the full character reliably
      - Pure white background → rembg removes it cleanly, no artefacts bleed into mesh
      - Full body head-to-toe → no cropping that would confuse the 3D model
      - Vibrant colours + negative monochrome → prevents sketch/line-art output

    Style notes:
      - "Clash-of-Clans style" removed: too generic, renders barrel-chested barbarians
        and overrides specific costume elements (skull masks, tribal furs, etc.)
      - Only visually meaningful style_tags included; topology tags ("mobile-optimized",
        "clean quad-based topology") filtered out — they confuse SDXL.

    Returns (prompt, prompt_2, negative_prompt).
    """
    char = parsed.character_description
    obj_str = f", holding {parsed.rigid_object}" if parsed.rigid_object else ""

    # Translate "semi-voxel" tag into SDXL-friendly visual terms.
    # Avoid "Clash-of-Clans style" — too dominant, produces generic barbarians regardless
    # of the character description. Use softer stylization terms instead.
    # Only pass visually meaningful tags — topology/tech tags ("mobile-optimized",
    # "clean quad-based topology") confuse SDXL and waste the 77-token budget.
    _VISUAL_TAGS = {"low-poly", "stylized", "game-ready", "cartoon", "cel-shaded", "vibrant"}
    has_voxel = any("voxel" in t.lower() for t in parsed.style_tags)
    style_suffix = (
        "semi-voxel stylized game art, exaggerated proportions, bold flat colours"
        if has_voxel else "stylized game art, bold flat colours"
    )
    extra_tags = ", ".join(
        t for t in parsed.style_tags
        if "voxel" not in t.lower() and t.lower() in _VISUAL_TAGS
    )
    if extra_tags:
        style_suffix = f"{style_suffix}, {extra_tags}"

    # ── Primary prompt (≤77 tokens) ──────────────────────────────────────────
    # Order: pose/bg > CHARACTER IDENTITY > style
    # Character description MUST come before style to prevent generic warrior output.
    # CLIP truncation (77 tokens) is not a risk here (~55 tokens typical), but
    # attention weight distribution still favours earlier tokens.
    prompt = (
        "T-pose, pure white background, front view, full body, "
        f"{char}{obj_str}, "
        f"{style_suffix}, vibrant colours"
    )

    # ── Secondary prompt (CLIP-G ≤77 tokens) — single-character framing ────────
    # WARNING: do NOT use "design sheet", "turnaround", or "multiple views" here —
    # SDXL will generate a reference sheet with many small characters instead of one.
    prompt_2 = (
        "single character, isolated on white background, head to toe, "
        "front-facing, symmetrical T-pose, full body visible, "
        "hero game character, strong silhouette, bold flat colours, high contrast"
    )

    negative = (
        # Background
        "grey background, gradient background, beige background, brown background"
        ", colored background, dark background, shadows, cast shadows"
        # Multi-view / design sheet — KEY additions to prevent reference sheet output
        ", character sheet, design sheet, turnaround, multiple views, multiple poses"
        ", reference sheet, model sheet, ortho views, panel layout"
        # Sketch / monochrome — prevent line-art output
        ", sketch, line art, pencil drawing, wireframe, monochrome, black and white"
        ", greyscale, grayscale, outline only, ink drawing"
        # Bad poses for 3D reconstruction
        ", 3/4 view, side view, back view, action pose, running, jumping"
        ", crouching, leaning, tilted, perspective distortion, foreshortening"
        # Quality / style rejections
        ", photorealistic, hyperrealistic, photograph"
        ", multiple characters, watermark, text, logo, frame, border"
        ", cropped, partial body, blurry, low quality, ugly, deformed"
        ", extra limbs, missing limbs, nsfw"
    )
    return prompt, prompt_2, negative


# ---------------------------------------------------------------------------
# Sub-step A: Text → Image (FLUX.2-klein-4B)
# ---------------------------------------------------------------------------

def _generate_concept_image(
    parsed: ParsedPrompt,
    output_path: str,
    num_steps: int = 28,
) -> Image.Image:
    """
    Use FLUX.2-klein-4B to generate a full-body character concept image.
    Saves the result to output_path and returns the PIL Image.

    Why FLUX.2-klein-4B over SD 3.5 Medium / SDXL:
      - 4B parameters fits natively and lightning-fast on a 10GB RTX 3080.
      - Maintains Flux's legendary prompt adherence and language understanding
        without the 24GB VRAM footprint of larger models.
      - Can follow complex multi-sentence paragraphs perfectly.

    VRAM management (10 GB RTX 3080):
      - `enable_model_cpu_offload()` ensures the model fits comfortably within VRAM.
      - All components deleted + CUDA cache cleared before returning.

    HF model: black-forest-labs/FLUX.2-klein-4B
    License: Apache 2.0
    """
    import torch
    from diffusers import DiffusionPipeline

    prompt, prompt_2, negative = _build_image_prompt(parsed)
    # Combine prompts since FLUX.2 Klein has a single text encoder
    full_prompt = f"{prompt}, {prompt_2}"
    print(f"[Stage 2] Prompt: {full_prompt[:120]}…")

    model_id = "black-forest-labs/FLUX.2-klein-4B"
    print(f"[Stage 2] Loading {model_id}…")
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )
    # CPU offload to keep VRAM strictly under 10 GB
    pipe.enable_model_cpu_offload()

    print(f"[Stage 2] Generating concept image ({num_steps} steps, 1024×1024)…")
    image = pipe(
        prompt=full_prompt,
        num_inference_steps=num_steps,
        width=1024,
        height=1024,
    ).images[0]

    image.save(output_path)
    print(f"[Stage 2] Concept image saved: {output_path}")

    # Free VRAM before loading TRELLIS.2
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return image


def _generate_concept_image_sdxl_lightning(
    parsed: ParsedPrompt,
    output_path: str,
    num_steps: int = 8,
) -> Image.Image:
    """
    DEPRECATED — kept for reference / emergency fallback only.

    SDXL Lightning (ByteDance) with the SDXL base text encoders.
    Replaced by FLUX.2-klein-4B because CLIP's 77-token limit causes complex
    costume descriptions (skull masks, tribal furs) to collapse into generic
    warriors regardless of prompt wording.

    Usage: call explicitly if FLUX.2-klein-4B fails to download.
    """
    import torch
    from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    prompt, prompt_2, negative = _build_image_prompt(parsed)
    base_model = "stabilityai/stable-diffusion-xl-base-1.0"
    lightning_repo = "ByteDance/SDXL-Lightning"
    lightning_ckpt = f"sdxl_lightning_{num_steps}step_unet.safetensors"

    print(f"[Stage 2][SDXL-Lightning fallback] Loading UNet ({num_steps}-step)…")
    unet = UNet2DConditionModel.from_config(base_model, subfolder="unet").to("cuda", torch.float16)
    unet.load_state_dict(load_file(hf_hub_download(lightning_repo, lightning_ckpt), device="cuda"))

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model, unet=unet, torch_dtype=torch.float16, variant="fp16",
    ).to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )

    image = pipe(
        prompt, prompt_2=prompt_2, negative_prompt=negative,
        num_inference_steps=num_steps, guidance_scale=0,
        width=1024, height=1024,
    ).images[0]

    image.save(output_path)
    del pipe, unet
    gc.collect()
    torch.cuda.empty_cache()
    return image


# ---------------------------------------------------------------------------
# Sub-step A.5: Background removal
# ---------------------------------------------------------------------------

def _remove_background(image: Image.Image) -> Image.Image:
    """
    Strip the white background using TRELLIS.2's bundled BiRefNet rembg module.

    Returns an RGBA PIL Image with the subject on a transparent background.
    TRELLIS.2's `preprocess_image()` accepts RGBA directly, skipping its own
    rembg step (we load the pipeline with skip_rembg=True to save VRAM).
    """
    import sys
    trellis_root = str(Path("external/TRELLIS.2").resolve())
    if trellis_root not in sys.path:
        sys.path.insert(0, trellis_root)
    # Must be set before the first `import trellis2` — trellis2/__init__.py
    # imports trellis2.modules which initialises the attention backend at import time.
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    from trellis2.pipelines.rembg import BiRefNet
    remover = BiRefNet()
    remover.cuda()
    return remover(image)


# ---------------------------------------------------------------------------
# Sub-step B: Image → 3D (TRELLIS.2-4B)
# ---------------------------------------------------------------------------

def _image_to_3d(
    image: Image.Image,
    glb_path: str,
    obj_path: str,
    decimation_target: int = 1_000_000,
    texture_size: int = 2048,
) -> None:
    """
    Run TRELLIS.2 to produce a GLB + OBJ from a background-removed RGBA image.

    Pipeline steps:
      1. Load Trellis2ImageTo3DPipeline (512-only, no rembg) — fits in 10 GB VRAM
      2. `pipeline.run(image, pipeline_type='512', preprocess_image=False)`
         → returns MeshWithVoxel (geometry + volumetric PBR latent)
      3. mesh.simplify(16_777_216) — nvdiffrast requires ≤16M triangles
      4. Export raw OBJ from mesh.vertices / mesh.faces (untextured, for Stage 3)
      5. `o_voxel.postprocess.to_glb()` — PBR texture bake + GLB export

    VRAM management (10 GB RTX 3080):
      - pipeline_512_only=True skips loading the 1024-res models (~5 GB saving)
      - skip_rembg=True skips BiRefNet loading (caller already removed background)
      - low_vram=True (default) uses sequential CPU offloading between model stages
      - Pipeline deleted + CUDA cache cleared after export

    Args:
        image: RGBA PIL Image with transparent background.
        glb_path: Destination path for the PBR GLB file.
        obj_path: Destination path for the untextured OBJ file.
        decimation_target: Target triangle count for o_voxel texture baking.
        texture_size: PBR texture atlas resolution in pixels (2048 or 4096).
    """
    import sys
    import torch
    import trimesh

    trellis_root = str(Path("external/TRELLIS.2").resolve())
    if trellis_root not in sys.path:
        sys.path.insert(0, trellis_root)

    # Use PyTorch's built-in SDPA — flash_attn is not installed.
    # Must be set before trellis2 attention modules are imported/initialised.
    os.environ.setdefault("ATTN_BACKEND", "sdpa")

    import o_voxel
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    print("[Stage 2] Loading TRELLIS.2-4B pipeline (512-only, no rembg)…")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
        "microsoft/TRELLIS.2-4B",
        pipeline_512_only=True,
        skip_rembg=True,
    )
    pipeline.cuda()

    print("[Stage 2] Running TRELLIS.2 image-to-3D…")
    mesh = pipeline.run(image, pipeline_type="512", preprocess_image=False)[0]

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    # Simplify to nvdiffrast's 16M triangle limit before baking
    print("[Stage 2] Simplifying mesh…")
    mesh.simplify(16_777_216)

    # ── Export raw OBJ (untextured, for Stage 3 cleanup) ────────────────────
    print(f"[Stage 2] Exporting raw OBJ → {obj_path}")
    verts_np = mesh.vertices.cpu().numpy()
    faces_np = mesh.faces.cpu().numpy()
    raw_mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
    raw_mesh.export(obj_path)

    # ── PBR texture baking + GLB export via o_voxel ─────────────────────────
    print(f"[Stage 2] Baking PBR textures + exporting GLB → {glb_path}")
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    glb.export(glb_path)
    print("[Stage 2] TRELLIS.2 image-to-3D complete.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_3d_mesh(
    parsed_prompt: ParsedPrompt,
    output_dir: str = "output",
    output_name: str = "character",
) -> Stage2Output:
    """
    Takes a ParsedPrompt and generates a UV-textured 3D mesh.

    1. FLUX.2-klein-4B → concept PNG (white background, 1024×1024)
    2. BiRefNet (TRELLIS.2 rembg) → RGBA with transparent background
    3. TRELLIS.2-4B → MeshWithVoxel (geometry + volumetric PBR latent)
    4. mesh.simplify(16M) → within nvdiffrast limits
    5. Export raw OBJ (untextured) for Stage 3
    6. o_voxel.postprocess.to_glb() → PBR UV-textured GLB
    """
    out_path = Path(output_dir) / output_name / "intermediate"
    out_path.mkdir(parents=True, exist_ok=True)

    concept_png  = str(out_path / f"{output_name}_concept.png")
    obj_filename = str(out_path / f"{output_name}_raw.obj")
    glb_filename = str(out_path / f"{output_name}_raw.glb")

    # ── Real generation ───────────────────────────────────────────────────────
    print("[Stage 2] Starting real generation pipeline…")

    # A: Text → Image
    concept_image = _generate_concept_image(parsed_prompt, concept_png)

    # A.5: Remove background
    print("[Stage 2] Removing background…")
    concept_image_nobg = _remove_background(concept_image)

    # B: Image → 3D
    _image_to_3d(concept_image_nobg, glb_filename, obj_filename)

    print("[Stage 2] Text-to-3D complete.")
    return Stage2Output(
        obj_path=os.path.abspath(obj_filename),
        glb_path=os.path.abspath(glb_filename),
        concept_image_path=os.path.abspath(concept_png),
        output_name=output_name,
    )
