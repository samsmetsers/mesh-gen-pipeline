"""
Patch TRELLIS.2 pipeline for 10GB VRAM support.

Adds `pipeline_512_only` and `skip_rembg` flags to from_pretrained:
  - pipeline_512_only=True: only load 512-res flow models (saves ~5 GB)
  - skip_rembg=True: skip BiRefNet background removal model
  - Sets default_pipeline_type to '512' when pipeline_512_only=True

Usage:
    python scripts/patch_trellis_pipeline.py external/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py
"""

import sys
from pathlib import Path


PATCH_MARKER = "# [PATCHED: 10GB VRAM support]"

OLD_FROM_PRETRAINED = '''    def from_pretrained(cls, path: str, config_file: str = "pipeline.json") -> "Trellis2ImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super().from_pretrained(path, config_file)
        args = pipeline._pretrained_args'''

NEW_FROM_PRETRAINED = '''    def from_pretrained(  # [PATCHED: 10GB VRAM support]
        cls,
        path: str,
        config_file: str = "pipeline.json",
        pipeline_512_only: bool = False,
        skip_rembg: bool = False,
    ) -> "Trellis2ImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
            pipeline_512_only (bool): Only load 512-res models (saves ~5 GB VRAM on 10GB GPUs).
            skip_rembg (bool): Skip loading BiRefNet background removal model.
        """
        # Temporarily override model_names_to_load if 512-only mode
        _orig_names = cls.model_names_to_load
        if pipeline_512_only:
            cls.model_names_to_load = [m for m in cls.model_names_to_load if '1024' not in m]
        try:
            pipeline = super().from_pretrained(path, config_file)
        finally:
            cls.model_names_to_load = _orig_names
        args = pipeline._pretrained_args'''

OLD_REMBG_LINE = "        pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])"

NEW_REMBG_BLOCK = """        if skip_rembg:
            pipeline.rembg_model = None
        else:
            pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])"""

OLD_DEFAULT_PIPELINE = "        pipeline.default_pipeline_type = args.get('default_pipeline_type', '1024_cascade')"

NEW_DEFAULT_PIPELINE = """        pipeline.default_pipeline_type = '512' if pipeline_512_only else args.get('default_pipeline_type', '1024_cascade')"""


def patch_file(path: str) -> None:
    content = Path(path).read_text()

    if PATCH_MARKER in content:
        print(f"[patch] {path} is already patched.")
        return

    # Apply from_pretrained signature patch
    if OLD_FROM_PRETRAINED not in content:
        print(f"[patch] WARNING: could not find from_pretrained signature to patch in {path}")
        print("[patch] The TRELLIS.2 pipeline may have been restructured.")
        return

    content = content.replace(OLD_FROM_PRETRAINED, NEW_FROM_PRETRAINED)

    # Apply rembg skip patch
    content = content.replace(OLD_REMBG_LINE, NEW_REMBG_BLOCK)

    # Apply default_pipeline_type patch
    content = content.replace(OLD_DEFAULT_PIPELINE, NEW_DEFAULT_PIPELINE)

    Path(path).write_text(content)
    print(f"[patch] Successfully patched {path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python patch_trellis_pipeline.py <path_to_trellis2_image_to_3d.py>")
        sys.exit(1)
    patch_file(sys.argv[1])
