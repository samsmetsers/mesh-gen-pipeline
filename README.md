# Mesh-Gen-Pipeline

A free, state-of-the-art, prompt-to-3D-rigged-game-character pipeline.
This pipeline generates 3D rigged, game-ready characters from a text prompt in 5 stages.

## Features
- **Prompt to 3D**: Go from a description to a fully textured, low-poly 3D character.
- **Auto-Rigging**: Automatically rigs the generated mesh with a humanoid skeleton.
- **Motion Generation**: Applies AI-generated animations (idle, walk, attack) to the rigged character.
- **Modular Architecture**: Run the pipeline end-to-end or stage-by-stage.
- **Hardware constraints**: Fits on a consumer GPU (10 GB VRAM, RTX 3080).

## Pipeline Stages
1. **Stage 1 (Prompt Parsing)**: Uses Llama-3.3-70B to extract character details and desired animation.
2. **Stage 2 (Text-to-3D)**: Uses FLUX.2-klein-4B to generate a concept image, BiRefNet for background removal, and TRELLIS.2-4B for generating a 3D textured mesh.
3. **Stage 3 (Mesh Optimization)**: Decimates the mesh using PyMeshLab to target low polygon counts.
4. **Stage 4 (Auto-Rigging)**: Uses P3-SAM to segment the mesh and UniRig for autoregressive skeleton prediction and skinning. Standardized in Blender.
5. **Stage 5 (Animation)**: Uses MotionGPT3 for text-to-motion generation, mapped back to the Blender rig.

## Setup
Ensure you have Python 3.13 and `uv` installed.
```bash
# Install base deps
uv sync

# Full setup (requires GPU)
./scripts/setup_all.sh
# Note: You may also need to run setup_flux.sh and setup_trellis.sh manually.
```

## Usage
```bash
# Mock run (no GPU required):
uv run python main.py --prompt "Game-ready prehistoric shaman character..." -n shaman --mock

# Real run:
uv run python main.py --prompt "Game-ready prehistoric shaman character..." -n shaman
```

## Licenses and Acknowledgements

This project's code is licensed under the [MIT License](LICENSE). However, the generated assets and the runtime dependencies are governed by the licenses of the underlying models:

- **Llama-3.3-70B**: [Meta Llama 3 Community License](https://llama.meta.com/llama3/license/)
- **FLUX.2-klein-4B**: [Apache 2.0 License](https://github.com/black-forest-labs/flux) / [FLUX.1 [dev] Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md)
- **TRELLIS.2-4B**: [MIT License](https://github.com/microsoft/TRELLIS)
- **BiRefNet**: [MIT License](https://github.com/ZhengPeng7/BiRefNet)
- **PyMeshLab**: [GPL](https://github.com/cnr-isti-vclab/PyMeshLab)
- **P3-SAM**: [Apache 2.0 / MIT](https://github.com/dvlab-research/P3-SAM)
- **UniRig**: [Apache 2.0 / MIT](https://github.com/dvlab-research/UniRig)
- **MotionGPT3**: [Apache 2.0 / MIT](https://github.com/OpenMotionLab/MotionGPT)
- **SMPL / HumanML3D**: The SMPL body model and HumanML3D dataset motions are restricted to **Non-Commercial / Academic use only**. Please refer to the [SMPL License](https://smpl.is.tue.mpg.de/) and [HumanML3D](https://github.com/EricGuo5513/HumanML3D) for details.
