# Mesh Generation & Animation Pipeline

An autonomous pipeline for generating high-quality rigged and animated 3D characters from text prompts.

## 🚀 Overview

The pipeline consists of 4 specialized stages:
1.  **Stage 1: Vision Prior (FLUX):** Generates a high-quality 2D reference image from your prompt.
2.  **Stage 2: 3D Generation (TRELLIS.2):** Converts the 2D image into a high-fidelity 3D mesh.
3.  **Stage 3: Refinement:** Optimizes the mesh and handles format conversions (OBJ to GLB).
4.  **Stage 4: Animation Routing (UniRig + Blender):** Automatically extracts a skeleton, predicts skinning weights, and assembles a rigged GLB with **T-Pose** and **WalkCycle** animations.

---

## 💻 Environment Requirements

This pipeline requires a Linux environment (WSL2 supported) and an NVIDIA GPU (RTX 3080+ recommended).

### 1. WSL2 Installation (Windows Users)
If you are on Windows, you MUST use WSL2 (Ubuntu 22.04 or 24.04).
- Open PowerShell as Admin and run: `wsl --install`
- Follow the official guide: [Install WSL](https://learn.microsoft.com/en-us/windows/wsl/install)

### 2. CUDA & Drivers
Ensure you have the latest NVIDIA drivers installed on Windows. WSL2 will share the GPU with Windows.
- Inside WSL, you need the CUDA toolkit. Our setup script handles this, but your driver version should support **CUDA 12.4+**.

### 3. Dual Conda Environments
Due to dependency conflicts between deep learning libraries and Blender's bundled Python, the pipeline uses two environments:
- **`meshgen` (Python 3.10):** Main pipeline environment (Stage 1-4 logic).
- **`blender_env` (Python 3.12):** Specifically for Blender assembly to provide compatible `numpy` versions.

---

## 🛠️ Installation

1.  **Clone the repo and submodules:**
    ```bash
    git clone --recursive <repo-url>
    cd mesh-gen-pipeline
    ```

2.  **Run the automated setup:**
    ```bash
    bash scripts/setup_wsl.sh
    ```
    This script will:
    - Install Miniconda
    - Create the `meshgen` and `blender_env` environments
    - Install PyTorch with CUDA 12.4 support
    - Install all dependencies from `requirements.txt`

3.  **Setup external repos:**
    ```bash
    bash scripts/setup_external_repos.sh
    ```

---

## 🏃 Usage

Run the complete pipeline with a single command:

```bash
conda activate meshgen
python main.py --prompt "A mystical shaman with a staff" --output_name shaman --type character
```

### Options:
- `--type character`: Enables Stage 4 rigging and animation.
- `--type rigid`: Enables rigid-body physics simulation (coming soon).
- `--skip_gen`: Use an existing reference image in `output/<name>/`.
- `--skip_trellis`: Use an existing raw mesh.
- `--skip_refine`: Use an existing refined mesh.

---

## 📂 Project Structure

- `src/`: Core pipeline logic.
  - `extract_trimesh.py`: Mesh preprocessing.
  - `calculate_high_poly_skin.py`: High-resolution skin weight prediction.
  - `assemble_character.py`: Blender script for rigging and NLA animation assembly.
- `TRELLIS.2/`: 3D generation engine.
- `UniRig/`: Skeleton and skinning prediction engine.
- `output/`: Generated assets, logs, and final GLB files.

---

## 🧪 Viewing Animations in Blender

To check the rigged character:
1.  Import `output/<name>/<name>_final.glb` into Blender.
2.  Select the **Armature**.
3.  Change the bottom panel to the **Action Editor** (inside Dope Sheet).
4.  Select **WalkCycle** or **TPose** from the dropdown.
5.  Press **Spacebar** to play.

---

## 📝 Troubleshooting & Learned Fixes

During development, several critical fixes were applied to the base libraries:
- **UniRig Model Loading:** Fixed `UnpicklingError` by allowing `Box` objects in `torch.load`.
- **Spatial Alignment:** Implemented automatic axis-mapping between UniRig's Y-up space and Blender's Z-up space.
- **Bone Visualization:** Fixed "little balls" armature issue by correctly connecting parent joints to children or limb tips.
- **Headless Blender:** Solved `bpy` dependency issues by linking to a specialized `blender_env`.
