#!/bin/bash

# setup_wsl.sh - Environment setup for mesh-gen-pipeline in WSL2
# This script sets up a Conda environment and installs PyTorch with CUDA 12.4 support.

set -e

echo "Starting WSL2 Environment Setup..."

# 1. Install Miniconda if not present
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh
    ~/miniconda3/bin/conda init bash
    fi

    # Ensure conda is available in the current shell
    CONDA_BASE=$(conda info --base)
    source "$CONDA_BASE/etc/profile.d/conda.sh"

    # Accept Anaconda Terms of Service (required for some versions of Miniconda/Conda)
    echo "Accepting Anaconda Terms of Service..."
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

    # 2. Create Conda Environment
    echo "Creating Conda environment 'meshgen' with Python 3.10..."
    if ! conda info --envs | grep -q "^meshgen "; then
        conda create -n meshgen python=3.10 -y
    fi
    conda activate meshgen

    # 3. Install PyTorch with CUDA 12.4 support
    echo "Installing PyTorch 2.4+ with CUDA 12.4..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    # Use conda to install a matching CUDA toolkit to PyTorch's CUDA version for building extensions
    conda install -n meshgen -c nvidia cuda-version=12.4 cuda-toolkit=12.4 -y
    export CUDA_HOME="$CONDA_PREFIX"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LIBRARY_PATH="$CUDA_HOME/lib/stubs:/usr/lib/wsl/lib:$LIBRARY_PATH"
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
    # Aggressively fix the conda ld issue in WSL2
    [ -f "$CONDA_PREFIX/compiler_compat/ld" ] && rm -f "$CONDA_PREFIX/compiler_compat/ld"

    # 4. Install other dependencies
    echo "Installing pipeline dependencies..."
    # Ensure the correct architecture for RTX 3080
    export TORCH_CUDA_ARCH_LIST="8.6"
    export ALLOW_CUDA_VERSION_MISMATCH=1
    # Limit parallel jobs to prevent OOM
    export MAX_JOBS=1
    pip install -r requirements.txt

echo "WSL2 Setup Complete! Please restart your terminal or run 'source ~/.bashrc' and 'conda activate meshgen'."
