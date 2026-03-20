#!/bin/bash
# fix_build.sh - Consolidated environment fix and TRELLIS.2 setup

# 1. Ensure WSL2 driver libraries are in the linker path
export LIBRARY_PATH=/usr/lib/wsl/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# 2. Set CUDA environment to use the conda-installed toolkit (12.4.1)
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH

# 3. Optimization flags for RTX 3080 and memory management
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS=1

# 4. Install system dependency (requires sudo)
echo "Installing system dependencies..."
sudo apt install -y libjpeg-dev

# 5. Clean up previous build attempts
echo "Cleaning up /tmp/extensions..."
rm -rf /tmp/extensions

# 6. Run the actual TRELLIS.2 setup
echo "Starting TRELLIS.2 setup in the current environment..."
cd TRELLIS.2
bash setup.sh --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
cd ..

echo "Setup attempt complete."
