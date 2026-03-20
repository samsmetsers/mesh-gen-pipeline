#!/bin/bash

# setup_external_repos.sh - Clones and sets up TRELLIS.2 and UniRig

set -e

# Stage 2: TRELLIS.2
echo "Setting up TRELLIS.2..."
if [ ! -d "TRELLIS.2" ]; then
    git clone --recursive https://github.com/microsoft/TRELLIS.2
fi
cd TRELLIS.2
bash setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
cd ..

# Stage 4: UniRig
echo "Cloning UniRig..."
git clone https://github.com/VAST-AI-Research/UniRig

echo "External repositories cloned and setup instructions for TRELLIS.2 executed."
