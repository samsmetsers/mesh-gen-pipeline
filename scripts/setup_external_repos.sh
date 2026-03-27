#!/bin/bash

# setup_external_repos.sh - Clone external repositories required by the pipeline
# Run this BEFORE setup_uv_envs.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Stage 2: TRELLIS.2
echo "Setting up TRELLIS.2..."
if [ ! -d "TRELLIS.2" ]; then
    git clone --recursive https://github.com/microsoft/TRELLIS.2
fi

# Stage 4b: Puppeteer (UniRig) — cloned as extern/Puppeteer
echo "Setting up Puppeteer (UniRig)..."
if [ ! -d "extern/Puppeteer" ]; then
    mkdir -p extern
    git clone https://github.com/VAST-AI-Research/UniRig.git extern/Puppeteer
fi

# Stage 4a: SAMPart3D (used for PartSAM pointops extension)
echo "Setting up SAMPart3D..."
if [ ! -d "extern/SAMPart3D" ]; then
    mkdir -p extern
    git clone https://github.com/Pointcept/SAMPart3D.git extern/SAMPart3D
fi

echo "External repositories setup complete."
