#!/bin/bash
set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -d ".venv_motiongpt" ]; then
    uv venv .venv_motiongpt --python 3.11
fi
source .venv_motiongpt/bin/activate
uv pip install -r external/MotionGPT3/requirements.txt || true
uv pip install setuptools gdown smplx
python -m spacy download en_core_web_sm || true

cd external/MotionGPT3

# Download SMPL models, GPT2 tokenizer, and pretrained MotionGPT3/MLD ckpts
bash prepare/download_smpl_model.sh || true
bash prepare/prepare_gpt2.sh || true
bash prepare/download_pretrained_motiongpt3_model.sh || true
bash prepare/download_mld_pretrained_models.sh || true

# Download t2m evaluators (contains glove + Comp_v6_KLD01 mean/std needed by inference)
bash prepare/download_t2m_evaluators.sh || true

# Flatten t2m nesting so paths match assets.yaml (deps/t2m/t2m/Comp_v6_KLD01/meta/*)
if [ -d deps/t2m/t2m/t2m ]; then
    mkdir -p deps/_t2m_fixed/t2m
    mv deps/t2m/t2m/t2m/* deps/_t2m_fixed/t2m/ 2>/dev/null || true
    mv deps/t2m/t2m/kit deps/_t2m_fixed/t2m/kit 2>/dev/null || true
    rm -rf deps/t2m
    mv deps/_t2m_fixed deps/t2m
fi

# Move glove to expected location
if [ -d deps/t2m/glove ] && [ ! -d deps/glove ]; then
    mv deps/t2m/glove deps/glove
fi

# Set up minimal datasets/humanml3d stub so demo.py dataset bootstrap succeeds.
# Inference only needs Mean.npy/Std.npy; the sample-set requires one motion+text file.
mkdir -p datasets/humanml3d/new_joint_vecs datasets/humanml3d/texts
if [ -f deps/t2m/t2m/Comp_v6_KLD01/meta/mean.npy ]; then
    cp deps/t2m/t2m/Comp_v6_KLD01/meta/mean.npy datasets/humanml3d/Mean.npy
    cp deps/t2m/t2m/Comp_v6_KLD01/meta/std.npy datasets/humanml3d/Std.npy
fi
echo "fake_id" > datasets/humanml3d/test.txt
echo "fake_id" > datasets/humanml3d/val.txt
echo "fake_id" > datasets/humanml3d/train.txt
echo "fake_id" > datasets/humanml3d/val_tiny.txt
python -c "
import numpy as np, os
if not os.path.exists('datasets/humanml3d/new_joint_vecs/fake_id.npy'):
    np.save('datasets/humanml3d/new_joint_vecs/fake_id.npy', np.random.randn(40, 263).astype(np.float32))
" || true
if [ ! -f datasets/humanml3d/texts/fake_id.txt ]; then
    echo "a person walks forward#0/1/OTHER/OTHER#0.0#40.0" > datasets/humanml3d/texts/fake_id.txt
fi
