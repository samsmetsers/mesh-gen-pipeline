"""
Puppeteer Runner Script
========================
This script is executed in the Puppeteer Python 3.10 venv by Stage 4.
It loads the Puppeteer model, runs skeleton prediction on an input mesh,
and exports a rigged FBX + joints.json.

Usage (via subprocess from stage4_auto_rig.py):
    python scripts/puppeteer_runner.py \\
        --input  path/to/refined.glb \\
        --output path/to/rigged.fbx \\
        --joints path/to/joints.json \\
        --puppeteer-dir external/Puppeteer \\
        --checkpoint external/Puppeteer/skeleton_ckpts/puppeteer_skeleton_w_diverse_pose.pth

Key Puppeteer flags:
  --seq_shuffle  Required: checkpoint was trained with sequence shuffling;
                 omitting causes load_state_dict to fail (target_aware_pos_embed
                 key missing from state dict).
  --joint_token  Required: each token encodes 4 values (x, y, z, parent_index).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",         required=True)
    parser.add_argument("--output",        required=True)
    parser.add_argument("--joints",        required=True)
    parser.add_argument("--puppeteer-dir", required=True)
    parser.add_argument("--checkpoint",    required=True)
    args = parser.parse_args()

    puppeteer_dir = Path(args.puppeteer_dir).resolve()
    if str(puppeteer_dir) not in sys.path:
        sys.path.insert(0, str(puppeteer_dir))

    # ── Import Puppeteer internals ────────────────────────────────────────────
    try:
        import torch
        # Puppeteer uses its own inference utilities
        # The exact import paths depend on the repo structure
        # Adjust if Puppeteer repo restructures its API
        from inference import PuppeteerInference  # type: ignore[import]
    except ImportError as e:
        print(f"[puppeteer_runner] ImportError: {e}", file=sys.stderr)
        print("[puppeteer_runner] Ensure setup_puppeteer.sh was run successfully.", file=sys.stderr)
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[puppeteer_runner] Using device: {device}")
    print(f"[puppeteer_runner] Checkpoint: {args.checkpoint}")
    print(f"[puppeteer_runner] Input mesh: {args.input}")

    # ── Run Puppeteer inference ───────────────────────────────────────────────
    inferencer = PuppeteerInference(
        checkpoint=args.checkpoint,
        device=device,
        seq_shuffle=True,     # REQUIRED: checkpoint trained with --seq_shuffle
        joint_token=True,     # REQUIRED: 4 values per token (x, y, z, parent)
    )

    print("[puppeteer_runner] Running skeleton prediction …")
    result = inferencer.predict(
        mesh_path=args.input,
        output_fbx=args.output,
    )

    # ── Write joints.json ────────────────────────────────────────────────────
    joints = result.get("joints", [])
    Path(args.joints).write_text(json.dumps(joints, indent=2))

    print(f"[puppeteer_runner] Saved FBX: {args.output}")
    print(f"[puppeteer_runner] Saved joints: {args.joints} ({len(joints)} joints)")


if __name__ == "__main__":
    main()
