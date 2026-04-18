"""
Stage 5: Generative Animation
=============================
Utilizes MotionGPT3 to generate a library of specific combat or interaction animations based on text prompts.
Exports into the final GLB.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from pydantic import BaseModel, Field

from src.stage4_auto_rig import Stage4Output

class Stage5Output(BaseModel):
    animated_glb_path: str = Field(description="Path to the final animated GLB file.")
    animations_json_path: str = Field(description="Path to the animations metadata.")
    output_name: str = Field(description="Short identifier used for file naming.")

_PROJECT_ROOT = Path(__file__).parent.parent
_MOTIONGPT_VENV = _PROJECT_ROOT / ".venv_motiongpt"

def _run_motiongpt3(rigged_glb: str, output_glb: str, original_prompt: str) -> dict:
    print(f"[Stage 5] Generating animations via MotionGPT3 for prompt: '{original_prompt}'")
    python_bin = str(_MOTIONGPT_VENV / "bin" / "python")
    script_path = _PROJECT_ROOT / "scripts" / "motiongpt_inference.py"
    
    cmd = [
        python_bin, str(script_path),
        "--input-glb", str(Path(rigged_glb).resolve()),
        "--output-glb", str(Path(output_glb).resolve()),
        "--prompt", original_prompt
    ]
    subprocess.run(cmd, check=True)
    
    return {"animations": ["generated_via_motiongpt3"]}

def run_stage5(stage4_output: Stage4Output, original_prompt: str, output_dir: str = "output") -> Stage5Output:
    name = stage4_output.output_name
    rigged_glb = stage4_output.glb_path
    
    out_root = Path(output_dir) / name
    intermediate = out_root / "intermediate"
    intermediate.mkdir(parents=True, exist_ok=True)
    
    animated_glb = str(out_root / f"{name}_animated.glb")
    animations_json = str(intermediate / "animations.json")
    
    anim_data = _run_motiongpt3(rigged_glb, animated_glb, original_prompt)
    
    with open(animations_json, "w") as f:
        json.dump(anim_data, f, indent=2)
        
    return Stage5Output(
        animated_glb_path=os.path.abspath(animated_glb),
        animations_json_path=os.path.abspath(animations_json),
        output_name=name,
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 5: Generative Animation")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to stage4_output.json")
    parser.add_argument("--prompt", "-p", type=str, required=True, help="Original user prompt")
    parser.add_argument("--output-dir", "-o", type=str, default="output")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        s4_data = json.load(f)
    s4 = Stage4Output(**s4_data)

    result = run_stage5(s4, args.prompt, output_dir=args.output_dir)
    json_path = Path(args.output_dir) / result.output_name / "intermediate" / "stage5_output.json"
    json_path.write_text(result.model_dump_json(indent=2))
    print(f"\n[Stage 5] Complete. Output JSON: {json_path}")
    print(result.model_dump_json(indent=2))
