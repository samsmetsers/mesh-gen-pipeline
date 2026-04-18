import os
import argparse
import subprocess
import json
from pathlib import Path
import shutil
import yaml

# Add MotionGPT3 to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MGPT_DIR = PROJECT_ROOT / "external" / "MotionGPT3"
MGPT_PYTHON = str(PROJECT_ROOT / ".venv_motiongpt" / "bin" / "python")

def run_cmd(cmd, cwd=str(MGPT_DIR)):
    print(f"Executing: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(MGPT_DIR) + ":" + env.get("PYTHONPATH", "")
    # Force single GPU for RTX 3080
    env["CUDA_VISIBLE_DEVICES"] = "0"
    subprocess.run(cmd, check=True, cwd=cwd, env=env)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glb", required=True)
    parser.add_argument("--output-glb", required=True)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    # Define base animations to generate for every character
    base_prompts = [
        "a character standing idle",
        "a character walking forward",
        "a character attacking with a weapon",
        "a character dying and falling to the ground",
    ]
    # Add the user's specific prompt as the last one
    all_prompts = base_prompts + [args.prompt]

    # Step 1: Write prompt to temp txt file (MotionGPT3 demo expects one prompt per line)
    temp_input_txt = PROJECT_ROOT / "tmp_mgpt_input.txt"
    temp_input_txt.write_text("\n".join(all_prompts) + "\n")

    # Step 2: Resolve checkpoint path (motiongpt3.ckpt preferred; mld_humanml3d.ckpt fallback)
    ckpt_path = MGPT_DIR / "checkpoints" / "motiongpt3.ckpt"
    if not ckpt_path.exists():
        ckpt_path = MGPT_DIR / "checkpoints" / "mld_humanml3d.ckpt"
    if not ckpt_path.exists():
        raise RuntimeError(f"No MotionGPT3 checkpoint found in {MGPT_DIR / 'checkpoints'}")

    # Step 3: Create override config that extends MoT_vae_stage3 with single-GPU + our checkpoint
    base_cfg_path = MGPT_DIR / "configs" / "MoT_vae_stage3.yaml"
    with open(base_cfg_path, "r") as f:
        base_cfg = yaml.safe_load(f)
    base_cfg["DEVICE"] = [0]
    base_cfg["NUM_NODES"] = 1
    base_cfg.setdefault("TEST", {})["CHECKPOINTS"] = str(ckpt_path.resolve())
    # Increase batch size so it can do all 5 prompts if possible, or just leave at 1. demo.py handles it.
    base_cfg.setdefault("TEST", {})["BATCH_SIZE"] = 1
    base_cfg.setdefault("DEMO", {})["TASK"] = "t2m"
    override_cfg_path = MGPT_DIR / "configs" / "_runtime_override.yaml"
    with open(override_cfg_path, "w") as f:
        yaml.safe_dump(base_cfg, f)

    # Step 4: Prepare output directory
    output_dir = PROJECT_ROOT / "tmp_mgpt_output"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Step 5: Run MotionGPT3 demo.py
    print(f"[MotionGPT3] Generating {len(all_prompts)} animations...")
    demo_cmd = [
        MGPT_PYTHON, "demo.py",
        "--cfg", "configs/_runtime_override.yaml",
        "--cfg_assets", "configs/assets.yaml",
        "--example", str(temp_input_txt.resolve()),
        "--out_dir", str(output_dir.resolve()),
    ]

    try:
        run_cmd(demo_cmd)
    except subprocess.CalledProcessError as e:
        print(f"[MotionGPT3] Inference failed: {e}. Falling back to no-animation GLB.")
        shutil.copy(args.input_glb, args.output_glb)
        return

    # Step 6: Find generated animations (.npy)
    # demo.py writes to: {out_dir}/motgpt/MoT_vae_stage3/samples_{TIME}/{idx}_out.npy
    # The indices will be 0, 1, 2, 3, 4 corresponding to our prompts.
    npy_files = sorted(output_dir.rglob("*_out.npy"))

    if not npy_files:
        print("[MotionGPT3] Warning: No .npy animations generated. Copying input as output.")
        shutil.copy(args.input_glb, args.output_glb)
        return

    output_glb_path = Path(args.output_glb).resolve()
    output_glb_path.parent.mkdir(parents=True, exist_ok=True)

    # We collect all generated motions to pass to blender_retarget_motion.py
    motion_args = []
    anim_metadata = []
    
    action_names = ["idle", "walk", "attack", "die", "custom"]
    for i, npy_file in enumerate(npy_files):
        if i >= len(all_prompts):
            break
        action_name = action_names[i]
        npy_out = output_glb_path.parent / f"{output_glb_path.stem}_{action_name}.npy"
        shutil.copy(npy_file, npy_out)
        motion_args.extend(["--motion-npy", f"{action_name}={str(npy_out)}"])
        anim_metadata.append({
            "action_name": action_name,
            "motion_npy": str(npy_out),
            "prompt": all_prompts[i],
        })

    # Step 7: Retarget the SMPL motions onto the UniRig skeleton via Blender.
    retarget_script = PROJECT_ROOT / "scripts" / "blender_retarget_motion.py"
    retarget_cmd = [
        "blender", "--background", "--python", str(retarget_script), "--",
        "--input-glb", str(Path(args.input_glb).resolve()),
        "--output-glb", str(output_glb_path),
    ] + motion_args
    
    print(f"[MotionGPT3] Retargeting SMPL → UniRig: {' '.join(retarget_cmd)}")
    try:
        subprocess.run(retarget_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[MotionGPT3] Retarget failed: {e}; falling back to static copy.")
        shutil.copy(args.input_glb, output_glb_path)

    anim_meta = output_glb_path.parent / f"{output_glb_path.stem}_motions.json"
    with open(anim_meta, "w") as f:
        json.dump({
            "source_ckpt": str(ckpt_path),
            "motions": anim_metadata,
        }, f, indent=2)

    # Clean up temp folders inside script so they don't accumulate.
    try:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if temp_input_txt.exists():
            temp_input_txt.unlink()
    except Exception as exc:
        print(f"[MotionGPT3] Temp cleanup warning: {exc}")

    print(f"[MotionGPT3] Complete. Animated GLB: {output_glb_path}")

if __name__ == "__main__":
    main()
