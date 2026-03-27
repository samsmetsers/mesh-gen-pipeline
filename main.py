"""
Mesh Generation & Animation Pipeline
=====================================
Stage 1: Vision Prior (FLUX)        - Text → 2D reference image
Stage 2: 3D Generation (TRELLIS.2)  - Image → raw 3D mesh
Stage 3: Mesh Refinement            - Taubin Smooth + Topology-Preserved Retopo
Stage 4: Character Rigging & Animation
  4a: Prop Segmentation             - Separate body and props
  4b: Universal Rigging (UniRig)    - body.obj → rigged_body.glb
  4c: Motion Generation             - Text → walk/attack animations
  4d: Final Assembly (Blender)      - Rig + props + TRELLIS color → .fbx/.glb

Usage:
  python main.py --prompt "A deer with antlers" --output_name deer
  python main.py --prompt "A warrior with a sword" --stage 3 4 --output_name warrior
  python main.py --prompt "quadruped beast walking" --stage 4 --output_name beast
"""
import os
import sys
import argparse
import subprocess
import shutil


def run_stage(env_name, script_path, args_list):
    """Runs a stage script in a specific uv virtualenv."""
    import subprocess
    import sys
    
    # Mapping conda env names to uv venv paths
    env_map = {
        "flux_env": ".venv",
        "meshgen": ".venv",
        "stage34_env": ".venv",
        "PartSAM": ".venv_PartSAM",
        "unirig": ".venv_unirig",
        "motion": ".venv",
        "riganything": ".venv_riganything",
        "sampart3d": ".venv_sampart3d"
    }
    
    venv_dir = env_map.get(env_name, ".venv")
    python_exe = os.path.join(venv_dir, "bin", "python")

    # Set environment variables for the subprocess
    env = os.environ.copy()
    env["TORCH_WEIGHTS_ONLY"] = "0"  # Fix for PyTorch 2.6+ loading old checkpoints

    if not os.path.exists(python_exe):
        print(f"Warning: {python_exe} not found, falling back to sys.executable")
        python_exe = sys.executable

    cmd = [python_exe, script_path] + args_list
    
    print(f"\n[Environment: {env_name} (using {venv_dir})] Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def main():
    parser = argparse.ArgumentParser(description="Mesh Generation & Animation Pipeline")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for character generation")
    parser.add_argument("--output_name", type=str, default="creature", help="Base name for output files")
    parser.add_argument("--stage", type=int, nargs="+", default=[1, 2, 3, 4], help="Stages to run (1-4)")
    parser.add_argument("--creature_type", type=str, choices=["biped", "quadruped"], help="Override creature type")
    parser.add_argument("--target_faces", type=int, default=10000, help="Target face count for refinement")
    parser.add_argument("--decimate", action="store_true", help="Enable mesh decimation in Stage 3 (reduces to --target_faces)")
    args = parser.parse_args()

    # ── Paths ───────────────────────────────────────────────────────────────
    project_dir  = os.path.join("output", args.output_name)
    inter_dir    = os.path.join(project_dir, "intermediate")
    os.makedirs(project_dir, exist_ok=True)
    os.makedirs(inter_dir, exist_ok=True)

    base_name      = args.output_name
    ref_image_path = os.path.join(inter_dir, f"{base_name}_reference.png")
    raw_mesh_obj   = os.path.join(inter_dir, f"{base_name}_raw.obj")
    raw_mesh_glb   = os.path.join(inter_dir, f"{base_name}_raw.glb")
    refined_path   = os.path.join(inter_dir, f"{base_name}_refined.glb")
    stage3_body    = os.path.join(inter_dir, "body.glb")
    stage3_props   = os.path.join(inter_dir, "props.glb")
    joints_json    = os.path.join(inter_dir, "joints.json")
    rigged_fbx     = os.path.join(inter_dir, "rigged_body.fbx")
    motion_dir     = os.path.join(inter_dir, "motions")
    final_fbx      = os.path.join(project_dir, f"{base_name}_final.fbx")
    final_glb      = os.path.join(project_dir, f"{base_name}_final.glb")

    stages = set(args.stage)

    # ═══════════════════════════════════════════════════════════════════════
    # Stage 1: Vision Prior (FLUX) -> flux_env
    # ═══════════════════════════════════════════════════════════════════════
    if 1 in stages:
        print("\n" + "=" * 60)
        print("  Stage 1: Vision Prior")
        print("=" * 60)
        run_stage("flux_env", "src/stage1_vision_prior.py", ["--prompt", args.prompt, "--output", ref_image_path])
    else:
        print("\n--- Skipping Stage 1 ---")

    # ═══════════════════════════════════════════════════════════════════════
    # Stage 2: 3D Generation (TRELLIS.2) -> meshgen
    # ═══════════════════════════════════════════════════════════════════════
    if 2 in stages:
        print("\n" + "=" * 60)
        print("  Stage 2: 3D Generation")
        print("=" * 60)
        run_stage("meshgen", "src/stage2_trellis_wrapper.py", ["--image", ref_image_path, "--output", raw_mesh_obj])
    else:
        print("\n--- Skipping Stage 2 ---")

    # ═══════════════════════════════════════════════════════════════════════
    # Stage 3: Mesh Optimization -> stage34_env
    # ═══════════════════════════════════════════════════════════════════════
    if 3 in stages:
        print("\n" + "=" * 60)
        print("  Stage 3: Mesh Optimization")
        print("=" * 60)
        stage3_args = [
            "--input", raw_mesh_obj, "--output", refined_path,
            "--target_faces", str(args.target_faces)
        ]
        if args.decimate:
            stage3_args.append("--decimate")
        run_stage("stage34_env", "src/stage3_mesh_optimization.py", stage3_args)
    else:
        print("\n--- Skipping Stage 3 ---")

    # ═══════════════════════════════════════════════════════════════════════
    # Stage 4: Character Rigging & Animation
    # ═══════════════════════════════════════════════════════════════════════
    if 4 in stages:
        print("\n" + "=" * 60)
        print("  Stage 4: Rigging, Animation & Final Assembly")
        print("=" * 60)

        # ── Step 4a: Skip Segmentation (Alternative 1: Rig-Aware Rigid Weighting) ────────
        # We no longer use PartSAM/SAMPart3D for segmentation. 
        # Instead, we rig the combined mesh and handle props via bone weighting in Stage 4d.
        print("\n--- Step 4a: Skipping Segmentation (using Rig-Aware Weighting instead) ---")
        body_obj = refined_path
        props_obj = None

        # ── Step 4b: Rigging -> unirig ───────────────────────────────────────
        print("\n--- Step 4b: Topology-Agnostic Rigging ---")
        try:
            run_stage("unirig", "src/stage4_topology_agnostic_rig.py", [
                "--input", body_obj, "--output_dir", inter_dir
            ])
        except subprocess.CalledProcessError as e:
            print(f"  Warning: Rigging step exited with error ({e.returncode}). Continuing if joints.json exists.")

        if not os.path.exists(joints_json):
            print(f"  Rigging did not produce {joints_json}. Skipping motion synthesis and assembly.")
        else:
            # ── Step 4c: Motion Synthesis (procedural, no external dependency) ──
            print("\n--- Step 4c: Motion Synthesis ---")
            try:
                run_stage("motion", "src/stage4_motion_synthesis.py", [
                    "--rig_file", joints_json, "--output_dir", motion_dir, "--prompt", args.prompt, "--props", props_obj if props_obj else ""
                ])
            except subprocess.CalledProcessError as e:
                print(f"  Warning: Motion synthesis failed ({e.returncode}). Continuing to assembly.")

            # ── Step 4d: Assembly -> riganything ─────────────────────────────
            if not os.path.exists(rigged_fbx):
                print(f"  Warning: {rigged_fbx} not found. Skipping assembly.")
            else:
                print("\n--- Step 4d: STaR Motion Retargeting & Assembly ---")
                # When decimating, use the decimated refined mesh as the body/colour source
                # so the final display mesh has the reduced geometry, not the full-res raw.
                # Without decimation, prefer the raw TRELLIS GLB for richer colour data.
                if args.decimate:
                    color_src = refined_path
                    print(f"  Using decimated refined mesh for color: {color_src}")
                else:
                    color_src = raw_mesh_glb if os.path.exists(raw_mesh_glb) else refined_path
                    print(f"  Using raw TRELLIS GLB for color: {color_src}")
                run_stage("riganything", "src/stage4_assemble_final.py", [
                    "--rigged_body", rigged_fbx,
                    "--motion_dir", motion_dir,
                    "--output_fbx", final_fbx,
                    "--output_glb", final_glb,
                    "--joints_json", joints_json,
                    "--color_source", color_src,
                    "--props", props_obj if props_obj and os.path.exists(props_obj) else ""
                ])
    else:
        print("\n--- Skipping Stage 4 ---")

    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print("=" * 60)
    print(f"  Final FBX : {final_fbx}")
    print(f"  Final GLB : {final_glb}")

    # ═══════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print("=" * 60)
    print(f"  Project directory : {project_dir}")
    print(f"  Intermediate files: {inter_dir}")
    if 4 in stages:
        if os.path.exists(final_fbx):
            print(f"  Final FBX : {final_fbx}")
        if os.path.exists(final_glb):
            print(f"  Final GLB : {final_glb}")
    print()


if __name__ == "__main__":
    main()
