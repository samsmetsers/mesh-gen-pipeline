"""
Mesh-Gen-Pipeline — Main Orchestrator
========================================
Runs all stages of the prompt-to-3D-rigged-character pipeline in sequence.

Stages:
  1. Prompt Parsing   — text → parsed JSON
  2. Text-to-3D       — parsed JSON  → raw OBJ/GLB
  3. Mesh Optimization— raw GLB → repaired & decimated GLB (PyMeshLab)
  4. Auto-Rigging     — GLB  → rigged FBX + joints.json (UniRig + P3-SAM)
  5. Animation        — Rigged GLB → animated GLB (MotionGPT3)

Usage:
  # Mock mode (no GPU, tests the full pipeline):
  uv run python main.py --prompt "..." --mock -n character

Output structure:
  output/<name>/
  ├── <name>_rigged.fbx         ← Stage 4 primary output
  ├── <name>_final.glb          ← Stage 4 secondary output
  ├── <name>_animated.glb       ← Stage 5 output
  └── intermediate/
      ├── parsed_prompt.json    ← Stage 1
      ├── <name>_raw.obj        ← Stage 2
      ├── <name>_raw.glb        ← Stage 2
      ├── <name>_refined.glb    ← Stage 3
      ├── joints.json           ← Stage 4
      ├── stage1_output.json
      ├── stage2_output.json
      ├── stage3_output.json
      ├── stage4_output.json
      └── stage5_output.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _load_stage_json(output_dir: str, name: str, stage: int) -> dict:
    """Load intermediate JSON for a given stage."""
    path = Path(output_dir) / name / "intermediate" / f"stage{stage}_output.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Stage {stage} output not found at {path}. "
            f"Run stages 1-{stage} first or use --resume-from {stage}."
        )
    return json.loads(path.read_text())


def run_pipeline(
    prompt: str,
    output_name: str,
    output_dir: str = "output",
    stages: list[int] | None = None,
    resume_from: int | None = None,
    quality: str = "standard",
    target_faces: int | None = None,
) -> dict:
    """
    Run the full mesh-gen pipeline.

    Returns a dict with all stage outputs.
    """
    if stages is None:
        stages = [1, 2, 3, 4, 5]

    # Adjust stages for resume
    if resume_from is not None:
        stages = [s for s in stages if s >= resume_from]

    results: dict = {}

    # ── Stage 1: Prompt Parsing ─────────────────────────────────────────────
    if 1 in stages:
        print("\n" + "=" * 60)
        print("STAGE 1: Prompt Parsing (text → JSON)")
        print("=" * 60)
        t0 = time.time()

        from src.stage1_prompt_parsing import parse_prompt
        s1 = parse_prompt(prompt=prompt)
        
        # Save JSON
        _save_json(output_dir, output_name, 1, s1.model_dump())
        results["stage1"] = s1.model_dump()
        print(f"[main] Stage 1 done in {time.time() - t0:.1f}s")
    elif resume_from and resume_from > 1:
        data = _load_stage_json(output_dir, output_name, 1)
        from src.stage1_prompt_parsing import ParsedPrompt
        results["stage1"] = ParsedPrompt(**data).model_dump()

    # ── Stage 2: Text-to-3D ─────────────────────────────────────────────────
    if 2 in stages:
        print("\n" + "=" * 60)
        print("STAGE 2: Text-to-3D (parsed prompt → 3D mesh)")
        print("=" * 60)
        t0 = time.time()

        from src.stage1_prompt_parsing import ParsedPrompt
        from src.stage2_text_to_3d import generate_3d_mesh
        s1_out = ParsedPrompt(**results["stage1"])
        s2 = generate_3d_mesh(
            parsed_prompt=s1_out,
            output_dir=output_dir,
            output_name=output_name,
        )
        _save_json(output_dir, output_name, 2, s2.model_dump())
        results["stage2"] = s2.model_dump()
        print(f"[main] Stage 2 done in {time.time() - t0:.1f}s")
    elif resume_from and resume_from > 2:
        data = _load_stage_json(output_dir, output_name, 2)
        from src.stage2_text_to_3d import Stage2Output
        results["stage2"] = Stage2Output(**data).model_dump()

    # ── Stage 3: Mesh Optimization ──────────────────────────────────────────
    if 3 in stages:
        print("\n" + "=" * 60)
        print("STAGE 3: Mesh Optimization & Decimation")
        print("=" * 60)
        t0 = time.time()

        from src.stage2_text_to_3d import Stage2Output
        from src.stage3_mesh_optimization import run_stage3
        s2_out = Stage2Output(**results["stage2"])
        s3 = run_stage3(
            stage2_output=s2_out,
            output_dir=output_dir,
            quality=quality,
            target_faces=target_faces,
        )
        _save_json(output_dir, output_name, 3, s3.model_dump())
        results["stage3"] = s3.model_dump()
        print(f"[main] Stage 3 done in {time.time() - t0:.1f}s → {s3.face_count} faces")
    elif resume_from and resume_from > 3:
        data = _load_stage_json(output_dir, output_name, 3)
        from src.stage3_mesh_optimization import Stage3Output
        results["stage3"] = Stage3Output(**data).model_dump()

    # ── Stage 4: Auto-Rigging ───────────────────────────────────────────────
    if 4 in stages:
        print("\n" + "=" * 60)
        print("STAGE 4: Auto-Rigging (UniRig + P3-SAM)")
        print("=" * 60)
        t0 = time.time()

        from src.stage3_mesh_optimization import Stage3Output
        from src.stage4_auto_rig import run_stage4
        s3_out = Stage3Output(**results["stage3"])
        s4 = run_stage4(
            stage3_output=s3_out,
            output_dir=output_dir,
        )
        _save_json(output_dir, output_name, 4, s4.model_dump())
        results["stage4"] = s4.model_dump()
        print(
            f"[main] Stage 4 done in {time.time() - t0:.1f}s "
            f"({s4.joint_count} joints, method={s4.rigging_method})"
        )
    elif resume_from and resume_from > 4:
        data = _load_stage_json(output_dir, output_name, 4)
        from src.stage4_auto_rig import Stage4Output
        results["stage4"] = Stage4Output(**data).model_dump()

    # ── Stage 5: Animation ──────────────────────────────────────────────────
    if 5 in stages:
        print("\n" + "=" * 60)
        print("STAGE 5: Animation (MotionGPT3)")
        print("=" * 60)
        t0 = time.time()

        from src.stage4_auto_rig import Stage4Output
        from src.stage5_animation import run_stage5
        s4_out = Stage4Output(**results["stage4"])
        s5 = run_stage5(
            stage4_output=s4_out,
            original_prompt=prompt,
            output_dir=output_dir,
        )
        _save_json(output_dir, output_name, 5, s5.model_dump())
        results["stage5"] = s5.model_dump()
        print(f"[main] Stage 5 done in {time.time() - t0:.1f}s")
    elif resume_from and resume_from > 5:
        data = _load_stage_json(output_dir, output_name, 5)
        from src.stage5_animation import Stage5Output
        results["stage5"] = Stage5Output(**data).model_dump()

    return results


def _save_json(output_dir: str, name: str, stage: int, data: dict) -> None:
    path = Path(output_dir) / name / "intermediate" / f"stage{stage}_output.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _print_summary(results: dict, output_dir: str, name: str) -> None:
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    if "stage1" in results:
        print(f"  Parsed prompt   : {results['stage1']}")
    if "stage2" in results:
        print(f"  Concept image   : {results['stage2'].get('concept_image_path', '')}")
        print(f"  Raw OBJ         : {results['stage2'].get('obj_path', '')}")
        print(f"  Raw GLB         : {results['stage2'].get('glb_path', '')}")
    if "stage3" in results:
        print(f"  Refined GLB     : {results['stage3']['refined_glb_path']}")
        print(f"  Refined OBJ     : {results['stage3']['refined_obj_path']}")
        print(f"  Face count      : {results['stage3']['face_count']}")
    if "stage4" in results:
        print(f"  Rigged FBX      : {results['stage4']['fbx_path']}")
        print(f"  Final GLB       : {results['stage4']['glb_path']}")
        print(f"  Joints JSON     : {results['stage4']['joints_path']}")
        print(f"  Rigging method  : {results['stage4']['rigging_method']}")
    if "stage5" in results:
        print(f"  Animated GLB    : {results['stage5']['animated_glb_path']}")
        print(f"  Animations JSON : {results['stage5']['animations_json_path']}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mesh-Gen-Pipeline: text prompt → rigged 3D game character",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--prompt", "-p", type=str, required=True,
                        help="Natural language character description")
    parser.add_argument("--output-name", "-n", type=str, default="character",
                        help="Short name for output files (default: character)")
    parser.add_argument("--output-dir", "-o", type=str, default="output",
                        help="Output root directory (default: output)")
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                        choices=[1, 2, 3, 4, 5],
                        help="Stages to run (default: 1 2 3 4 5)")
    parser.add_argument("--resume-from", type=int, default=None, choices=[2, 3, 4, 5],
                        help="Resume from this stage (loads prior stage outputs from disk)")
    parser.add_argument("--quality", "-q", type=str, default="standard",
                        choices=["mobile", "standard", "high"],
                        help="Mesh quality preset for Stage 3 (default: standard)")
    parser.add_argument("--faces", type=int, default=None,
                        help="Override target face count for Stage 3")
    args = parser.parse_args()

    print(f"\nMesh-Gen-Pipeline")
    print(f"  Prompt : {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    print(f"  Name   : {args.output_name}")
    print(f"  Stages : {args.stages}")

    results = run_pipeline(
        prompt=args.prompt,
        output_name=args.output_name,
        output_dir=args.output_dir,
        stages=args.stages,
        resume_from=args.resume_from,
        quality=args.quality,
        target_faces=args.faces,
    )

    _print_summary(results, args.output_dir, args.output_name)


if __name__ == "__main__":
    main()
