"""
Puppeteer Runner
================
Orchestrates the Puppeteer 3-stage rigging pipeline for a single OBJ mesh:

  Stage 1 — Skeleton generation  (skeleton/demo.py via SkeletonGPT)
  Stage 2 — Skinning weights      (skinning/main.py via torchrun)
  Stage 3 — FBX export            (export.py via Blender)

Inputs / outputs
----------------
  --input         OBJ mesh path (required; must be .obj — skinning only reads .obj)
  --output        FBX output path
  --joints        joints.json output path
  --puppeteer-dir Path to the cloned Puppeteer repo (external/Puppeteer)

Checkpoint auto-discovery
--------------------------
Skeleton:  <puppeteer-dir>/skeleton_ckpts/puppeteer_skeleton_w_diverse_pose.pth
           (downloaded by setup_puppeteer.sh into the repo root)
Skinning:  <puppeteer-dir>/skinning/skinning/skinning_ckpts/puppeteer_skin_w_diverse_pose_depth1.pth
           (embedded inside the skinning Python package tree)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def run_cmd(
    cmd: list[str],
    cwd: str | None = None,
    env: dict | None = None,
    timeout: int = 900,
) -> None:
    print(f"[puppeteer_runner] cwd={cwd or '.'}: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout
    )
    if result.stdout:
        print(result.stdout[-4000:])
    if result.returncode != 0:
        print(result.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(
            f"Command failed (code {result.returncode}): {' '.join(cmd)}"
        )


def _find_checkpoint(puppeteer_dir: Path, *relative_candidates: str) -> Path:
    """Return the first candidate path that exists, or raise."""
    for rel in relative_candidates:
        p = puppeteer_dir / rel
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No checkpoint found in {puppeteer_dir}. Tried:\n"
        + "\n".join(f"  {c}" for c in relative_candidates)
        + "\nRun scripts/setup_puppeteer.sh to download checkpoints."
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",         required=True,  help="Input OBJ mesh path")
    parser.add_argument("--output",        required=True,  help="Output FBX path")
    parser.add_argument("--output-glb",    required=False, default="", help="Also export GLB directly from Blender (bypasses FBX round-trip)")
    parser.add_argument("--joints",        required=True,  help="Output joints.json path")
    parser.add_argument("--puppeteer-dir", required=True,  help="Path to Puppeteer repo root")
    args = parser.parse_args()

    puppeteer_dir = Path(args.puppeteer_dir).resolve()
    input_path    = Path(args.input).resolve()
    output_fbx    = Path(args.output).resolve()
    output_glb    = Path(args.output_glb).resolve() if args.output_glb else None
    joints_path   = Path(args.joints).resolve()

    if input_path.suffix.lower() != ".obj":
        raise ValueError(
            f"Puppeteer skinning only accepts .obj files; got: {input_path}. "
            "Ensure Stage 3 has written a refined .obj before calling Stage 4."
        )

    # Python / torchrun from the venv that invoked this script
    python_bin   = sys.executable
    torchrun_bin = str(Path(python_bin).parent / "torchrun")
    if not Path(torchrun_bin).exists():
        raise RuntimeError(
            f"torchrun not found at {torchrun_bin}. "
            "Activate the .venv_puppeteer or run setup_puppeteer.sh."
        )

    # ── Checkpoint discovery ───────────────────────────────────────────────────
    # Skeleton: downloaded by setup_puppeteer.sh into <puppeteer-dir>/skeleton_ckpts/
    skel_ckpt = _find_checkpoint(
        puppeteer_dir,
        "skeleton_ckpts/puppeteer_skeleton_w_diverse_pose.pth",
        # Also check the embedded copy inside the skeleton package tree
        "skeleton/skeleton/skeleton_ckpts/puppeteer_skeleton_w_diverse_pose.pth",
    )

    # Skinning: lives inside the skinning Python package tree
    skin_ckpt = _find_checkpoint(
        puppeteer_dir,
        "skinning/skinning/skinning_ckpts/puppeteer_skin_w_diverse_pose_depth1.pth",
        # Fallback: downloaded to repo root by setup_puppeteer.sh
        "skinning_ckpts/puppeteer_skin_w_diverse_pose_depth1.pth",
    )

    print(f"[puppeteer_runner] Skeleton ckpt : {skel_ckpt}")
    print(f"[puppeteer_runner] Skinning ckpt : {skin_ckpt}")

    # ── Working directory ──────────────────────────────────────────────────────
    tmp_dir = Path("tmp_puppeteer").resolve()
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    # Copy input mesh + any sidecar files (MTL, textures) into examples folder.
    # The textured FBX export script (puppeteer_blend_export.py) uses Blender's
    # OBJ importer which resolves mtllib / map_Kd relative to the OBJ file, so
    # all referenced files must sit alongside the OBJ copy.
    examples_dir = tmp_dir / "examples"
    examples_dir.mkdir()
    shutil.copy(input_path, examples_dir / input_path.name)
    # Copy .mtl file(s) and texture images from the OBJ's directory
    obj_dir = input_path.parent
    for sidecar in obj_dir.iterdir():
        if sidecar.suffix.lower() in (".mtl", ".png", ".jpg", ".jpeg") and sidecar != input_path:
            shutil.copy(sidecar, examples_dir / sidecar.name)

    base_name = input_path.stem  # e.g. "shaman_refined"

    # ── Base env ───────────────────────────────────────────────────────────────
    # Build a clean environment stripped of any parent-venv contamination.
    # This runner is invoked by .venv_puppeteer/bin/python but the parent process
    # (stage4_auto_rig.py) runs inside .venv, so VIRTUAL_ENV / PYTHONHOME / stray
    # PYTHONPATH values from .venv would otherwise pollute the Puppeteer workers.
    env = os.environ.copy()
    # Remove virtual-env indicators from the project's main .venv
    for _key in ("PYTHONHOME", "VIRTUAL_ENV", "VIRTUAL_ENV_PROMPT", "PYTHONPATH"):
        env.pop(_key, None)
    # Anchor the Python executable to the venv that launched this runner so that
    # torchrun's worker processes inherit the correct interpreter path.
    env["PYTHONEXECUTABLE"] = python_bin  # some launchers honour this
    # We'll set PYTHONPATH per stage below.

    # ── Stage 1: Skeleton generation ──────────────────────────────────────────
    print("[puppeteer_runner] Stage 1: Skeleton generation …")

    skel_dir        = puppeteer_dir / "skeleton"
    skel_third_par  = skel_dir / "third_partys"
    skel_michelangelo = skel_third_par / "Michelangelo"
    # Ensure a 'third_party' symlink exists (Puppeteer has an inconsistent naming)
    tp_alt = skel_dir / "third_party"
    if not tp_alt.exists() and skel_third_par.exists():
        try:
            os.symlink("third_partys", tp_alt)
        except OSError:
            pass  # race condition / already exists

    skel_output_root = tmp_dir / "skel_results"
    save_name        = "skel_run"

    skel_env = env.copy()
    skel_env["PYTHONPATH"] = ":".join([
        str(skel_dir),
        str(skel_third_par),
        str(skel_michelangelo),
    ])

    run_cmd(
        [
            python_bin, "demo.py",
            "--input_path",        str(examples_dir / input_path.name),
            "--pretrained_weights", str(skel_ckpt),
            "--output_dir",         str(skel_output_root),
            "--save_name",          save_name,
            "--input_pc_num",       "8192",
            "--joint_token",
            "--seq_shuffle",
            # Marching Cubes makes the surface smooth and watertight before point
            # cloud sampling.  Without it, blocky/semi-voxel style meshes produce
            # very sparse skeletons (2-4 joints vs 20+ with MC on).
            "--apply_marching_cubes",
        ],
        cwd=str(skel_dir),
        env=skel_env,
    )

    # Skeleton output: <output_dir>/<save_name>/<base_name>_pred.txt
    pred_txt = skel_output_root / save_name / f"{base_name}_pred.txt"
    if not pred_txt.exists():
        # Some Puppeteer versions omit the _pred suffix
        pred_txt = skel_output_root / save_name / f"{base_name}.txt"
    if not pred_txt.exists():
        raise RuntimeError(
            f"Stage 1 failed: skeleton prediction not found.\n"
            f"Expected: {skel_output_root / save_name / f'{base_name}_pred.txt'}"
        )

    # Stage 2 expects: <skeletons_dir>/<base_name>.txt  (no _pred suffix)
    skeletons_dir = tmp_dir / "skeletons"
    skeletons_dir.mkdir()
    shutil.copy(pred_txt, skeletons_dir / f"{base_name}.txt")

    # ── Stage 2: Skinning weights ──────────────────────────────────────────────
    print("[puppeteer_runner] Stage 2: Skinning weights …")

    skin_dir       = puppeteer_dir / "skinning"
    skin_third_par = skin_dir / "third_partys"
    tp_alt_skin    = skin_dir / "third_party"
    if not tp_alt_skin.exists() and skin_third_par.exists():
        try:
            os.symlink("third_partys", tp_alt_skin)
        except OSError:
            pass

    # skinning_models/models.py imports third_partys.Michelangelo, but Michelangelo
    # only lives under skeleton/third_partys/.  Ensure a symlink exists so the
    # import resolves correctly without modifying Puppeteer source.
    mich_link = skin_third_par / "Michelangelo"
    if not mich_link.exists():
        try:
            os.symlink(str(skel_michelangelo), str(mich_link))
        except OSError:
            pass

    skin_save_dir = tmp_dir / "skin_results"
    skin_env      = env.copy()
    skin_env["PYTHONPATH"] = ":".join([
        str(skin_dir),
        str(skin_third_par),
    ])

    run_cmd(
        [
            torchrun_bin,
            "--nproc_per_node=1",
            "--master_port=10009",
            "main.py",
            "--num_workers",        "1",
            "--batch_size",         "1",
            "--generate",
            "--pretrained_weights", str(skin_ckpt),
            "--input_skel_folder",  str(skeletons_dir),
            "--mesh_folder",        str(examples_dir),
            "--post_filter",
            "--depth",              "1",
            "--save_folder",        str(skin_save_dir),
        ],
        cwd=str(skin_dir),
        env=skin_env,
    )

    # Skinning output: <save_folder>/generate/<base_name>_skin.txt
    skin_txt = skin_save_dir / "generate" / f"{base_name}_skin.txt"
    if not skin_txt.exists():
        raise RuntimeError(
            f"Stage 2 failed: skinning result not found.\n"
            f"Expected: {skin_txt}"
        )

    # ── Stage 3: FBX export via Blender ───────────────────────────────────────
    # Use our textured export script instead of Puppeteer's export.py.
    # Upstream export.py calls from_pydata() which strips UV maps and materials,
    # producing a grey FBX.  puppeteer_blend_export.py uses Blender's own OBJ
    # importer so the Stage 3 texture atlas survives into the FBX/GLB.
    print("[puppeteer_runner] Stage 3: FBX export via Blender (textured) …")

    _scripts_dir = Path(__file__).parent
    blend_cmd = [
        "blender", "--background", "--python",
        str(_scripts_dir / "puppeteer_blend_export.py"),
        "--",
        "--mesh",   str(examples_dir / input_path.name),
        "--rig",    str(skin_txt),
        "--output", str(output_fbx),
    ]
    if output_glb:
        blend_cmd += ["--output-glb", str(output_glb)]

    run_cmd(blend_cmd, cwd=str(puppeteer_dir))

    if not output_fbx.exists():
        raise RuntimeError(f"Stage 3 failed: FBX not produced at {output_fbx}")

    # ── Parse rig → joints.json ────────────────────────────────────────────────
    print("[puppeteer_runner] Building joints.json …")
    joints: list[dict] = []
    with open(skin_txt) as f:
        lines = f.readlines()

    for line in lines:
        parts = line.split()
        if parts and parts[0] == "joints":
            joints.append({
                "name":     parts[1],
                "position": [float(parts[2]), float(parts[3]), float(parts[4])],
                "parent":   None,
            })

    # Second pass: fill parents from hierarchy lines
    joint_by_name = {j["name"]: j for j in joints}
    for line in lines:
        parts = line.split()
        if parts and parts[0] == "hier":
            parent_name, child_name = parts[1], parts[2]
            if child_name in joint_by_name:
                joint_by_name[child_name]["parent"] = parent_name

    joints_path.write_text(json.dumps(joints, indent=2))

    # Cleanup temp directory
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(
        f"[puppeteer_runner] Done.\n"
        f"  FBX:    {output_fbx}\n"
        f"  Joints: {joints_path} ({len(joints)} joints)"
    )


if __name__ == "__main__":
    main()
