"""
Stage 5: Procedural Animation Generation
==========================================
Generates 3 animations for the rigged character, automatically inferred from the
original prompt:

  1. **idle**  — Subtle breathing + sway (all characters)
  2. **walk**  — Biped walk cycle with arm swing (movement for bipeds)
  3. **attack** — Context-aware attack motion inferred from prompt:
       - staff/wand/magical → "staff raise + summon" (shaman, mage class)
       - sword/axe/blade    → "overhead slash"
       - bow/arrow          → "draw and fire"
       - claws/unarmed      → "lunge + claw swipe"
       - default            → "forward strike"

Animation engine: **Blender 4.0.2** via subprocess.
  - Blender script (`scripts/blender_animate.py`) receives the rigged GLB + skeleton
    info and generates keyframed NLA action clips.
  - Output: animated GLB with 3 embedded animation tracks.

Blender call is: `blender --background --python scripts/blender_animate.py -- <args>`

Mock mode: generates a stub animated GLB (valid binary glTF) + animation manifest JSON.

Output:
  - <name>_animated.glb  — GLB with idle/walk/attack NLA clips
  - animations.json       — manifest: clip name, frame range, fps
  - stage5_output.json
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from src.stage4_auto_rig import Stage4Output


# ---------------------------------------------------------------------------
# Attack type inference
# ---------------------------------------------------------------------------

AttackType = Literal[
    "staff_summon",   # raise staff / summon magic — shaman, mage, warlock
    "sword_slash",    # overhead or diagonal sword/axe strike
    "bow_fire",       # draw bow, aim, fire
    "gun_fire",       # point and fire weapon
    "claw_swipe",     # unarmed lunge and claw swipe
    "generic_strike", # default forward strike
]

_STAFF_KEYWORDS = {
    "staff", "stave", "wand", "scepter", "sceptre", "orb", "tome", "spellbook",
    "shaman", "mage", "sorcerer", "sorceress", "wizard", "witch", "warlock",
    "magic", "magical", "spell", "summon", "glowing", "mushroom",
}
_SWORD_KEYWORDS = {
    "sword", "blade", "katana", "sabre", "saber", "axe", "hatchet",
    "scythe", "cleaver", "machete", "greatsword", "longsword", "dagger",
    "warrior", "knight", "paladin", "barbarian",
}
_BOW_KEYWORDS = {
    "bow", "arrow", "crossbow", "quiver", "ranger", "hunter", "archer",
}
_GUN_KEYWORDS = {
    "gun", "rifle", "pistol", "revolver", "shotgun", "soldier", "military",
    "marine", "sniper", "blaster", "automatic",
}
_CLAW_KEYWORDS = {
    "claw", "fist", "punch", "unarmed", "martial", "monk", "beast",
    "animal", "creature", "zombie", "undead",
}


def infer_attack_type(prompt: str) -> AttackType:
    """Infer the most appropriate attack animation from the character prompt."""
    words = set(prompt.lower().replace(",", " ").replace(".", " ").split())
    if words & _STAFF_KEYWORDS:
        return "staff_summon"
    if words & _SWORD_KEYWORDS:
        return "sword_slash"
    if words & _BOW_KEYWORDS:
        return "bow_fire"
    if words & _GUN_KEYWORDS:
        return "gun_fire"
    if words & _CLAW_KEYWORDS:
        return "claw_swipe"
    return "generic_strike"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class AnimationClip(BaseModel):
    name: str = Field(description="Clip name (idle, walk, attack).")
    attack_type: Optional[AttackType] = Field(
        default=None,
        description="Attack sub-type (only set for attack clip).",
    )
    frame_start: int
    frame_end: int
    fps: int = 24


class Stage5Output(BaseModel):
    animated_glb_path: str = Field(description="Path to animated GLB file.")
    animations_json_path: str = Field(description="Path to animations manifest JSON.")
    clips: list[AnimationClip] = Field(description="List of animation clips.")
    output_name: str = Field(description="Short identifier used for file naming.")


# ---------------------------------------------------------------------------
# Animation clip definitions
# ---------------------------------------------------------------------------

# Frame layout: idle 0-47, walk 48-95, attack 96-143 (all at 24fps = 2s each)
_CLIP_LAYOUT = [
    AnimationClip(name="idle",   frame_start=0,   frame_end=47,  fps=24),
    AnimationClip(name="walk",   frame_start=48,  frame_end=95,  fps=24),
    AnimationClip(name="attack", frame_start=96,  frame_end=143, fps=24),
]


# ---------------------------------------------------------------------------
# Blender subprocess call
# ---------------------------------------------------------------------------

def _run_blender_animation(
    rigged_glb: str,
    joints_json: str,
    output_glb: str,
    attack_type: AttackType,
    blender_script: str,
    blender_bin: str = "blender",
) -> None:
    """Invoke Blender in background mode to generate animations."""
    cmd = [
        blender_bin, "--background",
        "--python", blender_script,
        "--",
        "--input",       rigged_glb,
        "--joints",      joints_json,
        "--output",      output_glb,
        "--attack-type", attack_type,
    ]
    print(f"[Stage 5] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(
            f"Blender animation script failed (exit {result.returncode}):\n"
            f"STDOUT: {result.stdout[-2000:]}\n"
            f"STDERR: {result.stderr[-2000:]}"
        )
    print("[Stage 5] Blender animation complete.")


# ---------------------------------------------------------------------------
# Mock animation GLB (minimal but valid binary glTF with animation data)
# ---------------------------------------------------------------------------

def _write_mock_animated_glb(output_path: str) -> None:
    """Write a minimal valid GLB with a stub animation track."""
    import struct as _struct

    # Minimal glTF JSON with 1 animation (3 channels: idle/walk/attack as names)
    gltf_json = (
        '{"asset":{"version":"2.0","generator":"mesh-gen-pipeline-mock"},'
        '"scene":0,"scenes":[{"nodes":[0]}],'
        '"nodes":[{"mesh":0,"name":"character"}],'
        '"meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}],'
        '"animations":[{"name":"idle","channels":[],"samplers":[]},'
        '{"name":"walk","channels":[],"samplers":[]},'
        '{"name":"attack","channels":[],"samplers":[]}],'
        '"accessors":['
        '{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3",'
        '"min":[-0.5,-0.5,0.0],"max":[0.5,0.5,0.0]},'
        '{"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}],'
        '"bufferViews":['
        '{"buffer":0,"byteOffset":0,"byteLength":36},'
        '{"buffer":0,"byteOffset":36,"byteLength":6}],'
        '"buffers":[{"byteLength":44}]}'
    )
    json_bytes = gltf_json.encode("utf-8")
    padded_json = json_bytes + b" " * ((4 - len(json_bytes) % 4) % 4)

    vertices = _struct.pack("<9f", -0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.0, 0.5, 0.0)
    indices = _struct.pack("<3H", 0, 1, 2)
    bin_data = vertices + indices + b"\x00\x00"

    total_length = 12 + 8 + len(padded_json) + 8 + len(bin_data)
    with open(output_path, "wb") as f:
        f.write(_struct.pack("<III", 0x46546C67, 2, total_length))
        f.write(_struct.pack("<II", len(padded_json), 0x4E4F534A))
        f.write(padded_json)
        f.write(_struct.pack("<II", len(bin_data), 0x004E4942))
        f.write(bin_data)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_stage5(
    stage4_output: Stage4Output,
    original_prompt: str,
    output_dir: str = "output",
    blender_bin: str = "blender",
    blender_script: Optional[str] = None,
    mock: bool = False,
) -> Stage5Output:
    """
    Run Stage 5: generate idle / walk / attack animations for the rigged character.

    Args:
        stage4_output:    Output from Stage 4 (rigged GLB + joints.json).
        original_prompt:  The original text prompt (used to infer attack style).
        output_dir:       Root output directory.
        blender_bin:      Path to Blender executable (default: "blender" on PATH).
        blender_script:   Path to blender_animate.py (default: scripts/blender_animate.py).
        mock:             If True, write stub animated GLB without running Blender.

    Returns:
        Stage5Output with paths and animation clip manifest.
    """
    name = stage4_output.output_name
    out_path = Path(output_dir) / name / "intermediate"
    out_path.mkdir(parents=True, exist_ok=True)

    animated_glb = str(out_path / f"{name}_animated.glb")
    animations_json = str(out_path / "animations.json")

    attack_type = infer_attack_type(original_prompt)
    print(f"[Stage 5] Inferred attack type: {attack_type}")

    # Tag the attack clip with the inferred type
    clips = [
        _CLIP_LAYOUT[0],  # idle
        _CLIP_LAYOUT[1],  # walk
        AnimationClip(
            name="attack",
            attack_type=attack_type,
            frame_start=96,
            frame_end=143,
            fps=24,
        ),
    ]

    if mock:
        print(f"[Stage 5] Mock mode: writing stub animated GLB to {animated_glb}")
        _write_mock_animated_glb(animated_glb)
    else:
        if blender_script is None:
            blender_script = str(
                Path(__file__).parent.parent / "scripts" / "blender_animate.py"
            )
        _run_blender_animation(
            rigged_glb=stage4_output.glb_path,
            joints_json=stage4_output.joints_path,
            output_glb=animated_glb,
            attack_type=attack_type,
            blender_script=blender_script,
            blender_bin=blender_bin,
        )

    # Write animations manifest
    manifest = [c.model_dump(exclude_none=False) for c in clips]
    Path(animations_json).write_text(json.dumps(manifest, indent=2))
    print(f"[Stage 5] Animations manifest: {animations_json}")

    return Stage5Output(
        animated_glb_path=os.path.abspath(animated_glb),
        animations_json_path=os.path.abspath(animations_json),
        clips=clips,
        output_name=name,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 5: Animation Generation")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to stage4_output.json")
    parser.add_argument("--prompt", "-p", type=str, default="",
                        help="Original character prompt (for attack type inference)")
    parser.add_argument("--output-dir", "-o", type=str, default="output")
    parser.add_argument("--blender-bin", type=str, default="blender")
    parser.add_argument("--blender-script", type=str, default=None)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text())
    s4 = Stage4Output(**data)

    result = run_stage5(
        s4,
        original_prompt=args.prompt,
        output_dir=args.output_dir,
        blender_bin=args.blender_bin,
        blender_script=args.blender_script,
        mock=args.mock,
    )

    json_path = (
        Path(args.output_dir) / s4.output_name / "intermediate" / "stage5_output.json"
    )
    json_path.write_text(result.model_dump_json(indent=2))
    print(f"\n[Stage 5] Complete. Output JSON: {json_path}")
    print(result.model_dump_json(indent=2))
