"""
Stage 4c: Character-Aware Procedural Motion Synthesis

Generates character-class-specific animations purely procedurally — no HOI-Diff
dependency. The dummy mean/std approach produces garbage when real checkpoints
exist, so we rely entirely on hand-crafted generators that always produce
correct, sensible motion.

Animation routing by character type
────────────────────────────────────
  Any character      →  idle  +  move (walk / trot / fly)
  Biped mage/shaman  →  + attack: dramatic two-hand staff raise + burst
  Biped melee        →  + attack: one-hand sword slash (wind-up → slash → follow-through)
  Biped archer       →  + attack: bow draw → aim → release → reset
  Biped generic      →  + attack: upper-body lunge
  Quadruped          →  move = diagonal trot
  Flying             →  move = wing-flap locomotion
"""

import os
import argparse
import json
import numpy as np


# ─── Character Detection ───────────────────────────────────────────────────

def detect_creature_type(prompt: str) -> str:
    """Returns 'biped', 'quadruped', or 'flying'."""
    p = prompt.lower()

    flying_words = [
        # Specific birds (not "bird" alone — too generic, fires on "bird mask" etc.)
        "eagle", "hawk", "raven", "crow", "falcon", "owl",
        "dragon", "wyvern", "griffin", "bat", "phoenix", "angel",
        "winged", "flying", "flies",
    ]
    quad_words = [
        "horse", "deer", "wolf", "dog", "cat", "lion", "tiger", "bear",
        "fox", "cow", "bull", "beast", "creature", "monster", "dinosaur",
        "raptor", "steed", "pony", "goat", "sheep", "boar", "elk", "moose",
        "quadruped", "four-legged", "four legged",
    ]
    biped_words = [
        "human", "man", "woman", "person", "warrior", "mage", "shaman",
        "knight", "archer", "wizard", "fighter", "elf", "orc", "dwarf",
        "ranger", "rogue", "paladin", "sorcerer", "priest", "cleric",
        "monk", "druid", "necromancer", "hunter", "soldier", "guard",
        "barbarian", "samurai", "ninja", "pirate", "bandit", "mercenary",
        "warlock", "witch", "swordsman", "bard", "assassin",
        "doctor", "medic", "plague", "alchemist", "apothecary",
    ]

    for w in flying_words:
        if w in p:
            return "flying"
    for w in quad_words:
        if w in p:
            return "quadruped"
    for w in biped_words:
        if w in p:
            return "biped"
    return "biped"   # default — most game characters are bipeds


def detect_biped_class(prompt: str) -> str:
    """
    Returns one of: 'mage_staff', 'melee', 'archer', 'generic_biped'.
    Only meaningful when creature_type == 'biped'.
    """
    p = prompt.lower()

    staff_words = [
        "mage", "wizard", "shaman", "sorcerer", "witch", "druid",
        "warlock", "necromancer", "staff", "wand", "rod", "scepter",
        "cane", "spell", "magic", "priest", "cleric", "ritual",
    ]
    throw_words = [
        "vial", "potion", "elixir", "flask", "bottle", "phial",
        "alchemist", "apothecary", "plague", "doctor", "medic",
        "throw", "toss", "lob",
    ]
    archer_words = [
        "archer", "ranger", "hunter", "bow", "crossbow", "gun", "rifle",
        "sniper", "pistol", "shoot", "ranged",
    ]
    melee_words = [
        "warrior", "knight", "fighter", "soldier", "barbarian", "samurai",
        "sword", "axe", "blade", "dagger", "hammer", "mace", "spear",
        "melee", "slash", "berserker",
    ]

    # Order matters: staff > throw > archer > melee
    for w in staff_words:
        if w in p:
            return "mage_staff"
    for w in throw_words:
        if w in p:
            return "throw_potion"
    for w in archer_words:
        if w in p:
            return "archer"
    for w in melee_words:
        if w in p:
            return "melee"
    return "generic_biped"


# ─── Skeleton Classification ───────────────────────────────────────────────

def _classify_skeleton(joints: dict, hierarchy: dict, root: str):
    """
    Hierarchy-based skeleton classification.

    The naive position-relative-to-root approach breaks when the rig root sits
    at the very bottom of the skeleton (as Puppeteer places it at feet level):
    every joint is "above" the root → classified as "hand" → no foot joints at
    all → walk looks like arm-flapping.

    Instead we:
    1. Find leaf nodes (end-effectors).
    2. Classify leaves by BOUNDING-BOX position: highest+central → head,
       most lateral (normalised) → hand, low+less-lateral → foot.
    3. Walk the tree upward. A chain whose leaves all share one role inherits
       that role.  A branch whose subtree mixes roles → "spine".

    Returns (roles, sides) where
      roles[name] ∈ {'root','spine','head','hand','foot'}
      sides[name] ∈ {0=left/center, 1=right}
    """
    all_pos   = np.array(list(joints.values()))
    ranges    = all_pos.max(0) - all_pos.min(0)

    # Use root→children centroid direction to detect vertical axis.
    # Range-based detection fails for T-pose bipeds where lateral arm span ≈
    # body height — argmax(ranges) picks the arm-span axis as "vertical".
    # For Puppeteer skeletons the root sits at foot level; children are all
    # above it, so up_vec clearly points along the true vertical axis.
    root_pos = np.array(joints[root])
    children_of_root = [c for c in hierarchy.get(root, []) if c in joints]
    vert = None
    if children_of_root:
        child_centroid = np.mean([np.array(joints[c]) for c in children_of_root], axis=0)
        up_vec = child_centroid - root_pos
        up_abs = np.abs(up_vec)
        dominant = int(np.argmax(up_abs))
        second = float(np.sort(up_abs)[-2])
        if up_abs[dominant] > 0 and (second == 0 or up_abs[dominant] / second > 2.0):
            vert = dominant
    if vert is None:
        vert = int(np.argmax(ranges))  # fallback for heuristic skeletons

    depth = int(np.argmin(ranges))
    if depth == vert:
        other = [i for i in range(3) if i != vert]
        depth = other[0] if ranges[other[0]] < ranges[other[1]] else other[1]
    lat = 3 - vert - depth

    axis_names = {0: 'X', 1: 'Y', 2: 'Z'}
    print(f"  [classify_skeleton] axes: vert={axis_names[vert]}, depth={axis_names[depth]}, lat={axis_names[lat]}")

    vert_min   = float(all_pos[:, vert].min())
    vert_range = float(ranges[vert]) or 1.0

    lat_center = float((all_pos[:, lat].min() + all_pos[:, lat].max()) / 2)
    lat_half   = float((all_pos[:, lat].max() - all_pos[:, lat].min()) / 2) or 1.0

    # ── Find leaf nodes ────────────────────────────────────────────────────
    leaves: set = set()
    for name in joints:
        if not hierarchy.get(name):
            leaves.add(name)
    if not leaves:          # degenerate: treat all as leaves
        leaves = set(joints.keys())

    # ── Classify leaves ────────────────────────────────────────────────────
    def _classify_leaf(name):
        pos      = np.array(joints[name])
        vert_rel = (pos[vert] - vert_min) / vert_range   # 0=bottom … 1=top
        lat_val  = pos[lat] - lat_center
        lat_abs  = abs(lat_val) / lat_half                # 0=center … 1=edge

        if vert_rel > 0.75 and lat_abs < 0.40:
            return "head", 0
        elif lat_abs > 0.60:                              # clearly lateral → arm
            return "hand", (1 if lat_val > 0 else 0)
        elif vert_rel < 0.30:                             # clearly low → leg
            return "foot", (1 if lat_val > 0 else 0)
        else:
            return "hand", (1 if lat_val > 0 else 0)     # default: arm

    leaf_roles: dict = {}
    leaf_sides: dict = {}
    for n in leaves:
        leaf_roles[n], leaf_sides[n] = _classify_leaf(n)

    # ── Subtree-leaf cache ─────────────────────────────────────────────────
    _cache: dict = {}
    def _subtree_leaves(name) -> set:
        if name in _cache:
            return _cache[name]
        ch = hierarchy.get(name, [])
        result = {name} if name in leaves else set()
        for c in ch:
            result |= _subtree_leaves(c)
        _cache[name] = result
        return result

    # ── Assign roles ───────────────────────────────────────────────────────
    roles: dict = {}
    sides: dict = {}

    for name, pos in joints.items():
        pos     = np.array(pos)
        lat_val = pos[lat] - lat_center

        if name == root:
            roles[name] = "root";  sides[name] = 0;  continue

        if name in leaves:
            roles[name] = leaf_roles[name]
            sides[name] = leaf_sides[name]
            continue

        sub      = _subtree_leaves(name)
        role_set = {leaf_roles[l] for l in sub if l in leaf_roles}

        # Inherit side from subtree leaves (handles hips/shoulders that sit
        # at x≈0 but belong to a left or right limb chain).
        sub_sides = [leaf_sides[l] for l in sub if l in leaf_sides]
        sub_side  = 1 if sub_sides and sum(sub_sides) > len(sub_sides) / 2 else 0

        if not role_set:
            roles[name] = "spine";  sides[name] = 0
        elif len(role_set) == 1:
            roles[name] = list(role_set)[0]
            sides[name] = sub_side
        elif role_set <= {"hand", "head"}:
            # Upper-body branch = spine
            roles[name] = "spine";  sides[name] = 0
        elif "hand" in role_set and "foot" not in role_set:
            roles[name] = "hand";   sides[name] = sub_side
        elif "foot" in role_set and "hand" not in role_set:
            roles[name] = "foot";   sides[name] = sub_side
        else:
            roles[name] = "spine";  sides[name] = 0

    return roles, sides


# ─── Procedural Walk / Biped ──────────────────────────────────────────────

def _gen_walk(joint_names, roles, sides, fps=20, duration=2.0):
    """Standard biped walk cycle: legs stride, arms swing opposite.

    Leg amplitude is kept moderate (0.28 rad ≈ 16°) so the motion reads
    as a steady walk rather than an exaggerated march.
    """
    n = int(fps * duration)
    motion = np.zeros((n, len(joint_names), 3), dtype=np.float32)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)

    for i, name in enumerate(joint_names):
        role  = roles.get(name, "other")
        side  = sides.get(name, 0)
        phase = 0.0 if side == 0 else np.pi

        if role == "foot":
            motion[:, i, 0] = 0.28 * np.sin(t + phase)            # stride (was 0.55)
            motion[:, i, 1] = 0.06 * np.abs(np.sin(t + phase))    # lift (was 0.12)
        elif role == "hand":
            motion[:, i, 0] = 0.18 * np.sin(t + phase + np.pi)    # arm swing (was 0.30)
        elif role == "spine":
            motion[:, i, 0] = 0.02 * np.sin(t)                    # subtle sway (was 0.04)
        elif role == "head":
            motion[:, i, 0] = 0.01 * np.sin(t)
    return motion


def _gen_idle_biped(joint_names, roles, sides, fps=20, duration=3.0):
    """Idle: slow breathing sway on spine/head, subtle arm float."""
    n = int(fps * duration)
    motion = np.zeros((n, len(joint_names), 3), dtype=np.float32)
    t_b = np.linspace(0, 2 * np.pi * 0.5, n, endpoint=False)

    for i, name in enumerate(joint_names):
        role = roles.get(name, "other")
        side = sides.get(name, 0)

        if role == "spine":
            motion[:, i, 0] = 0.035 * np.sin(t_b)
            motion[:, i, 2] = 0.020 * np.sin(t_b * 1.3)
        elif role == "head":
            motion[:, i, 0] = 0.04 * np.sin(t_b)
            motion[:, i, 2] = 0.03 * np.sin(t_b * 0.7)
        elif role == "hand":
            ph = 0 if side == 0 else np.pi * 0.3
            motion[:, i, 0] = 0.06 * np.sin(t_b + ph)
    return motion


# ─── Attack: Mage / Shaman — Staff Raise ──────────────────────────────────

def _gen_attack_staff_raise(joint_names, roles, sides, fps=20, duration=3.5):
    """
    Shaman/mage staff raise — capped at 0.85 rad (≈49°) peak so arms
    never overshoot past vertical and end up behind the head.

      0.00–0.20  gather   — arms drift slightly down, body leans forward
      0.20–0.45  raise    — arms sweep upward to ~45° above horizontal
      0.45–0.65  channel  — arms held high with magical tremor
      0.65–0.80  burst    — brief extra push to peak (0.85 rad), then snap
      0.80–1.00  recoil   — arms settle back to rest pose

    Arm joints rotate on channel 0 (primary bend axis for Puppeteer FBX bones).
    Spine and head get sympathetic secondary motion.
    """
    n = int(fps * duration)
    motion = np.zeros((n, len(joint_names), 3), dtype=np.float32)
    t = np.linspace(0.0, 1.0, n, endpoint=False)

    def _env(t_val):
        """Piecewise envelope — peak is 0.85 (never exceeds ~49°)."""
        if   t_val < 0.20: return t_val / 0.20 * (-0.07)           # gather: slight drop
        elif t_val < 0.45: return -0.07 + (t_val - 0.20) / 0.25 * 0.82  # raise to 0.75
        elif t_val < 0.65: return 0.75                              # channel plateau
        elif t_val < 0.80: return 0.75 + (t_val - 0.65) / 0.15 * 0.10   # burst to 0.85
        else:              return 0.85 - (t_val - 0.80) / 0.20 * 0.85    # recoil to 0

    env = np.vectorize(_env)(t)

    # Tremor only during channel phase (0.45–0.65)
    ch_mask = np.clip((t - 0.45) / 0.08, 0, 1) * np.clip((0.65 - t) / 0.08, 0, 1)
    tremor  = 0.06 * np.sin(t * 30 * np.pi) * ch_mask

    for i, name in enumerate(joint_names):
        role = roles.get(name, "other")
        side = sides.get(name, 0)

        if role == "hand":
            # Channel 0: primary arm raise (rotation around local-X of arm bone)
            motion[:, i, 0] = (env + tremor) * 0.95
            # Channel 2: very slight outward spread during raise, converge at burst
            lat_sign = 1 if side == 1 else -1
            spread = env * 0.12 * np.clip(1.0 - (t - 0.65) / 0.15, 0, 1)
            motion[:, i, 2] = lat_sign * spread

        elif role == "spine":
            # Forward lean grows with raise, releases at burst
            motion[:, i, 0] = env * 0.08 * np.clip(1.0 - (t - 0.65) / 0.35, 0, 1)

        elif role == "head":
            # Head tilts back slightly as arms go up (looking toward the sky)
            motion[:, i, 0] = -env * 0.12 + tremor * 0.2

    return motion


# ─── Attack: Melee — Sword Slash ──────────────────────────────────────────

def _gen_attack_slash(joint_names, roles, sides, fps=20, duration=2.5):
    """
    One-hand sword slash (dominant = right hand):
      0.00–0.25  wind-up  — sword arm pulls back/up, body rotates away
      0.25–0.50  slash    — explosive swing forward-across (fastest phase)
      0.50–0.70  follow   — arm extends past body
      0.70–1.00  reset    — return to guard stance
    """
    n = int(fps * duration)
    motion = np.zeros((n, len(joint_names), 3), dtype=np.float32)
    t = np.linspace(0.0, 1.0, n, endpoint=False)

    # Smoothstep helper
    def _ss(a, b, x):
        v = np.clip((x - a) / max(b - a, 1e-6), 0, 1)
        return v * v * (3 - 2 * v)

    windup  =  _ss(0.00, 0.25, t)
    slash   =  _ss(0.25, 0.50, t)
    follow  =  _ss(0.50, 0.70, t)
    reset_v =  _ss(0.70, 1.00, t)

    # Sword arm: raises during wind-up, snaps down-forward during slash
    sword_raise  = windup * 0.80 - slash * 1.40 + follow * 0.20 + reset_v * (-0.20 * (1 - reset_v))
    sword_twist  = windup * (-0.30) + slash * 0.80 - follow * 0.20

    # Off-hand (left): slight counter-balance lift
    offhand_raise = slash * 0.25 - follow * 0.15

    for i, name in enumerate(joint_names):
        role = roles.get(name, "other")
        side = sides.get(name, 0)   # 0=left, 1=right

        if role == "hand":
            if side == 1:  # right = sword arm
                motion[:, i, 0] = sword_raise
                motion[:, i, 2] = sword_twist
            else:          # left = off-hand balance
                motion[:, i, 0] = offhand_raise

        elif role == "spine":
            # Body rotates into the slash, then snaps back
            motion[:, i, 2] = windup * (-0.25) + slash * 0.35 - reset_v * 0.10

        elif role == "head":
            # Track the target
            motion[:, i, 0] = -0.10 * slash

    return motion


# ─── Attack: Archer — Bow Draw & Release ──────────────────────────────────

def _gen_attack_bow_shoot(joint_names, roles, sides, fps=20, duration=3.0):
    """
    Bow aim → draw → hold → release → reset:
      0.00–0.20  raise     — both arms lift bow to shoulder height
      0.20–0.40  draw      — draw arm (right) pulls string back to cheek
      0.40–0.55  hold/aim  — steady; very slight breathing sway
      0.55–0.65  release   — draw arm snaps forward (recoil)
      0.65–1.00  reset     — arms lower back to rest
    """
    n = int(fps * duration)
    motion = np.zeros((n, len(joint_names), 3), dtype=np.float32)
    t = np.linspace(0.0, 1.0, n, endpoint=False)

    def _ss(a, b, x):
        v = np.clip((x - a) / max(b - a, 1e-6), 0, 1)
        return v * v * (3 - 2 * v)

    raise_v   = _ss(0.00, 0.20, t)
    draw_v    = _ss(0.20, 0.40, t)
    hold_v    = _ss(0.40, 0.55, t)   # plateau
    release_v = _ss(0.55, 0.65, t)
    reset_v   = _ss(0.65, 1.00, t)

    aim_height = raise_v * 0.60   # shoulder height raise
    aim_sway   = 0.015 * np.sin(t * 12 * np.pi) * hold_v  # breathing during aim

    for i, name in enumerate(joint_names):
        role = roles.get(name, "other")
        side = sides.get(name, 0)

        if role == "hand":
            # Both arms rise during raise phase
            base_raise = aim_height - reset_v * aim_height

            if side == 1:  # right = draw arm
                # Pulls back during draw, snaps forward at release
                draw_back = draw_v * 0.55 - release_v * 0.45
                motion[:, i, 0] = base_raise + aim_sway
                motion[:, i, 1] = draw_back   # backward pull
            else:          # left = bow arm (extends forward)
                motion[:, i, 0] = base_raise + aim_sway
                motion[:, i, 1] = -draw_v * 0.20   # slight forward extension

        elif role == "spine":
            # Slight forward lean into aim
            motion[:, i, 0] = aim_height * 0.12 - reset_v * 0.12

        elif role == "head":
            # Look along the arrow
            motion[:, i, 0] = -aim_height * 0.15

    return motion


# ─── Attack: Potion Throw ─────────────────────────────────────────────────

def _gen_attack_throw_potion(joint_names, roles, sides, fps=20, duration=3.0):
    """
    Plague-doctor / alchemist elixir throw:
      0.00–0.15  reach    — dominant hand draws vial from belt (slight forward dip)
      0.15–0.40  wind-up  — arm sweeps back and up, body rotates away
      0.40–0.60  throw    — explosive forward arc + wrist snap (fastest phase)
      0.60–0.75  release  — arm snaps through, shoulder follows
      0.75–1.00  reset    — arm settles back; body unwinds

    Dominant = right hand (side == 1).  Off-hand gives counterbalance.
    Spine rotates into the throw for power.  Head tracks the target.
    """
    n = int(fps * duration)
    motion = np.zeros((n, len(joint_names), 3), dtype=np.float32)
    t = np.linspace(0.0, 1.0, n, endpoint=False)

    def _ss(a, b, x):
        v = np.clip((x - a) / max(b - a, 1e-6), 0, 1)
        return v * v * (3 - 2 * v)

    reach   = _ss(0.00, 0.15, t)
    windup  = _ss(0.15, 0.40, t)
    throw_v = _ss(0.40, 0.60, t)
    release = _ss(0.60, 0.75, t)
    reset_v = _ss(0.75, 1.00, t)

    # Throw arm trajectory: dip → back/up → snap forward → follow-through
    throw_arm_ch0 = (reach * (-0.15)           # reach down to belt
                   + windup * 0.70             # sweep up and back
                   - throw_v * 1.20            # explosive snap forward
                   + release * 0.20            # follow-through extension
                   - reset_v * (-0.15))        # return
    throw_arm_ch1 = (windup * (-0.35)          # pull back behind body
                   + throw_v * 0.55            # snap forward during release
                   - release * 0.20)

    # Off-hand counterbalance
    off_arm_ch0 = windup * 0.20 - throw_v * 0.15

    for i, name in enumerate(joint_names):
        role = roles.get(name, "other")
        side = sides.get(name, 0)   # 0=left, 1=right

        if role == "hand":
            if side == 1:   # right = throw arm
                motion[:, i, 0] = throw_arm_ch0
                motion[:, i, 1] = throw_arm_ch1
            else:           # left = counterbalance
                motion[:, i, 0] = off_arm_ch0

        elif role == "spine":
            # Body winds up away, then snaps through
            motion[:, i, 2] = windup * (-0.30) + throw_v * 0.40 - reset_v * 0.10

        elif role == "head":
            # Look toward target throughout
            motion[:, i, 0] = -windup * 0.08 + throw_v * (-0.05)

    return motion


# ─── Attack: Generic Biped ─────────────────────────────────────────────────

def _gen_attack_generic(joint_names, roles, sides, fps=20, duration=2.0):
    """Generic upper-body lunge: lean forward, punch/strike with dominant arm."""
    n = int(fps * duration)
    motion = np.zeros((n, len(joint_names), 3), dtype=np.float32)
    t = np.linspace(0.0, 1.0, n, endpoint=False)

    def _ss(a, b, x):
        v = np.clip((x - a) / max(b - a, 1e-6), 0, 1)
        return v * v * (3 - 2 * v)

    lunge  = _ss(0.0, 0.4, t) - _ss(0.6, 1.0, t)
    strike = _ss(0.2, 0.5, t)

    for i, name in enumerate(joint_names):
        role = roles.get(name, "other")
        side = sides.get(name, 0)

        if role == "hand":
            if side == 1:
                motion[:, i, 0] = strike * 0.60   # right arm punch up/forward
                motion[:, i, 1] = strike * 0.40
            else:
                motion[:, i, 0] = strike * 0.25

        elif role == "spine":
            motion[:, i, 0] = lunge * 0.18

        elif role == "head":
            motion[:, i, 0] = -lunge * 0.10

    return motion


# ─── Locomotion: Quadruped Trot ───────────────────────────────────────────

def _gen_trot(joint_names, roles, sides, fps=20, duration=2.0):
    """
    Diagonal trot: front-left + back-right move together (phase 0),
    front-right + back-left move together (phase π).

    'foot' joints are split into front/back by vertical position relative to
    the skeleton's vertical midpoint. If we can't tell, we treat them all as
    alternating legs.
    """
    n = int(fps * duration)
    motion = np.zeros((n, len(joint_names), 3), dtype=np.float32)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # Use index order as proxy for front/back (earlier indices = front in most rigs)
    foot_indices = [i for i, nm in enumerate(joint_names) if roles.get(nm) == "foot"]

    for k, i in enumerate(foot_indices):
        name = joint_names[i]
        side = sides.get(name, k % 2)   # 0=left, 1=right

        # Diagonal pairing: front-left(k=0) phase=0, back-right(k=1) phase=0,
        #                   front-right(k=2) phase=π, back-left(k=3) phase=π
        phase = 0.0 if (k % 2 == 0) else np.pi

        motion[:, i, 0] = 0.50 * np.sin(t + phase)              # stride (fore-aft)
        motion[:, i, 1] = 0.10 * np.abs(np.sin(t + phase))      # lift
        lat_sign = 1 if side == 1 else -1
        motion[:, i, 2] = lat_sign * 0.03 * np.sin(t + phase)   # tiny splay

    # Body bob and head nod
    for i, name in enumerate(joint_names):
        role = roles.get(name, "other")
        if role == "spine":
            motion[:, i, 1] = 0.06 * np.abs(np.sin(t * 2))   # vertical bounce
        elif role == "head":
            motion[:, i, 0] = 0.05 * np.sin(t * 2)            # nod

    return motion


def _gen_idle_quad(joint_names, roles, sides, fps=20, duration=3.0):
    """Quadruped idle: breathing body sway, occasional head movement."""
    n = int(fps * duration)
    motion = np.zeros((n, len(joint_names), 3), dtype=np.float32)
    t = np.linspace(0, 2 * np.pi * 0.4, n, endpoint=False)

    for i, name in enumerate(joint_names):
        role = roles.get(name, "other")
        if role == "spine":
            motion[:, i, 1] = 0.025 * np.sin(t)    # breathing lift
        elif role == "head":
            motion[:, i, 0] = 0.06 * np.sin(t * 0.6)
            motion[:, i, 2] = 0.03 * np.sin(t * 0.4)

    return motion


# ─── Locomotion: Flying — Wing Flap ───────────────────────────────────────

def _gen_fly(joint_names, roles, sides, fps=20, duration=2.0):
    """
    Wing-flap locomotion: 'hand' joints are wings.
    Down-stroke is fast (power), up-stroke is slow (recovery).
    """
    n = int(fps * duration)
    motion = np.zeros((n, len(joint_names), 3), dtype=np.float32)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # Asymmetric flap: fast down, slow up
    # Use sawtooth-like wave: cos with bias
    flap = -(np.cos(t) - 0.3 * np.cos(2 * t)) * 0.5   # range roughly [-0.9, 0.7]

    for i, name in enumerate(joint_names):
        role = roles.get(name, "other")
        side = sides.get(name, 0)
        lat_sign = 1 if side == 1 else -1

        if role == "hand":
            motion[:, i, 0] = flap * 0.80      # up/down
            motion[:, i, 2] = lat_sign * np.abs(flap) * 0.25  # wingtip angle

        elif role == "spine":
            motion[:, i, 0] = -flap * 0.08     # body pitches slightly with stroke
        elif role == "head":
            motion[:, i, 0] = -flap * 0.05

    return motion


def _gen_idle_flying(joint_names, roles, sides, fps=20, duration=3.0):
    """Perched idle: slow wing-fold twitch, head scanning."""
    n = int(fps * duration)
    motion = np.zeros((n, len(joint_names), 3), dtype=np.float32)
    t = np.linspace(0, 2 * np.pi * 0.35, n, endpoint=False)

    for i, name in enumerate(joint_names):
        role = roles.get(name, "other")
        side = sides.get(name, 0)
        lat_sign = 1 if side == 1 else -1

        if role == "hand":
            motion[:, i, 0] = 0.04 * np.sin(t)
            motion[:, i, 2] = lat_sign * 0.03 * np.sin(t * 1.3)
        elif role == "head":
            motion[:, i, 2] = 0.12 * np.sin(t * 0.5)   # look left/right
            motion[:, i, 0] = 0.05 * np.sin(t * 1.1)   # nod

    return motion


# ─── Dispatcher ───────────────────────────────────────────────────────────

def _pick_motions(creature_type: str, biped_class: str):
    """
    Returns a list of (motion_key, generator_name) pairs to generate.
    motion_key is the filename stem: motion_{key}.npy
    """
    if creature_type == "flying":
        return [
            ("idle",   "idle_flying"),
            ("walk",   "fly"),
        ]
    if creature_type == "quadruped":
        return [
            ("idle",   "idle_quad"),
            ("walk",   "trot"),
        ]
    # biped
    attack_gen = {
        "mage_staff":    "staff_raise",
        "throw_potion":  "throw_potion",
        "melee":         "slash",
        "archer":        "bow_shoot",
        "generic_biped": "generic_attack",
    }.get(biped_class, "generic_attack")

    return [
        ("idle",   "idle_biped"),
        ("walk",   "walk"),
        ("action", attack_gen),
    ]


def _generate(gen_name: str, joint_names, roles, sides):
    dispatch = {
        "idle_biped":    _gen_idle_biped,
        "idle_quad":     _gen_idle_quad,
        "idle_flying":   _gen_idle_flying,
        "walk":          _gen_walk,
        "trot":          _gen_trot,
        "fly":           _gen_fly,
        "staff_raise":   _gen_attack_staff_raise,
        "throw_potion":  _gen_attack_throw_potion,
        "slash":         _gen_attack_slash,
        "bow_shoot":     _gen_attack_bow_shoot,
        "generic_attack":_gen_attack_generic,
    }
    fn = dispatch.get(gen_name)
    if fn is None:
        return np.zeros((40, len(joint_names), 3), dtype=np.float32)
    return fn(joint_names, roles, sides)


# ─── Main Entry Point ─────────────────────────────────────────────────────

def synthesize_motion(joints_json: str, output_dir: str,
                      prompt: str = "", props_path=None):
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Motion Synthesis: '{prompt}' ---")

    with open(joints_json) as f:
        j_data = json.load(f)

    joints    = {k: list(v) for k, v in j_data["joints"].items()}
    hierarchy = j_data.get("hierarchy", {})
    root      = j_data.get("root", next(iter(joints)))
    j_names   = list(joints.keys())

    roles, sides = _classify_skeleton(joints, hierarchy, root)

    role_counts = {}
    for r in roles.values():
        role_counts[r] = role_counts.get(r, 0) + 1
    print(f"  Skeleton: {len(j_names)} joints — {role_counts}")

    creature_type = detect_creature_type(prompt)
    biped_class   = detect_biped_class(prompt) if creature_type == "biped" else "n/a"
    print(f"  Creature type: {creature_type}  |  Class: {biped_class}")

    motion_plan = _pick_motions(creature_type, biped_class)
    print(f"  Animations to generate: {[k for k, _ in motion_plan]}")

    for mkey, gen_name in motion_plan:
        out_path = os.path.join(output_dir, f"motion_{mkey}.npy")
        print(f"  [{mkey}] generating '{gen_name}'…")
        motion = _generate(gen_name, j_names, roles, sides)
        np.save(out_path, {"motion": motion, "joint_names": j_names, "fps": 20})
        print(f"  [{mkey}] saved  shape={motion.shape}  →  {out_path}")

    print("--- Motion Synthesis complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rig_file",   required=True,
                        help="Path to joints.json from rigging stage")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt",     default="")
    parser.add_argument("--props",      default=None)
    args = parser.parse_args()
    synthesize_motion(args.rig_file, args.output_dir, args.prompt, args.props)
