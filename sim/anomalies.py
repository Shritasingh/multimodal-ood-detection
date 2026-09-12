"""One function per injectable anomaly type, so `scripts/run_sim.py` can
dispatch on `--scenario` without branching on implementation details.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

import carla

# Props unlikely to appear in normal CARLA street scenes -> stand-ins for
# "novel semantic object" (plastic-bag-shaped small/lightweight props from
# the default blueprint library; swap in a custom mesh later if needed).
NOVEL_OBJECT_BLUEPRINTS = [
    "static.prop.trafficcone01",
    "static.prop.plasticbag",
    "static.prop.box01",
    "static.prop.mattress",
]

CUTIN_DURATION_TICKS = 40  # ~4s hard lateral cut at 10Hz


def inject_semantic(world: "carla.World", ego: "carla.Actor", tick_idx: int, n_ticks: int) -> dict:
    """Spawns a novel static prop ~15m ahead of the ego. The prop is never
    removed, so it's anomalous for the rest of the run, not just a short
    window after onset."""
    bp_name = NOVEL_OBJECT_BLUEPRINTS[tick_idx % len(NOVEL_OBJECT_BLUEPRINTS)]
    bp = world.get_blueprint_library().find(bp_name)
    ego_tf = ego.get_transform()
    fwd = ego_tf.get_forward_vector()
    spawn_loc = ego_tf.location + carla.Location(x=fwd.x * 15, y=fwd.y * 15, z=0.3)
    actor = world.try_spawn_actor(bp, carla.Transform(spawn_loc, ego_tf.rotation))
    return {
        "blueprint": bp_name,
        "actor_id": actor.id if actor else None,
        "onset_frame": tick_idx,
        "offset_frame": n_ticks,
    }


def find_adjacent_lead_vehicle(ego: "carla.Actor", vehicles, max_ahead_m: float = 25.0):
    ego_loc = ego.get_transform().location
    ego_fwd = ego.get_transform().get_forward_vector()
    best, best_dist = None, max_ahead_m
    for v in vehicles:
        d = v.get_transform().location - ego_loc
        along = d.x * ego_fwd.x + d.y * ego_fwd.y
        if 3.0 < along < best_dist:
            best, best_dist = v, along
    return best


def start_cutin(world, ego, vehicles, tm, tick_idx: int):
    """Hijacks a nearby lead vehicle's control for a scripted hard-lateral
    cut-in. Returns (cutin_state, anomaly_fields); cutin_state is None if no
    lead vehicle was available to hijack (anomaly_fields then carries
    anomaly_injection_failed=True instead of onset/offset -- this run has no
    actual anomaly and should be discarded by anything scoring it)."""
    lead = find_adjacent_lead_vehicle(ego, vehicles)
    if lead is None:
        return None, {"anomaly_injection_failed": True}
    lead.set_autopilot(False, tm.get_port())
    state = {"actor": lead, "steps_left": CUTIN_DURATION_TICKS}
    fields = {
        "actor_id": lead.id,
        "onset_frame": tick_idx,
        "offset_frame": tick_idx + CUTIN_DURATION_TICKS,
    }
    return state, fields


def step_cutin(state: dict, tm) -> bool:
    """Applies one tick of hard lateral+forward control. Returns whether the
    cut-in is still active (caller should stop calling this once False)."""
    state["actor"].apply_control(carla.VehicleControl(throttle=0.9, steer=-0.35, brake=0.0))
    state["steps_left"] -= 1
    if state["steps_left"] <= 0:
        state["actor"].set_autopilot(True, tm.get_port())
        return False
    return True


def _apply_flare(img: np.ndarray, rng: np.random.RandomState, strength: float = 0.8) -> np.ndarray:
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w * rng.uniform(0.3, 0.7), h * rng.uniform(0.1, 0.4)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    radius = 0.5 * min(h, w)
    glow = np.clip(1.0 - dist / radius, 0, 1) ** 2
    out = img.astype(np.float32) + strength * 255.0 * glow[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_brightness_spike(img: np.ndarray, rng: np.random.RandomState, factor: float = 2.2) -> np.ndarray:
    out = img.astype(np.float32) * factor
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_blur(img: np.ndarray, rng: np.random.RandomState, radius: float = 8.0) -> np.ndarray:
    return np.array(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=radius)))


def _apply_mixed(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    return _apply_flare(_apply_brightness_spike(img, rng, factor=1.4), rng, strength=0.5)


VISUAL_CORRUPTIONS = {
    "flare": _apply_flare,
    "brightness": _apply_brightness_spike,
    "blur": _apply_blur,
    "mixed": _apply_mixed,
}


def inject_visual(run_dir: Path, corruption: str, onset_frame: int, duration: int, seed: int = 0) -> dict:
    """In-place pixel-space corruption of run_dir/frames/*.jpg over
    [onset_frame, onset_frame + duration). `seed` makes the corruption (e.g.
    flare position) reproducible, matching every other seeded source of
    randomness in a run."""
    rng = np.random.RandomState(seed)
    frame_paths = sorted((run_dir / "frames").glob("*.jpg"))
    onset, offset = onset_frame, min(onset_frame + duration, len(frame_paths))
    corrupt_fn = VISUAL_CORRUPTIONS[corruption]
    for idx, frame_path in enumerate(frame_paths):
        if onset <= idx < offset:
            img = corrupt_fn(np.array(Image.open(frame_path).convert("RGB")), rng)
            Image.fromarray(img).save(frame_path)
    return {
        "corruption": corruption,
        "onset_frame": onset,
        "offset_frame": offset,
        "n_frames": len(frame_paths),
    }
