"""One function per injectable anomaly type, so `scripts/run_sim.py` can
dispatch on `--scenario` without branching on implementation details.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

import carla

from encoders.physics_encoder import RolePhysicsEncoder
from sim import placement

# Props unlikely to appear in normal CARLA street scenes -> stand-ins for
# "novel semantic object" (plastic-bag-shaped small/lightweight props from
# the default blueprint library; swap in a custom mesh later if needed).
# static.prop.mattress does not exist in CARLA 0.9.16.
NOVEL_OBJECT_BLUEPRINTS = [
    "static.prop.box01",
    "static.prop.plasticbag",
    "static.prop.trafficcone01",
]

CUTIN_DURATION_TICKS = 40  # ~4s hard lateral cut at 10Hz


def inject_semantic(
    world: "carla.World",
    ego: "carla.Actor",
    tick_idx: int,
    n_ticks: int,
    blueprint: str | None = None,
    spawn_distance_m: float = 15.0,
) -> dict:
    """Spawn a prop on the ego's lane spawn_distance_m ahead with its base on the road (no scene checks; the caller decides when)."""
    bp_name = blueprint or NOVEL_OBJECT_BLUEPRINTS[tick_idx % len(NOVEL_OBJECT_BLUEPRINTS)]
    bp = world.get_blueprint_library().find(bp_name)
    ego_tf = ego.get_transform()
    wp = placement.ego_lane_waypoint(world, ego)
    ahead = placement.lane_point_ahead(wp, spawn_distance_m) if wp is not None else None
    if ahead is not None:
        x, y, yaw = ahead.transform.location.x, ahead.transform.location.y, ahead.transform.rotation.yaw
        fallback_z, lane = ahead.transform.location.z, {"road_id": ahead.road_id, "lane_id": ahead.lane_id}
    else:
        fwd = ego_tf.get_forward_vector()
        x, y, yaw = ego_tf.location.x + fwd.x * spawn_distance_m, ego_tf.location.y + fwd.y * spawn_distance_m, ego_tf.rotation.yaw
        fallback_z, lane = ego_tf.location.z, None
    ground_z, ground_src = placement.ground_height(world, x, y, ego_tf.location.z, fallback_z)
    actor, info = placement.spawn_on_ground(world, bp, x, y, yaw, ground_z)
    fields = {
        "blueprint": bp_name,
        "actor_id": actor.id if actor else None,
        "onset_frame": tick_idx,
        "offset_frame": n_ticks,
        "spawn_distance_m": spawn_distance_m,
        "spawn_xy": [round(x, 2), round(y, 2)],
        "lane": lane,
        "ground_source": ground_src,
    }
    if actor is None:
        fields["spawn_failed"] = info
    else:
        fields.update(info)
    return fields


def find_lead_vehicle(
    world,
    ego,
    vehicles,
    states: list,
    min_ahead_m: float = 8.0,
    max_ahead_m: float = 30.0,
    max_lateral_m: float = 1.5,
    min_lead_speed_mps: float = 1.0,
):
    """Return (vehicle, info) for the ego's lead (the physics_roles `lead`, in-lane, 8-30 m ahead, moving, outside junctions) or (None, reason)."""
    enc = RolePhysicsEncoder()
    if len(states) < enc.window:
        return None, "not enough state history yet"
    _, extras = enc.encode(states[-enc.window :])
    lead_id = extras["roles"].get("lead")
    if lead_id is None:
        return None, "no agent in the encoder's lead role"
    lead = next((v for v in vehicles if v.id == lead_id), None)
    if lead is None:
        return None, "the lead role is held by a non-background agent (e.g. a pedestrian)"

    e = states[-1]["ego"]
    o = next(a for a in states[-1]["others"] if a["id"] == lead_id)
    c, sn = math.cos(e["yaw"]), math.sin(e["yaw"])
    dx, dy = o["x"] - e["x"], o["y"] - e["y"]
    rel_x, rel_y = dx * c + dy * sn, -dx * sn + dy * c          # +y = ego's right
    align = math.cos(o["yaw"] - e["yaw"])
    speed = lead.get_velocity().length()
    ego_wp = placement.ego_lane_waypoint(world, ego)
    off_lane = placement.distance_to_path(placement.lane_path(ego_wp, max_ahead_m + 5.0), lead.get_location()) if ego_wp else 99.0
    lead_wp = world.get_map().get_waypoint(lead.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
    if not (min_ahead_m <= rel_x <= max_ahead_m):
        return None, f"lead is {rel_x:.0f} m ahead (need {min_ahead_m:.0f}-{max_ahead_m:.0f})"
    if abs(rel_y) > max_lateral_m or off_lane > 1.5:
        return None, f"lead is off the ego's lane centreline ({max(abs(rel_y), off_lane):.1f} m)"
    if align < 0.9:
        return None, "lead is not aligned with the ego's lane"
    if speed < min_lead_speed_mps:
        return None, f"lead is not moving ({speed:.1f} m/s)"
    if lead_wp is None or lead_wp.is_junction:
        return None, "lead is inside a junction"
    return lead, {
        "lead_role_verified": True,
        "lead_rel_x": round(rel_x, 2),
        "lead_rel_y": round(rel_y, 2),
        "lead_speed_mps": round(speed, 2),
        "ego_speed_mps": round(math.hypot(e["vx"], e["vy"]), 2),
        "lane": {"road_id": lead_wp.road_id, "lane_id": lead_wp.lane_id},
    }


def add_lead_vehicle(world, ego, distance_m: float, seed: int):
    """Spawn a vehicle on the ego's lane distance_m ahead, moving at the ego's velocity, as a dedicated lead; returns (vehicle, info) or (None, reason)."""
    wp = placement.ego_lane_waypoint(world, ego)
    ahead = placement.lane_point_ahead(wp, distance_m) if wp is not None else None
    if ahead is None:
        return None, "no lane ahead of the ego"
    bps = [b for b in world.get_blueprint_library().filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4]
    bp = random.Random(seed).choice(bps)
    loc, rot = ahead.transform.location, ahead.transform.rotation
    ground_z, _ = placement.ground_height(world, loc.x, loc.y, ego.get_location().z, loc.z)
    lead = world.try_spawn_actor(bp, carla.Transform(carla.Location(loc.x, loc.y, ground_z + 0.3), rot))
    if lead is None:
        return None, "spawn blocked (something is at the lead spawn point)"
    lead.set_target_velocity(ego.get_velocity())
    return lead, {"lead_added": True, "lead_blueprint": bp.id, "lead_rel_x": round(distance_m, 2), "lead_rel_y": 0.0,
                  "lane": {"road_id": ahead.road_id, "lane_id": ahead.lane_id}}


def start_cutin(world, ego, lead, lead_info: dict, tm, tick_idx: int):
    """Hijack the lead for a hard lateral swerve; returns (cutin_state, anomaly_fields)."""
    lead.set_autopilot(False, tm.get_port())
    steer = 0.35 if lead_info["lead_rel_y"] < 0 else -0.35
    state = {"actor": lead, "steps_left": CUTIN_DURATION_TICKS, "steer": steer}
    fields = {
        "actor_id": lead.id,
        "onset_frame": tick_idx,
        "offset_frame": tick_idx + CUTIN_DURATION_TICKS,
        **lead_info,
    }
    return state, fields


def step_cutin(state: dict, tm) -> bool:
    """Applies one tick of hard lateral+forward control. Returns whether the
    cut-in is still active (caller should stop calling this once False)."""
    state["actor"].apply_control(carla.VehicleControl(throttle=0.9, steer=state["steer"], brake=0.0))
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
