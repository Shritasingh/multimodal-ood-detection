"""Read-only scene checks and ground-aware spawning for sim/anomalies.py (CARLA is left-handed: +y is right)."""
from __future__ import annotations

import carla

_ROAD_LABELS = (
    carla.CityObjectLabel.Roads,
    carla.CityObjectLabel.RoadLines,
    carla.CityObjectLabel.Ground,
    carla.CityObjectLabel.Terrain,
    carla.CityObjectLabel.Sidewalks,
)


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def ego_yaw_rate_dps(ego: "carla.Actor") -> float:
    """|Ego yaw rate| in deg/s from the simulator's angular velocity."""
    return abs(ego.get_angular_velocity().z)


def ego_lane_waypoint(world: "carla.World", ego: "carla.Actor"):
    """Driving-lane waypoint under the ego, or None."""
    return world.get_map().get_waypoint(ego.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)


def lane_point_ahead(wp: "carla.Waypoint", distance_m: float):
    """Waypoint distance_m ahead along wp's lane (the branch closest to its heading), or None if the lane ends."""
    nxt = wp.next(distance_m)
    if not nxt:
        return None
    return min(nxt, key=lambda w: abs(_wrap_deg(w.transform.rotation.yaw - wp.transform.rotation.yaw)))


def lane_path(wp: "carla.Waypoint", length_m: float, step_m: float = 5.0):
    """Lane centreline ahead of wp as waypoints step_m apart."""
    path, travelled = [wp], 0.0
    while travelled < length_m:
        nxt = lane_point_ahead(path[-1], step_m)
        if nxt is None:
            break
        path.append(nxt)
        travelled += step_m
    return path


def distance_to_path(path, location: "carla.Location") -> float:
    """Distance from location to the nearest waypoint of path."""
    return min(w.transform.location.distance(location) for w in path)


def agents_in_corridor(world: "carla.World", ego: "carla.Actor", length_m: float, half_width_m: float):
    """Vehicles and pedestrians within length_m ahead of the ego and half_width_m to either side."""
    tf = ego.get_transform()
    fwd, loc = tf.get_forward_vector(), tf.location
    found = []
    for a in list(world.get_actors().filter("vehicle.*")) + list(world.get_actors().filter("walker.pedestrian.*")):
        if a.id == ego.id:
            continue
        d = a.get_location() - loc
        along, lateral = d.x * fwd.x + d.y * fwd.y, -d.x * fwd.y + d.y * fwd.x
        if 0.0 < along <= length_m and abs(lateral) <= half_width_m:
            found.append(a)
    return found


def ground_height(world: "carla.World", x: float, y: float, z_hint: float, fallback: float, search_m: float = 10.0) -> tuple[float, str]:
    """Road surface z under (x, y) from a downward ray (lane waypoint z is ~0.22 m too low in Town02), else fallback."""
    lp = world.ground_projection(carla.Location(x, y, z_hint + 2.0), search_m)
    if lp is not None and lp.label in _ROAD_LABELS:
        return lp.location.z, f"ray:{str(lp.label)}"
    return fallback, "waypoint" if lp is None else f"waypoint (ray hit {lp.label})"


def spawn_on_ground(world: "carla.World", bp, x: float, y: float, yaw_deg: float, ground_z: float):
    """Spawn bp with its bounding-box bottom on ground_z (blueprint origins differ); returns (actor, info) or (None, reason)."""
    rot = carla.Rotation(yaw=yaw_deg)
    actor = world.try_spawn_actor(bp, carla.Transform(carla.Location(x, y, ground_z + 0.5), rot))
    if actor is None:
        return None, "spawn blocked (something is at the spawn point)"
    bb = actor.bounding_box
    bottom_local = bb.location.z - bb.extent.z
    z = ground_z - bottom_local
    actor.set_transform(carla.Transform(carla.Location(x, y, z), rot))
    return actor, {"ground_z": round(ground_z, 3), "actor_z": round(z, 3), "bbox_bottom_local": round(bottom_local, 3)}
