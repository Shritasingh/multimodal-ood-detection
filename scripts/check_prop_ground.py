"""Compare old (ego z + 0.3) vs bounding-box prop placement against the road: .venv/bin/python scripts/check_prop_ground.py [prop ...] (needs CARLA)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import carla

from sim import placement
from sim.anomalies import NOVEL_OBJECT_BLUEPRINTS


def lowest_point_m(world, actor, ground_z: float, span_m: float = 2.5) -> float | None:
    """Height above ground_z of the lowest horizontal ray that hits the prop (cm resolution), or None."""
    tf = actor.get_transform()
    right = tf.get_right_vector()
    c = tf.location
    for h in np.arange(0.0, 1.2, 0.01):
        a = carla.Location(c.x - right.x * span_m, c.y - right.y * span_m, ground_z + h)
        b = carla.Location(c.x + right.x * span_m, c.y + right.y * span_m, ground_z + h)
        for hit in world.cast_ray(a, b):
            if hit.label not in placement._ROAD_LABELS and hit.location.distance(carla.Location(c.x, c.y, ground_z + h)) < span_m - 0.2:
                return float(h)
    return None


def main():
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(60.0)
    world = client.load_world("Town02")
    settings = world.get_settings()
    settings.synchronous_mode, settings.fixed_delta_seconds = True, 0.1
    world.apply_settings(settings)
    lib = world.get_blueprint_library()
    sp = world.get_map().get_spawn_points()[0]

    ego = world.spawn_actor(lib.filter("vehicle.tesla.model3")[0], sp)
    for _ in range(5):
        world.tick()
    ego_tf = ego.get_transform()
    fwd = ego_tf.get_forward_vector()
    x, y = ego_tf.location.x + fwd.x * 15, ego_tf.location.y + fwd.y * 15
    wp = world.get_map().get_waypoint(carla.Location(x, y, ego_tf.location.z), project_to_road=True)
    gz, how = placement.ground_height(world, x, y, ego_tf.location.z, wp.transform.location.z)
    print(f"ego z {ego_tf.location.z:.3f} | lane waypoint z {wp.transform.location.z:.3f} | ground under prop spot {gz:.3f} ({how})\n")
    print(f"{'blueprint':30s} {'bbox ctr z':>10s} {'half h':>7s} | {'OLD (ego z+0.3)':>16s} {'lowest pt above ground':>24s} | {'CALIBRATED':>10s} {'lowest pt above ground':>24s}")

    for name in (sys.argv[1:] or NOVEL_OBJECT_BLUEPRINTS):
        name = name if name.startswith("static.") else f"static.prop.{name}"
        bp = lib.find(name)
        old = world.try_spawn_actor(bp, carla.Transform(ego_tf.location + carla.Location(x=fwd.x * 15, y=fwd.y * 15, z=0.3), ego_tf.rotation))
        world.tick()
        old_low = lowest_point_m(world, old, gz) if old else None
        bb = old.bounding_box if old else None
        old_z = old.get_transform().location.z if old else float("nan")
        if old:
            old.destroy()
            world.tick()
        new, info = placement.spawn_on_ground(world, bp, x, y, ego_tf.rotation.yaw, gz)
        world.tick()
        new_low = lowest_point_m(world, new, gz) if new else None
        top_h = None
        if new:
            fmt = lambda v: "no ray hit" if v is None else f"{v * 100:5.1f} cm"
            print(f"{name:30s} {bb.location.z:10.3f} {bb.extent.z:7.3f} | z={old_z - gz:+.3f} m {fmt(old_low):>24s} | z={new.get_transform().location.z - gz:+.3f} m {fmt(new_low):>24s}")
            new.destroy()
            world.tick()
    ego.destroy()
    settings.synchronous_mode, settings.fixed_delta_seconds = False, None
    world.apply_settings(settings)


if __name__ == "__main__":
    main()
