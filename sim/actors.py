"""Actor setup for a CARLA collection run: connecting, spawning the ego +
camera + background traffic and teardown.
"""
from __future__ import annotations

import random

import carla

EGO_BLUEPRINT = "vehicle.tesla.model3"
EGO_CAMERA_TRANSFORM = carla.Transform(carla.Location(x=1.6, z=1.7))


def connect(host: str, port: int, town: str | None, fixed_delta: float, seed: int):
    """Connects, loads/reuses the map, and puts the world + traffic manager
    into synchronous mode. Returns (client, world, tm, settings) -- keep
    `settings` around to pass to teardown()."""
    client = carla.Client(host, port)
    client.set_timeout(30.0)
    if town is None:
        world = client.get_world()
        print(f"Reusing already-loaded map: {world.get_map().name}")
    else:
        world = client.load_world(town)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_delta
    world.apply_settings(settings)

    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(seed)
    # Seeds CARLA's own pedestrian/nav-mesh RNG -- without this,
    # get_random_location_from_navigation() (walker spawn/goal points) is
    # not reproducible even with everything else seeded.
    world.set_pedestrians_seed(seed)
    return client, world, tm, settings


def spawn_ego(world: "carla.World", tm: "carla.TrafficManager") -> "carla.Actor":
    """Spawns the autopilot ego at the map's first spawn point (reserved for
    it -- spawn_background_traffic skips index 0)."""
    bp = world.get_blueprint_library().filter(EGO_BLUEPRINT)[0]
    spawn_point = world.get_map().get_spawn_points()[0]

    # CARLA map spawn points are supposed to always sit on a Driving lane, but
    # verify it explicitly -- a bad/edited map or an unfamiliar town (this
    # index isn't hand-checked per-town) failing silently here would show up
    # later as a confusingly stuck/non-navigating autopilot, not a clear error.
    waypoint = world.get_map().get_waypoint(
        spawn_point.location, project_to_road=False, lane_type=carla.LaneType.Driving
    )
    if waypoint is None:
        raise RuntimeError(
            f"Ego spawn point {spawn_point.location} is not on a drivable lane "
            f"for map {world.get_map().name} -- autopilot would have nowhere to go."
        )

    ego = world.spawn_actor(bp, spawn_point)
    ego.set_autopilot(True, tm.get_port())
    return ego


def spawn_ego_camera(world: "carla.World", ego: "carla.Actor", width: int, height: int, fov: int = 90) -> "carla.Actor":
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(width))
    bp.set_attribute("image_size_y", str(height))
    bp.set_attribute("fov", str(fov))
    return world.spawn_actor(bp, EGO_CAMERA_TRANSFORM, attach_to=ego)


def spawn_background_traffic(client, world, tm, n_vehicles: int, n_walkers: int, seed: int):
    # A local Random instance, not the `random` module -- module-level
    # random.choice()/random.random() calls draw from global interpreter
    # state that's never seeded, so blueprint/speed choices wouldn't be
    # reproducible even though the spawn-point shuffle below was.
    rng = random.Random(seed)

    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    rng.shuffle(spawn_points)

    vehicle_bps = bp_lib.filter("vehicle.*")
    vehicle_bps = [b for b in vehicle_bps if int(b.get_attribute("number_of_wheels")) == 4]

    # fill exactly n_vehicles so a blocked spawn point cannot leave one seed with fewer agents
    vehicles = []
    for sp in spawn_points[1:]:  # reserve spawn_points[0] for ego
        if len(vehicles) >= n_vehicles:
            break
        bp = rng.choice(vehicle_bps)
        actor = world.try_spawn_actor(bp, sp)
        if actor is not None:
            actor.set_autopilot(True, tm.get_port())
            vehicles.append(actor)
    if len(vehicles) < n_vehicles:
        print(f"WARNING: only {len(vehicles)}/{n_vehicles} vehicles could be spawned")

    walker_bps = bp_lib.filter("walker.pedestrian.*")
    walkers = []
    controllers = []
    walker_controller_bp = bp_lib.find("controller.ai.walker")
    attempts = 0
    while len(walkers) < n_walkers and attempts < 20 * max(n_walkers, 1):
        attempts += 1
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        bp = rng.choice(walker_bps)
        actor = world.try_spawn_actor(bp, carla.Transform(loc))
        if actor is None:
            continue
        walkers.append(actor)
    if len(walkers) < n_walkers:
        print(f"WARNING: only {len(walkers)}/{n_walkers} walkers could be spawned")
    world.tick()  # walkers need to exist in the world before attaching controllers
    for w in walkers:
        controller = world.spawn_actor(walker_controller_bp, carla.Transform(), attach_to=w)
        controller.start()
        controller.go_to_location(world.get_random_location_from_navigation())
        controller.set_max_speed(1.0 + rng.random())
        controllers.append(controller)

    return vehicles, walkers, controllers


def teardown(client, world, tm, settings, camera, ego, vehicles, walkers, controllers) -> None:
    """Fixed order -- don't reorder without retesting (see module docstring)."""
    camera.stop()
    world.tick()  # flush any in-flight callback
    camera.destroy()
    ego.set_autopilot(False, tm.get_port())
    ego.destroy()
    for c in controllers:
        c.stop()
        c.destroy()
    for w in walkers:
        w.destroy()
    client.apply_batch([carla.command.DestroyActor(v) for v in vehicles])

    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    tm.set_synchronous_mode(False)
