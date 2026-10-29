"""Record one randomised run (sim/scenarios.py): nominal, or with one semantic / physics / visual anomaly.

    .venv/bin/python scripts/run_scenario.py nominal 7                      # -> data/batch1/nominal_007
    .venv/bin/python scripts/run_scenario.py semantic 3 --subtype crowd     # -> data/batch1/semantic_003
    .venv/bin/python scripts/run_scenario.py physics 12 --out-root data/proto

Writes frames/*.jpg, states.jsonl (same schema as run_sim.py), run_info.json (every sampled choice) and, for
anomaly runs, anomalies.json with onset_frame / offset_frame and `success`. Exit code 3 = the anomaly did not
happen (no suitable agent, or the object never came on screen): discard the run.
Needs a CARLA server (../carla_sim/launch_carla.sh -RenderOffScreen); the Traffic Manager uses port 8100 so it
does not collide with a local web server on 8000.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import queue
import random
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import carla  # noqa: E402

from sim import placement, scenarios as S  # noqa: E402
from sim.state_extractor import GroundTruthStateHistory  # noqa: E402

N_TICKS = 600
FIXED_DELTA = 0.1
CAMERA = dict(width=800, height=600, fov=90, transform=carla.Transform(carla.Location(x=1.6, z=1.7)))
TM_PORT = 8100
EGO_BLUEPRINT = "vehicle.tesla.model3"


def connect(town: str, seed: int):
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(120.0)
    world = client.get_world()
    if not world.get_map().name.endswith(town):
        world = client.load_world(town)
    else:   # same map as the last run: clear anything a crashed run left behind
        leftovers = [a for f in ("vehicle.*", "walker.*", "controller.*", "sensor.*", "static.prop.*") for a in world.get_actors().filter(f)]
        client.apply_batch_sync([carla.command.DestroyActor(a) for a in leftovers])
    settings = world.get_settings()
    settings.synchronous_mode, settings.fixed_delta_seconds = True, FIXED_DELTA
    world.apply_settings(settings)
    tm = client.get_trafficmanager(TM_PORT)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(seed)
    world.set_pedestrians_seed(seed)
    return client, world, tm, settings


def camera(world, ego, kind: str):
    bp = world.get_blueprint_library().find(kind)
    bp.set_attribute("image_size_x", str(CAMERA["width"]))
    bp.set_attribute("image_size_y", str(CAMERA["height"]))
    bp.set_attribute("fov", str(CAMERA["fov"]))
    cam = world.spawn_actor(bp, CAMERA["transform"], attach_to=ego)
    q: queue.Queue = queue.Queue()
    cam.listen(q.put)
    return cam, q


def spawn_traffic(world, tm, sc: S.Scenario, rng: random.Random, reserved: list):
    """Background vehicles at free spawn points (none near the ego / lead) and walkers with AI controllers."""
    bp_lib = world.get_blueprint_library()
    vehicle_bps = [b for b in bp_lib.filter("vehicle.*") if b.get_attribute("number_of_wheels").as_int() == 4 and b.id not in S.NOT_IN_TRAFFIC]
    points = world.get_map().get_spawn_points()
    rng.shuffle(points)
    vehicles = []
    for sp in points:
        if len(vehicles) >= sc.n_vehicles:
            break
        if any(sp.location.distance(r) < 15.0 for r in reserved):
            continue
        v = world.try_spawn_actor(rng.choice(vehicle_bps), sp)
        if v is not None:
            v.set_autopilot(True, tm.get_port())
            vehicles.append(v)
    walker_bps = list(bp_lib.filter("walker.pedestrian.*"))
    walkers, controllers = [], []
    for _ in range(20 * sc.n_walkers):
        if len(walkers) >= sc.n_walkers:
            break
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            w = world.try_spawn_actor(rng.choice(walker_bps), carla.Transform(loc))
            if w is not None:
                walkers.append(w)
    world.tick()
    ctrl_bp = bp_lib.find("controller.ai.walker")
    for w in walkers:
        c = world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=w)
        c.start()
        c.go_to_location(world.get_random_location_from_navigation())
        c.set_max_speed(1.0 + rng.random())
        controllers.append(c)
    return vehicles, walkers, controllers


def place_semantic(world, route: list, sc: S.Scenario, rng: random.Random) -> tuple[list, dict]:
    """The semantic anomaly's actors, placed on the planned route before the first tick."""
    kind = sc.anomaly["kind"]
    bps, (lo, hi), spread, band = S.SEMANTIC_KINDS[kind]
    i = S.route_point(route, sc.anomaly["at_m"])
    wp = route[i]
    bp_lib = world.get_blueprint_library()
    half = wp.lane_width / 2
    actors, placed = [], []
    names = [rng.choice(bps) for _ in range(rng.randint(lo, hi))]
    if kind == "construction":
        names += rng.sample(S.CONSTRUCTION_EXTRAS, 2)
    grid = rng.sample([(a, l) for a in np.arange(-spread / 2, spread / 2, 0.9) for l in np.arange(band[0], band[1], 0.9)], len(names)) if kind == "crowd" else None
    for j, name in enumerate(names):
        along = (j / max(1, len(names) - 1) - 0.5) * spread if kind in ("construction",) else rng.uniform(-spread / 2, spread / 2)
        if kind == "crowd":
            bp = rng.choice(list(bp_lib.filter("walker.pedestrian.*")))
        else:
            bp = bp_lib.find(name)
        lateral = half + rng.uniform(*band)
        if grid is not None:
            along, lateral = grid[j][0], half + grid[j][1]
        if kind == "emergency":
            lateral = half + 0.9   # parked on the kerb, nose along the road; widened below by the bbox
        loc = S.offset_location(wp, lateral, along)
        yaw = wp.transform.rotation.yaw + (rng.uniform(-180, 180) if kind in ("debris", "out_of_place", "crowd") else 0.0)
        ground_z, _ = placement.ground_height(world, loc.x, loc.y, loc.z, loc.z)
        if kind == "crowd":
            a = world.try_spawn_actor(bp, carla.Transform(carla.Location(loc.x, loc.y, ground_z + 1.0), carla.Rotation(yaw=yaw)))
            info = {}
        else:
            a, info = placement.spawn_on_ground(world, bp, loc.x, loc.y, yaw, ground_z)
            if a is None:
                continue
            if kind == "emergency":   # push it out until its inner side sits 0.3 m inside the lane edge
                ext = a.bounding_box.extent.y
                loc = S.offset_location(wp, half - 0.3 + ext)
                a.set_transform(carla.Transform(carla.Location(loc.x, loc.y, a.get_location().z), carla.Rotation(yaw=yaw)))
                a.set_light_state(carla.VehicleLightState(carla.VehicleLightState.Special1 | carla.VehicleLightState.Position))
                a.apply_control(carla.VehicleControl(hand_brake=True))
        if a is not None:
            actors.append(a)
            placed.append({"blueprint": bp.id, "id": a.id, "xy": [round(loc.x, 2), round(loc.y, 2)], **info})
    return actors, {"route_index": i, "route_m": round(i * 2.0, 1), "anchor_xy": [round(wp.transform.location.x, 2), round(wp.transform.location.y, 2)],
                    "lane": {"road_id": wp.road_id, "lane_id": wp.lane_id}, "actors": placed}


def window_from_visibility(px: list[int]) -> tuple[int | None, int | None]:
    vis = [t for t, p in enumerate(px) if p >= S.MIN_VISIBLE_PX]
    return (vis[0], vis[-1] + 1) if len(vis) >= S.MIN_VISIBLE_TICKS else (None, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["nominal", "semantic", "physics", "visual"])
    ap.add_argument("seed", type=int)
    ap.add_argument("--subtype", help="pin the anomaly kind (e.g. crowd, lead_brake, dirty_camera)")
    ap.add_argument("--name", help="run name (default <kind>_<seed:03d>)")
    ap.add_argument("--out-root", default="data/batch1")
    ap.add_argument("--n-ticks", type=int, default=N_TICKS)
    args = ap.parse_args()

    sc = S.sample(args.kind, args.seed, args.subtype)
    name = args.name or f"{args.kind}_{args.seed:03d}"
    out = ROOT / args.out_root / name
    shutil.rmtree(out, ignore_errors=True)
    (out / "frames").mkdir(parents=True)
    rng = random.Random(f"run:{args.kind}:{args.seed}")
    print(f"{name}: {sc.town} · {sc.weather} · {sc.n_vehicles} vehicles · {sc.n_walkers} walkers · lead {sc.lead} · {sc.anomaly}", flush=True)

    client, world, tm, settings = connect(sc.town, args.seed)
    world.set_weather(getattr(carla.WeatherParameters, sc.weather))
    cams, ego, lead, vehicles, walkers, controllers, anomaly_actors = [], None, None, [], [], [], []
    anomaly = {"anomaly_type": args.kind, **sc.anomaly} if args.kind != "nominal" else None
    status = 0
    try:
        bp_lib, m = world.get_blueprint_library(), world.get_map()
        # ego at a random spawn point, on a random route long enough for the whole run
        for sp in rng.sample(m.get_spawn_points(), len(m.get_spawn_points())):
            start = m.get_waypoint(sp.location, lane_type=carla.LaneType.Driving)
            route = S.plan_route(world, start, S.ROUTE_M, rng)
            if args.kind == "physics" and sc.anomaly["kind"] == "cut_in":
                # start on a multi-lane stretch: a neighbour lane along most of the first 100 m
                if sum(S.adjacent_lane(w) is not None for w in route[:50]) < 40:
                    continue
            if len(route) * 2.0 >= 0.8 * S.ROUTE_M and (ego := world.try_spawn_actor(bp_lib.find(EGO_BLUEPRINT), sp)) is not None:
                break
        tm.auto_lane_change(ego, False)
        ego.set_autopilot(True, tm.get_port())
        tm.set_path(ego, S.path_points(route))
        if sc.lead:   # shares the ego's route from tick 0
            k = S.route_point(route, sc.lead_gap_m, avoid_junction=False)
            bps = [b for b in bp_lib.filter("vehicle.*") if b.get_attribute("number_of_wheels").as_int() == 4 and b.id not in S.NOT_IN_TRAFFIC]
            tf = route[k].transform
            lead = world.try_spawn_actor(rng.choice(bps), carla.Transform(tf.location + carla.Location(z=0.4), tf.rotation))
            if lead is not None:
                lead.set_autopilot(True, tm.get_port())
                tm.auto_lane_change(lead, False)
                tm.vehicle_percentage_speed_difference(lead, sc.lead_speed_diff_pct)
                tm.set_path(lead, S.path_points(route[k:]))
        companion = None
        if args.kind == "physics" and sc.anomaly["kind"] == "cut_in":   # a car in the neighbour lane from tick 0, to cut in later
            k = S.route_point(route, sc.anomaly["companion_gap_m"], avoid_junction=False)
            side = S.adjacent_lane(route[k])
            if side is not None:
                bps = [b for b in bp_lib.filter("vehicle.*") if b.get_attribute("number_of_wheels").as_int() == 4 and b.id not in S.NOT_IN_TRAFFIC]
                tf = side.transform
                companion = world.try_spawn_actor(rng.choice(bps), carla.Transform(tf.location + carla.Location(z=0.4), tf.rotation))
            if companion is not None:
                companion.set_autopilot(True, tm.get_port())
                tm.auto_lane_change(companion, False)
                # same speed as the ego, so it stays alongside / just ahead (the ego follows the lead when there is one)
                tm.vehicle_percentage_speed_difference(companion, sc.lead_speed_diff_pct if lead is not None else 0.0)
                tm.set_path(companion, S.path_points(S.companion_route(route[k:])))
                vehicles.append(companion)
            anomaly["companion_id"] = companion.id if companion else None
        if args.kind == "physics" and sc.anomaly["kind"] == "oncoming_drift":   # a car coming down the route towards the ego
            k = S.route_point(route, sc.anomaly["oncoming_gap_m"])
            opp = S.opposite_lane(route[k])
            if opp is not None:
                bps = [b for b in bp_lib.filter("vehicle.*") if b.get_attribute("number_of_wheels").as_int() == 4 and b.id not in S.NOT_IN_TRAFFIC]
                tf = opp.transform
                companion = world.try_spawn_actor(rng.choice(bps), carla.Transform(tf.location + carla.Location(z=0.4), tf.rotation))
            if companion is not None:
                companion.set_autopilot(True, tm.get_port())
                tm.auto_lane_change(companion, False)
                tm.set_path(companion, S.path_points(S.oncoming_route(route[: k + 1])))
                vehicles.append(companion)
            anomaly["companion_id"] = companion.id if companion else None
        if args.kind == "semantic":
            anomaly_actors, info = place_semantic(world, route, sc, rng)
            anomaly.update(info)
        traffic, walkers, controllers = spawn_traffic(world, tm, sc, rng, [a.get_location() for a in (ego, lead, companion) if a is not None])
        vehicles += traffic
        rgb, q_rgb = camera(world, ego, "sensor.camera.rgb")
        seg, q_seg = camera(world, ego, "sensor.camera.instance_segmentation")
        cams = [rgb, seg]
        (out / "run_info.json").write_text(json.dumps({
            "run_name": name, "scenario": args.kind, **dataclasses.asdict(sc), "lead_id": lead.id if lead else None,
            "lead_spawned": lead is not None, "n_vehicles_spawned": len(vehicles), "n_walkers_spawned": len(walkers),
            "ego_id": ego.id, "n_ticks": args.n_ticks, "fixed_delta": FIXED_DELTA,
            "route_xy": [[round(w.transform.location.x, 1), round(w.transform.location.y, 1)] for w in route[::5]]}))

        history = GroundTruthStateHistory(max_len=4)
        anomaly_ids = {a.id for a in anomaly_actors}
        px, px_ids, takeover, phys_rng = [], None, None, random.Random(f"phys:{args.seed}")
        attempts, attempt, tried = [], None, {}   # physics: every takeover tried, the current one, agent id -> tick it failed
        next_try = sc.anomaly.get("trigger_tick", 0)
        with open(out / "states.jsonl", "w") as f:
            for t in range(args.n_ticks):
                if (args.kind == "physics" and takeover is None and "onset_frame" not in anomaly and next_try <= t < S.PHYSICS_GIVE_UP
                        and px_ids is not None
                        and (ego.get_velocity().length() >= S.MIN_EGO_SPEED or sc.anomaly["kind"] == "oncoming_drift")):
                    # (an oncoming car drifting into the lane is anomalous even when the ego is waiting)
                    # keep trying the requested kind from the trigger tick on, on a fresh (or long-rested) agent each time
                    kind = sc.anomaly["kind"]
                    exclude = {i for i, when in tried.items() if t - when < S.RETRY_SAME_AGENT_AFTER}
                    cands = S.physics_candidates(world, ego, px_ids, lead.id if lead else None, exclude)[kind]
                    if cands:
                        actor = cands[0] if kind != "random_swerve" else phys_rng.choice(cands[:4])
                        takeover = S.PhysicsTakeover(kind, actor, ego, tm, sc.anomaly["strength"], sc.anomaly.get("sign", 1.0))
                        attempt = {"start": t, "actor_id": actor.id, "actor_blueprint": actor.type_id, "px": int((px_ids == actor.id).sum())}
                        anomaly_ids = {actor.id}
                        print(f"tick {t}: attempt {len(attempts) + 1}: {kind} on {actor.type_id} ({actor.id})", flush=True)
                if takeover is not None and not takeover.step():
                    attempt.update(end=t, **takeover.summary())
                    attempts.append(attempt)
                    if takeover.success:
                        anomaly.update(kind_requested=sc.anomaly["kind"], actor_id=attempt["actor_id"], actor_blueprint=attempt["actor_blueprint"],
                                       onset_frame=attempt["start"], offset_frame=min(args.n_ticks, t + 10))
                        print(f"tick {t}: {sc.anomaly['kind']} succeeded ({takeover.summary()})", flush=True)
                    else:
                        tried[attempt["actor_id"]] = t
                        next_try, anomaly_ids = t + S.RETRY_COOLDOWN, set()
                        print(f"tick {t}: attempt failed ({takeover.summary()}), trying again", flush=True)
                    takeover = None
                world.tick()
                q_rgb.get().save_to_disk(str(out / "frames" / f"{t:06d}.jpg"))
                px_ids = S.instance_ids(q_seg.get())
                px.append(S.visible_px(px_ids, anomaly_ids))
                history.push(world, ego, sim_time=world.get_snapshot().timestamp.elapsed_seconds)
                f.write(json.dumps(history.buffer[-1]) + "\n")
                if t % 100 == 0:
                    print(f"tick {t}/{args.n_ticks}", flush=True)

        if args.kind == "semantic":
            onset, offset = window_from_visibility(px)
            anomaly.update(onset_frame=onset, offset_frame=offset, success=onset is not None, visible_px=px)
        elif args.kind == "physics":
            anomaly.update(success="onset_frame" in anomaly, attempts=attempts, visible_px=px)
            if anomaly["success"]:
                anomaly["visible_px_at_onset"] = px[anomaly["onset_frame"]]
        elif args.kind == "visual":
            nprng, state = np.random.RandomState(args.seed), {}
            from PIL import Image
            for t in range(anomaly["onset_frame"], min(anomaly["offset_frame"], args.n_ticks)):
                p = out / "frames" / f"{t:06d}.jpg"
                Image.fromarray(S.corrupt(np.array(Image.open(p).convert("RGB")), anomaly["kind"], anomaly["strength"], nprng, state)).save(p, quality=95)
            anomaly["success"] = True
        if anomaly is not None and not anomaly["success"]:
            status = 3
            print(f"FAILED: the {args.kind} anomaly did not happen -- discard {name}", flush=True)
    finally:
        if anomaly is not None:
            anomaly["n_frames"] = args.n_ticks
            (out / "anomalies.json").write_text(json.dumps(anomaly, indent=1))
        for c in cams:
            c.stop()
        if cams:
            world.tick()
        for c in controllers:
            c.stop()
        doomed = cams + controllers + walkers + vehicles + anomaly_actors + [a for a in (lead, ego) if a is not None]
        client.apply_batch_sync([carla.command.DestroyActor(a) for a in doomed])
        settings.synchronous_mode, settings.fixed_delta_seconds = False, None
        world.apply_settings(settings)
        tm.set_synchronous_mode(False)
    print(f"done: {out}", flush=True)
    sys.exit(status)


if __name__ == "__main__":
    main()
