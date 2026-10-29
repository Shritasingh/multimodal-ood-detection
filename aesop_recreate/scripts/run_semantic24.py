"""Record the semantic24 runs (sim/semantic24.py): Town01 / Town02, street furniture, parked cars, one anomaly per test run.

    .venv/bin/python scripts/run_semantic24.py                    # every run not yet recorded, then the data configs
    .venv/bin/python scripts/run_semantic24.py t2_haybale         # one run (re-recorded)
    .venv/bin/python scripts/run_semantic24.py --config-only      # rewrite aesop_recreate/config/data_semantic24*.json

Per run, aesop_recreate/data/semantic24/<run>/: frames/*.jpg, inst/*.png (instance segmentation, BGRA as CARLA sends it),
depth/*.png (uint16 centimetres), camera.jsonl (camera pose per tick), actors.json (instance id -> blueprint of
every spawned actor), states.jsonl, run_info.json and, for anomaly runs, anomalies.json (onset / offset = the
ticks the anomaly is on screen). data/semantic24/_towns/ keeps each town's OpenDRIVE map and environment objects,
so scripts/extract_gt_concepts.py can label concepts offline. An anomaly that never comes on screen is re-recorded
with a fresh seed (up to 4 times). Needs a CARLA server; one subprocess per run, so a crash costs one run.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import queue
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent      # aesop_recreate/
MAIN = ROOT.parent                                   # multimodal-ood-detection/: sim/ and encoders/ merge with ours
sys.path[:0] = [str(ROOT), str(MAIN), str(MAIN / "scripts")]
import carla  # noqa: E402

from run_batch import carla_up, restart_carla  # noqa: E402
from run_scenario import CAMERA, EGO_BLUEPRINT, FIXED_DELTA, connect, spawn_traffic  # noqa: E402
from sim import concepts as K, placement, scenarios as S, semantic24 as P  # noqa: E402
from sim.state_extractor import GroundTruthStateHistory  # noqa: E402

OUT = ROOT / "data" / "semantic24"
TOWNS = OUT / "_towns"
RUN_TIMEOUT_S = 1200
SEED_RETRIES = 4


def save_town(world) -> None:
    """The town's OpenDRIVE map and environment objects (boxes in world coordinates), once per town."""
    town = world.get_map().name.split("/")[-1]
    if (TOWNS / f"{town}_env.json").exists():
        return
    TOWNS.mkdir(parents=True, exist_ok=True)
    (TOWNS / f"{town}.xodr").write_text(world.get_map().to_opendrive())
    objs = [{"name": o.name, "type": str(o.type), "center": [o.bounding_box.location.x, o.bounding_box.location.y, o.bounding_box.location.z],
             "extent": [o.bounding_box.extent.x, o.bounding_box.extent.y, o.bounding_box.extent.z], "yaw": o.bounding_box.rotation.yaw}
            for o in world.get_environment_objects(carla.CityObjectLabel.Any)]
    (TOWNS / f"{town}_env.json").write_text(json.dumps(objs))


def sidewalk(wp, side: str):
    """The sidewalk lane on the given side ('right' / 'left') of the ego's lane at wp, or None. Lanes are walked
    outwards by world position, since CARLA's left / right flip once the walk crosses into the opposite direction."""
    right = wp.transform.get_right_vector()
    sign = 1.0 if side == "right" else -1.0
    off = lambda l: sign * ((l.transform.location - wp.transform.location).x * right.x + (l.transform.location - wp.transform.location).y * right.y)  # noqa: E731
    lane = wp
    for _ in range(6):
        nxt = [l for l in (lane.get_left_lane(), lane.get_right_lane()) if l is not None and off(l) > off(lane) + 0.5]
        if not nxt:
            return None
        lane = nxt[0]
        if lane.lane_type == carla.LaneType.Sidewalk:
            return lane
    return None


def spawn_prop(world, name: str, wp, lateral: float = 0.0, along: float = 0.0, yaw_offset: float = 0.0):
    loc = S.offset_location(wp, lateral, along)
    ground_z, _ = placement.ground_height(world, loc.x, loc.y, loc.z, loc.z)
    a, _ = placement.spawn_on_ground(world, world.get_blueprint_library().find(name), loc.x, loc.y, wp.transform.rotation.yaw + yaw_offset, ground_z)
    return a


def park_vehicle(world, bp, wp, along: float = 0.0, lights: bool = False):
    """A stopped vehicle against the right kerb, its inner side on the lane edge, so the ego can pass (in Town02 the
    shoulder is 0.3 m, so it stands partly on the kerb)."""
    loc = S.offset_location(wp, wp.lane_width / 2 + 1.2, along)
    ground_z, _ = placement.ground_height(world, loc.x, loc.y, loc.z, loc.z)
    a, _ = placement.spawn_on_ground(world, bp, loc.x, loc.y, wp.transform.rotation.yaw, ground_z)
    if a is None:
        return None
    loc = S.offset_location(wp, wp.lane_width / 2 + a.bounding_box.extent.y + 0.05, along)
    a.set_transform(carla.Transform(carla.Location(loc.x, loc.y, a.get_location().z), carla.Rotation(yaw=wp.transform.rotation.yaw)))
    a.apply_control(carla.VehicleControl(hand_brake=True))
    if lights:
        a.set_light_state(carla.VehicleLightState(carla.VehicleLightState.Special1 | carla.VehicleLightState.Position))
    return a


def place_furniture(world, route: list, rng: random.Random) -> list:
    """Street furniture on the sidewalks along the route: one item every 12-28 m, either side."""
    actors, m = [], rng.uniform(*P.FURNITURE_GAP_M)
    while m < min(P.FURNITURE_UNTIL_M, 2.0 * (len(route) - 1)):
        wp = route[S.route_point(route, m)]
        side = sidewalk(wp, rng.choice(("right", "left")))
        if side is not None:
            obj = rng.choice(P.FURNITURE)
            a = spawn_prop(world, "static.prop." + rng.choice(K.PROPS[obj]), side, rng.uniform(-0.4, 0.4), yaw_offset=rng.choice((0, 90, 180, 270)))
            if a is not None:
                actors.append(a)
        m += rng.uniform(*P.FURNITURE_GAP_M)
    return actors


def place_anomaly(world, route: list, kind: str, at_m: float, rng: random.Random) -> list:
    wp = route[S.route_point(route, at_m)]
    bp_lib = world.get_blueprint_library()
    actors = []
    if kind == "haybale":
        side = sidewalk(wp, "right")
        if side is not None:
            n = rng.randint(2, 3)
            for j in range(n):
                actors.append(spawn_prop(world, "static.prop." + rng.choice(K.PROPS["hay bale"]), side, rng.uniform(-0.3, 0.3),
                                         (j - (n - 1) / 2) * 2.5, rng.uniform(0, 360)))
    elif kind == "pedestrian":
        loc = S.offset_location(wp, rng.uniform(-0.4, 0.4))
        ground_z, _ = placement.ground_height(world, loc.x, loc.y, loc.z, loc.z)
        actors.append(world.try_spawn_actor(rng.choice(list(bp_lib.filter("walker.pedestrian.*"))),
                                            carla.Transform(carla.Location(loc.x, loc.y, ground_z + 1.0), carla.Rotation(yaw=wp.transform.rotation.yaw + 90))))
    elif kind == "emergency":
        actors.append(park_vehicle(world, bp_lib.find("vehicle.ford.ambulance"), wp, 0.0, lights=True))
        actors.append(park_vehicle(world, bp_lib.find("vehicle.dodge.charger_police_2020"), wp, -9.0, lights=True))
        side = sidewalk(wp, "right")
        if side is not None:
            actors.append(spawn_prop(world, "static.prop.warningaccident", side, -0.6, -16.0, 180.0))
    return [a for a in actors if a is not None]


def cameras(world, ego) -> list:
    """RGB, instance segmentation and depth from the same pose, each with its own queue."""
    out = []
    for kind in ("sensor.camera.rgb", "sensor.camera.instance_segmentation", "sensor.camera.depth"):
        bp = world.get_blueprint_library().find(kind)
        bp.set_attribute("image_size_x", str(CAMERA["width"]))
        bp.set_attribute("image_size_y", str(CAMERA["height"]))
        bp.set_attribute("fov", str(CAMERA["fov"]))
        cam = world.spawn_actor(bp, CAMERA["transform"], attach_to=ego)
        q: queue.Queue = queue.Queue()
        cam.listen(q.put)
        out.append((cam, q))
    return out


def record(p: P.Plan, seed: int) -> int:
    """Record one run; exit status 3 if its anomaly never came on screen."""
    sc = P.sample(p, seed)
    out = OUT / p.name
    shutil.rmtree(out, ignore_errors=True)
    for d in ("frames", "inst", "depth"):
        (out / d).mkdir(parents=True)
    rng = random.Random(f"semantic24-run:{p.name}:{seed}")
    print(f"{p.name}: {sc}", flush=True)

    client, world, tm, settings = connect(p.town, seed)
    save_town(world)
    world.set_weather(getattr(carla.WeatherParameters, P.WEATHER))
    world.set_pedestrians_cross_factor(0.0)
    cams, ego, lead, vehicles, walkers, controllers, props, anomaly_actors = [], None, None, [], [], [], [], []
    status = 0
    try:
        bp_lib, m = world.get_blueprint_library(), world.get_map()
        for sp in rng.sample(m.get_spawn_points(), len(m.get_spawn_points())):
            start = m.get_waypoint(sp.location, lane_type=carla.LaneType.Driving)
            route = S.plan_route(world, start, S.ROUTE_M, rng)
            if len(route) * 2.0 >= 0.8 * S.ROUTE_M and (ego := world.try_spawn_actor(bp_lib.find(EGO_BLUEPRINT), sp)) is not None:
                break
        tm.auto_lane_change(ego, False)
        ego.set_autopilot(True, tm.get_port())
        tm.set_path(ego, S.path_points(route))
        car_bps = [b for b in bp_lib.filter("vehicle.*") if b.get_attribute("number_of_wheels").as_int() == 4 and b.id not in S.NOT_IN_TRAFFIC]
        if sc["lead"] and p.kind in ("nominal", "haybale"):   # a lead car would hide an in-lane anomaly
            k = S.route_point(route, sc["lead_gap_m"], avoid_junction=False)
            tf = route[k].transform
            lead = world.try_spawn_actor(rng.choice(car_bps), carla.Transform(tf.location + carla.Location(z=0.4), tf.rotation))
            if lead is not None:
                lead.set_autopilot(True, tm.get_port())
                tm.auto_lane_change(lead, False)
                tm.vehicle_percentage_speed_difference(lead, sc["lead_speed_diff_pct"])
                tm.set_path(lead, S.path_points(route[k:]))
        anomaly = None
        if p.kind != "nominal":
            anomaly_actors = place_anomaly(world, route, p.kind, sc["anomaly_at_m"], rng)
            anomaly = {"anomaly_type": "semantic", "kind": p.kind, "at_m": sc["anomaly_at_m"],
                       "actors": [{"id": a.id, "blueprint": a.type_id} for a in anomaly_actors]}
        props = place_furniture(world, route, rng)
        # no background car may start on the route up to the anomaly: one queued in front of it would hide it
        clear = [w.transform.location for w in route[: S.route_point(route, sc["anomaly_at_m"] + 20.0) : 5]] if anomaly else []
        traffic, walkers, controllers = spawn_traffic(world, tm, SimpleNamespace(n_vehicles=sc["n_vehicles"], n_walkers=sc["n_walkers"]), rng,
                                                      [a.get_location() for a in (ego, lead) if a is not None] + clear)
        vehicles += traffic
        # walkers whose spawn point is on a driving lane would be nominal 'pedestrian on the road': remove them
        on_road = [i for i, w in enumerate(walkers)
                   if m.get_waypoint(w.get_location(), project_to_road=False, lane_type=carla.LaneType.Driving) is not None]
        for i in reversed(on_road):
            controllers[i].stop()
            client.apply_batch_sync([carla.command.DestroyActor(controllers.pop(i)), carla.command.DestroyActor(walkers.pop(i))])
        cams = cameras(world, ego)
        world.tick()
        for _, q in cams:   # drop the frames of the settling tick
            q.get()
        everyone = [a for a in [lead] + vehicles + walkers + props + anomaly_actors if a is not None]
        (out / "actors.json").write_text(json.dumps({str(a.id & 0xFFFF): {"actor_id": a.id, "type_id": a.type_id,
                                                                          "base_type": a.attributes.get("base_type", "")} for a in everyone}))
        (out / "run_info.json").write_text(json.dumps({
            "run_name": p.name, **dataclasses.asdict(p), **sc, "ego_id": ego.id, "lead_id": lead.id if lead else None,
            "n_vehicles_spawned": len(vehicles), "n_walkers_spawned": len(walkers), "n_walkers_removed_on_road": len(on_road), "n_props": len(props),
            "n_ticks": P.N_TICKS, "fixed_delta": FIXED_DELTA, "camera": {k: CAMERA[k] for k in ("width", "height", "fov")},
            "route_xy": [[round(w.transform.location.x, 1), round(w.transform.location.y, 1)] for w in route[::5]]}))

        history = GroundTruthStateHistory(max_len=4)
        anomaly_ids = [a.id & 0xFFFF for a in anomaly_actors]
        px = []
        with open(out / "states.jsonl", "w") as fs, open(out / "camera.jsonl", "w") as fc:
            for t in range(P.N_TICKS):
                world.tick()
                rgb, inst, depth = (q.get() for _, q in cams)
                rgb.save_to_disk(str(out / "frames" / f"{t:06d}.jpg"))
                bgra = np.frombuffer(inst.raw_data, np.uint8).reshape(inst.height, inst.width, 4)
                cv2.imwrite(str(out / "inst" / f"{t:06d}.png"), bgra)
                d = K.decode_depth(np.frombuffer(depth.raw_data, np.uint8).reshape(depth.height, depth.width, 4))
                cv2.imwrite(str(out / "depth" / f"{t:06d}.png"), np.clip(d * 100, 0, 65535).astype(np.uint16))
                tf = rgb.transform
                fc.write(json.dumps({"t": t, "x": tf.location.x, "y": tf.location.y, "z": tf.location.z,
                                     "pitch": tf.rotation.pitch, "yaw": tf.rotation.yaw, "roll": tf.rotation.roll}) + "\n")
                ids = bgra[..., 1].astype(np.int32) + bgra[..., 0].astype(np.int32) * 256
                px.append(S.visible_px(ids, anomaly_ids))
                history.push(world, ego, sim_time=world.get_snapshot().timestamp.elapsed_seconds)
                fs.write(json.dumps(history.buffer[-1]) + "\n")
                if t % 100 == 0:
                    print(f"tick {t}/{P.N_TICKS}", flush=True)
        if anomaly is not None:
            vis = [t for t, n in enumerate(px) if n >= P.MIN_VISIBLE_PX]
            ok = len(vis) >= P.MIN_VISIBLE_TICKS
            anomaly.update(onset_frame=vis[0] if ok else None, offset_frame=vis[-1] + 1 if ok else None, success=ok, visible_px=px, n_frames=P.N_TICKS)
            (out / "anomalies.json").write_text(json.dumps(anomaly, indent=1))
            if not ok:
                status = 3
                print(f"FAILED: the {p.kind} anomaly was never on screen", flush=True)
    finally:
        for c, _ in cams:
            c.stop()
        if cams:
            world.tick()
        for c in controllers:
            c.stop()
        doomed = [c for c, _ in cams] + controllers + walkers + vehicles + props + anomaly_actors + [a for a in (lead, ego) if a is not None]
        client.apply_batch_sync([carla.command.DestroyActor(a) for a in doomed])
        settings.synchronous_mode, settings.fixed_delta_seconds = False, None
        world.apply_settings(settings)
        tm.set_synchronous_mode(False)
    print(f"done: {out}", flush=True)
    return status


def done(name: str) -> bool:
    d = OUT / name
    if not (d / "camera.jsonl").exists() or len((d / "camera.jsonl").read_text().splitlines()) != P.N_TICKS:
        return False
    a = d / "anomalies.json"
    return not a.exists() or json.loads(a.read_text()).get("success", False)


def write_configs() -> None:
    """config/data_semantic24.json (withheld routes) and config/data_semantic24_incl.json (included routes)."""
    # paths relative to multimodal-ood-detection/, like its own data configs, so its scoring and dashboard code read them as-is
    entry = lambda name: {"run_name": name, "data_path": f"aesop_recreate/data/semantic24/{name}", "out_path": f"aesop_recreate/data/semantic24/{name}/embeddings"}  # noqa: E731
    runs = P.plan()
    anomaly = []
    for p in runs:
        if p.kind != "nominal" and done(p.name):
            a = json.loads((OUT / p.name / "anomalies.json").read_text())
            anomaly.append({**entry(p.name), "onset_tick": a["onset_frame"], "offset_tick": a["offset_frame"], "anomaly_type": "semantic", "anomaly_kind": p.kind})
    for pool, path in (("withheld", "data_semantic24.json"), ("included", "data_semantic24_incl.json")):
        cfg = {"reference_runs": [entry(p.name) for p in runs if pool in p.pools], "negative_runs": [entry("t2_test_nominal")],
               "anomaly_runs": anomaly, "replay_twin": False}
        (ROOT / "config" / path).write_text(json.dumps(cfg, indent=2))
        print(f"wrote config/{path}: {len(cfg['reference_runs'])} reference, 1 negative, {len(anomaly)} anomaly runs")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run names (default: every run not yet recorded)")
    ap.add_argument("--seed", type=int, help="internal: record exactly this run with this seed")
    ap.add_argument("--config-only", action="store_true")
    args = ap.parse_args()
    plans = {p.name: p for p in P.plan()}
    if args.seed is not None:
        sys.exit(record(plans[args.runs[0]], args.seed))
    if not args.config_only:
        log = OUT / "record.log"
        OUT.mkdir(parents=True, exist_ok=True)
        todo = [plans[n] for n in args.runs] if args.runs else [p for p in plans.values() if not done(p.name)]
        todo.sort(key=lambda p: p.town)   # one map load per town
        for i, p in enumerate(todo):
            for k in range(SEED_RETRIES + 1):
                seed = p.seed + 1000 * k
                if not carla_up():
                    restart_carla(log)
                t0 = time.time()
                try:
                    rc = subprocess.run([sys.executable, __file__, p.name, "--seed", str(seed)], timeout=RUN_TIMEOUT_S,
                                        stdout=log.open("a"), stderr=subprocess.STDOUT).returncode
                except subprocess.TimeoutExpired:
                    rc = -1
                print(f"[{i + 1}/{len(todo)}] {p.name} seed {seed}: rc {rc} ({(time.time() - t0) / 60:.1f} min)", flush=True)
                if rc == 0:
                    break
                if rc != 3:
                    restart_carla(log)
    write_configs()


if __name__ == "__main__":
    main()
