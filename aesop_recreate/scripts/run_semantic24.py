"""Record the semantic24 runs (sim/semantic24.py): Town01 / Town02, street furniture, parked cars; the test runs are long,
the anomaly runs inject their anomaly P.INJECTIONS times.

    .venv/bin/python scripts/run_semantic24.py                    # every run not yet recorded, then the data configs
    .venv/bin/python scripts/run_semantic24.py t2_haybale         # one run (re-recorded)
    .venv/bin/python scripts/run_semantic24.py --config-only      # rewrite aesop_recreate/config/data_semantic24*.json

Per run, aesop_recreate/data/semantic24_long/<run>/ (the short test runs of the first version stay in data/semantic24/): frames/*.jpg, inst/*.png (instance segmentation, BGRA as CARLA sends it),
depth/*.png (uint16 centimetres), camera.jsonl (camera pose per tick), actors.json (instance id -> blueprint of
every spawned actor), states.jsonl, run_info.json and, for anomaly runs, anomalies.json (per injection, onset / offset =
the ticks it is on screen; for the pedestrian, the ticks he is on the road and on screen; "windows" lists the on-screen
ones; the data configs take each injection's window from the ground truth once it is extracted, gt_windows()). data/semantic24/_towns/ keeps each town's OpenDRIVE map and environment objects,
so scripts/extract_gt_concepts.py can label concepts offline. An anomaly run with fewer than P.INJECTIONS on-screen injections
is re-recorded with a fresh seed (up to 4 times), and so is any run in which a vehicle is thrown into the air (CARLA's physics
occasionally launches one, e.g. the Mitsubishi bus in Town02). Needs a CARLA server; one subprocess per run, so a crash costs one run.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import math
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

OUT = ROOT / "data" / "semantic24_long"
TOWNS = OUT / "_towns"
RUN_TIMEOUT_S = 1200
LONG_RUN_TIMEOUT_S = 5400
SEED_RETRIES = 4
# CARLA's only bus is thrown into the air by its physics now and then (t2_nominal_08 and t2_pedestrian, Town02): runs recorded
# after 2026-10-30 leave it out of the traffic (spawn_traffic and the lead car read S.NOT_IN_TRAFFIC)
S.NOT_IN_TRAFFIC = S.NOT_IN_TRAFFIC | {"vehicle.mitsubishi.fusorosa"}
AIRBORNE_TILT_DEG = 30.0   # a vehicle rolled or pitched beyond this, or rising faster than AIRBORNE_VZ, fails the run
AIRBORNE_VZ = 3.0


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
    """A stopped vehicle at the right kerb, its inner side P.PARK_KERB_M past the lane edge so the ego can pass (in Town02
    the shoulder is 0.3 m, so it stands partly on the kerb), physics off so it stays put and upright."""
    loc = S.offset_location(wp, wp.lane_width / 2 + 1.2, along)
    ground_z, _ = placement.ground_height(world, loc.x, loc.y, loc.z, loc.z)
    a, _ = placement.spawn_on_ground(world, bp, loc.x, loc.y, wp.transform.rotation.yaw, ground_z)
    if a is None:
        return None
    loc = S.offset_location(wp, wp.lane_width / 2 + a.bounding_box.extent.y + P.PARK_KERB_M, along)
    a.set_simulate_physics(False)
    a.set_transform(carla.Transform(carla.Location(loc.x, loc.y, a.get_location().z), carla.Rotation(yaw=wp.transform.rotation.yaw)))
    if lights:
        a.set_light_state(carla.VehicleLightState(carla.VehicleLightState.Special1 | carla.VehicleLightState.Position))
    return a


def place_furniture(world, route: list, rng: random.Random, until_m: float, from_m: float = 0.0) -> list:
    """Street furniture on the sidewalks along the route from from_m up to until_m: one item every 12-28 m, either side."""
    actors, m = [], from_m + rng.uniform(*P.FURNITURE_GAP_M)
    while m < min(until_m, 2.0 * (len(route) - 1)):
        wp = route[S.route_point(route, m)]
        side = sidewalk(wp, rng.choice(("right", "left")))
        if side is not None:
            obj = rng.choice(P.FURNITURE)
            a = spawn_prop(world, "static.prop." + rng.choice(K.PROPS[obj]), side, rng.uniform(-0.4, 0.4), yaw_offset=rng.choice((0, 90, 180, 270)))
            if a is not None:
                actors.append(a)
        m += rng.uniform(*P.FURNITURE_GAP_M)
    return actors


def pre_turn_point(route: list, at_m: float, rng: random.Random, step_m: float = 2.0) -> int:
    """Route index P.HAY_PRE_TURN_M before the last junction the ego crosses before at_m (in view while it waits there),
    or 25 m before at_m without one."""
    k = S.route_point(route, at_m)
    j = k
    while j > 0 and not route[j].is_junction and (k - j) * step_m < 60.0:
        j -= 1
    if not route[j].is_junction:
        return max(0, k - int(25.0 / step_m))
    while j > 0 and route[j - 1].is_junction:   # back to where the ego enters the junction
        j -= 1
    return max(0, j - int(rng.uniform(*P.HAY_PRE_TURN_M) / step_m))


def place_anomaly(world, route: list, kind: str, at_m: float, rng: random.Random) -> list:
    wp = route[S.route_point(route, at_m)]
    bp_lib = world.get_blueprint_library()
    actors = []
    if kind == "haybale":
        for at in (wp, route[pre_turn_point(route, at_m, rng)]):
            side = sidewalk(at, "right")
            if side is not None:
                n = rng.randint(2, 3)
                for j in range(n):
                    actors.append(spawn_prop(world, "static.prop." + rng.choice(K.PROPS["hay bale"]), side, rng.uniform(-0.3, 0.3),
                                             (j - (n - 1) / 2) * 2.5, rng.uniform(0, 360)))
    elif kind == "pedestrian":
        loc = S.offset_location(wp, wp.lane_width / 2 + P.PED_KERB_M)
        ground_z, _ = placement.ground_height(world, loc.x, loc.y, loc.z, loc.z)
        actors.append(world.try_spawn_actor(rng.choice(list(bp_lib.filter("walker.pedestrian.*"))),
                                            carla.Transform(carla.Location(loc.x, loc.y, ground_z + 1.0), carla.Rotation(yaw=wp.transform.rotation.yaw - 90))))
    elif kind == "emergency":
        actors.append(park_vehicle(world, bp_lib.find("vehicle.ford.ambulance"), wp, 0.0, lights=True))
        for along in P.POLICE_ALONG_M:   # the first free spot near the ambulance
            if (police := park_vehicle(world, bp_lib.find("vehicle.dodge.charger_police_2020"), wp, along, lights=True)) is not None:
                actors.append(police)
                break
        else:
            actors.append(None)
        side = sidewalk(wp, "right")
        if side is not None:
            actors.append(spawn_prop(world, "static.prop.warningaccident", side, -0.6, -16.0, 180.0))
    return actors


class Crossing:
    """The pedestrian anomaly, stepped every tick: waits on the right sidewalk, walks to the centre of the ego's lane
    once he is in the ego camera's view within P.PED_TRIGGER_M, stands there P.PED_PAUSE_S, then crosses to just past
    the far lane's edge (a guard rail can block the far sidewalk) and stops, or stops after P.PED_CROSS_S."""

    def __init__(self, walker, wp):
        self.walker, self.phase = walker, "wait"
        self.wait, self.cross_left = int(P.PED_PAUSE_S / FIXED_DELTA), int(P.PED_CROSS_S / FIXED_DELTA)
        self.goals = {"in": wp.transform.location, "cross": S.offset_location(wp, -(1.5 * wp.lane_width + 0.5))}

    def _in_view(self, ego) -> bool:
        here, there, fwd = ego.get_location(), self.walker.get_location(), ego.get_transform().get_forward_vector()
        dx, dy = there.x - here.x, there.y - here.y
        d = (dx * dx + dy * dy) ** 0.5
        return 0 < d < P.PED_TRIGGER_M and (dx * fwd.x + dy * fwd.y) / d > math.cos(math.radians(P.PED_VIEW_DEG))

    def _walk(self, goal) -> bool:
        """Walk towards goal; True once there."""
        here = self.walker.get_location()
        dx, dy = goal.x - here.x, goal.y - here.y
        d = (dx * dx + dy * dy) ** 0.5
        if d < 0.3:
            self.walker.apply_control(carla.WalkerControl(speed=0.0))
            return True
        self.walker.apply_control(carla.WalkerControl(direction=carla.Vector3D(dx / d, dy / d, 0.0), speed=P.PED_SPEED))
        return False

    def step(self, ego) -> str:
        if self.phase == "wait" and self._in_view(ego):
            self.phase = "in"
        if self.phase == "in" and self._walk(self.goals["in"]):
            self.phase = "pause"
        elif self.phase == "pause":
            self.wait -= 1
            if self.wait <= 0:
                self.phase = "cross"
        if self.phase == "cross":
            self.cross_left -= 1
            if self._walk(self.goals["cross"]) or self.cross_left <= 0:
                self.walker.apply_control(carla.WalkerControl(speed=0.0))
                self.phase = "done"
        return self.phase


class RouteTracker:
    """A long run's ego along its planned route, stepped every tick: its route distance (the nearest route point,
    searched forwards since the route can pass a place twice), and a new route from where the ego is once it has left
    the planned one (P.OFF_ROUTE_M for P.OFF_ROUTE_S) or nears its end. Street furniture is placed along the route as
    the ego goes, up to FURNISH_AHEAD_M ahead of it."""
    FURNISH_AHEAD_M = 300.0

    def __init__(self, world, tm, ego, route: list, rng: random.Random):
        self.world, self.tm, self.ego, self.route, self.rng = world, tm, ego, route, rng
        self.k, self.off, self.replans, self.props, self.furnished = 0, 0, 0, [], 0.0
        self._furnish()

    def _furnish(self) -> None:
        if self.at_m + self.FURNISH_AHEAD_M > self.furnished:
            until = self.furnished + 2 * self.FURNISH_AHEAD_M
            self.props += place_furniture(self.world, self.route, self.rng, until, self.furnished)
            self.furnished = until

    def step(self) -> bool:
        """True if the route was replaced this tick."""
        loc, ahead = self.ego.get_location(), self.route[self.k : self.k + 10]   # 20 m: further, a route doubling back would match
        d = [loc.distance(w.transform.location) for w in ahead]
        j = int(np.argmin(d))
        self.k, self.off = (self.k + j, 0) if d[j] < P.OFF_ROUTE_M else (self.k, self.off + 1)
        if self.off < P.OFF_ROUTE_S / FIXED_DELTA and self.k < len(self.route) - 100:
            self._furnish()
            return False
        start = self.world.get_map().get_waypoint(loc, lane_type=carla.LaneType.Driving)
        self.route, self.k, self.off, self.furnished = S.plan_route(self.world, start, P.LONG_ROUTE_M, self.rng), 0, 0, 0.0
        self.tm.set_path(self.ego, S.path_points(self.route))
        self._furnish()
        self.replans += 1
        return True

    @property
    def at_m(self) -> float:
        return 2.0 * self.k


class Injector:
    """A long anomaly run's injections, stepped every tick: one at a time, placed ahead of the ego on its route, removed
    once the ego is P.REMOVE_PAST_M past it (or after P.INJECTION_TIMEOUT_S, or when the ego's route is replaced),
    until P.INJECTIONS came on screen."""

    def __init__(self, world, client, kind: str, rng: random.Random):
        self.world, self.client, self.kind, self.rng = world, client, kind, rng
        self.cur, self.done = None, []   # the live injection, the finished ones

    @property
    def n_visible(self) -> int:
        return sum(r["visible"] for r in self.done)

    def _place(self, t: int, route: list, at_m: float) -> None:
        at_m = 2.0 * S.route_point(route, at_m)   # where it really goes: route_point moves it past any junction
        actors = place_anomaly(self.world, route, self.kind, at_m, self.rng)
        missing = not actors or None in actors   # a scene vehicle or the walker found no free spot
        actors = [a for a in actors if a is not None]
        crossing = Crossing(actors[0], route[S.route_point(route, at_m)]) if self.kind == "pedestrian" and actors else None
        self.cur = {"at_m": round(at_m, 1), "start": t, "actors": actors, "ids": [a.id & 0xFFFF for a in actors],
                    "crossing": crossing, "missing": missing, "px": [], "on_road": [], "phases": []}

    def _finish(self, t: int) -> None:
        c, self.cur = self.cur, None
        vis = [c["start"] + j for j, (n, r) in enumerate(zip(c["px"], c["on_road"])) if n >= P.MIN_VISIBLE_PX and r]
        ok = len(vis) >= P.MIN_VISIBLE_TICKS and not c["missing"]
        self.done.append({"i": len(self.done), "at_m": c["at_m"], "start": c["start"], "end": t, "visible": ok,
                          "onset": vis[0] if ok else None, "offset": vis[-1] + 1 if ok else None, "n_visible_ticks": len(vis),
                          "missing": c["missing"], "actors": [{"id": a.id, "blueprint": a.type_id} for a in c["actors"]],
                          **({"phases": c["phases"]} if c["crossing"] is not None else {})})
        print(f"injection {len(self.done)}: ticks {c['start']}-{t} at {c['at_m']} m, on screen {len(vis)} ticks, peak {max(c['px'], default=0)} px"
              + (f", phases {dict(collections.Counter(c['phases']))}, {c['crossing'].walker.get_location().distance(c['crossing'].goals['in']):.1f} m from the lane centre" if c["phases"] else "")
              + (" MISSING" if c["missing"] else ""), flush=True)
        self.client.apply_batch_sync([carla.command.DestroyActor(a) for a in c["actors"]])

    def step(self, t: int, ego, track: RouteTracker, replanned: bool) -> None:
        """Before the world tick: step the live injection, remove it once passed (or off the ego's new route), place the next."""
        at = track.at_m
        if (c := self.cur) is not None:
            if c["crossing"] is not None:
                c["phases"].append(c["crossing"].step(ego))
            if replanned or at > c["at_m"] + P.REMOVE_PAST_M or t - c["start"] > P.INJECTION_TIMEOUT_S / FIXED_DELTA:
                self._finish(t)
        if self.cur is None and self.n_visible < P.INJECTIONS and len(self.done) < P.MAX_INJECTIONS:
            self._place(t, track.route, at + self.rng.uniform(*(P.PED_AT_M if self.kind == "pedestrian" else P.INJECT_AHEAD_M)))
            if self.cur["missing"]:
                self._finish(t)

    def observe(self, m, ids: np.ndarray) -> None:
        """After the world tick: the live injection's pixels on screen (and, for the pedestrian, whether he is on the road)."""
        if (c := self.cur) is not None:
            c["px"].append(S.visible_px(ids, c["ids"]))
            c["on_road"].append(c["crossing"] is None or m.get_waypoint(c["crossing"].walker.get_location(), project_to_road=False,
                                                                         lane_type=carla.LaneType.Driving) is not None)

    def close(self, t: int) -> list:
        if self.cur is not None:
            self._finish(t)
        return self.done


def on_road_lane(m, loc) -> bool:
    """A walker here is 'on the road' in the ground truth: a road lane other than the kerb-side shoulder."""
    wp = m.get_waypoint(loc, project_to_road=False, lane_type=carla.LaneType.Any)
    return wp is not None and str(wp.lane_type) in K.ROAD_LANES and str(wp.lane_type) != "Shoulder"


def freeze_walkers(m, walkers: list, controllers: list, frozen: set, last: dict) -> None:
    """Stop for good every walker whose position P.FREEZE_LOOKAHEAD_S ahead (from its last tick's step: get_velocity()
    reads 0 for AI-controlled walkers) is on a road lane (a crosswalk): CARLA's cross factor 0 only stops jaywalking,
    its walker navigation still crosses roads at crosswalks. last: walker id -> its previous location."""
    k = P.FREEZE_LOOKAHEAD_S / FIXED_DELTA
    for w, c in zip(walkers, controllers):
        if w.id in frozen:
            continue
        loc = w.get_location()
        prev = last.get(w.id, loc)
        last[w.id] = loc
        dx, dy = (loc.x - prev.x) * k, (loc.y - prev.y) * k
        if any(on_road_lane(m, carla.Location(loc.x + f * dx, loc.y + f * dy, loc.z)) for f in (0.5, 1.0)):
            c.stop()
            w.apply_control(carla.WalkerControl(speed=0.0))
            frozen.add(w.id)


def break_jam(client, ego, vehicles: list) -> int:
    """Remove the background cars within P.JAM_AHEAD_M ahead of the ego; how many."""
    loc, fwd = ego.get_location(), ego.get_transform().get_forward_vector()
    ahead = [v for v in vehicles if v.is_alive and 0 < (v.get_location().x - loc.x) * fwd.x + (v.get_location().y - loc.y) * fwd.y
             and v.get_location().distance(loc) < P.JAM_AHEAD_M]
    client.apply_batch_sync([carla.command.DestroyActor(v) for v in ahead])
    return len(ahead)


def airborne(vehicles: list) -> list:
    """The vehicles CARLA's physics has thrown into the air or over on their side this tick."""
    out = []
    for v in vehicles:
        if not v.is_alive:   # CARLA now and then destroys a background car (e.g. one that fell through the map)
            continue
        r, vz = v.get_transform().rotation, v.get_velocity().z
        if abs(r.roll) > AIRBORNE_TILT_DEG or abs(r.pitch) > AIRBORNE_TILT_DEG or vz > AIRBORNE_VZ:
            out.append(v.type_id)
    return out


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
    """Record one run; exit status 3 if fewer than P.INJECTIONS of its anomalies came on screen."""
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
    cams, ego, lead, vehicles, walkers, controllers, props, injector, track = [], None, None, [], [], [], [], None, None
    status = 0
    try:
        bp_lib, m = world.get_blueprint_library(), world.get_map()
        for sp in rng.sample(m.get_spawn_points(), len(m.get_spawn_points())):
            start = m.get_waypoint(sp.location, lane_type=carla.LaneType.Driving)
            route = S.plan_route(world, start, P.LONG_ROUTE_M if p.long else S.ROUTE_M, rng)
            if len(route) * 2.0 >= 0.8 * (P.LONG_ROUTE_M if p.long else S.ROUTE_M) and (ego := world.try_spawn_actor(bp_lib.find(EGO_BLUEPRINT), sp)) is not None:
                break
        tm.auto_lane_change(ego, False)
        ego.set_autopilot(True, tm.get_port())
        tm.set_path(ego, S.path_points(route))
        if not p.pools:   # a test run: the ego drives slower all run
            tm.vehicle_percentage_speed_difference(ego, 100.0 - P.TEST_SPEED_PCT)
        car_bps = [b for b in bp_lib.filter("vehicle.*") if b.get_attribute("number_of_wheels").as_int() == 4 and b.id not in S.NOT_IN_TRAFFIC]
        if sc["lead"] and p.kind == "nominal" and not p.long:   # a lead car would hide the test runs' anomalies
            k = S.route_point(route, sc["lead_gap_m"], avoid_junction=False)
            tf = route[k].transform
            lead = world.try_spawn_actor(rng.choice(car_bps), carla.Transform(tf.location + carla.Location(z=0.4), tf.rotation))
            if lead is not None:
                lead.set_autopilot(True, tm.get_port())
                tm.auto_lane_change(lead, False)
                tm.vehicle_percentage_speed_difference(lead, sc["lead_speed_diff_pct"])
                tm.set_path(lead, S.path_points(route[k:]))
        props = [] if p.long else place_furniture(world, route, rng, P.FURNITURE_UNTIL_M)   # long runs: the route tracker's
        track = RouteTracker(world, tm, ego, route, rng) if p.long else None
        traffic, walkers, controllers = spawn_traffic(world, tm, SimpleNamespace(n_vehicles=sc["n_vehicles"], n_walkers=sc["n_walkers"]), rng,
                                                      [a.get_location() for a in (ego, lead) if a is not None])
        vehicles += traffic
        if p.kind != "nominal":
            injector = Injector(world, client, p.kind, rng)
        # walkers whose spawn point is on a road lane would be nominal 'pedestrian on the road': remove them
        on_road = [i for i, w in enumerate(walkers) if on_road_lane(m, w.get_location())]
        for i in reversed(on_road):
            controllers[i].stop()
            client.apply_batch_sync([carla.command.DestroyActor(controllers.pop(i)), carla.command.DestroyActor(walkers.pop(i))])
        cams = cameras(world, ego)
        world.tick()
        for _, q in cams:   # drop the frames of the settling tick
            q.get()
        everyone = [a for a in [lead] + vehicles + walkers + props if a is not None]
        actors = {str(a.id & 0xFFFF): {"actor_id": a.id, "type_id": a.type_id, "base_type": a.attributes.get("base_type", "")} for a in everyone}
        (out / "run_info.json").write_text(json.dumps({
            "run_name": p.name, **dataclasses.asdict(p), **sc, "ego_id": ego.id, "lead_id": lead.id if lead else None,
            "n_vehicles_spawned": len(vehicles), "n_walkers_spawned": len(walkers), "n_walkers_removed_on_road": len(on_road), "n_props": len(props),
            "n_ticks": p.n_ticks, "fixed_delta": FIXED_DELTA, "camera": {k: CAMERA[k] for k in ("width", "height", "fov")},
            "route_xy": [[round(w.transform.location.x, 1), round(w.transform.location.y, 1)] for w in route[::5]]}))

        history = GroundTruthStateHistory(max_len=4)
        flying, frozen, last, freeze, n_props = [], set(), {}, p.town in P.FREEZE_TOWNS, 0
        still_since, still_at, jam_removed = 0, ego.get_location(), 0
        with open(out / "states.jsonl", "w") as fs, open(out / "camera.jsonl", "w") as fc:
            for t in range(p.n_ticks):
                replanned = track.step() if track is not None else False
                if track is not None and len(track.props) > n_props:   # new street furniture joins actors.json
                    actors.update({str(a.id & 0xFFFF): {"actor_id": a.id, "type_id": a.type_id, "base_type": ""} for a in track.props[n_props:]})
                    n_props = len(track.props)
                if injector is not None:
                    injector.step(t, ego, track, replanned)
                    if injector.cur is not None:   # injection actors join actors.json (the ground truth's instance ids)
                        actors.update({str(a.id & 0xFFFF): {"actor_id": a.id, "type_id": a.type_id, "base_type": a.attributes.get("base_type", "")}
                                       for a in injector.cur["actors"]})
                if p.long:
                    here = ego.get_location()
                    if here.distance(still_at) > 1.0:
                        still_since, still_at = t, here
                    elif t - still_since > P.JAM_S / FIXED_DELTA:
                        jam_removed += break_jam(client, ego, vehicles)
                        still_since = t
                if freeze:
                    freeze_walkers(m, walkers, controllers, frozen, last)
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
                if injector is not None:
                    injector.observe(m, bgra[..., 1].astype(np.int32) + bgra[..., 0].astype(np.int32) * 256)
                flying += [(t, v) for v in airborne([a for a in [ego, lead] + vehicles if a is not None])]
                history.push(world, ego, sim_time=world.get_snapshot().timestamp.elapsed_seconds)
                fs.write(json.dumps(history.buffer[-1]) + "\n")
                if t % 100 == 0:
                    print(f"tick {t}/{p.n_ticks}" + (f"  injections {injector.n_visible} on screen / {len(injector.done)}" if injector else "")
                          + (f"  replans {track.replans}  jam cars removed {jam_removed}" if track else "")
                          + (f"  walkers frozen {len(frozen)}" if freeze else ""), flush=True)
        (out / "actors.json").write_text(json.dumps(actors))
        info = json.loads((out / "run_info.json").read_text())
        (out / "run_info.json").write_text(json.dumps({**info, "n_walkers_frozen": len(frozen), **({"n_replans": track.replans, "n_jam_cars_removed": jam_removed} if track else {})}))
        if injector is not None:
            inj = injector.close(p.n_ticks)
            windows = [[r["onset"], r["offset"]] for r in inj if r["visible"]]
            ok = len(windows) >= P.INJECTIONS
            (out / "anomalies.json").write_text(json.dumps({
                "anomaly_type": "semantic", "kind": p.kind, "success": ok, "n_frames": p.n_ticks, "n_injected": len(inj),
                "windows": windows, "onset_frame": windows[0][0] if windows else None, "offset_frame": windows[-1][1] if windows else None,
                "injections": inj}, indent=1))
            print(f"{p.kind}: {len(windows)} of {len(inj)} injections on screen", flush=True)
            if not ok:
                status = 3
                print(f"FAILED: fewer than {P.INJECTIONS} {p.kind} injections came on screen", flush=True)
        if flying:
            (out / "airborne.json").write_text(json.dumps(flying))
            status = 3
            print(f"FAILED: vehicles thrown into the air: {sorted(set(v for _, v in flying))} from tick {flying[0][0]}", flush=True)
    finally:
        for c, _ in cams:
            c.stop()
        if cams:
            world.tick()
        for c in controllers:
            c.stop()
        live = injector.cur["actors"] if injector is not None and injector.cur is not None else []
        doomed = [c for c, _ in cams] + controllers + walkers + vehicles + props + (track.props if track else []) + live + [a for a in (lead, ego) if a is not None]
        client.apply_batch_sync([carla.command.DestroyActor(a) for a in doomed])
        settings.synchronous_mode, settings.fixed_delta_seconds = False, None
        world.apply_settings(settings)
        tm.set_synchronous_mode(False)
    print(f"done: {out}", flush=True)
    return status


def done(p: P.Plan) -> bool:
    d = OUT / p.name
    if not (d / "camera.jsonl").exists() or len((d / "camera.jsonl").read_text().splitlines()) != p.n_ticks:
        return False
    if (d / "airborne.json").exists():
        return False
    a = d / "anomalies.json"
    return not a.exists() or json.loads(a.read_text()).get("success", False)


def gt_windows(name: str, a: dict) -> list | None:
    """Each injection's window from the ground truth (concepts_gt_raw.jsonl): first to last tick in which the ground
    truth sees any of its actors (the pedestrian: on the road), so no frame naming the anomaly counts as nominal; at
    least P.MIN_VISIBLE_TICKS of them. None before extract_gt_concepts.py has run."""
    path = OUT / name / "concepts_gt_raw.jsonl"
    if not path.exists() or "injections" not in a:
        return None
    seen = {}   # instance id -> ticks the ground truth labels it (on the road, for the pedestrian)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            for i in r["instances"]:
                if a["kind"] != "pedestrian" or i["predicate"] in ("on the road", "at the intersection"):
                    seen.setdefault(i["id"], []).append(r["t"])
    out = []
    for inj in a["injections"]:
        ticks = sorted(t for x in inj["actors"] for t in seen.get(x["id"] & 0xFFFF, []) if inj["start"] <= t < inj["end"])
        if len(ticks) >= P.MIN_VISIBLE_TICKS:
            out.append([ticks[0], ticks[-1] + 1])
    return out


def write_configs() -> None:
    """config/data_semantic24_long.json (withheld routes) and config/data_semantic24_long_incl.json (included routes)."""
    # paths relative to multimodal-ood-detection/, like its own data configs, so its scoring and dashboard code read them as-is
    entry = lambda name: {"run_name": name, "data_path": f"aesop_recreate/data/semantic24_long/{name}", "out_path": f"aesop_recreate/data/semantic24_long/{name}/embeddings"}  # noqa: E731
    runs = P.plan()
    anomaly = []
    for p in runs:
        if p.kind != "nominal" and done(p):
            a = json.loads((OUT / p.name / "anomalies.json").read_text())
            # windows: each injection's ticks in the ground truth (else the recorder's on-screen ticks); onset / offset
            # span them all (for readers that take one window)
            gt = gt_windows(p.name, a)
            w = gt or a.get("windows", [[a["onset_frame"], a["offset_frame"]]])
            anomaly.append({**entry(p.name), "onset_tick": w[0][0], "offset_tick": w[-1][1], "windows": w,
                            "windows_from": "ground truth" if gt else "recorder", "anomaly_type": "semantic", "anomaly_kind": p.kind})
    for pool, path in (("withheld", "data_semantic24_long.json"), ("included", "data_semantic24_long_incl.json")):
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
        todo = [plans[n] for n in args.runs] if args.runs else [p for p in plans.values() if not done(p)]
        todo.sort(key=lambda p: p.town)   # one map load per town
        for i, p in enumerate(todo):
            for k in range(SEED_RETRIES + 1):
                seed = p.seed + 1000 * k
                if not carla_up():
                    restart_carla(log)
                t0 = time.time()
                try:
                    rc = subprocess.run([sys.executable, __file__, p.name, "--seed", str(seed)], timeout=LONG_RUN_TIMEOUT_S if p.long else RUN_TIMEOUT_S,
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
