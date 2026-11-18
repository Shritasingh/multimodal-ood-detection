"""Record the semantic24 runs (sim/semantic24.py): Town01 / Town02, street furniture, parked cars, an agent-driven ego;
each test anomaly run shows its anomaly once.

    .venv/bin/python aesop_recreate/scripts/run_semantic24.py                    # every run not yet recorded, then the data config
    .venv/bin/python aesop_recreate/scripts/run_semantic24.py t2_haybale         # one run (re-recorded)
    .venv/bin/python aesop_recreate/scripts/run_semantic24.py --config-only      # rewrite aesop_recreate/config/data_semantic24.json

Per run, aesop_recreate/data/semantic24/<run>/: frames/*.jpg, inst/*.png (instance segmentation, BGRA as CARLA sends it),
depth/*.png (uint16 centimetres), camera.jsonl (camera pose per tick), actors.json (instance id -> blueprint of
every spawned actor), states.jsonl, run_info.json and, for anomaly runs, anomalies.json (onset / offset = the ticks the
anomaly is on screen; for the pedestrian, the ticks he is on the road and on screen; the data config takes the window
from the ground truth once it is extracted, gt_windows()). data/semantic24/_towns/ keeps each town's OpenDRIVE map and
environment objects, so scripts/extract_gt_concepts.py can label concepts offline. A run is re-recorded with a fresh
seed (up to SEED_RETRIES times) if its anomaly is not clearly visible (P.MIN_VISIBLE_TICKS, P.MIN_PEAK_PX), a car
collides or leaves its lane, or a vehicle is thrown into the air (CARLA's physics occasionally launches one).
Needs a CARLA server; one subprocess per run, so a crash costs one run.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import math
import os
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

sys.path.append(str(MAIN.parent / "carla_sim" / "CARLA_0.9.16" / "PythonAPI" / "carla"))   # CARLA's agents package
from agents.navigation.behavior_agent import BehaviorAgent  # noqa: E402
from agents.navigation.local_planner import RoadOption  # noqa: E402

AGENT_LOG = os.environ.get("AGENT_LOG")   # debugging: AGENT_LOG=<file> logs the ego driver's state every tick
OUT = ROOT / "data" / P.DATASET
TOWNS = OUT / "_towns"
RUN_TIMEOUT_S = 1200
SEED_RETRIES = 4
# CARLA's only bus is thrown into the air by its physics now and then (Town02), and the Cybertruck rubs kerbs / guard
# rails on tight corners and is too: both are left out of the traffic (spawn_traffic and the lead car read S.NOT_IN_TRAFFIC)
S.NOT_IN_TRAFFIC = S.NOT_IN_TRAFFIC | {"vehicle.mitsubishi.fusorosa", "vehicle.tesla.cybertruck"}
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


def clear_point(route: list, at_m: float, step_m: float = 2.0) -> float:
    """at_m moved forward to the first route point with no junction within P.SCENE_CLEAR_M behind or ahead of it."""
    n, i = int(P.SCENE_CLEAR_M / step_m), int(at_m / step_m)
    while i < len(route) - 1 and any(w.is_junction for w in route[max(0, i - n) : i + n + 1]):
        i += 1
    return i * step_m


_BARRIERS: dict = {}   # map name -> [(location, extent)] of its guard rails, fences and walls


def kerb_open(world, wp, along_m: float = 6.0) -> bool:
    """No guard rail, fence or wall between the right sidewalk and the road within along_m of wp (the pedestrian
    anomaly's way onto the road)."""
    m = world.get_map().name
    if m not in _BARRIERS:
        _BARRIERS[m] = [(o.bounding_box.location, o.bounding_box.extent) for lab in (carla.CityObjectLabel.GuardRail, carla.CityObjectLabel.Fences, carla.CityObjectLabel.Walls)
                        for o in world.get_environment_objects(lab)]
    loc, fwd, right = wp.transform.location, wp.transform.get_forward_vector(), wp.transform.get_right_vector()
    for c, e in _BARRIERS[m]:
        dx, dy = c.x - loc.x, c.y - loc.y
        a, r = dx * fwd.x + dy * fwd.y, dx * right.x + dy * right.y
        # a rail / fence runs along the kerb: its box centre is beside the kerb, up to its half length along the road
        if abs(a) < along_m + max(e.x, e.y) and wp.lane_width / 2 - 0.5 < r < wp.lane_width / 2 + P.PED_KERB_M + 1.0:
            return False
    return True


def anomaly_spot(world, route: list, kind: str, at_m: float, clear_kinds: tuple, open_kerb: bool) -> float:
    """Route distance where an anomaly goes: past any junction (route_point), for clear_kinds where no junction is within
    P.SCENE_CLEAR_M, and (open_kerb) for the pedestrian where nothing blocks his way onto the road."""
    at_m = 2.0 * S.route_point(route, at_m)
    if kind in clear_kinds:
        at_m = clear_point(route, at_m)
    for _ in range(40 if open_kerb and kind == "pedestrian" else 0):
        if kerb_open(world, route[S.route_point(route, at_m)]):
            break
        at_m = clear_point(route, at_m + 4.0)
    return at_m


def clear_kerb(world, client, wp) -> list:
    """Room for the emergency scene at wp: destroy the spawned props on the right kerb within P.SCENE_CLEAR_M of it and
    hide the town's own small objects there; returns the hidden ones' ids (enable_environment_objects shows them again)."""
    loc, fwd, right = wp.transform.location, wp.transform.get_forward_vector(), wp.transform.get_right_vector()

    def on_kerb(p) -> bool:
        dx, dy = p.x - loc.x, p.y - loc.y
        return abs(dx * fwd.x + dy * fwd.y) < P.SCENE_CLEAR_M and wp.lane_width / 2 - 0.5 < dx * right.x + dy * right.y < wp.lane_width / 2 + P.KERB_CLEAR_M

    client.apply_batch_sync([carla.command.DestroyActor(a) for a in world.get_actors().filter("static.prop.*") if on_kerb(a.get_location())])
    labels = (carla.CityObjectLabel.Poles, carla.CityObjectLabel.Static, carla.CityObjectLabel.Dynamic, carla.CityObjectLabel.Vegetation, carla.CityObjectLabel.Other)
    hidden = [o.id for lab in labels for o in world.get_environment_objects(lab) if on_kerb(o.transform.location)]
    world.enable_environment_objects(set(hidden), False)
    return hidden


def park_vehicle(world, bp, wp, along: float = 0.0, lights: bool = False):
    """A stopped vehicle against the right kerb, its inner side on the lane edge, so the ego can pass (in Town02 the
    shoulder is 0.3 m, so it stands partly on the kerb). Physics off: the kerb overlap would otherwise tip it over."""
    loc = S.offset_location(wp, wp.lane_width / 2 + 1.2, along)
    ground_z, _ = placement.ground_height(world, loc.x, loc.y, loc.z, loc.z)
    a, _ = placement.spawn_on_ground(world, bp, loc.x, loc.y, wp.transform.rotation.yaw, ground_z)
    if a is None:
        return None
    loc = S.offset_location(wp, wp.lane_width / 2 + a.bounding_box.extent.y + 0.05, along)
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


class AgentDriver:
    """The ego: CARLA's BehaviorAgent following the planned route point by point, stepped every tick.
    On top of the agent's own speed choice (P.SPEED_PCT of the limit), a cap that slows it at <= P.SLOW_A for
    what lies on its route ahead (at least P.LOOKAHEAD_WPS points, and as far as it needs to stop from its speed):
    curves (taken at <= P.CURVE_A_LAT), red / yellow lights
    (stop 2 m before the line), walkers on its lane and cars on its path (stop P.STOP_GAP_M short, or follow the car
    at its speed). The agent's own emergency stop stays as the fallback."""

    def __init__(self, world, m, ego, route: list):
        self.world, self.m, self.ego = world, m, ego
        self.agent = BehaviorAgent(ego, behavior="cautious", map_inst=m,
                                   opt_dict={**P.AGENT, "dt": FIXED_DELTA,
                                             "longitudinal_control_dict": {**P.AGENT["longitudinal_control_dict"], "dt": FIXED_DELTA}})
        self.stop_lines = [(tl, [w.transform.location for w in tl.get_stop_waypoints()]) for tl in world.get_actors().filter("traffic.traffic_light")]
        self.set_route(route)

    def set_route(self, route: list) -> None:
        self.agent.set_global_plan([(w, RoadOption.LANEFOLLOW) for w in route[1:]], clean_queue=True)

    def _cap(self) -> float:
        """Speed cap in m/s from what lies ahead: min over constraints of sqrt(v_i^2 + 2 a d_i)."""
        a, cap, here = P.SLOW_A, 1e9, self.ego.get_location()
        vel = self.ego.get_velocity()
        n = max(P.LOOKAHEAD_WPS, math.ceil(((vel.x ** 2 + vel.y ** 2) / (2 * a) + 15.0) / 2.0))   # stopping distance + 15 m
        q = [w for w, _ in list(self.agent._local_planner._waypoints_queue)[:n]]
        if len(q) < 2:
            return 0.0
        dist, d, prev = [], 0.0, here
        for w in q:
            d += prev.distance(w.transform.location)
            prev = w.transform.location
            dist.append(d)
        ahead = lambda v, d_stop: math.sqrt(v * v + 2 * a * max(0.0, d_stop))  # noqa: E731
        for w0, w1, d0 in zip(q, q[1:], dist):   # curves
            step = w0.transform.location.distance(w1.transform.location)
            k = abs(math.radians((w1.transform.rotation.yaw - w0.transform.rotation.yaw + 180) % 360 - 180)) / step if step > 0.1 else 0.0
            if k > 1e-3:
                cap = min(cap, ahead(math.sqrt(P.CURVE_A_LAT / k), d0))
        for tl, lines in self.stop_lines:   # red / yellow lights whose stop line is on the route ahead
            if tl.state in (carla.TrafficLightState.Red, carla.TrafficLightState.Yellow):
                for loc in lines:
                    hit = [d0 for w, d0 in zip(q, dist) if w.transform.location.distance(loc) < 1.5]
                    if hit:
                        cap = min(cap, ahead(0.0, hit[0] - 2.0))
        on_path = lambda loc, r: next((d0 for w, d0 in zip(q, dist) if w.transform.location.distance(loc) < r), None)  # noqa: E731
        for wk in self.world.get_actors().filter("walker.pedestrian.*"):   # walkers on a driving lane of the route ahead
            loc = wk.get_location()
            if loc.distance(here) < dist[-1] + 5 and self.m.get_waypoint(loc, project_to_road=False, lane_type=carla.LaneType.Driving) is not None:
                if (d0 := on_path(loc, 2.5)) is not None:
                    cap = min(cap, ahead(0.0, d0 - P.STOP_GAP_M))
        for v in self.world.get_actors().filter("vehicle.*"):   # cars on the route ahead: follow them
            if v.id == self.ego.id or v.get_location().distance(here) > dist[-1] + 5:
                continue
            if (d0 := on_path(v.get_location(), 1.8)) is not None:
                vel = v.get_velocity()
                cap = min(cap, ahead(math.hypot(vel.x, vel.y), d0 - P.STOP_GAP_M))
        return cap

    def _drop_passed(self) -> None:
        """Drop the route points already behind the ego. The agent's emergency stop returns before its local planner
        pops the points it passes, so an ego braking past one keeps it as its target and, moving off again, steers
        back towards it, off the road (t2_nominal_10, 2026-11-04)."""
        q, tf = self.agent._local_planner._waypoints_queue, self.ego.get_transform()
        f, loc = tf.get_forward_vector(), tf.location
        while len(q) > 1:
            w = q[0][0].transform.location
            if (w.x - loc.x) * f.x + (w.y - loc.y) * f.y >= 0.0:
                break
            q.popleft()

    def step(self) -> None:
        self._drop_passed()
        limit = self.ego.get_speed_limit()   # km/h
        self.agent._behavior.speed_lim_dist = (1 - P.SPEED_PCT / 100) * limit
        cap = self._cap()
        self.agent._behavior.max_speed = min(limit, 3.6 * cap)
        control = self.agent.run_step()
        self.ego.apply_control(control)
        if AGENT_LOG:
            lp, loc = self.agent._local_planner, self.ego.get_location()
            tw = lp.target_waypoint.transform.location if lp.target_waypoint is not None else None
            with open(AGENT_LOG, "a") as f:
                f.write(json.dumps({"queue": len(lp._waypoints_queue), "x": round(loc.x, 2), "y": round(loc.y, 2), "cap": round(cap, 2),
                                    "target": [round(tw.x, 1), round(tw.y, 1)] if tw else None, "limit": limit,
                                    "steer": round(control.steer, 3), "throttle": round(control.throttle, 3), "brake": round(control.brake, 3)}) + "\n")


class RouteTracker:
    """The ego along its planned route, stepped every tick: its route distance (the nearest route point, searched
    forwards since the route can pass a place twice), and a new route from where the ego is once it has left the
    planned one (P.OFF_ROUTE_M for P.OFF_ROUTE_S) or nears its end. Street furniture is placed along the route as the
    ego goes, up to FURNISH_AHEAD_M ahead of it. set_route(route) hands a new route to the ego's driver."""
    FURNISH_AHEAD_M = 300.0

    def __init__(self, world, set_route, ego, route: list, rng: random.Random):
        self.world, self.set_route, self.ego, self.route, self.rng = world, set_route, ego, route, rng
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
        self.route, self.k, self.off, self.furnished = S.plan_route(self.world, start, P.ROUTE_M, self.rng), 0, 0, 0.0
        self.set_route(self.route)
        self._furnish()
        self.replans += 1
        return True

    @property
    def at_m(self) -> float:
        return 2.0 * self.k


class Injector:
    """An anomaly run's injection, stepped every tick: placed once at the run's anomaly spot (at_m) on the ego's route,
    removed once the ego is P.REMOVE_PAST_M past it (or after P.INJECTION_TIMEOUT_S, or when the ego's route is
    replaced). It counts once it is on screen on P.MIN_VISIBLE_TICKS ticks and reaches P.MIN_PEAK_PX."""
    n = 1   # injections needed on screen

    def __init__(self, world, client, kind: str, rng: random.Random, at_m: float):
        self.world, self.client, self.kind, self.rng, self.at_m = world, client, kind, rng, at_m
        self.cur, self.done = None, []   # the live injection, the finished one

    @property
    def n_visible(self) -> int:
        return sum(r["visible"] for r in self.done)

    def _place(self, t: int, route: list, at_m: float) -> None:
        at_m = anomaly_spot(self.world, route, self.kind, at_m, P.ANOMALIES, True)
        hidden = []
        if self.kind == "emergency":   # on a cleared kerb
            hidden = clear_kerb(self.world, self.client, route[S.route_point(route, at_m)])
        actors = place_anomaly(self.world, route, self.kind, at_m, self.rng)
        missing = not actors or None in actors   # a scene vehicle or the walker found no free spot
        actors = [a for a in actors if a is not None]
        crossing = Crossing(actors[0], route[S.route_point(route, at_m)]) if self.kind == "pedestrian" and actors else None
        self.cur = {"at_m": round(at_m, 1), "start": t, "actors": actors, "ids": [a.id & 0xFFFF for a in actors],
                    "crossing": crossing, "missing": missing, "hidden": hidden, "px": [], "on_road": [], "phases": []}

    def _finish(self, t: int) -> None:
        c, self.cur = self.cur, None
        vis = [c["start"] + j for j, (n, r) in enumerate(zip(c["px"], c["on_road"])) if n >= P.MIN_VISIBLE_PX and r]
        ok = len(vis) >= P.MIN_VISIBLE_TICKS and max(c["px"], default=0) >= P.MIN_PEAK_PX and not c["missing"]
        self.done.append({"i": len(self.done), "at_m": c["at_m"], "start": c["start"], "end": t, "visible": ok,
                          "onset": vis[0] if ok else None, "offset": vis[-1] + 1 if ok else None, "n_visible_ticks": len(vis),
                          "peak_px": max(c["px"], default=0),
                          "missing": c["missing"], "actors": [{"id": a.id, "blueprint": a.type_id} for a in c["actors"]],
                          **({"phases": c["phases"]} if c["crossing"] is not None else {})})
        print(f"injection {len(self.done)}: ticks {c['start']}-{t} at {c['at_m']} m, on screen {len(vis)} ticks, peak {max(c['px'], default=0)} px"
              + (f", phases {dict(collections.Counter(c['phases']))}, {c['crossing'].walker.get_location().distance(c['crossing'].goals['in']):.1f} m from the lane centre" if c["phases"] else "")
              + (" MISSING" if c["missing"] else ""), flush=True)
        self.client.apply_batch_sync([carla.command.DestroyActor(a) for a in c["actors"]])
        self.world.enable_environment_objects(set(c["hidden"]), True)

    def step(self, t: int, ego, track: RouteTracker, replanned: bool) -> None:
        """Before the world tick: step the live injection, remove it once passed (or off the ego's new route), place the next."""
        at = track.at_m
        if (c := self.cur) is not None:
            if c["crossing"] is not None:
                c["phases"].append(c["crossing"].step(ego))
            if replanned or at > c["at_m"] + P.REMOVE_PAST_M or t - c["start"] > P.INJECTION_TIMEOUT_S / FIXED_DELTA:
                self._finish(t)
        if self.cur is None and not self.done:
            self._place(t, track.route, self.at_m)
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


def smooth_tm(tm, v, speed_diff_pct: float = 100.0 - P.SPEED_PCT) -> None:
    """A Traffic Manager car that keeps its lane, a P.LEADING_GAP_M gap and every rule."""
    tm.auto_lane_change(v, False)
    tm.random_left_lanechange_percentage(v, 0.0)
    tm.random_right_lanechange_percentage(v, 0.0)
    tm.distance_to_leading_vehicle(v, P.LEADING_GAP_M)
    tm.vehicle_percentage_speed_difference(v, speed_diff_pct)
    for ignore in (tm.ignore_lights_percentage, tm.ignore_signs_percentage, tm.ignore_vehicles_percentage, tm.ignore_walkers_percentage):
        ignore(v, 0.0)


def collision_sensors(world, cars: list, events: list, ego) -> list:
    """A collision sensor on every car; each event appends (sim time, car, what it hit) to events, the
    car being "ego" or its blueprint (states.jsonl's "t" is the same clock)."""
    bp = world.get_blueprint_library().find("sensor.other.collision")
    out = []
    for car in cars:
        s = world.spawn_actor(bp, carla.Transform(), attach_to=car)
        name = "ego" if car.id == ego.id else car.type_id
        s.listen(lambda e, c=name: events.append((round(e.timestamp, 2), c, e.other_actor.type_id if e.other_actor else "?")))
        out.append(s)
    return out


def off_lane(m, actor) -> bool:
    return m.get_waypoint(actor.get_location(), project_to_road=False, lane_type=carla.LaneType.Driving) is None


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
    """Record one run; exit status 3 if its anomaly was not clearly on screen, a vehicle was thrown into the air or a
    car collided or left its lane."""
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
    cams, ego, lead, vehicles, walkers, controllers, injector, track = [], None, None, [], [], [], None, None
    sensors, crashes = [], []
    status = 0
    try:
        bp_lib, m = world.get_blueprint_library(), world.get_map()
        for sp in rng.sample(m.get_spawn_points(), len(m.get_spawn_points())):
            start = m.get_waypoint(sp.location, lane_type=carla.LaneType.Driving)
            route = S.plan_route(world, start, S.ROUTE_M, rng)
            if len(route) * 2.0 >= 0.8 * S.ROUTE_M and (ego := world.try_spawn_actor(bp_lib.find(EGO_BLUEPRINT), sp)) is not None:
                break
        driver = AgentDriver(world, m, ego, route)
        car_bps = [b for b in bp_lib.filter("vehicle.*") if b.get_attribute("number_of_wheels").as_int() == 4 and b.id not in S.NOT_IN_TRAFFIC]
        if sc["lead"] and p.kind == "nominal":   # a lead car would hide the test runs' anomalies
            k = S.route_point(route, sc["lead_gap_m"], avoid_junction=False)
            tf = route[k].transform
            lead = world.try_spawn_actor(rng.choice(car_bps), carla.Transform(tf.location + carla.Location(z=0.4), tf.rotation))
            if lead is not None:
                lead.set_autopilot(True, tm.get_port())
                tm.auto_lane_change(lead, False)
                smooth_tm(tm, lead)
                tm.vehicle_percentage_speed_difference(lead, sc["lead_speed_diff_pct"])
                tm.set_path(lead, S.path_points(route[k:]))
        track = RouteTracker(world, driver.set_route, ego, route, rng)   # it places the street furniture
        reserved = [a.get_location() for a in (ego, lead) if a is not None]
        anomaly_at = None
        if p.kind != "nominal":   # one anomaly: its junction-free spot, no traffic spawned on the way there
            anomaly_at = anomaly_spot(world, route, p.kind, sc["anomaly_at_m"], P.ANOMALIES, True)
            reserved += [w.transform.location for w in route[: S.route_point(route, anomaly_at + 20.0, avoid_junction=False) : 5]]
        traffic, walkers, controllers = spawn_traffic(world, tm, SimpleNamespace(n_vehicles=sc["n_vehicles"], n_walkers=sc["n_walkers"]), rng, reserved)
        vehicles += traffic
        for v in vehicles:
            smooth_tm(tm, v)
        if p.kind != "nominal":
            injector = Injector(world, client, p.kind, rng, anomaly_at)
        # walkers whose spawn point is on a road lane would be nominal 'pedestrian on the road': remove them
        on_road = [i for i, w in enumerate(walkers) if on_road_lane(m, w.get_location())]
        for i in reversed(on_road):
            controllers[i].stop()
            client.apply_batch_sync([carla.command.DestroyActor(controllers.pop(i)), carla.command.DestroyActor(walkers.pop(i))])
        cams = cameras(world, ego)
        sensors = collision_sensors(world, [a for a in [ego, lead] + vehicles if a is not None], crashes, ego)
        world.tick()
        for _, q in cams:   # drop the frames of the settling tick
            q.get()
        everyone = [a for a in [lead] + vehicles + walkers if a is not None]
        actors = {str(a.id & 0xFFFF): {"actor_id": a.id, "type_id": a.type_id, "base_type": a.attributes.get("base_type", "")} for a in everyone}
        (out / "run_info.json").write_text(json.dumps({
            "run_name": p.name, **dataclasses.asdict(p), **sc, "ego_id": ego.id, "lead_id": lead.id if lead else None,
            "n_vehicles_spawned": len(vehicles), "n_walkers_spawned": len(walkers), "n_walkers_removed_on_road": len(on_road),
            "n_ticks": p.n_ticks, "fixed_delta": FIXED_DELTA, "camera": {k: CAMERA[k] for k in ("width", "height", "fov")},
            "route_xy": [[round(w.transform.location.x, 1), round(w.transform.location.y, 1)] for w in route[::5]]}))

        history = GroundTruthStateHistory(max_len=4)
        flying, frozen, last, freeze, n_props = [], set(), {}, p.town in P.FREEZE_TOWNS, 0
        off_since, off_road = {}, []   # car id -> first tick off the driving lanes; (tick, car) that stayed off too long
        with open(out / "states.jsonl", "w") as fs, open(out / "camera.jsonl", "w") as fc:
            for t in range(p.n_ticks):
                replanned = track.step()
                if len(track.props) > n_props:   # new street furniture joins actors.json
                    actors.update({str(a.id & 0xFFFF): {"actor_id": a.id, "type_id": a.type_id, "base_type": ""} for a in track.props[n_props:]})
                    n_props = len(track.props)
                if injector is not None:
                    injector.step(t, ego, track, replanned)
                    if injector.cur is not None:   # injection actors join actors.json (the ground truth's instance ids)
                        actors.update({str(a.id & 0xFFFF): {"actor_id": a.id, "type_id": a.type_id, "base_type": a.attributes.get("base_type", "")}
                                       for a in injector.cur["actors"]})
                if freeze:
                    freeze_walkers(m, walkers, controllers, frozen, last)
                driver.step()
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
                for car in [a for a in [ego, lead] + vehicles if a is not None and a.is_alive]:   # the ego off its lanes on any tick, another car for P.OFF_LANE_S
                    if not off_lane(m, car):
                        off_since.pop(car.id, None)
                    elif t - off_since.setdefault(car.id, t) >= (0 if car is ego else P.OFF_LANE_S / FIXED_DELTA) and not any(c == car.id for _, c, _ in off_road):
                        off_road.append((t, car.id, "ego" if car is ego else car.type_id))
                history.push(world, ego, sim_time=world.get_snapshot().timestamp.elapsed_seconds)
                fs.write(json.dumps(history.buffer[-1]) + "\n")
                if t % 100 == 0:
                    print(f"tick {t}/{p.n_ticks}" + (f"  injections {injector.n_visible} on screen / {len(injector.done)}" if injector else "")
                          + f"  replans {track.replans}"
                          + (f"  walkers frozen {len(frozen)}" if freeze else ""), flush=True)
        (out / "actors.json").write_text(json.dumps(actors))
        info = json.loads((out / "run_info.json").read_text())
        (out / "run_info.json").write_text(json.dumps({**info, "n_walkers_frozen": len(frozen), "n_replans": track.replans}))
        if injector is not None:
            inj = injector.close(p.n_ticks)
            windows = [[r["onset"], r["offset"]] for r in inj if r["visible"]]
            ok = len(windows) >= injector.n
            (out / "anomalies.json").write_text(json.dumps({
                "anomaly_type": "semantic", "kind": p.kind, "success": ok, "n_frames": p.n_ticks, "n_injected": len(inj),
                "windows": windows, "onset_frame": windows[0][0] if windows else None, "offset_frame": windows[-1][1] if windows else None,
                "injections": inj}, indent=1))
            print(f"{p.kind}: {len(windows)} of {len(inj)} injections on screen", flush=True)
            if not ok:
                status = 3
                print(f"FAILED: the {p.kind} anomaly was not clearly on screen", flush=True)
        if crashes or off_road:
            (out / "collisions.json").write_text(json.dumps({"collisions": crashes, "off_lane": off_road}))
            status = 3
            print(f"FAILED: {len(crashes)} collision events {sorted(set((a, b) for _, a, b in crashes))[:5]}, "
                  f"{len(off_road)} cars off their lanes {off_road[:5]}", flush=True)
        if flying:
            (out / "airborne.json").write_text(json.dumps(flying))
            status = 3
            print(f"FAILED: vehicles thrown into the air: {sorted(set(v for _, v in flying))} from tick {flying[0][0]}", flush=True)
    finally:
        for c, _ in cams:
            c.stop()
        for c in sensors:
            c.stop()
        if cams:
            world.tick()
        for c in controllers:
            c.stop()
        live = injector.cur["actors"] if injector is not None and injector.cur is not None else []
        if live:   # a run that ended mid-injection: show the town's objects its scene hid
            world.enable_environment_objects(set(injector.cur["hidden"]), True)
        doomed = [c for c, _ in cams] + sensors + controllers + walkers + vehicles + (track.props if track else []) + live + [a for a in (lead, ego) if a is not None]
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
    if (d / "airborne.json").exists() or (d / "collisions.json").exists():
        return False
    a = d / "anomalies.json"
    return not a.exists() or json.loads(a.read_text()).get("success", False)


def gt_windows(name: str, a: dict) -> list | None:
    """Each injection's window from the ground truth (concepts_gt_raw.jsonl): first to last tick in which the ground
    truth sees any of its actors (the pedestrian: on the road), so no frame naming the anomaly counts as nominal; at
    least P.GT_WINDOW_MIN_TICKS of them. None before extract_gt_concepts.py has run."""
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
        if len(ticks) >= P.GT_WINDOW_MIN_TICKS:
            out.append([ticks[0], ticks[-1] + 1])
    return out


def write_configs() -> None:
    """config/data_semantic24.json: the withheld-routes pool, the held-out nominal run and the anomaly runs with their
    windows (score.py and the dashboard read the windows; the pools themselves come from sim/semantic24.plan())."""
    # paths relative to multimodal-ood-detection/, like its own data configs, so its scoring and dashboard code read them as-is
    entry = lambda name: {"run_name": name, "data_path": f"aesop_recreate/data/{P.DATASET}/{name}", "out_path": f"aesop_recreate/data/{P.DATASET}/{name}/embeddings"}  # noqa: E731
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
    path = f"data_{P.DATASET}.json"
    cfg = {"reference_runs": [entry(p.name) for p in runs if "withheld" in p.pools], "negative_runs": [entry("t2_test_nominal")],
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
