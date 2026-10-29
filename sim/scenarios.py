"""Randomised scenarios for batch collection (scripts/run_scenario.py, scripts/run_batch.py).

Every run samples, from its seed alone: town, weather, traffic density, the ego's spawn point and route, and whether
a lead vehicle shares the ego's route from the first tick. Anomaly runs also sample one anomaly. Nominal and anomaly
runs draw these nominal factors from the same distribution, so the only systematic difference is the anomaly.

Nothing appears mid-run:
  semantic  objects / vehicles / pedestrians are placed on the ego's planned route before the first tick; the
            anomaly window is when they are actually on screen (instance-segmentation pixel count)
  physics   an agent that has been driving since the start (the route-sharing lead, an adjacent-lane car, an
            oncoming car, or any visible car) is taken over at a random tick for a swerve / hard brake / drift
  visual    a camera corruption over a random window, applied to the recorded frames afterwards
A run whose anomaly never happens (no suitable agent, object never on screen) is marked failed.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

TOWNS = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD"]
# daylight and dusk, dry to moderate rain; night / heavy rain / dust storm leave the camera too dark or occluded
WEATHERS = ["ClearNoon", "CloudyNoon", "WetNoon", "WetCloudyNoon", "SoftRainNoon", "MidRainyNoon",
            "ClearSunset", "CloudySunset", "WetSunset", "WetCloudySunset", "SoftRainSunset"]
MULTI_LANE_TOWNS = ["Town03", "Town04", "Town05", "Town10HD"]   # have roads with 2+ lanes per direction (cut-ins need them)
VEHICLES = (40, 80)          # background vehicles, uniform
WALKERS = (20, 60)           # background pedestrians, uniform
LEAD_PROB = 0.6              # a vehicle shares the ego's route from tick 0 (nominal and anomaly runs alike)
LEAD_GAP_M = (12.0, 25.0)
ROUTE_M = 1500.0

# emergency vehicles are a semantic anomaly, so they never appear in background traffic
EMERGENCY = ["vehicle.carlamotors.firetruck", "vehicle.ford.ambulance", "vehicle.dodge.charger_police_2020"]
NOT_IN_TRAFFIC = set(EMERGENCY) | {"vehicle.dodge.charger_police"}

SEMANTIC_KINDS = {
    # kind: (blueprints, count range, spread along the road in m, lateral band in m from the lane edge: - = in lane)
    "debris": (["static.prop.box01", "static.prop.box02", "static.prop.box03", "static.prop.creasedbox01", "static.prop.creasedbox02",
                "static.prop.trashbag", "static.prop.plasticbag", "static.prop.travelcase", "static.prop.dirtdebris01",
                "static.prop.dirtdebris02", "static.prop.brokentile01", "static.prop.garbage01", "static.prop.garbage03"], (3, 6), 8.0, (-0.6, 0.2)),
    "construction": (["static.prop.constructioncone", "static.prop.trafficcone01", "static.prop.trafficcone02"], (4, 7), 14.0, (-0.6, 0.0)),
    "out_of_place": (["static.prop.haybale", "static.prop.shoppingcart", "static.prop.trampoline", "static.prop.swingcouch",
                      "static.prop.vendingmachine", "static.prop.doghouse", "static.prop.barbeque", "static.prop.slide",
                      "static.prop.foodcart", "static.prop.kiosk_01", "static.prop.plastictable"], (1, 2), 6.0, (-0.4, 0.4)),
    "emergency": (EMERGENCY, (1, 1), 0.0, (-0.5, 0.0)),
    "crowd": (["walker.pedestrian.*"], (12, 20), 6.0, (-0.6, 2.7)),   # spills off the kerb; placed on a 0.9 m grid, see run_scenario
}
CONSTRUCTION_EXTRAS = ["static.prop.streetbarrier", "static.prop.warningconstruction", "static.prop.trafficwarning"]
SEMANTIC_AT_M = (45.0, 110.0)     # route distance from the ego's start to the anomaly
MIN_VISIBLE_PX = 1000           # pixels of the anomaly actors for a frame to count as showing it (~32x32)
MIN_VISIBLE_TICKS = 15

PHYSICS_KINDS = ("lead_swerve", "lead_brake", "cut_in", "oncoming_drift", "random_swerve")
PHYSICS_TRIGGER = (80, 220)     # earliest tick to start trying, uniform; from then on it keeps trying until it works
CUT_IN_TRIGGER = (40, 120)      # cut_in starts earlier: its neighbour-lane companion is reliably alongside early on
PHYSICS_GIVE_UP = 540           # no successful takeover by then: the run is failed
MIN_AGENT_PX = 800              # the taken-over agent must be clearly on screen (a car at ~20 m)
MIN_AGENT_SPEED = 3.0           # m/s: only agents in flowing traffic (a car queued at a light cannot cut in)
MIN_EGO_SPEED = 2.0             # m/s
RETRY_COOLDOWN = 10             # ticks between a failed attempt and the next search
RETRY_SAME_AGENT_AFTER = 50     # ticks before a failed agent may be picked again
TAKEOVER_MAX_TICKS = 70

VISUAL_KINDS = ("flare", "blur", "brightness", "noise", "dirty_camera")
VISUAL_ONSET = (180, 380)
VISUAL_DURATION = (100, 200)


@dataclass
class Scenario:
    seed: int
    kind: str                         # nominal / semantic / physics / visual
    town: str
    weather: str
    n_vehicles: int
    n_walkers: int
    lead: bool
    lead_gap_m: float
    lead_speed_diff_pct: float        # Traffic Manager speed difference (+ = slower than the limit), so the ego catches up
    anomaly: dict = field(default_factory=dict)


def sample(kind: str, seed: int, subtype: str | None = None) -> Scenario:
    """Everything random about a run, from its seed (subtype pins the anomaly kind)."""
    rng = random.Random(f"{kind}:{seed}")
    sc = Scenario(seed, kind, rng.choice(TOWNS), rng.choice(WEATHERS), rng.randint(*VEHICLES), rng.randint(*WALKERS),
                  rng.random() < LEAD_PROB, round(rng.uniform(*LEAD_GAP_M), 1), round(rng.uniform(5, 25), 1))
    if kind == "semantic":
        k = subtype or rng.choice(list(SEMANTIC_KINDS))
        sc.anomaly = {"kind": k, "at_m": round(rng.uniform(*SEMANTIC_AT_M), 1)}
    elif kind == "physics":
        k = subtype or rng.choice(PHYSICS_KINDS)
        if k.startswith("lead"):
            sc.lead = True   # these kinds need the route-sharing lead
        if k == "cut_in":    # needs a same-direction neighbour lane, and a car in it (placed by run_scenario)
            sc.town = rng.choice(MULTI_LANE_TOWNS)
        sc.anomaly = {"kind": k, "trigger_tick": rng.randint(*PHYSICS_TRIGGER), "strength": round(rng.uniform(0.6, 1.0), 2),
                      "sign": rng.choice((-1.0, 1.0)),   # swerve direction for the swerve kinds
                      "companion_gap_m": round(rng.uniform(5.0, 15.0), 1),   # cut_in: neighbour-lane car this far ahead at tick 0
                      "oncoming_gap_m": round(rng.uniform(120.0, 220.0), 1)}  # oncoming_drift: opposite-lane car this far up the route
        if k == "cut_in":
            sc.anomaly["trigger_tick"] = rng.randint(*CUT_IN_TRIGGER)
    elif kind == "visual":
        k = subtype or rng.choice(VISUAL_KINDS)
        onset = rng.randint(*VISUAL_ONSET)
        sc.anomaly = {"kind": k, "onset_frame": onset, "offset_frame": onset + rng.randint(*VISUAL_DURATION),
                      "strength": round(rng.uniform(0.6, 1.0), 2)}
    return sc


# --- CARLA-side helpers (imported lazily so sample() works without the carla package) -----------------------

def plan_route(world, start_wp, length_m: float, rng: random.Random, step_m: float = 2.0) -> list:
    """Random walk over the lane graph: a waypoint every step_m, a random branch at every junction."""
    route, travelled = [start_wp], 0.0
    while travelled < length_m:
        nxt = route[-1].next(step_m)
        if not nxt:
            break
        route.append(rng.choice(nxt))
        travelled += step_m
    return route


def adjacent_lane(wp):
    """The neighbouring driving lane in the same direction (right preferred), or None."""
    import carla

    for side in (wp.get_right_lane(), wp.get_left_lane()):
        if side is not None and side.lane_type == carla.LaneType.Driving and side.lane_id * wp.lane_id > 0:
            return side
    return None


def opposite_lane(wp):
    """The nearest lane to the left that runs the other way (right-hand traffic), or None."""
    import carla

    lane = wp.get_left_lane()
    for _ in range(4):
        if lane is None or lane.lane_type != carla.LaneType.Driving:
            return None
        if lane.lane_id * wp.lane_id < 0:
            return lane
        lane = lane.get_left_lane()
    return None


def oncoming_route(route: list) -> list:
    """The route driven backwards in the opposite lane (outside junctions), for a car coming towards the ego."""
    out = []
    for w in reversed(route):
        o = opposite_lane(w) if not w.is_junction else None
        if o is not None:
            out.append(o)
    return out


def companion_route(route: list) -> list:
    """The route shifted one lane over wherever a same-direction neighbour lane exists (outside junctions)."""
    out = []
    for w in route:
        a = adjacent_lane(w) if not w.is_junction else None
        if a is not None:
            out.append(a)
    return out


def path_points(route: list, spacing: int = 5) -> list:
    """Traffic Manager path for a route: every spacing-th waypoint outside junctions, plus the first one after each
    junction. Points inside a junction sit where turn lanes overlap, and the Traffic Manager may resolve them to the
    wrong branch."""
    pts, last = [], -spacing
    for i, w in enumerate(route[1:], 1):
        exit_point = route[i - 1].is_junction and not w.is_junction
        if not w.is_junction and (i - last >= spacing or exit_point):
            pts.append(w.transform.location)
            last = i
    return pts


def route_point(route: list, distance_m: float, step_m: float = 2.0, avoid_junction: bool = True):
    """Index of the route waypoint at about distance_m, moved forward past any junction."""
    i = min(int(distance_m / step_m), len(route) - 1)
    while avoid_junction and i < len(route) - 1 and (route[i].is_junction or any(w.is_junction for w in route[i : i + 8])):
        i += 1
    return i


def offset_location(wp, lateral_m: float, along_m: float = 0.0):
    """Location lateral_m to the right of wp's lane centre (negative = left), along_m further along its heading."""
    import carla

    tf = wp.transform
    f, r = tf.get_forward_vector(), tf.get_right_vector()
    loc = tf.location
    return carla.Location(loc.x + f.x * along_m + r.x * lateral_m, loc.y + f.y * along_m + r.y * lateral_m, loc.z)


def ego_frame(ego_tf, loc) -> tuple[float, float]:
    """(ahead, right) of loc in the ego's frame."""
    f, r = ego_tf.get_forward_vector(), ego_tf.get_right_vector()
    d = loc - ego_tf.location
    return d.x * f.x + d.y * f.y, d.x * r.x + d.y * r.y


def instance_ids(image) -> np.ndarray:
    """Actor id per pixel of an instance-segmentation image (G + 256 B)."""
    a = np.frombuffer(image.raw_data, np.uint8).reshape(image.height, image.width, 4)
    return a[..., 1].astype(np.int32) + a[..., 0].astype(np.int32) * 256


def visible_px(ids: np.ndarray, actor_ids) -> int:
    return int(np.isin(ids, list(actor_ids)).sum()) if actor_ids else 0


# --- physics anomalies ------------------------------------------------------------------------------------

def physics_candidates(world, ego, ids: np.ndarray, lead_id: int | None, exclude: set[int] = frozenset()) -> dict[str, list]:
    """Visible, moving vehicles outside junctions that could carry out each physics kind right now, nearest first."""
    import carla

    tf, m = ego.get_transform(), world.get_map()
    out: dict[str, list] = {k: [] for k in PHYSICS_KINDS}
    counts = dict(zip(*np.unique(ids, return_counts=True)))
    for v in world.get_actors().filter("vehicle.*"):
        if v.id == ego.id or v.id in exclude or counts.get(v.id, 0) < MIN_AGENT_PX or v.get_velocity().length() < MIN_AGENT_SPEED:
            continue
        ahead, right = ego_frame(tf, v.get_location())
        if ahead <= 3.0:
            continue
        wp = m.get_waypoint(v.get_location(), lane_type=carla.LaneType.Driving)
        if wp is None or wp.is_junction:
            continue
        rel_yaw = math.radians(v.get_transform().rotation.yaw - tf.rotation.yaw)
        same_dir, opposite = math.cos(rel_yaw) > 0.9, math.cos(rel_yaw) < -0.9
        dist = math.hypot(ahead, right)
        if v.id == lead_id and same_dir and abs(right) < 1.8 and 6.0 < ahead < 25.0:
            out["lead_swerve"].append((dist, v))
            out["lead_brake"].append((dist, v))
        if same_dir and 2.0 <= abs(right) <= 5.0 and 3.0 < ahead < 20.0:   # the next lane over, just ahead of the ego
            out["cut_in"].append((dist, v))
        if opposite and 8.0 < ahead < 35.0 and 2.0 < abs(right) < 9.0:
            out["oncoming_drift"].append((dist, v))
        if dist < 25.0 and (same_dir or opposite):
            out["random_swerve"].append((dist, v))
    return {k: [v for _, v in sorted(c, key=lambda x: x[0])] for k, c in out.items()}


class PhysicsTakeover:
    """Closed-loop control of one agent: pure-pursuit steering towards a target lateral line (in the agent's frame
    at the start of the takeover) while holding its speed, or hard braking. Each kind has a success test; on
    success or after TAKEOVER_MAX_TICKS the agent goes back to autopilot.

      cut_in          move into the ego's lane ahead of it and stay there (within 1.0 m of its centre for 3 ticks)
      oncoming_drift  drift over the centre line into the ego's lane (within 1.8 m of its centre), then back
      lead_swerve,
      random_swerve   swerve 2-3.5 m sideways, then back (success: >= 1.5 m excursion)
      lead_brake      brake hard (success: speed drops by >= max(3 m/s, 50%))
    """

    WHEELBASE = 2.8

    def __init__(self, kind: str, actor, ego, tm, strength: float, sign: float):
        self.kind, self.actor, self.ego, self.tm, self.strength = kind, actor, ego, tm, strength
        tf = actor.get_transform()
        self.origin, self.fwd, self.right = tf.location, tf.get_forward_vector(), tf.get_right_vector()
        self.v0 = actor.get_velocity().length()
        self.phase, self.ticks, self.phase_ticks, self.in_target = "out", 0, 0, 0
        self.swing = sign * (2.0 + 1.5 * strength)
        self.max_steer = math.radians(max(w.max_steer_angle for w in actor.get_physics_control().wheels) or 70.0)
        self.success, self.max_lat, self.min_gap_to_ego_lane, self.min_speed = False, 0.0, float("inf"), self.v0
        tm.auto_lane_change(actor, False)
        actor.set_autopilot(False, tm.get_port())

    def _local(self, loc) -> tuple[float, float]:
        d = loc - self.origin
        return d.x * self.fwd.x + d.y * self.fwd.y, d.x * self.right.x + d.y * self.right.y

    def _ego_lane_point(self, extra_m: float = 0.0):
        """The ego's lane centreline (from the map, so it follows curves) level with the agent, plus extra_m further on."""
        from sim import placement

        wp = placement.ego_lane_waypoint(self.ego.get_world(), self.ego)
        ahead = ego_frame(self.ego.get_transform(), self.actor.get_location())[0]
        pt = placement.lane_point_ahead(wp, max(0.5, ahead + extra_m)) if wp is not None else None
        return pt.transform.location if pt is not None else None

    def _ego_lane_gap(self) -> float:
        """Distance from the agent to the ego's lane centreline."""
        pt = self._ego_lane_point()
        return self.actor.get_location().distance(pt) if pt is not None else float("inf")

    def _control(self, target_y: float | None = None, brake_only: bool = False, target=None):
        """Pure pursuit to `target` (a world point), or to the line target_y in the agent's start frame; speed held."""
        import carla

        v = self.actor.get_velocity().length()
        if brake_only:
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.6 + 0.4 * self.strength)
        look = max(6.0, 1.2 * v)
        if target is None:
            x, _ = self._local(self.actor.get_location())
            target = self.origin + self.fwd * (x + look) + self.right * target_y
        tf = self.actor.get_transform()
        to = target - tf.location
        heading = math.atan2(tf.get_forward_vector().y, tf.get_forward_vector().x)
        alpha = math.atan2(to.y, to.x) - heading
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))
        steer = math.atan2(2 * self.WHEELBASE * math.sin(alpha), look) / self.max_steer
        speed_err = max(self.v0, 4.0) - v
        return carla.VehicleControl(throttle=float(np.clip(0.35 + 0.4 * speed_err, 0, 1)), steer=float(np.clip(steer, -1, 1)),
                                    brake=float(np.clip(-0.3 * (speed_err + 1.0), 0, 1)))

    def step(self) -> bool:
        """Apply this tick's control; False once finished (success or timeout; autopilot restored)."""
        self.ticks += 1
        self.phase_ticks += 1
        _, y = self._local(self.actor.get_location())
        v = self.actor.get_velocity().length()
        self.max_lat, self.min_speed = max(self.max_lat, abs(y)), min(self.min_speed, v)
        done = self.ticks >= TAKEOVER_MAX_TICKS
        if self.kind == "lead_brake":
            control = self._control(0.0, brake_only=True)
            if self.v0 - v >= max(3.0, 0.5 * self.v0):
                self.success, done = True, True
        elif self.kind == "cut_in":
            gap = self._ego_lane_gap()
            ahead_of_ego = ego_frame(self.ego.get_transform(), self.actor.get_location())[0] > 2.0
            self.min_gap_to_ego_lane = min(self.min_gap_to_ego_lane, gap)
            self.in_target = self.in_target + 1 if gap < 1.0 and ahead_of_ego else 0
            control = self._control(target=self._ego_lane_point(max(6.0, 1.2 * v)) or self.actor.get_location())
            if self.in_target >= 3:
                self.success, done = True, True   # handed back in the ego's lane; lane changes stay off
        else:
            if self.kind == "oncoming_drift":   # head for the ego's lane (the point level with it: it is coming towards us)
                gap = self._ego_lane_gap()
                self.min_gap_to_ego_lane = min(self.min_gap_to_ego_lane, gap)
                reached = gap < 1.8
                out = {"target": self._ego_lane_point(-max(6.0, 1.2 * v)) or self.actor.get_location()}
            else:
                reached = abs(y - self.swing) < 0.5
                out = {"target_y": self.swing}
            if self.phase == "out" and (reached or self.phase_ticks > 30):
                self.phase, self.phase_ticks = "back", 0
            control = self._control(**out) if self.phase == "out" else self._control(0.0)
            if self.phase == "back" and (abs(y) < 0.5 or self.phase_ticks > 30):
                done = True
            self.success = self.max_lat >= 1.5 if self.kind != "oncoming_drift" else self.min_gap_to_ego_lane < 1.8
        if done:
            self.actor.set_autopilot(True, self.tm.get_port())
            return False
        self.actor.apply_control(control)
        return True

    def summary(self) -> dict:
        return {"max_lateral_m": round(self.max_lat, 2), "speed_start": round(self.v0, 2), "speed_min": round(self.min_speed, 2),
                "min_gap_to_ego_lane_m": None if math.isinf(self.min_gap_to_ego_lane) else round(self.min_gap_to_ego_lane, 2),
                "ticks": self.ticks, "success": self.success}


# --- visual anomalies (on recorded frames) ----------------------------------------------------------------

def corrupt(img: np.ndarray, kind: str, strength: float, rng: np.random.RandomState, state: dict) -> np.ndarray:
    """One corrupted frame; `state` keeps per-run choices (flare position, occluder mask) fixed over the window."""
    from PIL import Image, ImageFilter

    h, w = img.shape[:2]
    x = img.astype(np.float32)
    if kind == "flare":
        if "c" not in state:
            state["c"] = (w * rng.uniform(0.25, 0.75), h * rng.uniform(0.1, 0.45))
        yy, xx = np.mgrid[0:h, 0:w]
        glow = np.clip(1 - np.hypot(xx - state["c"][0], yy - state["c"][1]) / (0.55 * min(h, w)), 0, 1) ** 2
        x = x + (0.5 + 0.4 * strength) * 255 * glow[..., None]
    elif kind == "blur":
        return np.array(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=3 + 7 * strength)))
    elif kind == "brightness":
        x = x * (1.5 + 1.0 * strength)
    elif kind == "noise":
        x = x + rng.normal(0, 25 + 35 * strength, x.shape)
    elif kind == "dirty_camera":   # mud / droplets on the lens: soft dark blobs that stay put
        if "mask" not in state:
            yy, xx = np.mgrid[0:h, 0:w]
            m = np.zeros((h, w), np.float32)
            for _ in range(rng.randint(4, 9)):
                cx, cy, r = rng.uniform(0, w), rng.uniform(0.3 * h, h), rng.uniform(0.06, 0.16) * w * (0.6 + strength)
                m = np.maximum(m, np.clip(1 - np.hypot(xx - cx, yy - cy) / r, 0, 1) ** 0.6)
            state["mask"] = m[..., None]
        x = x * (1 - 0.9 * state["mask"]) + 40 * state["mask"]
    return np.clip(x, 0, 255).astype(np.uint8)
