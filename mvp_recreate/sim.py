"""The scene and the dataset plans: single snapshots at an empty 4-way (Town04 junction 916, one lane each way,
crosswalks on all four sides). The ego is parked before the junction on road 52 (config.json: scene).

Every observation is ONE camera frame of the ego and at most ONE other agent, plus that agent's last `history` states,
built by assuming constant velocity with the heading along its path (no simulated history). Two datasets, every frame
deterministic (config.json: cyclist, pedestrian, droplet):

  cyclist     nominal: a cyclist close to the sidewalk (curb_offset_m right of its lane centre), at `speed`, in its
                lane's direction
                - each of the 9 `paths` (the three other approaches - oncoming, the crossing lane going right, the
                  crossing lane going left - x straight / right turn / left turn): path_snaps snapshots evenly along it.
                  A turn (one_turn) is straight on in the approach lane, ONE smooth turn where it meets the exit
                  lane's line (for left turns: at the far side, alongside the crosswalk ahead), then straight on.
                - our own lane: ego_snaps snapshots from the opposite crosswalk on, riding away in our direction
                - each turning path again: turn_snaps snapshots evenly inside the turn (arc start to `history` ticks past
                  its end), so a frame whose history sweeps through the curve has neighbours doing the same (appended
                  here so the earlier indices, and the droplet's source, stay put)
              wrongway (1): the cyclist at ego-lane snapshot wrong_at's place, riding towards us (physics anomaly)
              droplet (1): that same nominal snapshot with len(droplet.radii_px) blurred blobs on the lens
                           (vision anomaly; record.py makes it from the recorded frame)
  pedestrian  nominal: each of `bps` at `positions` positions evenly along the crosswalk nearest the ego x `orients` (standing
              facing the ego / walking right / walking left along it at `speed`)
              conductor (1): conductor_bp (a policeman, 0030) standing facing the ego at conductor_t (semantic anomaly)
              droplet (1): nominal droplet.pedestrian_src with blurred blobs on the lens, as the cyclist's (vision anomaly)

The cyclist is kinematic (physics off, placed); the pedestrian is a CARLA walker (WalkerControl for settle_ticks ticks
so its pose is a standing / walking one); its last state is its actual pose.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

import carla

from cfg import CACHES, CFG, SHIFTED  # noqa: F401  (re-exported for record / encode / score / the dashboard)

_SC, _CY, _PE, _DR = CFG["scene"], CFG["cyclist"], CFG["pedestrian"], CFG["droplet"]
TOWN, JUNCTION = _SC["town"], _SC["junction"]
EGO_ROAD, EGO_LANE, EGO_BACK_M, EGO_STRAIGHT, EGO_BP = _SC["ego_road"], _SC["ego_lane"], _SC["ego_back_m"], _SC["ego_straight"], _SC["ego_bp"]
CAMERA = _SC["camera"]
FIXED_DELTA, SETTLE_TICKS, VISIBLE_PX = _SC["fixed_delta"], _SC["settle_ticks"], _SC["visible_px"]
HISTORY = PHYS_HISTORY = CFG["history"]
STEP_M = _SC["route_step_m"]
BIKE_BP, CONDUCTOR_BP = _CY["bp"], _PE["conductor_bp"]
DROP_FROM = {"cyclist": ("cyclist", "nominal", len(_CY["paths"]) * _CY["path_snaps"] + _CY["wrong_at"]),   # each droplet frame's source
             "pedestrian": ("pedestrian", "nominal", _DR["pedestrian_src"])}


# --- geometry helpers ------------------------------------------------------------------------------------

@dataclass
class Route:
    """A lane route through the junction as a dense polyline (x, y, z, yaw) with arc length s."""
    name: str
    in_road: int
    in_lane: int
    out_road: int
    out_lane: int
    turn: str                        # straight / left / right
    xyz: np.ndarray                  # (N, 3)
    yaw: np.ndarray                  # (N,) degrees
    s: np.ndarray                    # (N,) arc length
    junction_s: tuple = (0.0, 0.0)   # arc span inside the junction

    def at(self, s: float):
        """(x, y, z, yaw_deg) at arc length s (clamped)."""
        s = float(np.clip(s, 0.0, self.s[-1]))
        i = min(int(np.searchsorted(self.s, s, side="right") - 1), len(self.s) - 2)
        f = (s - self.s[i]) / max(self.s[i + 1] - self.s[i], 1e-6)
        p = self.xyz[i] * (1 - f) + self.xyz[i + 1] * f
        dy = (self.yaw[i + 1] - self.yaw[i] + 180) % 360 - 180
        return p[0], p[1], p[2], self.yaw[i] + f * dy

    @property
    def length(self) -> float:
        return float(self.s[-1])


def in_poly(x: float, y: float, poly: np.ndarray) -> bool:
    inside, n = False, len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


def _chain(wp, dist: float, back: bool) -> list:
    """Waypoints every STEP_M for dist metres before (back) or after wp, staying on its road where possible."""
    out, cur = [], wp
    for _ in range(int(dist / STEP_M)):
        nxt = cur.previous(STEP_M) if back else cur.next(STEP_M)
        if not nxt:
            break
        cur = ([w for w in nxt if w.road_id == cur.road_id] or nxt)[0]
        out.append(cur)
    return out[::-1] if back else out


def _to_sidewalk(m, p: np.ndarray, d: np.ndarray, max_m: float = 10.0) -> np.ndarray:
    """p moved along d in 0.25 m steps until it is on a Sidewalk lane."""
    for k in range(int(max_m / 0.25) + 1):
        q = p + d * 0.25 * k
        wp = m.get_waypoint(carla.Location(float(q[0]), float(q[1]), 0.0), project_to_road=False, lane_type=carla.LaneType.Any)
        if wp is not None and wp.lane_type == carla.LaneType.Sidewalk:
            return q
    return p


# --- world -----------------------------------------------------------------------------------------------

def weather() -> carla.WeatherParameters:
    w = carla.WeatherParameters()
    for k, v in _SC["weather"].items():
        setattr(w, k, v)
    return w


def strip_world(world) -> int:
    """Hide every environment object except roads, road lines, sidewalks and ground (incl. Terrain: the hedges)."""
    ids = set()
    for name, lab in carla.CityObjectLabel.names.items():
        if name not in _SC["keep_labels"]:
            ids |= {o.id for o in world.get_environment_objects(lab)}
    world.enable_environment_objects(ids, False)
    return len(ids)


def junction(m):
    return next(wp.get_junction() for a, b in m.get_topology() for wp in (a, b) if wp.is_junction and wp.get_junction().id == JUNCTION)


def ego_transform(m) -> carla.Transform:
    """EGO_BACK_M before our lane's junction entry along the entry heading, facing it (road 52 curves behind: the lane
    itself turns away)."""
    ent = next(st for st, _ in junction(m).get_waypoints(carla.LaneType.Driving)
               if st.previous(1.0)[0].road_id == EGO_ROAD and st.previous(1.0)[0].lane_id == EGO_LANE).transform
    f = ent.get_forward_vector()
    x, y = ent.location.x - f.x * EGO_BACK_M, ent.location.y - f.y * EGO_BACK_M
    z = m.get_waypoint(carla.Location(x, y, ent.location.z)).transform.location.z
    return carla.Transform(carla.Location(x, y, z), carla.Rotation(yaw=ent.rotation.yaw))


def routes(m) -> dict[str, Route]:
    """Every lane connection through the junction (route_back_m in, the connector, route_ahead_m out), by name in:lane->out:lane."""
    out = {}
    for st, en in junction(m).get_waypoints(carla.LaneType.Driving):
        p, n = st.previous(1.0)[0], en.next(1.0)[0]
        conn, cur = [st], st
        for _ in range(400):
            if cur.transform.location.distance(en.transform.location) < STEP_M:
                break
            cur = cur.next(STEP_M)[0]
            conn.append(cur)
        back = _chain(st, _SC["route_back_m"], True)
        wps = back + conn + _chain(en, _SC["route_ahead_m"], False)
        xyz = np.array([[w.transform.location.x, w.transform.location.y, w.transform.location.z] for w in wps])
        yaw = np.array([w.transform.rotation.yaw for w in wps])
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1))])
        dyaw = (en.transform.rotation.yaw - st.transform.rotation.yaw + 180) % 360 - 180
        turn = "straight" if abs(dyaw) < 30 else ("right" if dyaw > 0 else "left")   # CARLA yaw is clockwise
        name = f"{p.road_id}:{p.lane_id}->{n.road_id}:{n.lane_id}"
        j0, j1 = float(s[len(back)]), float(s[len(back) + len(conn) - 1])
        out[name] = Route(name, p.road_id, p.lane_id, n.road_id, n.lane_id, turn, xyz, yaw, s, (j0, j1))
    return out


def reversed_route(r: Route) -> Route:
    xyz, yaw = r.xyz[::-1].copy(), (r.yaw[::-1] + 180.0) % 360.0
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1))])
    L = r.s[-1]
    return Route(r.name + ":reversed", r.out_road, r.out_lane, r.in_road, r.in_lane, r.turn, xyz, yaw, s, (L - r.junction_s[1], L - r.junction_s[0]))


def crosswalks(m) -> list[np.ndarray]:
    """The crosswalks near the junction as (4, 2) rectangles. Map.get_crosswalks() polygons stop 1-2 m short of the
    sidewalk, so each is stretched along its axis to the sidewalk on both sides."""
    pts, polys, cur = m.get_crosswalks(), [], []
    for p in pts:
        cur.append((p.x, p.y))
        if len(cur) > 2 and math.hypot(cur[0][0] - p.x, cur[0][1] - p.y) < 1e-3:
            polys.append(np.array(cur[:-1]))
            cur = []
    c = junction(m).bounding_box.location
    out = []
    for q in polys:
        if np.hypot(*(q.mean(0) - [c.x, c.y])) >= _SC["crosswalk_radius_m"]:
            continue
        edges = [(q[i], q[(i + 1) % len(q)]) for i in range(len(q))]
        short = sorted(edges, key=lambda e: np.linalg.norm(e[1] - e[0]))[:2]
        a, b = [(e[0] + e[1]) / 2 for e in short]
        half = np.mean([np.linalg.norm(e[1] - e[0]) for e in short]) / 2
        u = (b - a) / np.linalg.norm(b - a)
        n = np.array([-u[1], u[0]])
        a, b = _to_sidewalk(m, a, -u), _to_sidewalk(m, b, u)
        out.append(np.array([a + n * half, b + n * half, b - n * half, a - n * half]))
    return out


def _cross2(u: np.ndarray, v: np.ndarray) -> float:
    return float(u[0] * v[1] - u[1] * v[0])


def one_turn(r: Route) -> Route:
    """Turn r as straight on, then ONE smooth turn, then straight, curb_offset_m right
    of the lane centres. The cyclist rides straight on in its approach lane until it meets the line of its exit lane,
    turns there on a fillet of radius turn_radius_m, and rides straight on in the exit lane. For a left turn that
    meeting point is at the far side of the junction, alongside the crosswalk straight ahead, so no turn goes
    through the middle of the junction. junction_s spans from the junction entry to the junction exit."""
    off, R = _CY["curb_offset_m"], _CY["turn_radius_m"]

    def lane(s: float) -> tuple[np.ndarray, np.ndarray]:
        x, y, z, yaw = r.at(s)
        a = math.radians(yaw)
        return np.array([x - off * math.sin(a), y + off * math.cos(a), z]), np.array([math.cos(a), math.sin(a)])

    j0, j1 = r.junction_s
    p1, d1 = lane(j0)                                                       # where the approach lane enters
    p2, d2 = lane(j1)                                                       # where the exit lane leaves
    x = p1[:2] + d1 * (_cross2(p2[:2] - p1[:2], d2) / _cross2(d1, d2))     # the two lane lines meet
    T = R * math.tan(math.acos(float(np.clip(d1 @ d2, -1.0, 1.0))) / 2)
    a0, a1 = x - T * d1, x + T * d2
    z = float(p1[2])
    arc = [[*((1 - t) ** 2 * a0 + 2 * (1 - t) * t * x + t ** 2 * a1), z] for t in np.linspace(0, 1, 25)]
    raw = np.vstack([np.array([lane(s)[0] for s in np.arange(0.0, j0, STEP_M)] + [p1]), np.array(arc),
                     np.array([p2] + [lane(s)[0] for s in np.arange(j1 + STEP_M, r.length, STEP_M)])])
    raw = raw[np.concatenate([[True], np.linalg.norm(np.diff(raw[:, :2], axis=0), axis=1) > 1e-6])]
    s_raw = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(raw[:, :2], axis=0), axis=1))])
    s_new = np.arange(0.0, s_raw[-1], 0.25)
    xyz = np.stack([np.interp(s_new, s_raw, raw[:, d]) for d in range(3)], 1)
    xy = xyz[:, :2]
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    g = np.gradient(xy, axis=0)
    yaw = np.degrees(np.arctan2(g[:, 1], g[:, 0]))
    s0 = s[int(np.argmin(np.linalg.norm(xy - p1[:2], axis=1)))]
    s1 = s[int(np.argmin(np.linalg.norm(xy - p2[:2], axis=1)))]
    return Route(r.name + ":turn", r.in_road, r.in_lane, r.out_road, r.out_lane, r.turn, xyz, yaw, s, (float(s0), float(s1)))

def turn_arc(r: Route, min_deg_per_m: float = 2.0) -> tuple[float, float]:
    """(start s, end s) of the turn of route r: where, inside its junction span, its heading turns faster than
    min_deg_per_m (the roads themselves bend further out)."""
    ds = np.diff(r.s)
    rate = np.abs((np.diff(r.yaw) + 180) % 360 - 180) / np.maximum(ds, 1e-6)
    i = np.nonzero((rate > min_deg_per_m) & (r.s[:-1] >= r.junction_s[0]) & (r.s[1:] <= r.junction_s[1]))[0]
    return float(r.s[i[0]]), float(r.s[i[-1] + 1])


def predicate(m, cws: list[np.ndarray], x: float, y: float, z: float) -> str:
    """Where a ground point is: on the crosswalk / in the intersection / on the road / on the sidewalk / beside the road."""
    if any(in_poly(x, y, q) for q in cws):
        return "on the crosswalk"
    wp = m.get_waypoint(carla.Location(x, y, z), project_to_road=False, lane_type=carla.LaneType.Any)
    if wp is None:      # the map leaves unregistered strips at lane edges: asphalt within 1 m of a driving lane is road
        near = m.get_waypoint(carla.Location(x, y, z), project_to_road=True, lane_type=carla.LaneType.Driving)
        loc = near.transform.location if near else None
        if near is None or math.hypot(x - loc.x, y - loc.y) > near.lane_width / 2 + 1.0:
            return "beside the road"
        wp = near
    if wp.lane_type == carla.LaneType.Sidewalk:
        return "on the sidewalk"
    if wp.lane_type in (carla.LaneType.Driving, carla.LaneType.Bidirectional, carla.LaneType.Shoulder, carla.LaneType.Parking):
        return "in the intersection" if wp.is_junction and wp.get_junction().id == JUNCTION else "on the road"
    return "beside the road"


class Geometry:
    """Everything map-derived the plans need, computed once per process."""

    def __init__(self, m):
        self.m = m
        self.ego_tf = ego_transform(m)
        e = self.ego_tf
        f, r = e.get_forward_vector(), e.get_right_vector()
        self.cam = np.array([e.location.x + f.x * CAMERA["x"], e.location.y + f.y * CAMERA["x"]])
        self.fwd, self.right = np.array([f.x, f.y]), np.array([r.x, r.y])
        self.routes = routes(m)
        self.wrong = reversed_route(self.routes[EGO_STRAIGHT])
        self.cws = crosswalks(m)
        self.turn_routes, self.turn_of = {}, {}   # the turning paths as ridden (one_turn), by name
        for n in _CY["paths"]:
            if self.routes[n].turn != "straight":
                tr = one_turn(self.routes[n])
                self.turn_routes[tr.name], self.turn_of[n] = tr, tr.name

    def ego_frame(self, x: float, y: float) -> tuple[float, float]:
        """(forward, right) of a world point from the camera."""
        d = np.array([x, y]) - self.cam
        return float(d @ self.fwd), float(d @ self.right)


# --- plans -------------------------------------------------------------------------------------------------

@dataclass
class FramePlan:
    cache: str
    group: str
    idx: int
    note: str = ""
    bike: dict | None = None        # {route, s0 (at the frame), speed, offset (m right of the route heading)}
    ped: dict | None = None         # {bp, orient, t, x, y, z (at the frame), yaw (deg), speed}
    src: dict | None = None         # droplet: the nominal frame it is made from {cache, group, idx}

    def to_dict(self) -> dict:
        return asdict(self)


def bike_pose(g: Geometry, b: dict, k: int) -> tuple[float, float, float, float]:
    """(x, y, z, yaw deg) k ticks before the frame: constant speed along the route, `offset` m right of its heading."""
    r = g.wrong if b["route"] == g.wrong.name else (g.turn_routes.get(b["route"]) or g.routes[b["route"]])
    x, y, z, yaw = r.at(b["s0"] - b["speed"] * FIXED_DELTA * k)
    o, a = b["offset"], math.radians(yaw)
    return x - o * math.sin(a), y + o * math.cos(a), z, yaw        # right of heading = (-sin, cos) in CARLA's frame


def ped_back(p: dict, k: float) -> tuple[float, float]:
    """The pedestrian's position k ticks before the frame (constant velocity)."""
    a, d = math.radians(p["yaw"]), p["speed"] * FIXED_DELTA * k
    return p["x"] - d * math.cos(a), p["y"] - d * math.sin(a)


def _ped_crosswalk(g: Geometry) -> tuple[np.ndarray, np.ndarray]:
    """The pedestrians' crosswalk, the one nearest the ego (across our own road): its axis from the left end (ego view)
    to the right end."""
    q = min(g.cws, key=lambda c: g.ego_frame(*c.mean(0))[0])
    edges = [(q[i], q[(i + 1) % 4]) for i in range(4)]
    short = sorted(edges, key=lambda e: np.linalg.norm(e[1] - e[0]))[:2]
    a, b = [(e[0] + e[1]) / 2 for e in short]
    return (a, b) if g.ego_frame(*a)[1] < g.ego_frame(*b)[1] else (b, a)

def _ego_lane_s(g: Geometry) -> np.ndarray:
    """Arc positions on our lane's straight route: from where it enters the opposite crosswalk, ego_span_m on."""
    r = g.routes[EGO_STRAIGHT]
    far = max(g.cws, key=lambda q: g.ego_frame(*q.mean(0))[0])
    s_in = next(s for s in np.arange(0, r.length, 0.25) if g.ego_frame(*r.at(s)[:2])[0] > 5 and in_poly(*r.at(s)[:2], far))
    return np.linspace(s_in, s_in + _CY["ego_span_m"], _CY["ego_snaps"])


def _ped_spec(g: Geometry, bp: str, t: float, orient: str) -> dict:
    """At fraction t (left -> right) along the pedestrians' crosswalk; standing faces the ego camera, walking goes right /
    left along the crosswalk."""
    a, b = _ped_crosswalk(g)
    x, y = a + (b - a) * t
    z = g.m.get_waypoint(carla.Location(float(x), float(y), 0.0), project_to_road=True).transform.location.z
    if orient == "stand":
        yaw, speed = math.degrees(math.atan2(g.cam[1] - y, g.cam[0] - x)), 0.0
    else:
        d = (b - a) if orient == "walk_right" else (a - b)
        yaw, speed = math.degrees(math.atan2(d[1], d[0])), _PE["speed"]
    return {"bp": bp, "orient": orient, "t": round(t, 3), "x": round(float(x), 2), "y": round(float(y), 2), "z": round(z, 2),
            "yaw": round(yaw, 1), "speed": speed}


def plans(g: Geometry) -> list[FramePlan]:
    """Every observation to record with CARLA, then the ones made from recorded frames (derived_plans)."""
    out = []
    bike = lambda route, s, offset: {"route": route, "s0": round(float(s), 2), "speed": _CY["speed"], "offset": offset}  # noqa: E731
    for name in _CY["paths"]:                                    # cyclist / nominal: the other approaches' paths
        r, off = g.routes[name], _CY["curb_offset_m"]
        if name in g.turn_of:                                    # a turn (one_turn): the offset is built in
            r, off = g.turn_routes[g.turn_of[name]], 0.0
        for s in np.linspace(r.junction_s[0] - _CY["path_before_m"], r.junction_s[1] + _CY["path_after_m"], _CY["path_snaps"]):
            out.append(FramePlan("cyclist", "nominal", len(out), f"{r.turn} {r.name}", bike=bike(r.name, s, off)))
    ego_s = _ego_lane_s(g)
    for s in ego_s:                                              # our lane, riding away
        out.append(FramePlan("cyclist", "nominal", len(out), f"our lane {EGO_STRAIGHT}, away from us", bike=bike(EGO_STRAIGHT, s, _CY["curb_offset_m"])))
    for name in _CY["paths"]:                                    # extra snapshots inside each turn (turn_snaps)
        if name in g.turn_of and _CY["turn_snaps"]:
            r = g.turn_routes[g.turn_of[name]]
            a0, a1 = turn_arc(r)
            for s in np.linspace(a0, a1 + _CY["speed"] * FIXED_DELTA * (HISTORY - 1), _CY["turn_snaps"] + 2)[1:-1]:
                out.append(FramePlan("cyclist", "nominal", len(out), f"{r.turn} {r.name}, in the turn", bike=bike(r.name, s, 0.0)))
    # cyclist / wrongway: ego-lane snapshot wrong_at's place on the reversed route (same curb: left of its own heading)
    out.append(FramePlan("cyclist", "wrongway", 0, f"our lane, towards us (at nominal #{DROP_FROM['cyclist'][2]}'s place)",
                         bike=bike(g.wrong.name, g.wrong.length - float(ego_s[_CY["wrong_at"]]), -_CY["curb_offset_m"])))
    n = 0                                                        # pedestrian / nominal: kinds x positions x orientations
    for bp in _PE["bps"]:
        for ti, t in enumerate(np.linspace(*_PE["t"], _PE["positions"])):
            for o in _PE["orients"]:
                out.append(FramePlan("pedestrian", "nominal", n, f"{bp.split('.')[-1]} crosswalk position {ti} {o}", ped=_ped_spec(g, bp, float(t), o)))
                n += 1
    out.append(FramePlan("pedestrian", "conductor", 0, "conductor, middle of the crosswalk, standing", ped=_ped_spec(g, CONDUCTOR_BP, _PE["conductor_t"], "stand")))
    return out + derived_plans()


def derived_plans() -> list[FramePlan]:
    """Frames made from recorded ones (no CARLA): one droplet frame per dataset."""
    n = len(_DR["radii_px"])
    return [FramePlan(cache, "droplet", 0, f"nominal #{src[2]} + {n} blurred blobs on the lens", src=dict(zip(("cache", "group", "idx"), src)))
            for cache, src in DROP_FROM.items()]


DROPLET = _DR    # radii_px, min_sep, blur_sigma, seed (record.make_droplet)
