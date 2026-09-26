"""Physics encoders: ground-truth kinematic state -> embedding.

  GroundTruthPhysicsEncoder  (name "physics")        36-d: absolute ego pose + the 5 nearest agents
  RolePhysicsEncoder         (name "physics_roles")  36-d: ego kinematics + role slots (lead, follower, ...)

Input format is decoupled from the `carla` package so this module (and its
tests) don't need a running simulator: `sim/state_extractor.py` is
responsible for turning `carla.Actor` snapshots into the plain dicts these
encoders expect.
"""
from __future__ import annotations

from typing import Any, Sequence, TypedDict

import numpy as np


class AgentState(TypedDict):
    id: int
    x: float
    y: float
    yaw: float  # radians
    vx: float
    vy: float


class TimestepState(TypedDict):
    t: float
    ego: AgentState
    others: list[AgentState]


class GroundTruthPhysicsEncoder:
    name = "physics"

    def __init__(self, max_tracked_agents: int = 5, relative_range_m: float = 40.0, dt: float = 0.1):
        self.max_tracked_agents = max_tracked_agents
        self.relative_range_m = relative_range_m
        self.dt = dt  # seconds/tick; must match collection's --fixed-delta (default 0.1s = 10Hz)

    @staticmethod
    def _wrap(angle: float) -> float:
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    def _agent_yaw_rate(self, obs_history: Sequence[TimestepState], agent_id: int) -> float:
        """Same single-step-difference approach as the ego's own yaw_rate,
        just applied to one tracked other agent."""
        if len(obs_history) < 2:
            return 0.0

        def find(step: TimestepState):
            return next((o for o in step["others"] if o["id"] == agent_id), None)

        prev, cur = find(obs_history[-2]), find(obs_history[-1])
        if prev is None or cur is None:
            return 0.0
        return float(np.diff(np.unwrap([prev["yaw"], cur["yaw"]]))[0]) / self.dt

    def _relative_features(
        self, ego: AgentState, other: AgentState, ego_yaw_rate: float, other_yaw_rate: float
    ) -> np.ndarray:
        """Other agent's state in the ego's own frame: position and velocity
        rotated so the ego's heading is the x-axis (rel_x = ahead, rel_y =
        left), plus relative yaw and yaw_rate. Cartesian rather than
        range/bearing so a cut-in's lateral drift is a direct signed
        quantity (rel_y -> 0) instead of derived from two polar terms."""
        dx, dy = other["x"] - ego["x"], other["y"] - ego["y"]
        cos_e, sin_e = np.cos(ego["yaw"]), np.sin(ego["yaw"])
        rel_x = dx * cos_e + dy * sin_e
        rel_y = -dx * sin_e + dy * cos_e
        rel_yaw = self._wrap(other["yaw"] - ego["yaw"])
        dvx, dvy = other["vx"] - ego["vx"], other["vy"] - ego["vy"]
        rel_vx = dvx * cos_e + dvy * sin_e
        rel_vy = -dvx * sin_e + dvy * cos_e
        rel_yaw_rate = other_yaw_rate - ego_yaw_rate
        return np.array([rel_x, rel_y, rel_yaw, rel_vx, rel_vy, rel_yaw_rate])

    def encode(self, obs_history: Sequence[TimestepState]) -> tuple[np.ndarray, dict[str, Any]]:
        """obs_history: chronological list of TimestepState snapshots.
        Returns (embedding, extras)."""
        latest = obs_history[-1]
        ego = latest["ego"]
        # Raw instantaneous ego state, not windowed mean/std summary stats --
        # yaw_rate is the one derived quantity, a direct single-step
        # difference (not a mean/std over a diff array, which halves the
        # true value when the window is only 2 states).
        if len(obs_history) >= 2:
            prev_yaw = obs_history[-2]["ego"]["yaw"]
            yaw_rate = float(np.diff(np.unwrap([prev_yaw, ego["yaw"]]))[0]) / self.dt
        else:
            yaw_rate = 0.0
        ego_feats = np.array([ego["x"], ego["y"], ego["yaw"], ego["vx"], ego["vy"], yaw_rate])

        # Nearest-K other agents within relative_range_m at the latest timestep,
        # each described by its ego-relative {x, y, yaw, vx, vy, yaw_rate}.
        # (state_extractor.py's tracking radius is a separate, usually
        # larger, upstream cutoff on what even reaches this encoder.)
        def _range(a):
            return np.hypot(a["x"] - latest["ego"]["x"], a["y"] - latest["ego"]["y"])
        in_range = [a for a in latest["others"] if _range(a) <= self.relative_range_m]
        others_sorted = sorted(in_range, key=_range)[: self.max_tracked_agents]

        other_feat_blocks = [
            self._relative_features(ego, agent, yaw_rate, self._agent_yaw_rate(obs_history, agent["id"]))
            for agent in others_sorted
        ]

        pad_width = self.max_tracked_agents - len(other_feat_blocks)
        if other_feat_blocks:
            others_feats = np.concatenate(other_feat_blocks)
        else:
            others_feats = np.array([])
        if pad_width > 0:
            others_feats = np.concatenate([others_feats, np.zeros(pad_width * 6)])

        embedding = np.concatenate([ego_feats, others_feats])
        return embedding, {
            "num_tracked_others": len(other_feat_blocks),
            "tracked_agent_ids": [a["id"] for a in others_sorted],
        }


ROLES = ("lead", "follower", "left", "right", "oncoming", "crossing")
ROLE_MAX_RANGE_M = {"lead": 40.0, "follower": 25.0, "left": 25.0, "right": 25.0, "oncoming": 30.0, "crossing": 25.0}
ROLE_SLOT_FEATURES = ("present", "rel_x", "rel_y", "rel_vx", "rel_vy")
ROLE_EGO_FEATURES = ("v_long", "yaw_rate", "a_long", "a_lat", "jerk_long")
ROLE_FEATURE_NAMES = (
    [f"ego_{n}" for n in ROLE_EGO_FEATURES]
    + [f"{r}_{n}" for r in ROLES for n in ROLE_SLOT_FEATURES]
    + ["lead_ttc"]
)
ROLE_WINDOW = 3  # states needed: [t-2, t-1, t]


class RolePhysicsEncoder:
    """Ego kinematics plus lead/follower/left/right/oncoming/crossing slots (ego frame, +y = right), 36-d; other agents' yaw-rate unused."""
    name = "physics_roles"
    window = ROLE_WINDOW

    def __init__(
        self,
        max_range_m: dict[str, float] | None = None,
        lane_half_width_m: float = 2.0,
        adjacent_half_width_m: float = 5.5,
        same_dir_cos: float = 0.7,
        ttc_cap_s: float = 10.0,
        dt: float = 0.1,  # CARLA fixed_delta_seconds = 0.1 -> 10.00 Hz (verified from logged t)
    ):
        self.max_range_m = {**ROLE_MAX_RANGE_M, **(max_range_m or {})}
        self.lane_half_width_m = lane_half_width_m
        self.adjacent_half_width_m = adjacent_half_width_m
        self.same_dir_cos = same_dir_cos
        self.ttc_cap_s = ttc_cap_s
        self.dt = dt

    @staticmethod
    def _wrap(a: float) -> float:
        return float(np.arctan2(np.sin(a), np.cos(a)))

    @staticmethod
    def _v_long(agent) -> float:
        return agent["vx"] * np.cos(agent["yaw"]) + agent["vy"] * np.sin(agent["yaw"])

    def _ego_features(self, hist: Sequence[dict]) -> np.ndarray:
        e2, e1, e0 = (h["ego"] for h in hist[-3:])
        v0, v1, v2 = self._v_long(e0), self._v_long(e1), self._v_long(e2)
        a0, a1 = (v0 - v1) / self.dt, (v1 - v2) / self.dt
        yaw_rate = self._wrap(e0["yaw"] - e1["yaw"]) / self.dt
        return np.array([v0, yaw_rate, a0, v0 * yaw_rate, (a0 - a1) / self.dt])

    def _role(self, rel_x: float, rel_y: float, rel_yaw: float) -> str | None:
        c = np.cos(rel_yaw)
        if c > self.same_dir_cos:
            if abs(rel_y) <= self.lane_half_width_m:
                return "lead" if rel_x > 0 else "follower"
            if abs(rel_y) <= self.adjacent_half_width_m:
                return "left" if rel_y < 0 else "right"
            return None
        if c < -self.same_dir_cos:
            return "oncoming"
        return "crossing"

    def encode(self, obs_history: Sequence[dict]) -> tuple[np.ndarray, dict[str, Any]]:
        latest = obs_history[-1]
        ego = latest["ego"]
        cos_e, sin_e = np.cos(ego["yaw"]), np.sin(ego["yaw"])
        best: dict[str, tuple[float, np.ndarray, int]] = {}
        for o in latest["others"]:
            dx, dy = o["x"] - ego["x"], o["y"] - ego["y"]
            rel_x, rel_y = dx * cos_e + dy * sin_e, -dx * sin_e + dy * cos_e   # +y = ego's right (left-handed)
            rel_yaw = self._wrap(o["yaw"] - ego["yaw"])
            role = self._role(rel_x, rel_y, rel_yaw)
            dist = float(np.hypot(rel_x, rel_y))
            if role is None or dist > self.max_range_m[role]:
                continue
            dvx, dvy = o["vx"] - ego["vx"], o["vy"] - ego["vy"]
            feats = np.array([1.0, rel_x, rel_y, dvx * cos_e + dvy * sin_e, -dvx * sin_e + dvy * cos_e])
            key = abs(rel_x) if role in ("lead", "follower") else dist   # nearest along the lane / nearest overall
            if role not in best or key < best[role][0]:
                best[role] = (key, feats, o["id"])
        blocks = [best[r][1] if r in best else np.zeros(len(ROLE_SLOT_FEATURES)) for r in ROLES]
        ttc = self.ttc_cap_s
        if "lead" in best:
            rel_x, closing = best["lead"][1][1], -best["lead"][1][3]
            if closing > 0.1:
                ttc = float(np.clip(rel_x / closing, 0.0, self.ttc_cap_s))
        emb = np.concatenate([self._ego_features(obs_history), *blocks, [ttc]])
        return emb, {"roles": {r: best[r][2] for r in best}, "feature_names": ROLE_FEATURE_NAMES}
