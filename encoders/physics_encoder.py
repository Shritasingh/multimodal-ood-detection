"""E_Physics: ground-truth kinematic state -> x_hat.

Purpose (per motivating example): catch anomalies that are temporal/dynamic
-- a vehicle cutting in, erratic braking -- where color and lighting are
distractors and what matters is how an agent's pose/velocity evolves over the
window. Per the MVP plan, we start from CARLA's ground-truth actor state
(bypassing a learned tracker/state-estimator) to isolate "is this
representation the right one" from "is my state estimator any good".

Input format is decoupled from the `carla` package so this module (and its
tests) don't need a running simulator: `sim/state_extractor.py` is
responsible for turning `carla.Actor` snapshots into the plain dicts this
encoder expects.
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
    def _speed(agent: AgentState) -> float:
        return float(np.hypot(agent["vx"], agent["vy"]))

    def _relative_features(self, ego: AgentState, other: AgentState) -> np.ndarray:
        dx, dy = other["x"] - ego["x"], other["y"] - ego["y"]
        range_m = float(np.hypot(dx, dy))
        # Bearing of the other agent relative to ego heading.
        bearing = float(np.arctan2(dy, dx) - ego["yaw"])
        rel_vx, rel_vy = other["vx"] - ego["vx"], other["vy"] - ego["vy"]
        closing_speed = -float((dx * rel_vx + dy * rel_vy) / max(range_m, 1e-3))
        return np.array([range_m, bearing, closing_speed, self._speed(other)])

    def encode(self, obs_history: Sequence[TimestepState]) -> tuple[np.ndarray, dict[str, Any]]:
        """obs_history: chronological list of TimestepState snapshots.
        Returns (embedding, extras)."""
        ego_traj = np.array(
            [[s["ego"]["x"], s["ego"]["y"], s["ego"]["yaw"], self._speed(s["ego"])] for s in obs_history]
        )
        # Ego kinematic summary: speed/heading and their finite-difference
        # derivatives (accel, yaw rate) over the window.
        ego_speed = ego_traj[:, 3]
        ego_yaw = ego_traj[:, 2]
        ego_accel = np.diff(ego_speed, prepend=ego_speed[0]) / self.dt
        ego_yaw_rate = np.diff(np.unwrap(ego_yaw), prepend=ego_yaw[0]) / self.dt
        ego_feats = np.concatenate(
            [
                [ego_speed.mean(), ego_speed.std(), ego_accel.mean(), ego_accel.std()],
                [ego_yaw_rate.mean(), ego_yaw_rate.std()],
            ]
        )

        # Nearest-K other agents within relative_range_m at the latest timestep,
        # each described by relative range/bearing/closing-speed/speed, tracked
        # back through the window to capture e.g. a cut-in's lateral drift over
        # time. (state_extractor.py's tracking radius is a separate, usually
        # larger, upstream cutoff on what even reaches this encoder.)
        latest = obs_history[-1]
        def _range(a):
            return np.hypot(a["x"] - latest["ego"]["x"], a["y"] - latest["ego"]["y"])
        in_range = [a for a in latest["others"] if _range(a) <= self.relative_range_m]
        others_sorted = sorted(in_range, key=_range)[: self.max_tracked_agents]

        other_feat_blocks = []
        for agent in others_sorted:
            agent_id = agent["id"]
            track = []
            for step in obs_history:
                match = next((o for o in step["others"] if o["id"] == agent_id), None)
                if match is not None:
                    track.append(self._relative_features(step["ego"], match))
            if not track:
                continue
            track = np.stack(track)
            # Summarize this agent's relative track: latest state + how much
            # bearing/range changed over the window (cut-in signature).
            block = np.concatenate(
                [
                    track[-1],
                    [track[-1, 1] - track[0, 1]],  # bearing drift
                    [track[-1, 0] - track[0, 0]],  # range closure
                ]
            )
            other_feat_blocks.append(block)

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
