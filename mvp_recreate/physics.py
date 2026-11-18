"""The physics encoder: 2 role slots, each holding its agent's last HISTORY states in the ego frame, nothing else.

Slots: agent (the nearest non-walker: the cyclist / any vehicle) and pedestrian (the nearest walker). No lane logic:
where the agent is and which way it rides is in the features themselves. Every agent is encoded: there is no range
limit; per slot the nearest agent wins.

Per slot and history step (oldest first), in the ego frame at that step (+x forward, +y right):
    x, y, vx, vy, cos(yaw), sin(yaw), speed
(yaw as cos / sin so a heading near +-180 deg does not jump). Slots are assigned on the latest state; each slot's agent
is followed back by id. An empty slot or a step where its agent is missing is all zeros. 2 slots x 5 steps x 7 = 70-d.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from cfg import CFG

ROLES = ("agent", "pedestrian")
FEATURES = ("x", "y", "vx", "vy", "cos_yaw", "sin_yaw", "speed")
WALKER_TYPES = {"pedestrian"}


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def ego_frame(agent: dict, ego: dict) -> dict:
    """An agent's state in the ego frame (+x forward, +y right, as CARLA's left-handed world)."""
    c, s = math.cos(ego["yaw"]), math.sin(ego["yaw"])
    dx, dy = agent["x"] - ego["x"], agent["y"] - ego["y"]
    dvx, dvy = agent["vx"] - ego["vx"], agent["vy"] - ego["vy"]
    yaw = _wrap(agent["yaw"] - ego["yaw"])
    return {"x": dx * c + dy * s, "y": -dx * s + dy * c, "vx": dvx * c + dvy * s, "vy": -dvx * s + dvy * c,
            "yaw": yaw, "speed": math.hypot(agent["vx"], agent["vy"])}


def role_of(walker: bool) -> str:
    return "pedestrian" if walker else "agent"


def assign(state: dict, types: dict[int, str]) -> dict[int, dict]:
    """id -> {role (None if unassigned or beaten by a nearer agent), rel (ego-frame state)} for every agent of a state."""
    out, best = {}, {}
    for o in state["others"]:
        rel = ego_frame(o, state["ego"])
        role = role_of(types.get(o["id"]) in WALKER_TYPES)
        out[o["id"]] = {"role": None, "candidate": role, "rel": rel}
        key = math.hypot(rel["x"], rel["y"])
        if role not in best or key < best[role][0]:
            best[role] = (key, o["id"])
    for role, (_, i) in best.items():
        out[i]["role"] = role
    return out


class PhysicsEncoder:
    name = "physics_slots"

    def __init__(self, history: int = CFG["history"]):
        self.history = history
        self.dim = len(ROLES) * history * len(FEATURES)
        self.feature_names = [f"{r}_{f}@t-{history - 1 - k}" for r in ROLES for k in range(history) for f in FEATURES]

    @staticmethod
    def row(rel: dict) -> list[float]:
        return [rel["x"], rel["y"], rel["vx"], rel["vy"], math.cos(rel["yaw"]), math.sin(rel["yaw"]), rel["speed"]]

    def encode(self, hist: Sequence[dict], types: dict[int, str]) -> tuple[np.ndarray, dict]:
        hist = list(hist)[-self.history:]
        roles = {v["role"]: i for i, v in assign(hist[-1], types).items() if v["role"]}
        blocks = []
        for r in ROLES:
            rows = np.zeros((self.history, len(FEATURES)))
            if r in roles:
                for k, st in enumerate(hist):
                    o = next((a for a in st["others"] if a["id"] == roles[r]), None)
                    if o is not None:
                        rows[k] = self.row(ego_frame(o, st["ego"]))
            blocks.append(rows.ravel())
        return np.concatenate(blocks), {"roles": roles}
