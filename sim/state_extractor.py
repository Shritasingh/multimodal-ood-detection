"""Turns live `carla.Actor` state into the plain-dict format
`encoders.physics_encoder.GroundTruthPhysicsEncoder` expects.

Kept separate from the encoder itself so the encoder has no dependency on a
running simulator (and is unit-testable without CARLA installed/running).
"""
from __future__ import annotations

import collections
from typing import Optional

import carla


def actor_to_agent_state(actor: "carla.Actor") -> dict:
    t = actor.get_transform()
    v = actor.get_velocity()
    return {
        "id": actor.id,
        "x": t.location.x,
        "y": t.location.y,
        "yaw": t.rotation.yaw * 3.141592653589793 / 180.0,
        "vx": v.x,
        "vy": v.y,
    }


class GroundTruthStateHistory:
    """Rolling buffer of TimestepState snapshots for the physics encoder.

    Call `push(world, ego_actor)` once per tick (or per control step) and
    `window(n)` to pull the last n snapshots in the format
    GroundTruthPhysicsEncoder.encode expects.
    """

    def __init__(self, max_len: int = 50, other_actor_radius_m: float = 60.0):
        self.buffer: collections.deque = collections.deque(maxlen=max_len)
        self.radius = other_actor_radius_m

    def push(self, world: "carla.World", ego_actor: "carla.Actor", sim_time: Optional[float] = None) -> None:
        ego_state = actor_to_agent_state(ego_actor)
        others = []
        for actor in world.get_actors().filter("vehicle.*"):
            if actor.id == ego_actor.id:
                continue
            state = actor_to_agent_state(actor)
            if (state["x"] - ego_state["x"]) ** 2 + (state["y"] - ego_state["y"]) ** 2 <= self.radius ** 2:
                others.append(state)
        for actor in world.get_actors().filter("walker.pedestrian.*"):
            state = actor_to_agent_state(actor)
            if (state["x"] - ego_state["x"]) ** 2 + (state["y"] - ego_state["y"]) ** 2 <= self.radius ** 2:
                others.append(state)

        t = sim_time if sim_time is not None else world.get_snapshot().timestamp.elapsed_seconds
        self.buffer.append({"t": t, "ego": ego_state, "others": others})

    def window(self, n: int) -> list[dict]:
        return list(self.buffer)[-n:]

    def __len__(self) -> int:
        return len(self.buffer)
