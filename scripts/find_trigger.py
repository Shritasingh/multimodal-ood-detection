"""Picks the best anomaly-trigger tick from a clean run's states.jsonl.

Scores candidate ticks so the anomaly has maximum runway to stay visible
before the ego's route diverges (a turn) or the run ends:

  straight  -- longest gap until the next turn/yaw-change, for semantic
               (persists to end of run) and visual (pixel corruption,
               geometry-independent but scored the same way for consistency).
  physics   -- ticks where a genuine in-lane lead vehicle exists (small
               lateral offset, 5-20m ahead, both moving), scored by how much
               straight runway follows (needs >= CUTIN_DURATION_TICKS=40 to
               resolve before any turn) and how centered the candidate is.

Usage:
    .venv/bin/python scripts/find_trigger.py data/nominal_run/states.jsonl [--mode straight|physics]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

TURN_YAW_DELTA = 0.05
CUTIN_DURATION_TICKS = 40


def load_states(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def turn_ranges(states: list[dict]) -> list[tuple[int, int]]:
    prev = None
    turns = []
    for i, d in enumerate(states):
        yaw = d["ego"]["yaw"]
        if prev is not None and abs(yaw - prev) > TURN_YAW_DELTA:
            turns.append(i)
        prev = yaw
    ranges: list[list[int]] = []
    for t in turns:
        if ranges and t - ranges[-1][-1] <= 2:
            ranges[-1].append(t)
        else:
            ranges.append([t])
    return [(r[0], r[-1]) for r in ranges]


def ticks_to_next_turn(tick: int, turns: list[tuple[int, int]], n: int) -> int:
    for start, end in turns:
        if start > tick:
            return start - tick
    return n - tick


def find_best_straight(states: list[dict], warmup: int = 50, min_tick: int = 0) -> tuple[int, int]:
    """Returns (best_tick, runway) -- tick with the most straight road ahead,
    not earlier than max(warmup, min_tick) so there's a real nominal
    baseline before the anomaly starts."""
    turns = turn_ranges(states)
    n = len(states)
    floor = max(warmup, min_tick)
    best_tick, best_runway = floor, 0
    # Candidate ticks: right after each turn ends (or the floor), scored by runway.
    starts = [floor] + [end + 1 for _, end in turns if end + 1 < n and end + 1 >= floor]
    for s in starts:
        runway = ticks_to_next_turn(s, turns, n)
        if runway > best_runway:
            best_tick, best_runway = s, runway
    return best_tick, best_runway


SEMANTIC_SPAWN_M = 15.0  # inject_semantic places the prop 15m ahead of the ego


def find_best_physics(
    states: list[dict], min_tick: int = 0, require_unoccluded_semantic_spot: bool = False
) -> tuple[int, int, float]:
    """Returns (best_tick, actor_id, score). Requires >= CUTIN_DURATION_TICKS
    of straight runway after the tick, prefers small lateral offset and
    along distance near the middle of [5, 20]. `min_tick` enforces a real
    nominal baseline before the anomaly starts.

    With `require_unoccluded_semantic_spot`, also rejects ticks where any
    actor sits closer than SEMANTIC_SPAWN_M in-lane -- so the same tick can
    double as the semantic trigger without a car parked in front of the
    ego blocking the spawned prop. The lead-vehicle candidate itself must
    then sit *beyond* SEMANTIC_SPAWN_M (else it's the blocker)."""
    turns = turn_ranges(states)
    n = len(states)
    best = None  # (score, tick, actor_id)
    for i, d in enumerate(states):
        if i < min_tick:
            continue
        e = d["ego"]
        yaw = e["yaw"]
        fwd = (math.cos(yaw), math.sin(yaw))
        speed = math.hypot(e["vx"], e["vy"])
        if speed < 3.0:
            continue
        runway = ticks_to_next_turn(i, turns, n)
        if runway < CUTIN_DURATION_TICKS + 10:  # margin past the maneuver itself
            continue

        nearby = []
        for o in d["others"]:
            o_fwd = (math.cos(o["yaw"]), math.sin(o["yaw"]))
            heading_align = fwd[0] * o_fwd[0] + fwd[1] * o_fwd[1]
            if heading_align < 0.7:  # reject oncoming/parked-facing-away vehicles
                continue
            dx, dy = o["x"] - e["x"], o["y"] - e["y"]
            along = dx * fwd[0] + dy * fwd[1]
            lateral = dx * -fwd[1] + dy * fwd[0]
            if 0 < along < 25.0 and abs(lateral) < 2.0:
                nearby.append((o["id"], along, lateral))

        if require_unoccluded_semantic_spot:
            blockers = [a for a in nearby if a[1] < SEMANTIC_SPAWN_M]
            if blockers:
                continue
            lo = SEMANTIC_SPAWN_M
        else:
            lo = 5.0

        for actor_id, along, lateral in nearby:
            if not (lo < along < 20.0):
                continue
            # Lower is better: centeredness + how close along is to the
            # middle of the allowed window.
            target = (lo + 20.0) / 2
            score = abs(lateral) * 3 + abs(along - target)
            if best is None or score < best[0]:
                best = (score, i, actor_id)
    if best is None:
        raise RuntimeError("no valid lead-vehicle candidate found in this run")
    score, tick, actor_id = best
    return tick, actor_id, score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("states_path", type=Path)
    p.add_argument("--mode", choices=["straight", "physics"], default="straight")
    p.add_argument(
        "--min-tick", type=int, default=0, help="don't trigger before this tick (nominal baseline)"
    )
    p.add_argument(
        "--shared",
        action="store_true",
        help="physics mode: also require no actor closer than the semantic spawn point, "
        "so this same tick can be reused as the semantic/visual trigger",
    )
    args = p.parse_args()

    states = load_states(args.states_path)
    print(f"loaded {len(states)} states from {args.states_path}")
    print(f"turns: {turn_ranges(states)}")

    if args.mode == "straight":
        tick, runway = find_best_straight(states, min_tick=args.min_tick)
        print(f"best trigger tick: {tick} (runway to next turn: {runway} ticks)")
    else:
        tick, actor_id, score = find_best_physics(
            states, min_tick=args.min_tick, require_unoccluded_semantic_spot=args.shared
        )
        print(f"best trigger tick: {tick} (candidate actor {actor_id}, score {score:.2f})")


if __name__ == "__main__":
    main()
