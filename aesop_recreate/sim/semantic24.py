"""The semantic24 experiment: run plan and per-run sampling (recorded by scripts/run_semantic24.py).

34 runs, all ClearNoon, 600 ticks at 10 Hz:
  t1_nominal_01..20   Town01 nominal                     withheld-routes pool (all 20), included-routes pool (01..10)
  t2_nominal_01..10   Town02 nominal                     included-routes pool
  t2_test_nominal     Town02 nominal                     held-out nominal (false alarms)
  t2_haybale          2-3 hay bales on the right sidewalk               new object, nominal predicate
  t2_pedestrian       a pedestrian standing in the ego lane mid-block   nominal object, new predicate
  t2_emergency        ambulance + police car with lights at the kerb, warning sign    new concept

Every run also gets street furniture (sim/concepts.PROPS: trash can, garbage bag, recycling bin, mailbox, bus stop)
on the sidewalks along the ego's route; parked cars are the towns' own. Walkers never cross the road (cross factor 0)
and any that spawn on a driving lane are removed, so a pedestrian on the road only appears in t2_pedestrian.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

TOWN_POOL, TOWN_TEST = "Town01", "Town02"
WEATHER = "ClearNoon"
N_TICKS = 600
VEHICLES = (30, 50)
WALKERS = (30, 50)
LEAD_PROB = 0.6
LEAD_GAP_M = (12.0, 25.0)

FURNITURE = ["trash can", "garbage bag", "recycling bin", "mailbox", "bus stop"]
FURNITURE_GAP_M = (12.0, 28.0)    # route distance between furniture items
FURNITURE_UNTIL_M = 600.0         # the ego covers ~300-450 m in a run

ANOMALIES = ("haybale", "pedestrian", "emergency")
ANOMALY_AT_M = (60.0, 110.0)      # route distance from the ego's start
MIN_VISIBLE_PX = 600              # anomaly pixels for a frame to count as showing it
MIN_VISIBLE_TICKS = 15
STRIDE = 5                        # every encoder and score uses ticks 0, 5, 10, ...


@dataclass
class Plan:
    name: str
    town: str
    seed: int
    kind: str                     # nominal / haybale / pedestrian / emergency
    pools: list[str] = field(default_factory=list)   # reference pools this run belongs to: withheld / included


def plan() -> list[Plan]:
    runs = [Plan(f"t1_nominal_{i:02d}", TOWN_POOL, i, "nominal", ["withheld"] + (["included"] if i <= 10 else [])) for i in range(1, 21)]
    runs += [Plan(f"t2_nominal_{i:02d}", TOWN_TEST, 100 + i, "nominal", ["included"]) for i in range(1, 11)]
    runs += [Plan("t2_test_nominal", TOWN_TEST, 201, "nominal")]
    runs += [Plan(f"t2_{k}", TOWN_TEST, 202 + j, k) for j, k in enumerate(ANOMALIES)]
    return runs


def sample(p: Plan, seed: int | None = None) -> dict:
    """Everything random about a run, from its seed (a retried anomaly run passes a fresh seed)."""
    seed = p.seed if seed is None else seed
    rng = random.Random(f"semantic24:{p.kind}:{seed}")
    return {"seed": seed, "town": p.town, "weather": WEATHER, "kind": p.kind,
            "n_vehicles": rng.randint(*VEHICLES), "n_walkers": rng.randint(*WALKERS),
            "lead": rng.random() < LEAD_PROB, "lead_gap_m": round(rng.uniform(*LEAD_GAP_M), 1),
            "lead_speed_diff_pct": round(rng.uniform(5, 25), 1),
            "anomaly_at_m": round(rng.uniform(*ANOMALY_AT_M), 1) if p.kind != "nominal" else None}
