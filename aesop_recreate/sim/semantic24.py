"""The semantic24 experiment: run plan and per-run sampling (recorded by scripts/run_semantic24.py into data/semantic24/).

34 runs, all ClearNoon, at 10 Hz, N_TICKS each:
  t1_nominal_01..20   Town01 nominal                     withheld-routes pool (all 20), included-routes pool (01..10)
  t2_nominal_01..10   Town02 nominal                     included-routes pool
  t2_test_nominal     Town02 nominal                     held-out nominal (false alarms)
  t2_haybale          2-3 hay bales on the right sidewalk (and a second group before the junction on the way)   new object
  t2_pedestrian       a pedestrian stepping into the ego lane                                                    new predicate
  t2_emergency        ambulance + police car at the kerb, warning sign                                           new concept
Each test anomaly is placed once, anomaly_at_m along the ego's route, and must be clearly visible (>= MIN_VISIBLE_TICKS
ticks on screen, >= MIN_PEAK_PX at its closest), else the run is re-recorded.

Every run gets street furniture (sim/concepts.PROPS: trash can, garbage bag, recycling bin, mailbox, bus stop) on the
sidewalks along the ego's route; parked cars are the towns' own. Walkers never cross the road (cross factor 0) and any
that spawn on a road lane are removed; in FREEZE_TOWNS a walker about to step onto a road lane (its position
FREEZE_LOOKAHEAD_S ahead), e.g. at a crosswalk, is stopped there for the rest of the run, so a pedestrian on the road
only appears in t2_pedestrian. The ego is an agent (AGENT: smooth speed control, slows for curves, red lights, walkers
and cars ahead); the Traffic Manager background traffic keeps its lane and a LEADING_GAP_M gap; every car drives at
SPEED_PCT of the limit; any collision or car off its lane re-records the run.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

DATASET = "semantic24"            # aesop_recreate/data/<DATASET>/
TOWN_POOL, TOWN_TEST = "Town01", "Town02"
WEATHER = "ClearNoon"
N_TICKS = 600
REMOVE_PAST_M = 15.0              # the anomaly is removed once the ego is this far past it
INJECTION_TIMEOUT_S = 150.0       # or after this long (stuck in traffic)
OFF_ROUTE_M = 8.0                 # the Traffic Manager now and then takes another branch at a junction than the planned
OFF_ROUTE_S = 2.0                 # route's: an ego this far from the route this long gets a new route from where it is
ROUTE_M = 4500.0                  # length of a replanned route
FREEZE_TOWNS = ("Town01", "Town02")
FREEZE_LOOKAHEAD_S = 1.0
VEHICLES = (30, 50)
WALKERS = (30, 50)
LEAD_PROB = 0.6
LEAD_GAP_M = (12.0, 25.0)

FURNITURE = ["trash can", "garbage bag", "recycling bin", "mailbox", "bus stop"]
FURNITURE_GAP_M = (12.0, 28.0)    # route distance between furniture items

ANOMALIES = ("haybale", "pedestrian", "emergency")
ANOMALY_AT_M = (60.0, 110.0)      # route distance from the ego's start
HAY_PRE_TURN_M = (15.0, 30.0)     # a second group of bales on the right sidewalk this far before the junction the ego
                                  # turns at on its way to the first group (25 m before it if there is no junction)
PED_AT_M = (90.0, 130.0)          # the pedestrian waits at the right kerb (PED_KERB_M past the lane edge) this far along the route,
PED_KERB_M = 0.8                  # (mid-sidewalk, a fence or furniture can block his way to the road)
PED_TRIGGER_M = 40.0              # walks to the ego lane's centre once he is in the camera's view and this close,
PED_VIEW_DEG = 40.0               # (in view: within this angle of the ego's heading, ahead of it)
PED_PAUSE_S = 10.0                # stands there this long, then crosses to just past the far lane's edge
PED_CROSS_S = 15.0                # and stops there (or after this long, if something blocks his way)
PED_SPEED = 1.4                   # m/s
POLICE_ALONG_M = (-9.0, -12.0, -15.0, 9.0, 12.0)   # police car spots relative to the ambulance, tried in order
SCENE_CLEAR_M = 18.0              # the emergency scene (police car / sign up to 16 m from the ambulance) and the pedestrian go where the
KERB_CLEAR_M = 4.0                # route has no junction this far behind or ahead, after clearing the kerb (this far
                                  # past the lane edge) along it of props and the town's own small objects
MIN_VISIBLE_PX = 250              # anomaly pixels for a frame to count as showing it (concepts.MIN_PX, as the ground truth)
GT_WINDOW_MIN_TICKS = 15          # an anomaly window from the ground truth needs this many ticks
MIN_VISIBLE_TICKS = 30            # the anomaly is on screen (>= MIN_VISIBLE_PX; the pedestrian also on the road) this many
MIN_PEAK_PX = 3000                # ticks and this large at its closest, else the run is re-recorded
STRIDE = 5                        # every encoder and score uses ticks 0, 5, 10, ...

SPEED_PCT = 90.0                  # every car (ego and Traffic Manager traffic) at this % of the speed limit
LEADING_GAP_M = 4.0               # Traffic Manager cars keep this gap to the car ahead (gentler braking)
OFF_LANE_S = 1.0                  # a background car off the driving lanes this long fails the run (the ego: any tick)
AGENT = {                         # CARLA BehaviorAgent ('cautious') options, tuned on Town02's tightest corner (A/B test
    "max_throttle": 0.6,          # 2026-11-03: 0.27 m right of the lane centre vs 0.90 m for the Traffic Manager, jerk
    "max_brake": 0.3,             # p99 15 vs 32 m/s3): a short steering look-ahead and a gentle speed controller
    "base_min_distance": 1.5,
    "distance_ratio": 0.2,
    "longitudinal_control_dict": {"K_P": 0.15, "K_I": 0.02, "K_D": 0.0},
}
CURVE_A_LAT = 2.0                 # the ego takes curves at <= this lateral acceleration (m/s2)
SLOW_A = 1.5                      # and slows for curves, red lights, walkers on its lane and cars ahead at <= this (m/s2)
LOOKAHEAD_WPS = 25                # route points (2 m apart) the ego looks ahead for them, at least (else its stopping distance
                                  # + 15 m: Town02's 60 km/h roads need ~100 m)
STOP_GAP_M = 6.0                  # it stops this far before a walker or the car ahead, 2 m before a stop line


@dataclass
class Plan:
    name: str
    town: str
    seed: int
    kind: str                     # nominal / haybale / pedestrian / emergency
    pools: list[str] = field(default_factory=list)   # reference pools this run belongs to: withheld / included
    n_ticks: int = N_TICKS


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
            "anomaly_at_m": round(rng.uniform(*(PED_AT_M if p.kind == "pedestrian" else ANOMALY_AT_M)), 1) if p.kind != "nominal" else None}
