"""The semantic24 experiment: run plan and per-run sampling (recorded by scripts/run_semantic24.py).

34 runs, all ClearNoon, at 10 Hz; pool runs 600 ticks, test runs LONG_TICKS (long, as in Elhafsi et al. 2023, the CARLA data
AESOP evaluates on: anomalies injected one after another along one drive):
  t1_nominal_01..20   Town01 nominal                     withheld-routes pool (all 20), included-routes pool (01..10)
  t2_nominal_01..10   Town02 nominal                     included-routes pool
  t2_test_nominal     Town02 nominal, long               held-out nominal (false alarms)
  t2_haybale          long, INJECTIONS x 2-3 hay bales on the right sidewalk             new object, nominal predicate
  t2_pedestrian       long, INJECTIONS x a pedestrian stepping into the ego lane         nominal object, new predicate
  t2_emergency        long, INJECTIONS x ambulance + police car at the kerb, warning sign   new concept
In a long anomaly run each injection is placed INJECT_AHEAD_M ahead of the ego on its route, removed once the ego is
REMOVE_PAST_M past it, and the next one placed (one live on a route the ego has left is removed too); injections that
never come on screen do not count, and injecting stops
once INJECTIONS have (the rest of the run is nominal). Each injection's anomaly window = the ticks it is on screen.

Every run also gets street furniture (sim/concepts.PROPS: trash can, garbage bag, recycling bin, mailbox, bus stop)
on the sidewalks along the ego's route; parked cars are the towns' own. Walkers never cross the road (cross factor 0)
and any that spawn on a road lane are removed. In Town02 (FREEZE_TOWNS) a walker about to step onto a road lane (its
position FREEZE_LOOKAHEAD_S ahead), e.g. at a crosswalk, is stopped there for the rest of the run, so a pedestrian on the
road only appears in t2_pedestrian (Town01's runs, recorded before this rule, keep their crosswalk crossings).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

TOWN_POOL, TOWN_TEST = "Town01", "Town02"
WEATHER = "ClearNoon"
N_TICKS = 600
LONG_TICKS = 7200                 # test runs: 12 min
LONG_ROUTE_M = 4500.0             # the ego covers ~80-280 m a minute at TEST_SPEED_PCT (red lights, traffic)
INJECTIONS = 10                   # on-screen injections per long anomaly run
MAX_INJECTIONS = 30               # tried at most
INJECT_AHEAD_M = (60.0, 100.0)    # hay bales / emergency: route distance ahead of the ego (pedestrian: PED_AT_M)
REMOVE_PAST_M = 15.0
INJECTION_TIMEOUT_S = 150.0       # an injection the ego has not passed by then (stuck in traffic) is removed anyway
OFF_ROUTE_M = 8.0                 # the Traffic Manager now and then takes another branch at a junction than the planned
OFF_ROUTE_S = 2.0                 # route's: an ego this far from the route this long gets a new route from where it is
JAM_S = 70.0                      # an ego that has not moved this long: the background cars queued JAM_AHEAD_M ahead
JAM_AHEAD_M = 60.0                # of it are removed (the Traffic Manager destroys any vehicle blocked for 90 s, ego included)
FREEZE_TOWNS = ("Town02",)
FREEZE_LOOKAHEAD_S = 1.0
VEHICLES = (30, 50)
LONG_VEHICLES = (10, 15)          # long runs: 12 min behind 30-50 cars in Town02 is one long queue, and they hide the anomaly
WALKERS = (30, 50)
LEAD_PROB = 0.6
LEAD_GAP_M = (12.0, 25.0)

FURNITURE = ["trash can", "garbage bag", "recycling bin", "mailbox", "bus stop"]
FURNITURE_GAP_M = (12.0, 28.0)    # route distance between furniture items
FURNITURE_UNTIL_M = 600.0         # the ego covers ~300-450 m in a run

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
TEST_SPEED_PCT = 85.0             # in the test runs (t2_test_nominal and the anomalies) the ego drives at this % of its
                                  # normal speed all run, so each anomaly stays on screen longer
POLICE_ALONG_M = (-9.0, -12.0, -15.0, 9.0, 12.0)   # police car spots relative to the ambulance, tried in order
PARK_KERB_M = 0.5                 # parked emergency vehicles stand this far past the lane edge, physics off: on the lane
                                  # edge the Traffic Manager treats them as blocking it, and the kerb could flip them
MIN_VISIBLE_PX = 250              # anomaly pixels for a frame to count as showing it (concepts.MIN_PX, as the ground truth)
MIN_VISIBLE_TICKS = 15
STRIDE = 5                        # every encoder and score uses ticks 0, 5, 10, ...


@dataclass
class Plan:
    name: str
    town: str
    seed: int
    kind: str                     # nominal / haybale / pedestrian / emergency
    pools: list[str] = field(default_factory=list)   # reference pools this run belongs to: withheld / included
    long: bool = False

    @property
    def n_ticks(self) -> int:
        return LONG_TICKS if self.long else N_TICKS


def plan() -> list[Plan]:
    runs = [Plan(f"t1_nominal_{i:02d}", TOWN_POOL, i, "nominal", ["withheld"] + (["included"] if i <= 10 else [])) for i in range(1, 21)]
    runs += [Plan(f"t2_nominal_{i:02d}", TOWN_TEST, 100 + i, "nominal", ["included"]) for i in range(1, 11)]
    runs += [Plan("t2_test_nominal", TOWN_TEST, 201, "nominal", long=True)]
    runs += [Plan(f"t2_{k}", TOWN_TEST, 202 + j, k, long=True) for j, k in enumerate(ANOMALIES)]
    return runs


def sample(p: Plan, seed: int | None = None) -> dict:
    """Everything random about a run, from its seed (a retried anomaly run passes a fresh seed)."""
    seed = p.seed if seed is None else seed
    rng = random.Random(f"semantic24:{p.kind}:{seed}")
    return {"seed": seed, "town": p.town, "weather": WEATHER, "kind": p.kind,
            "n_vehicles": rng.randint(*(LONG_VEHICLES if p.long else VEHICLES)), "n_walkers": rng.randint(*WALKERS),
            "lead": rng.random() < LEAD_PROB, "lead_gap_m": round(rng.uniform(*LEAD_GAP_M), 1),
            "lead_speed_diff_pct": round(rng.uniform(5, 25), 1),
            "anomaly_at_m": round(rng.uniform(*(PED_AT_M if p.kind == "pedestrian" else ANOMALY_AT_M)), 1) if p.kind != "nominal" and not p.long else None}
