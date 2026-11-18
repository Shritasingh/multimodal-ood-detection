"""Every parameter of the pipeline, from config.json (one place to change them)."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFG: dict = json.loads((HERE / "config.json").read_text())
DATA = HERE / "data"
RESULTS = HERE / "results"

# the datasets and their groups (nominal first), and which agent type each anomaly changes
CACHES = {"cyclist": ["nominal", "wrongway", "droplet"], "pedestrian": ["nominal", "conductor", "droplet"]}
SHIFTED = {"wrongway": "cyclist", "conductor": "pedestrian"}
OWN = {"wrongway": "physics", "conductor": "semantic", "droplet": "vision"}   # each anomaly's own representation
REPS = ("physics", "physics_euclid", "semantic", "vision")   # physics: standardised Euclidean; physics_euclid: plain
