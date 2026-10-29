"""Summary of recorded scenario runs: what was sampled, whether the anomaly happened, and how close the ego came.

    .venv/bin/python scripts/helpers/inspect_scenario.py data/proto/*     # one line per run (+ details of the anomaly)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def seg_dist(p, a, b) -> float:
    """Distance from point p to segment ab."""
    ax, ay, bx, by = *a, *b
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy or 1.0)))
    return math.dist(p, (ax + t * dx, ay + t * dy))


def summary(run: Path) -> str:
    info = json.loads((run / "run_info.json").read_text())
    states = [json.loads(l) for l in (run / "states.jsonl").read_text().splitlines()]
    xy = [(s["ego"]["x"], s["ego"]["y"]) for s in states]
    travelled = sum(math.dist(a, b) for a, b in zip(xy, xy[1:]))
    stopped = sum(math.hypot(s["ego"]["vx"], s["ego"]["vy"]) < 0.5 for s in states) / len(states)
    route = info.get("route_xy")
    if route:
        dev = [min(seg_dist(p, a, b) for a, b in zip(route, route[1:])) for p in xy[::10]]
        off = next((i * 10 for i, d in enumerate(dev) if d > 4.0), None)
        route_note = f", left route at tick {off}" if off is not None else ", on route"
    else:
        route_note = ""
    line = (f"{run.name:28s} {info['town']:8s} {info['weather']:15s} veh {info['n_vehicles_spawned']:3d} walk {info['n_walkers_spawned']:3d} "
            f"lead {'y' if info['lead_spawned'] else 'n'}  ego {travelled:5.0f} m, stopped {stopped:4.0%}{route_note}")
    a_path = run / "anomalies.json"
    if a_path.exists():
        a = json.loads(a_path.read_text())
        px = a.get("visible_px") or []
        line += f"\n    {a['anomaly_type']}/{a.get('kind')}: success {a.get('success')}  window {a.get('onset_frame')}-{a.get('offset_frame')}  max px {max(px) if px else '-'}"
        if a.get("anchor_xy"):
            d = min(math.dist(p, a["anchor_xy"]) for p in xy)
            line += f"  placed {len(a.get('actors', []))} actors at route {a.get('route_m')} m, ego closest {d:.0f} m"
        if a.get("actor_blueprint"):
            line += f"  {a['actor_blueprint']} (requested {a.get('kind_requested')}), px at onset {a.get('visible_px_at_onset')}"
    return line


if __name__ == "__main__":
    for p in sys.argv[1:]:
        if (Path(p) / "run_info.json").exists():
            print(summary(Path(p)))
