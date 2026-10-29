"""Record a batch of randomised runs (scripts/run_scenario.py) and write its data config.

    .venv/bin/python scripts/run_batch.py                       # 100 nominal, 30 semantic, 30 physics tries, 30 visual
    .venv/bin/python scripts/run_batch.py --nominal 10 --semantic 5 --physics 5 --visual 5 --batch batch_test

Anomaly subtypes are cycled so each is equally represented. A semantic run whose object never came on screen is
replaced by a run with the next unused seed until --semantic runs succeed; physics runs get exactly --physics tries
(failures are discarded, not replaced); visual runs cannot fail. Runs are ordered by town so the map is reloaded
as rarely as possible, finished runs are skipped on a re-run, and a crashed CARLA server is restarted.

Writes data/<batch>/<run>/..., data/<batch>/batch_log.jsonl (one line per attempt) and config/data_<batch>.json:
nominal seeds are split into reference_runs (the kNN pool), calibration_runs (conformal thresholds) and
negative_runs (held-out false-alarm check) by --split; anomaly_runs are the successful anomaly runs.
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sim import scenarios as S  # noqa: E402

CARLA = ROOT.parent / "carla_sim" / "launch_carla.sh"
RUN_TIMEOUT_S = 900
SUBTYPES = {"semantic": list(S.SEMANTIC_KINDS), "physics": list(S.PHYSICS_KINDS), "visual": list(S.VISUAL_KINDS)}


def carla_up() -> bool:
    try:
        socket.create_connection(("127.0.0.1", 2000), timeout=2).close()
        return True
    except OSError:
        return False


def restart_carla(log: Path) -> None:
    subprocess.run(["pkill", "-f", "CarlaUE4-Linux-Shipping"])
    time.sleep(10)
    subprocess.Popen([str(CARLA), "-RenderOffScreen", "-nosound"], stdout=log.open("a"), stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(90):
        if carla_up():
            time.sleep(15)   # the RPC port opens before the world is ready
            return
        time.sleep(2)
    raise RuntimeError("CARLA did not come back up")


def status_of(run_dir: Path, n_ticks: int) -> str | None:
    """'ok' / 'failed' for a finished run, None if missing or incomplete."""
    states = run_dir / "states.jsonl"
    if not states.exists() or len(states.read_text().splitlines()) != n_ticks or len(list((run_dir / "frames").glob("*.jpg"))) != n_ticks:
        return None
    a = run_dir / "anomalies.json"
    if not a.exists():
        return "ok"
    return "ok" if json.loads(a.read_text()).get("success") else "failed"


def record(kind: str, seed: int, subtype: str | None, batch: str, n_ticks: int, log: Path) -> str:
    name = f"{kind}_{seed:03d}"
    run_dir = ROOT / "data" / batch / name
    done = status_of(run_dir, n_ticks)
    if done is not None:
        return done
    for attempt in range(2):
        if not carla_up():
            restart_carla(log)
        cmd = [sys.executable, str(ROOT / "scripts" / "run_scenario.py"), kind, str(seed), "--out-root", f"data/{batch}", "--n-ticks", str(n_ticks)]
        if subtype:
            cmd += ["--subtype", subtype]
        t0 = time.time()
        try:
            rc = subprocess.run(cmd, timeout=RUN_TIMEOUT_S, stdout=log.open("a"), stderr=subprocess.STDOUT).returncode
        except subprocess.TimeoutExpired:
            rc = -1
        status = status_of(run_dir, n_ticks) or ("failed" if rc == 3 else "crashed")
        with (ROOT / "data" / batch / "batch_log.jsonl").open("a") as f:
            f.write(json.dumps({"run": name, "kind": kind, "subtype": subtype, "seed": seed, "rc": rc, "status": status,
                                "minutes": round((time.time() - t0) / 60, 2), "attempt": attempt}) + "\n")
        if status != "crashed":
            return status
        restart_carla(log)   # a crash usually means the server is wedged
    return "crashed"


def write_config(batch: str, n_ticks: int, split: tuple[int, int]) -> Path:
    base = ROOT / "data" / batch
    entry = lambda d: {"run_name": d.name, "data_path": f"data/{batch}/{d.name}", "out_path": f"data/{batch}/{d.name}/embeddings"}  # noqa: E731
    nominal = sorted(d for d in base.glob("nominal_*") if status_of(d, n_ticks) == "ok")
    n_ref, n_cal = split
    anomaly = []
    for d in sorted(base.iterdir()):
        if d.is_dir() and not d.name.startswith("nominal_") and status_of(d, n_ticks) == "ok":
            a = json.loads((d / "anomalies.json").read_text())
            anomaly.append({**entry(d), "onset_tick": a["onset_frame"], "offset_tick": a["offset_frame"],
                            "anomaly_type": a["anomaly_type"], "anomaly_kind": a["kind"]})
    cfg = {"reference_runs": [entry(d) for d in nominal[:n_ref]], "calibration_runs": [entry(d) for d in nominal[n_ref : n_ref + n_cal]],
           "negative_runs": [entry(d) for d in nominal[n_ref + n_cal :]], "anomaly_runs": anomaly}
    path = ROOT / "config" / f"data_{batch}.json"
    path.write_text(json.dumps(cfg, indent=2))
    print(f"wrote {path}: {len(cfg['reference_runs'])} reference, {len(cfg['calibration_runs'])} calibration, "
          f"{len(cfg['negative_runs'])} negative, {len(anomaly)} anomaly runs")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", default="batch1")
    ap.add_argument("--nominal", type=int, default=100)
    ap.add_argument("--semantic", type=int, default=30, help="successful semantic runs to collect")
    ap.add_argument("--physics", type=int, default=30, help="successful physics runs to collect (failures are replaced, like semantic)")
    ap.add_argument("--only", help="comma-separated run kinds to record, e.g. physics (default: all)")
    ap.add_argument("--seed-start", type=int, default=1, help="first seed of each kind (use a fresh range to re-record a kind)")
    ap.add_argument("--subtypes", help="comma-separated anomaly subtypes to record, e.g. cut_in (default: all of each kind)")
    ap.add_argument("--visual", type=int, default=30)
    ap.add_argument("--split", default="60,30", help="nominal runs for the reference pool, calibration; the rest are held out")
    ap.add_argument("--n-ticks", type=int, default=600)
    ap.add_argument("--config-only", action="store_true", help="just rewrite the data config from what is on disk")
    args = ap.parse_args()
    split = tuple(int(x) for x in args.split.split(","))
    if args.subtypes:
        keep = set(args.subtypes.split(","))
        for kind in SUBTYPES:
            SUBTYPES[kind] = [s for s in SUBTYPES[kind] if s in keep] or SUBTYPES[kind]
    (ROOT / "data" / args.batch).mkdir(parents=True, exist_ok=True)
    log = ROOT / "data" / args.batch / "batch_runs.log"
    if args.config_only:
        write_config(args.batch, args.n_ticks, split)
        return

    only = set(args.only.split(",")) if args.only else {"nominal", "semantic", "physics", "visual"}
    s0 = args.seed_start
    plan = [("nominal", s, None) for s in range(s0, s0 + args.nominal)] if "nominal" in only else []
    targets = {k: n for k, n in (("semantic", args.semantic), ("physics", args.physics), ("visual", args.visual)) if k in only}
    for kind, n in targets.items():
        plan += [(kind, s, SUBTYPES[kind][(s - s0) % len(SUBTYPES[kind])]) for s in range(s0, s0 + n)]
    plan.sort(key=lambda p: (S.sample(p[0], p[1], p[2]).town, p[0], p[1]))   # one map load per town

    results: dict[tuple, str] = {}
    for i, (kind, seed, sub) in enumerate(plan):
        results[(kind, seed)] = record(kind, seed, sub, args.batch, args.n_ticks, log)
        print(f"[{i + 1}/{len(plan)}] {kind}_{seed:03d} {sub or ''}: {results[(kind, seed)]}", flush=True)

    # replace failed semantic / physics runs with fresh seeds of the same subtype until each subtype has its share
    for kind in ("semantic", "physics"):
        if kind not in targets:
            continue
        subs, n = SUBTYPES[kind], targets[kind]
        want = {sub: n // len(subs) + (i < n % len(subs)) for i, sub in enumerate(subs)}
        sub_of = {s: subs[(s - s0) % len(subs)] for (k, s) in results if k == kind}
        next_seed = s0 + n
        while next_seed < s0 + 5 * n:
            have = {sub: sum(1 for (k, s), r in results.items() if k == kind and r == "ok" and sub_of.get(s) == sub) for sub in subs}
            short = [sub for sub in subs if have[sub] < want[sub]]
            if not short:
                break
            sub = short[(next_seed - s0 - n) % len(short)]
            sub_of[next_seed] = sub
            results[(kind, next_seed)] = record(kind, next_seed, sub, args.batch, args.n_ticks, log)
            print(f"[retry] {kind}_{next_seed:03d} {sub}: {results[(kind, next_seed)]}", flush=True)
            next_seed += 1

    for kind in ("nominal", "semantic", "physics", "visual"):
        r = [v for (k, _), v in results.items() if k == kind]
        print(f"{kind:9s} {sum(x == 'ok' for x in r):3d} ok / {len(r)} attempted")
    write_config(args.batch, args.n_ticks, split)


if __name__ == "__main__":
    main()
