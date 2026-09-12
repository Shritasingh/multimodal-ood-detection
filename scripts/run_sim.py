"""Builds simulated runs, injects anomalies, and saves frames + ground truth states to /data/<run_name>

Usage (flags):
    .venv/bin/python scripts/run_sim.py --run-name nominal_run --n-ticks 2000

    .venv/bin/python scripts/run_sim.py --run-name anomaly_semantic \
        --scenario semantic --trigger-tick 600 --n-ticks 1200

Usage (config file -- see config/*.json for one per scenario):
    .venv/bin/python scripts/run_sim.py --config config/physics.json

    # flags still override individual values from the config file:
    .venv/bin/python scripts/run_sim.py --config config/physics.json --run-name my_run_v2

Config file keys match the flag names with dashes replaced by underscores
(e.g. --run-name -> "run_name", --n-ticks -> "n_ticks").
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # path to sim/ package (repo root)
from sim import actors, anomalies
from sim.state_extractor import GroundTruthStateHistory


def parse_args():
    # First pass: peek at --config only, so its values can become the
    # defaults for the real parser below (CLI flags still win over it since
    # argparse only applies a default when the flag itself wasn't passed).
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument("--config", type=Path, help="JSON file of args; CLI flags override its values")
    conf_args, _ = conf_parser.parse_known_args()

    p = argparse.ArgumentParser(parents=[conf_parser])
    p.add_argument("--scenario", choices=["nominal", "semantic", "physics", "visual"], default="nominal")
    p.add_argument("--run-name")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument(
        "--town",
        default=None,
        help="Map to load via client.load_world(). Default (None) reuses whatever map "
        "the server already has loaded. Segfaults the server if it was launched with "
        "-quality-level=Low (carla_sim/launch_carla.sh no longer passes that flag).",
    )
    p.add_argument("--n-ticks", type=int, default=2000)
    p.add_argument("--trigger-tick", type=int, default=600, help="semantic/physics scenarios only")
    p.add_argument("--fixed-delta", type=float, default=0.1)
    p.add_argument("--n-background-vehicles", type=int, default=30)
    p.add_argument("--n-background-walkers", type=int, default=15)
    p.add_argument("--camera-width", type=int, default=800)
    p.add_argument("--camera-height", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--corruption", choices=list(anomalies.VISUAL_CORRUPTIONS), default="flare", help="visual scenario only")
    p.add_argument("--onset-frame", type=int, default=800, help="visual scenario only")
    p.add_argument("--duration", type=int, default=150, help="visual scenario only")

    if conf_args.config:
        with open(conf_args.config) as f:
            p.set_defaults(**json.load(f))

    args = p.parse_args()
    if not args.run_name:
        p.error("--run-name is required (via CLI or config file)")
    return args


def main():
    args = parse_args()
    out_dir = Path(__file__).resolve().parent.parent / "data" / args.run_name
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    states_path = out_dir / "states.jsonl"

    client, world, tm, settings = actors.connect(args.host, args.port, args.town, args.fixed_delta, args.seed)
    ego = actors.spawn_ego(world, tm)
    camera = actors.spawn_ego_camera(world, ego, args.camera_width, args.camera_height)
    image_queue: "queue.Queue" = queue.Queue()
    camera.listen(image_queue.put)

    vehicles, walkers, controllers = actors.spawn_background_traffic(
        client, world, tm, args.n_background_vehicles, args.n_background_walkers, args.seed
    )
    print(f"Spawned {len(vehicles)} background vehicles, {len(walkers)} walkers.")

    history = GroundTruthStateHistory(max_len=args.n_ticks + 1)
    anomaly_info = {"anomaly_type": args.scenario} if args.scenario != "nominal" else None
    cutin_state = None

    try:
        with open(states_path, "w") as states_f:
            for tick_idx in range(args.n_ticks):
                if tick_idx == args.trigger_tick:
                    if args.scenario == "semantic":
                        anomaly_info.update(anomalies.inject_semantic(world, ego, tick_idx, args.n_ticks))
                        print(f"tick {tick_idx}: spawned novel object {anomaly_info['blueprint']}")
                    elif args.scenario == "physics":
                        cutin_state, fields = anomalies.start_cutin(world, ego, vehicles, tm, tick_idx)
                        anomaly_info.update(fields)
                        if cutin_state is not None:
                            print(f"tick {tick_idx}: triggering cut-in on actor {fields['actor_id']}")
                        else:
                            print(
                                f"WARNING: tick {tick_idx}: no lead vehicle found within range -- "
                                "no cut-in was triggered. This run has no actual anomaly; "
                                "anomalies.json is marked anomaly_injection_failed so any scoring done on "
                                "it should be discarded.",
                                file=sys.stderr,
                            )

                if cutin_state is not None:
                    if not anomalies.step_cutin(cutin_state, tm):
                        cutin_state = None

                world.tick()
                image = image_queue.get()
                image.save_to_disk(str(out_dir / "frames" / f"{tick_idx:06d}.jpg"))

                history.push(world, ego, sim_time=world.get_snapshot().timestamp.elapsed_seconds)
                states_f.write(json.dumps(history.buffer[-1]) + "\n")

                if tick_idx % 100 == 0:
                    print(f"tick {tick_idx}/{args.n_ticks}")

        if args.scenario == "visual":
            anomaly_info.update(
                anomalies.inject_visual(out_dir, args.corruption, args.onset_frame, args.duration, seed=args.seed)
            )
    finally:
        if anomaly_info is not None:
            anomaly_info.setdefault("n_frames", args.n_ticks)
            (out_dir / "anomalies.json").write_text(json.dumps(anomaly_info, indent=2))
        actors.teardown(client, world, tm, settings, camera, ego, vehicles, walkers, controllers)

    print(f"Done. Data written to {out_dir}")


if __name__ == "__main__":
    main()
