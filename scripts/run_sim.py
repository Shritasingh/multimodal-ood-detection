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
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # path to sim/ package (repo root)
from encoders.physics_encoder import RolePhysicsEncoder
from sim import actors, anomalies, placement
from sim.state_extractor import GroundTruthStateHistory


CLEAR_AHEAD_M = 10.0  # semantic: no agent closer than this ahead of the ego


def semantic_blocker(world, ego) -> str | None:
    """Reason not to inject the prop now (ego at an intersection, or an agent within CLEAR_AHEAD_M ahead), else None."""
    wp = placement.ego_lane_waypoint(world, ego)
    if wp is None or wp.is_junction:
        return "ego is at an intersection"
    agents = placement.agents_in_corridor(world, ego, CLEAR_AHEAD_M, half_width_m=2.5)
    if agents:
        return f"{len(agents)} agent(s) within {CLEAR_AHEAD_M:.0f} m ahead of the ego"
    return None


def physics_blocker(world, ego, clear_ahead_m: float = 0.0, max_yaw_rate_dps: float = 4.0) -> str | None:
    """Reason not to start the swerve now (ego at an intersection, turning, or an agent within clear_ahead_m ahead), else None."""
    wp = placement.ego_lane_waypoint(world, ego)
    if wp is None or wp.is_junction:
        return "ego is at an intersection"
    rate = placement.ego_yaw_rate_dps(ego)
    if rate > max_yaw_rate_dps:
        return f"ego is turning ({rate:.0f} deg/s)"
    if clear_ahead_m > 0 and placement.agents_in_corridor(world, ego, clear_ahead_m, half_width_m=2.5):
        return f"an agent is within {clear_ahead_m:.0f} m ahead of the ego"
    return None


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
    p.add_argument("--trigger-tick", type=int, default=600,
                   help="semantic/physics: earliest tick the anomaly may start. It starts at the first tick from here "
                   "on where the scene allows it (see sim/anomalies.py); the actual onset is written to anomalies.json")
    p.add_argument("--trigger-retry-ticks", type=int, default=None,
                   help="semantic/physics: ticks to wait before trying again when the scene does not allow the anomaly "
                   "(default: 100 = 10 s for semantic, 1 for physics)")
    p.add_argument("--object-blueprint", default=None, help="semantic only: e.g. static.prop.box01 (default: cycles by onset tick)")
    p.add_argument("--spawn-distance", type=float, default=15.0, help="semantic only: metres ahead on the ego lane")
    p.add_argument("--add-lead", action=argparse.BooleanOptionalAction, default=False,
                   help="physics only: spawn a dedicated lead vehicle on the ego lane and swerve it, instead of hijacking a natural lead")
    p.add_argument("--lead-distance", type=float, default=15.0, help="physics with --add-lead: metres ahead of the ego")
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
    # A re-run with the same --run-name must not leave stale frames behind:
    # states.jsonl is truncated below (open "w"), so old, longer-run frames
    # sitting past a new shorter run's last tick would silently desync from
    # states.jsonl (run_encoders.py's frame/state count assert exists to
    # catch exactly this).
    shutil.rmtree(out_dir / "frames", ignore_errors=True)
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
    (out_dir / "run_info.json").write_text(json.dumps({
        "run_name": args.run_name, "scenario": args.scenario, "seed": args.seed, "town": world.get_map().name.split("/")[-1],
        "n_ticks": args.n_ticks, "fixed_delta": args.fixed_delta,
        "n_vehicles_requested": args.n_background_vehicles, "n_vehicles_spawned": len(vehicles),
        "n_walkers_requested": args.n_background_walkers, "n_walkers_spawned": len(walkers),
    }, indent=2))

    history = GroundTruthStateHistory(max_len=args.n_ticks + 1)
    anomaly_info = {"anomaly_type": args.scenario} if args.scenario != "nominal" else None
    cutin_state = None
    pending = args.scenario in ("semantic", "physics")
    last_reason = None
    give_up_tick = args.n_ticks - 60
    next_attempt = args.trigger_tick
    retry_ticks = args.trigger_retry_ticks or (1 if args.scenario == "physics" and not args.add_lead else 100)
    verify_lead_id = None

    try:
        with open(states_path, "w") as states_f:
            for tick_idx in range(args.n_ticks):
                if pending and tick_idx >= next_attempt and tick_idx < give_up_tick:
                    fields, reason = None, None
                    if args.scenario == "semantic":
                        reason = semantic_blocker(world, ego)
                        if reason is None:
                            fields = anomalies.inject_semantic(world, ego, tick_idx, args.n_ticks, args.object_blueprint, args.spawn_distance)
                            if fields["actor_id"] is None:
                                fields, reason = None, fields["spawn_failed"]
                    else:
                        reason = physics_blocker(world, ego, args.lead_distance + 5.0 if args.add_lead else 0.0)
                        if reason is None:
                            if args.add_lead:
                                lead, info = anomalies.add_lead_vehicle(world, ego, args.lead_distance, args.seed)
                                if lead is not None:
                                    vehicles.append(lead)
                                    verify_lead_id = lead.id
                            else:
                                lead, info = anomalies.find_lead_vehicle(world, ego, vehicles, list(history.buffer)[-3:])
                            if lead is None:
                                reason = info
                            else:
                                cutin_state, fields = anomalies.start_cutin(world, ego, lead, info, tm, tick_idx)
                    if fields is not None:
                        anomaly_info.update(fields)
                        anomaly_info["trigger_wait_ticks"] = tick_idx - args.trigger_tick
                        pending = False
                        what = fields.get("blueprint", f"swerve on lead actor {fields.get('actor_id')}")
                        print(f"tick {tick_idx}: anomaly started ({what}), waited {tick_idx - args.trigger_tick} ticks")
                    else:
                        last_reason = reason
                        next_attempt = tick_idx + retry_ticks
                        if (tick_idx - args.trigger_tick) % 50 == 0 or retry_ticks > 1:
                            print(f"tick {tick_idx}: not starting the anomaly -- {reason}; next try at tick {next_attempt}")

                if cutin_state is not None:
                    if not anomalies.step_cutin(cutin_state, tm):
                        cutin_state = None

                world.tick()
                image = image_queue.get()
                image.save_to_disk(str(out_dir / "frames" / f"{tick_idx:06d}.jpg"))

                history.push(world, ego, sim_time=world.get_snapshot().timestamp.elapsed_seconds)
                states_f.write(json.dumps(history.buffer[-1]) + "\n")
                if verify_lead_id is not None and len(history.buffer) >= RolePhysicsEncoder.window:
                    roles = RolePhysicsEncoder().encode(list(history.buffer)[-RolePhysicsEncoder.window:])[1]["roles"]
                    anomaly_info["lead_role_verified"] = roles.get("lead") == verify_lead_id
                    print(f"tick {tick_idx}: added lead is the encoder's lead role: {anomaly_info['lead_role_verified']}")
                    verify_lead_id = None

                if tick_idx % 100 == 0:
                    print(f"tick {tick_idx}/{args.n_ticks}")

        if pending:
            anomaly_info.update(anomaly_injection_failed=True, last_reason=last_reason)
            print(
                f"WARNING: the {args.scenario} anomaly never started (last reason: {last_reason}). "
                "anomalies.json is marked anomaly_injection_failed; discard this run for scoring.",
                file=sys.stderr,
            )

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
