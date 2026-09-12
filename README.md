# Mixture-of-representations for OOD detection 

## Background

Implements a *mixture of representations* (vision / semantic / physics) to catch multiple types of OOD scenarios 

## Repo layout

```
encoders/       the three encoders + OOD scorer
sim/            actor setup, anomaly injection, ground-truth state extractor
scripts/        data collection, anomaly injection, run encoders
config/         one JSON preset per scenario
data/           collected runs 
```

- `encoders/{vision,semantic,physics}_encoder.py` — DINOv2 / OWL-ViT / ground-truth-kinematics, one anomaly type each.
- `encoders/ood_scorer.py` — `NoveltyScorer`, representation-agnostic novelty score (**stub, TODO**).
- `sim/actors.py` — connect/spawn/teardown for the ego, camera, and background traffic.
- `sim/anomalies.py` — one injection function per anomaly type (semantic/physics/visual).
- `sim/state_extractor.py` — `GroundTruthStateHistory`, turns live `carla.Actor` state into the plain-dict format the physics encoder expects.
- `../carla_sim/launch_carla.sh` — sibling repo: starts the CARLA 0.9.16 server.
- `scripts/run_sim.py` — single entry point for every scenario (nominal/semantic/physics/visual); see `config/*.json` for one preset per scenario.
- `scripts/inject_appearance_corruption.py` — derives an extra visual-anomaly variant from an already-collected run, without recollecting.
- `scripts/run_encoders.py` — featurize a run with all three encoders.
- `config/{nominal,semantic,physics,visual}.json` — one flag preset per scenario, loaded via `run_sim.py --config`.

## Setup

```bash
git clone <remote-url> multimodal-ood-detection && cd multimodal-ood-detection   

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# torch needs the cu128 index for Blackwell support 
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
pip install -r requirements.txt
```

```bash
# CARLA server, ~8GB (get 0.9.16 specifically -- 0.9.15's client wheels don't
# support py3.12): https://github.com/carla-simulator/carla/releases
# Lives in the sibling carla_sim repo, not this one.
mkdir -p ../carla_sim/CARLA_0.9.16
tar -xzf CARLA_0.9.16.tar.gz -C ../carla_sim/CARLA_0.9.16

# verify:
../carla_sim/launch_carla.sh &
python -c "
import carla
c = carla.Client('127.0.0.1', 2000); c.set_timeout(30.0)
print('server:', c.get_server_version(), 'client:', c.get_client_version())
print('map:', c.get_world().get_map().name)
"
```

## Usage guide

```bash
source .venv/bin/activate
../carla_sim/launch_carla.sh &   # start the simulator if it's not already running

# one command per scenario, via a config file (see config/*.json)
python scripts/run_sim.py --config config/nominal.json
python scripts/run_sim.py --config config/semantic.json
python scripts/run_sim.py --config config/physics.json
python scripts/run_sim.py --config config/visual.json
# flags override individual config values, e.g.:
python scripts/run_sim.py --config config/physics.json --run-name physics_run_v2 --seed 1

# derive an extra visual-anomaly variant from an existing nominal run, without recollecting
python scripts/inject_appearance_corruption.py --src-run nominal_run --dst-run anomaly_blur --corruption blur --onset-frame 800 --duration 150

# featurize every run with all three encoders
for run in nominal_run anomaly_semantic anomaly_physics anomaly_visual; do
  python scripts/run_encoders.py --run-name $run
done

# scoring: NoveltyScorer is a stub (TODO) -- not wired up yet
```

### CLI reference

| Script | Key flags | Notes |
|---|---|---|
| `run_sim.py` | `--config` `--scenario {nominal,semantic,physics,visual}` `--run-name` `--n-ticks` `--trigger-tick` `--town` `--corruption/--onset-frame/--duration` `--n-background-vehicles/walkers` `--camera-width/height` `--fixed-delta` `--seed` | `--config` loads a JSON file of these same flags (dashes -> underscores as keys); CLI flags override it |
| `inject_appearance_corruption.py` | `--src-run` `--dst-run` `--corruption {flare,brightness,blur,mixed}` `--onset-frame` `--duration` `--seed` | pure post-processing, no CARLA needed; for deriving extra visual variants from a run you already collected |
| `run_encoders.py` | `--run-name` `--history-len` `--stride` `--encoders` `--relative-range` `--dt` | `--dt` must match the collection run's `--fixed-delta` |

## Known Sim Issues

**A crashed/interrupted run leaves orphaned actors** in the world (actor
spawning happens before the crash-safe `try`/`finally` in `run_sim.py`,
and teardown order is fixed in `actors.teardown()` for a reason — don't
reorder it). Clean up before retrying:

```bash
python -c "
import carla
w = carla.Client('127.0.0.1', 2000).get_world()
for a in list(w.get_actors().filter('vehicle.*')) + list(w.get_actors().filter('sensor.*')):
    a.destroy()
"
```

If the server itself died (`pgrep -f CarlaUE4-Linux-Shipping` empty), relaunch `../carla_sim/launch_carla.sh`.

## Data schema

Each run lives at `data/<run-name>/`:

- `frames/000123.jpg` — egocentric RGB frames, one per tick.
- `states.jsonl` — one JSON object per tick: `{"t", "ego": {"id","x","y","yaw","vx","vy"}, "others": [...]}`. `others` = agents within 60m of ego.

