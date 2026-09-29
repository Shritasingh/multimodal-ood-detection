# Mixture-of-representations for OOD detection 

## Background

Implements a *mixture of representations* (vision / semantic / physics) to catch multiple types of OOD scenarios 

## Repo layout

```
encoders/       the three encoders + OOD scorer
sim/            actor setup, anomaly injection, ground-truth state extractor
scripts/        data collection, anomaly injection, run encoders
config/         sim_*.json scenario presets, data.json runs, encoders*.json encoder settings
data/           collected runs 
```


## Setup

```bash
git clone git@github.com:Shritasingh/multimodal-ood-detection.git && cd multimodal-ood-detection

conda create -n multim_ood python=3.12 -y
conda activate multim_ood
pip install --upgrade pip

# torch needs the cu128 index for Blackwell support
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
pip install -r requirements.txt
```

The vision and semantic encoders download public models from Hugging Face. To
avoid unauthenticated Hub requests and receive the higher rate limit, create a
read-only token at https://huggingface.co/settings/tokens and authenticate
before running the encoders:

```bash
hf auth login
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
conda activate multim_ood
../carla_sim/launch_carla.sh &   # start the simulator if it's not already running

# one command per scenario, via a config file (config/sim_*.json); run names are templated on the seed
python scripts/run_sim.py --config config/sim_nominal.json --seed 3          # -> data/nominal_seed03
python scripts/run_sim.py --config config/sim_anom_semantic.json --seed 11   # -> data/anom_sem_s11
python scripts/run_sim.py --config config/sim_anom_physics.json --seed 11    # -> data/anom_phys_s11
python scripts/run_sim.py --config config/sim_anom_visual.json --seed 11     # -> data/anom_flare_s11
# flags override individual config values, e.g.:
python scripts/run_sim.py --config config/sim_anom_physics.json --run-name physics_run_v2 --seed 1

# create visual-anomaly variant from an existing nominal run, without recollecting
python scripts/inject_appearance_corruption.py --src-run nominal_run --dst-run anomaly_blur --corruption blur --onset-frame 800 --duration 150

# encode + score every run in config/data.json with the encoders in config/encoders.json
# (no CLI flags -- edit data.json for runs/out paths, encoders.json for encoders/k/metric)
python scripts/run_encoders.py   # expensive: runs DINOv2/OWL-ViT/BERT/physics over every frame,
                                  # caches raw embeddings to each run's out_path
python scripts/run_metrics.py    # cheap: scores the cached embeddings via kNN distance
                                  # (cosine or Mahalanobis, per-encoder in config), writes
                                  # anomaly_scores.csv/.png to each anomaly run's out_path.
                                  # re-run this alone after editing k/metric -- no need to
                                  # re-run run_encoders.py unless the encoders themselves changed
```

### Nominal seeds, patch vision, role physics

```bash
python scripts/run_nominal_batch.py                          # nominal seeds 1-10 -> data/nominal_seedNN (needs CARLA)
python scripts/run_encoders.py                              # vision_patch + physics_roles + semantic (top-n labels)
python scripts/run_metrics.py                               # scores anomaly + negative runs against the pooled seeds
python scripts/run_conformal.py                             # conformal thresholds; FPR/AUROC vs data.json's negative run
python scripts/run_encoders.py config/encoders_qwen.json    # Qwen3-VL encoders (same runs, same scoring)
python scripts/run_sim.py --config config/sim_anom_semantic.json --seed 3   # anomaly replay of seed 3
```

Configs: `config/data.json` holds the runs (`reference_runs` = nominal pool, `anomaly_runs` with onset/offset ticks,
`negative_runs` = held-out nominal); `config/encoders.json` and `config/encoders_qwen.json` hold only encoder settings
(`encoders` maps each encoder to its scoring params). `text_embedding.embedders` picks the label embedder (`bert`,
`minilm`, `clip`) for `semantic` and the Qwen label encoders; listing several runs each as its own series
(`semantic@bert`, `semantic@clip`, ...). Superseded configs are in `config/archive/`.

`semantic` keeps the `top_n` labels whose best-box score reaches `min_score` (`semantic_encoder_params`, default 5 and 0.025).
`run_sim.py` starts the prop at the first try from `--trigger-tick` where the ego is not at an intersection and no agent is
within 10 m ahead (retrying every 10 s), and the swerve when the encoder's lead role is a background vehicle in the ego lane.
`analysis/` holds diagnostics; `scripts/check_prop_ground.py` checks prop heights.

### CLI reference

| Script | Key flags | Notes |
|---|---|---|
| `run_sim.py` | `--config` `--scenario {nominal,semantic,physics,visual}` `--run-name` `--n-ticks` `--trigger-tick` `--town` `--corruption/--onset-frame/--duration` `--n-background-vehicles/walkers` `--camera-width/height` `--fixed-delta` `--seed` | `--config` loads a JSON file of these same flags (dashes -> underscores as keys); CLI flags override it |
| `inject_appearance_corruption.py` | `--src-run` `--dst-run` `--corruption {flare,brightness,blur,mixed}` `--onset-frame` `--duration` `--seed` | pure post-processing, no CARLA needed; for deriving extra visual variants from a run you already collected |
| `run_encoders.py` | optional config path (default `config/encoders.json`, runs from `config/data.json`) | vision/semantic are per-frame (no history window); physics uses a 2-frame window so its finite-difference features stay defined |
| `run_metrics.py` | optional config path (as above) | reads the `.npz` files `run_encoders.py` wrote; each encoder's params in `encoders` (`k`, `metric`/`metrics`) control the kNN scoring |

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

