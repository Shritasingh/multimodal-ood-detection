# runs_aesop: semantic OOD detection with AESOP-style concepts

Recreates the fast anomaly detector of AESOP (Sinha et al., 2024, arXiv 2407.08735) on CARLA, for semantic anomalies only.
Each frame becomes a set of **concepts**: an object plus where it is, e.g. "bus stop on the sidewalk". The concepts go
into a fixed monitoring prompt, and a text model embeds the prompt. A frame's anomaly score is its distance to a cache of
nominal embeddings.

The code reuses `multimodal-ood-detection` (the parent repo). `sim/` and `encoders/` here merge with its `sim/` and
`encoders/` as namespace packages, and the recorder imports its helpers (`run_scenario`, `run_batch`, `sim.scenarios`).

## The experiment (`sim/semantic24.py`, `sim/concepts.py`)

| | |
|---|---|
| Towns | Town01 (nominal pool) and Town02 (test). ClearNoon, 10 Hz, 600 ticks per run. |
| Runs | 20 Town01 nominal runs, 10 Town02 nominal runs and 4 Town02 test runs. The test runs are one nominal run plus hay bales, pedestrian and emergency, one anomaly each, which must be clearly visible (≥ 30 ticks on screen, ≥ 3000 px). |
| Objects (39) | vehicles, people, traffic control, street furniture, barriers, built / nature, and the anomaly words: hay bale, ambulance, police car, warning sign. Trash cans, garbage bags, recycling bins, mailboxes and bus stops are also placed along every route. |
| Predicates (5) | on the road, on the sidewalk, at the intersection, beside the road, in the sky |
| Anomalies | **new object:** 2-3 hay bales on the right sidewalk. **New predicate:** a pedestrian stepping into the ego lane. **New concept:** an ambulance and a police car with lights at the kerb, plus an "accident ahead" sign. |
| Driving | The ego is a CARLA BehaviorAgent that slows smoothly for curves, red lights, walkers and cars ahead. Background traffic is smooth and keeps its lane, and every car drives 10 % below the limit. Any collision, or a car leaving its lane, re-records the run. |
| Walkers | They never jaywalk. A walker about to step onto a road lane (a crosswalk) is stopped at the kerb, so no nominal run has a person on the road. |
| Pools | **Withheld routes:** the 20 Town01 runs. **Included routes:** Town01 runs 01-10 plus the 10 Town02 nominal runs. Both pools use the same 4 test runs. |

**Ground truth** comes from instance segmentation, depth and the OpenDRIVE lanes, and can be corrected by hand in the
dashboard. In Town01/02 the map has no lane on a strip just behind the kerb line, where most street lamps, traffic
lights and signs stand. The kerb-strip rule calls any non-scenery, non-vehicle object with no lane under it, within
1.5 m of a driving lane's edge, "on the sidewalk" (`KERB_SIDEWALK_M`).

**Corpus and scoring:**
- Nominal pool runs are encoded from their hand-corrected ground truth.
- Test runs are encoded twice, from raw OWL-ViT concepts and from ground truth, and scored against the ground-truth cache.
- Text embedders: e5-mistral-7b-instruct (8-bit), MiniLM, MPNet and BERT-large.
- Scores: kNN cosine (k = 1, 3, 5, 10) and Mahalanobis.
- Threshold: run-level conformal. Each pool run's 95th-percentile leave-one-out score is one Q, and tau = the ⌈(N+1)·0.95⌉-th smallest Q.
- Metrics follow Elhafsi et al.: each anomaly window is one positive, detected if any of its frames is above tau, and every other test frame is one negative. Accuracy = (TPR + TNR) / 2.

## Running it (from `multimodal-ood-detection/`; `pip install -r aesop_recreate/requirements.txt` first)

```bash
.venv/bin/python aesop_recreate/scripts/run_semantic24.py               # CARLA server up: record every missing run, write config/data_semantic24.json
.venv/bin/python aesop_recreate/scripts/extract_gt_concepts.py          # ground-truth concepts, every frame (no CARLA needed)
.venv/bin/python aesop_recreate/scripts/owl_concepts.py                 # raw OWL-ViT concepts, every frame
.venv/bin/python aesop_recreate/scripts/run_semantic24.py --config-only # anomaly windows from the ground truth
.venv/bin/python aesop_recreate/scripts/build_corpus.py                 # prompts -> embeddings, per run: corpus/frames.jsonl, <embedder>@<src>.npy
.venv/bin/python aesop_recreate/scripts/score.py                        # scores, kNN, conformal tau, metrics -> _results/score_<pool>.json
.venv/bin/python aesop_recreate/scripts/experiment.py                   # Experiment tab sweeps -> _results/experiment_<pool>.json (~5 min)
.venv/bin/python aesop_recreate/scripts/embed_map.py                    # Experiment tab embedding view -> _results/embed_map_<pool>.json
.venv/bin/python aesop_recreate/dashboard/serve.py                      # http://127.0.0.1:8010 (--port to change)
```

The dashboard's Images tab saves ground-truth corrections to `data/semantic24/gt_overrides.json`. Re-running
`extract_gt_concepts.py` keeps them. After editing, re-run `build_corpus.py` and then `score.py`; prompts that didn't
change come from the cache.

## Layout

```
sim/concepts.py                  vocabulary, prompt template, ground-truth extraction, override application
sim/semantic24.py                run plan and per-run sampling, recording parameters
scripts/run_semantic24.py        CARLA recorder (frames, instance segmentation, depth, camera pose, actors) + data config
scripts/extract_gt_concepts.py   ground-truth concepts (offline, from the saved maps)
scripts/owl_concepts.py          raw OWL-ViT concepts
scripts/build_corpus.py          per-run corpus: concepts, prompts, text embeddings
scripts/score.py                 kNN / Mahalanobis scores, run-level conformal tau, Elhafsi metrics
scripts/experiment.py            pool-size, corpus-size and threshold sweeps, ROC
scripts/embed_map.py             t-SNE / PCA of the embedding space
scripts/metrics.py               shared: anomaly windows, metrics at a threshold, Ledoit-Wolf Mahalanobis
encoders/concept_encoders.py     frame -> concepts: OWL-ViT
encoders/text_embedders.py       prompt -> vector: MiniLM, MPNet, BERT-large, e5-mistral (8-bit)
dashboard/                       Images, Scores and Experiment tabs
config/data_semantic24.json      the runs and the anomaly windows
data/semantic24/                 runs, _towns/ (OpenDRIVE maps, environment objects), _results/ (git-ignored)
```
