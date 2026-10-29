# aesop_recreate: semantic OOD detection with AESOP-style concepts

Recreates the fast anomaly detector of AESOP (Sinha et al., 2024, arXiv 2407.08735) on CARLA, for semantic anomalies only:
each frame becomes a set of **concepts** (object + where it is, e.g. "bus stop on the sidewalk"), the concepts are written
into a fixed monitoring prompt, the prompt is embedded with a text model, and a frame's anomaly score is its distance to
a cache of nominal embeddings. The nominal runs are built to be concept-dense, so nominal scores stay low.

The code reuses `multimodal-ood-detection` (the parent repo): `sim/` and `encoders/` here merge with its `sim/` and
`encoders/` (namespace packages), and the recorder, scorer and dashboard import its helpers.

## The experiment (`sim/semantic24.py`, `sim/concepts.py`)

| | |
|---|---|
| Towns | Town01 (nominal pool), Town02 (test), ClearNoon, 600 ticks at 10 Hz |
| Runs | 20 Town01 nominal, 10 Town02 nominal, 4 Town02 test: nominal, hay bales, pedestrian, emergency |
| Objects (39) | vehicles, people, traffic control, street furniture (trash can, garbage bag, recycling bin, mailbox, bus stop are also placed along every route), barriers, built / nature, and the anomaly words hay bale, ambulance, police car, warning sign |
| Predicates (5) | on the road, on the sidewalk, at the intersection, beside the road, in the sky |
| Anomalies | new object: 2-3 hay bales on the right sidewalk · new predicate: a pedestrian standing in the ego lane · new concept: ambulance + police car with lights at the kerb and an "accident ahead" sign |
| Pools | **withheld routes** (`semantic_runs_24`): the 20 Town01 runs · **included routes** (`semantic_runs_24_incl`): Town01 01-10 + the 10 Town02 nominal runs; the same 4 test runs |

Concept sources: ground truth (instance segmentation + depth + the OpenDRIVE lanes, hand-correctable), OWL-ViT over all
object x predicate phrases (each box's best concept, score >= 0.05, at most 15), Qwen3-VL with the closed vocabulary,
Qwen3-VL open vocabulary. Text embedders: e5-mistral-7b-instruct (8-bit), MiniLM, MPNet, BERT-large. End-to-end
baselines: CLIP image embeddings, DINOv2 patch tokens. Scores: kNN cosine (k = 1, 3, 5, 10) and Mahalanobis to the
nominal mean; run-level conformal thresholds (leave-one-run-out over the pool).

The assessment compares three groups, each in both pools: **OWL-ViT** concepts (ground truth / raw detections) x the
four text embedders, **Qwen** concepts (closed / open vocabulary) x the four text embedders, and the **end-to-end**
CLIP / DINOv2 baselines. The dashboard's headline series (`headline_series` in the encoder configs) take one of each
at kNN k = 5 with e5-mistral; every other embedder / k / Mahalanobis combination is in the score csvs.

## Running it (from `multimodal-ood-detection/`, CARLA server up; `pip install -r aesop_recreate/requirements.txt` first)

```bash
.venv/bin/python aesop_recreate/scripts/run_semantic24.py               # record every missing run, write config/data_semantic24*.json
.venv/bin/python aesop_recreate/scripts/extract_gt_concepts.py          # ground-truth concepts, every frame (no CARLA needed)
.venv/bin/python aesop_recreate/scripts/run_concepts.py owl             # OWL-ViT concepts, every frame
.venv/bin/python analysis/serve.py                                      # Labels tab: review / correct the ground truth
.venv/bin/python aesop_recreate/scripts/run_concepts.py qwen            # Qwen closed + open concepts, every 5th frame
.venv/bin/python aesop_recreate/scripts/run_concepts.py embed           # descriptions -> text embeddings (applies the GT corrections)
.venv/bin/python aesop_recreate/scripts/run_concepts.py image           # CLIP + DINOv2 patch tokens
.venv/bin/python aesop_recreate/scripts/score_semantic24.py aesop_recreate/config/encoders_semantic24.json
.venv/bin/python aesop_recreate/scripts/score_semantic24.py aesop_recreate/config/encoders_semantic24_incl.json
```

Ground-truth corrections made in the dashboard's Labels tab are stored in `data/semantic24/gt_overrides.json`, apart
from the extracted labels, so re-running `extract_gt_concepts.py` after a rule change keeps them; re-run `embed` (and the
scoring) after correcting.

## Layout

```
sim/concepts.py              vocabulary, prompt template, ground-truth extraction, override application
sim/semantic24.py            run plan and per-run sampling
scripts/run_semantic24.py    CARLA recorder (frames, instance segmentation, depth, camera pose, actors)
scripts/extract_gt_concepts.py
scripts/run_concepts.py      OWL-ViT / Qwen concepts, text and image embeddings
scripts/score_semantic24.py  kNN / Mahalanobis scores, conformal thresholds, AUROC
encoders/concept_encoders.py     frame -> concepts: OWL-ViT, Qwen3-VL (closed / open vocabulary prompts)
encoders/text_embedders.py       description -> vector: MiniLM, MPNet, BERT-large, e5-mistral (8-bit)
encoders/end_to_end_encoders.py  frame -> vector: CLIP image, DINOv2 patch tokens
config/                      data + encoder configs of both pools
data/semantic24/             runs (git-ignored), _towns/ (OpenDRIVE maps, environment objects)
```
