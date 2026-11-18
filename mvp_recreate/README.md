# snaps_mvp: snapshots at an empty 4-way

The simplest motivating example: one anomaly per representation, each in its own minimal dataset of single snapshots.

## Pipeline

Every parameter is in `config.json`, loaded by `cfg.py`:
- `scene`: town, junction, ego placement, camera, weather;
- `cyclist`, `pedestrian`: the datasets (each with its own droplet frame);
- `scoring`: q, k, the vision top fraction, and `vision_knn_patches`, the worst patches outlined on the dashboard and used for the vision kNN;
- `dashboard`: port, heatmap opacity, heatmap max.

Change a parameter there, then re-run the steps it affects.

```
.venv/bin/python mvp_recreate/record.py [--force]   # CARLA (127.0.0.1:2000) -> data/<dataset>/<group>/{frames,states}/<idx>, data/_geometry.json
.venv/bin/python mvp_recreate/record.py --derived-only   # rebuild only the droplet frames (no CARLA)
.venv/bin/python mvp_recreate/encode.py             # -> data/<dataset>/_emb/<group>.npz
.venv/bin/python mvp_recreate/score.py              # -> data/<dataset>/_scores.npz, results/{metrics.json,results.md}
.venv/bin/python mvp_recreate/dashboard/serve.py    # http://127.0.0.1:8013
```

## Scene (`sim.py`)

- **Map and junction:** Town04_Opt junction 916, a 4-way with one lane each way and crosswalks on all four sides.
- **Hidden scenery:** every environment object except roads, road lines, sidewalks and ground, including Terrain (the hedges).
- **Weather:** noon (sun at 45°), with no clouds and no wind. Drifting clouds made consecutive frames' skies near-identical.
- **Ego:** parked 14 m before our lane's junction entry. It sits on the straight extension of the lane and faces along it, because road 52 curves behind. From there the 800×600, 60° camera sees the whole junction.
- **Agents:** each frame has the ego and at most one agent.
  - The cyclist (`vehicle.diamondback.century`: red bike, green trousers, chosen to stand out from the grey road) is kinematic.
  - The pedestrian is a CARLA walker, given a few ticks of WalkerControl so it has a standing or walking pose.
- **Physics history:** 5 states (0.5 s), built by assuming constant velocity with the heading along the agent's path. Nothing is simulated.
  - cyclist: along its route at 4 m/s;
  - pedestrian: backwards from its actual pose at the frame.

## Datasets

**cyclist.** The cyclist rides 3.5 m right of its lane centre (`cyclist.curb_offset_m`), on the sidewalk edge, at 4 m/s, in the lane's direction.

- **nominal (224 frames):**
  - The three other approaches (oncoming; the crossing lane going right; the crossing lane going left) each have three paths:
    - straight on, over the crosswalk, into the lane beyond;
    - a right turn into the adjacent lane;
    - a left turn into the other crossing lane.

    Turns are one smooth turn (`sim.one_turn`). The cyclist rides straight on in its approach lane until it meets the line of its exit lane. It turns there on an arc of radius 3 m (`cyclist.turn_radius_m`) and rides straight on in the exit lane. For a left turn that point is at the far side of the junction, alongside the crosswalk ahead, so no turn crosses the middle of the junction. Straights stay in their lane.

    Each of these 9 paths gets 20 snapshots, spread evenly from 15 m before the junction to 15 m after it (180 frames).
  - Our own lane: 20 snapshots from the opposite crosswalk onwards, riding away from us.
  - In-turn snapshots: each of the 6 turns gets 4 more snapshots, spread evenly from the start of its arc to 1.6 m (4 ticks) past its end (`cyclist.turn_snaps`, `sim.turn_arc`; 24 frames, #200-223). Without them each turn had one snapshot whose 5-step history sweeps through the curve, and those 6 frames scored above the wrong-way cyclist on standardised physics. With 4 per turn, 2 in-turn frames (#213, #221) still score just above it there; 16 per turn would put every nominal frame below it. They are appended after the ego-lane snapshots so the earlier indices (and the droplet's source, #188) stay put.
  - No empty frame: an agent-less frame was the only physics vector with no agent, so physics flagged it.
- **wrongway (1 frame):** the cyclist at nominal #188's place (ego-lane snapshot #8), riding towards us. This is the physics anomaly.
- **droplet (1 frame):** nominal #188 with 3 irregular, asymmetric blobs of the image Gaussian-blurred (radii 75 / 60 / 45 px, sigma 12 px, ~36k px in all), like water drops on the lens (`droplet`, `record.make_droplet`). They are placed one by one over the most detailed spots, each fully inside the image and not overlapping the others; they may cover the agent (here they miss the distant bike). Its physics and semantic inputs are #188's, and it is scored without #188. This is the vision anomaly.

**pedestrian.**

- **nominal (216 frames):** 6 walkers × 12 positions evenly across the crosswalk closest to the ego (15.7 m ahead; t = 0.1 to 0.98, every 0.88 m, the last next to the right sidewalk), so position 5 (t = 0.5) is where the conductor stands, × 3 poses: standing facing the ego, walking right, and walking left along the crosswalk at 1.3 m/s. The walking frames come from one slow continuous run per walker and direction (`pedestrian.record_speed` = 0.44 m/s, from `run_up_m` = 1 m before the first spot): each frame is the tick nearest its spot, so a walker is within ~2 cm of it and the spots are passed at equal time steps. (Before, each walking frame was a separate spawn 0.62 m behind its spot, and the walker, still accelerating, stopped 0.25–0.36 m short of it.) Their physics history still uses 1.3 m/s. The walkers are 0001, 0015, 0017, 0028, 0046 and 0040.
- **conductor (1 frame):** walker 0030 (a policeman) standing facing the ego in the middle of the crosswalk, labelled "traffic conductor". This is the semantic anomaly. The orange hi-vis walker 0052 was dropped because vision caught it (no other CARLA outfit resembles it).
- **droplet (1 frame):** nominal #15 (walker 0001 standing at the conductor's spot, t = 0.5; `droplet.pedestrian_src`) with the cyclist droplet's 3 blobs, reused at the same image spots (`droplet.blobs_from` = "cyclist"; ~36k px); the small one covers the walker's head and shoulders (21 % of its box). Its physics and semantic inputs are #15's, and it is scored without #15. This is the vision anomaly.

### Ground-truth concepts

The concept is `<object> <predicate>` for the agent, if it has ≥ 50 visible pixels.

| | values |
|---|---|
| objects | cyclist · pedestrian · traffic conductor |
| predicates | on the road · in the intersection · on the crosswalk · on the sidewalk · beside the road |

The map leaves unregistered strips at lane edges. Asphalt within 1 m of a driving lane counts as road.

## Representations (`physics.py`, `encode.py`)

- **Physics (`PhysicsEncoder`, 70-d):** 2 role slots × 5 steps × (x, y, vx, vy, cos yaw, sin yaw, speed), all in the ego frame. The slots are agent (the nearest non-walker, here the cyclist) and pedestrian (the nearest walker); there is no lane logic, so the wrong-way cyclist is told apart by its position and reversed heading.
  - Every agent is always encoded: there is no range limit, and every agent gets a role.
- **Semantic:** the frame's ground-truth labels (`encode.labels_of`): `<object> <predicate>` of the agent (≥ 50 visible pixels), plus "empty sidewalk" / "empty intersection" when empty and "mountains in the distance" always.
  - The labels are concatenated into one sentence per frame, e.g. "I see a pedestrian on the crosswalk, an empty sidewalk and mountains in the distance." (`encode.sentence`).
  - Each distinct sentence is embedded once with e5-mistral-7b-instruct (8-bit), into the sentence table `data/_sentences.npz`.
- **Vision:** DINOv2-base patch tokens (16 × 21 patches at 224 × 294).

## Scores and threshold (`score.py`)

- **Scores per representation, against the dataset's nominal frames:**
  - physics: standardized Euclidean distance to the 5 nearest entries of the physics cache.
  - physics_euclid: the same, with plain Euclidean distance (no per-dimension scaling).
    - The cache holds each distinct nominal physics vector once. Vectors within `duplicate_tol` count as one entry, and the entry remembers how many frames share it.
    - E.g. the 5 walkers standing on the same spot are 1 entry: the pedestrian set's 165 frames are 55 entries.
  - semantic: the semantic cache is a set of sentences: the nominal frames' distinct sentences (each once, remembering how many frames have it) plus one single-label sentence per `semantic.extra_labels` label ("I see an empty sidewalk."). A frame's score is the mean cosine distance from its sentence to the 5 nearest cache sentences (`semantic_k` = 5), so nominal frames score above 0 even though their own sentence is in the cache. A plain pedestrian is labelled man / woman / child from CARLA's gender and age tags (`semantic.walker_nouns`); the extra labels use the same words.
  - vision: per patch, the distance to the nearest nominal patch (k = 1). The frame score is the mean of its worst 15 % patches.
- **kNN neighbours (dashboard):**
  - physics: the 5 nearest cache entries by plain Euclidean distance, i.e. physics_euclid's neighbours. The standardized score's own neighbours are saved as `physics_std_knn`. With plain distance, the wrong-way cyclist's neighbours are cyclists at its spot riding away from us.
  - semantic: the 5 nearest cache sentences, each with its labels as colour-coded chips.
  - vision: each of the frame's `vision_knn_patches` (17) worst patches, the ones outlined on the dashboard, has a nearest nominal patch. The neighbours are the nominal frames those patches come from, largest overlap first. Each tile shows how many patches it supplied and their mean distance.
- **Leave-one-out:** a nominal frame is scored without its own copy in the cache.
  - An entry or sentence it shares with other nominal frames stays; one only it supplied is removed. The added labels' sentences always stay.
  - So a frame scores the same whether or not it is in the nominal set; the traffic conductor and the walkers on its spot score alike.
  - Each droplet frame is scored without its source frame's copy in the same way.
- **Threshold:** the conformal q-quantile (q = 0.95) of the nominal leave-one-out scores. A nominal frame's own threshold comes from the other nominal frames.
- **Plots:** z-scored with the nominal leave-one-out mean and std.

## Dashboard

`dashboard/serve.py` serves http://127.0.0.1:8013. The Images tab is a 2 × 2 grid:

| | left | right |
|---|---|---|
| top | the frame with the DINO patch heatmap (colour and opacity grow with the distance; the `vision_knn_patches` worst patches outlined) and the object label boxes (instance-mask boxes, visible agents only) | kNN neighbours, with a toggle per representation; each tile has the frame, the colour-coded concepts (green = same concept, amber = same object, red = not in this frame) and the top-down physics view |
| bottom | scores per representation: z-score chips and z-scored plots of every frame, with the dashed conformal thresholds | semantic labels (the concepts and the sentence) and the top-down physics view (ego frame, role-coloured agent, velocity arrow, history trail, lanes, crosswalks, camera wedge) |

The Scores tab shows each anomaly's score, z, tau, whether it is flagged, and the share of nominal frames below it.

The server never mixes old and new data:
- It reloads by itself when `score.py` rewrites the scores.
- It answers 409 (shown as a banner) while frames on disk are newer than the scores.
- Image URLs carry a data version, so the browser cannot show cached old frames.
