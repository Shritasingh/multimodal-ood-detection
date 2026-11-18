# snaps_mvp results

Conformal threshold q = 0.95; z = (s - mean) / std of the dataset's nominal leave-one-out scores.

## cyclist: 224 nominal frames (219 distinct physics vectors; 13 cache sentences)

Semantic cache (nominal frames having each sentence; + = an extra label):

- 60: I see a cyclist on the sidewalk, an empty intersection and mountains in the distance.
- 92: I see a cyclist on the crosswalk, an empty sidewalk and mountains in the distance.
- 72: I see an empty sidewalk, an empty intersection and mountains in the distance.
- 0 +: I see an empty intersection.
- 0 +: I see an empty sidewalk.
- 0 +: I see a man on the sidewalk.
- 0 +: I see a woman on the sidewalk.
- 0 +: I see a child on the sidewalk.
- 0 +: I see a cyclist on the sidewalk.
- 0 +: I see mountains in the distance.
- 0 +: I see a man on the crosswalk.
- 0 +: I see a woman on the crosswalk.
- 0 +: I see a child on the crosswalk.

| group | rep | score | tau | z | tau z | above tau | nominal below |
|---|---|---|---|---|---|---|---|
| nominal (false alarms, leave-one-out) | physics | | 1.8766 | | 3.00 | 0.049 | |
| nominal (false alarms, leave-one-out) | physics_euclid | | 5.8410 | | 1.67 | 0.049 | |
| nominal (false alarms, leave-one-out) | semantic | | 0.0910 | | 1.10 | 0.000 | |
| nominal (false alarms, leave-one-out) | vision | | 0.0269 | | 2.02 | 0.049 | |
| wrongway | physics **(own)** | 2.1904 | 1.8766 | 3.69 | 3.00 | **yes** | 0.991 |
| wrongway | physics_euclid | 18.7572 | 5.8410 | 12.26 | 1.67 | **yes** | 1.000 |
| wrongway | semantic | 0.0775 | 0.0910 | -1.33 | 1.10 | no | 0.000 |
| wrongway | vision | 0.0059 | 0.0269 | -0.51 | 2.02 | no | 0.487 |
| droplet | physics | 0.1875 | 1.8766 | -0.76 | 3.00 | no | 0.013 |
| droplet | physics_euclid | 2.0689 | 5.8410 | -1.43 | 1.67 | no | 0.027 |
| droplet | semantic | 0.0775 | 0.0910 | -1.33 | 1.10 | no | 0.000 |
| droplet | vision **(own)** | 0.1390 | 0.0269 | 15.56 | 2.02 | **yes** | 1.000 |

## pedestrian: 216 nominal frames (36 distinct physics vectors; 12 cache sentences)

Semantic cache (nominal frames having each sentence; + = an extra label):

- 108: I see a woman on the crosswalk, an empty sidewalk and mountains in the distance.
- 108: I see a man on the crosswalk, an empty sidewalk and mountains in the distance.
- 0 +: I see an empty intersection.
- 0 +: I see an empty sidewalk.
- 0 +: I see a man on the sidewalk.
- 0 +: I see a woman on the sidewalk.
- 0 +: I see a child on the sidewalk.
- 0 +: I see a cyclist on the sidewalk.
- 0 +: I see mountains in the distance.
- 0 +: I see a man on the crosswalk.
- 0 +: I see a woman on the crosswalk.
- 0 +: I see a child on the crosswalk.

| group | rep | score | tau | z | tau z | above tau | nominal below |
|---|---|---|---|---|---|---|---|
| nominal (false alarms, leave-one-out) | physics | | 1.6112 | | 2.15 | 0.028 | |
| nominal (false alarms, leave-one-out) | physics_euclid | | 3.3813 | | 2.36 | 0.028 | |
| nominal (false alarms, leave-one-out) | semantic | | 0.1059 | | 1.00 | 0.000 | |
| nominal (false alarms, leave-one-out) | vision | | 0.0254 | | 1.88 | 0.046 | |
| conductor | physics | 1.0115 | 1.6112 | -0.39 | 2.15 | no | 0.639 |
| conductor | physics_euclid | 2.3661 | 3.3813 | -0.54 | 2.36 | no | 0.583 |
| conductor | semantic **(own)** | 0.1401 | 0.1059 | 228.73 | 1.00 | **yes** | 1.000 |
| conductor | vision | 0.0186 | 0.0254 | 0.48 | 1.88 | no | 0.773 |
| droplet | physics | 1.0115 | 1.6112 | -0.39 | 2.15 | no | 0.639 |
| droplet | physics_euclid | 2.3661 | 3.3813 | -0.54 | 2.36 | no | 0.583 |
| droplet | semantic | 0.1056 | 0.1059 | -1.00 | 1.00 | no | 0.000 |
| droplet | vision **(own)** | 0.1456 | 0.0254 | 26.48 | 1.88 | **yes** | 1.000 |

