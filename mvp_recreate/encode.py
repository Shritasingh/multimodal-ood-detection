"""Encode every observation -> data/<dataset>/_emb/<group>.npz, and every semantic sentence -> data/_sentences.npz

    .venv/bin/python mvp_recreate/encode.py                          # groups missing / changed, missing sentences
    .venv/bin/python mvp_recreate/encode.py --force [--reps physics vision semantic]   # recompute those reps

  physics   PhysicsEncoder (physics.py): 2 role slots (agent, pedestrian) x the last `history` states x (x, y, vx, vy,
            cos yaw, sin yaw, speed): 70-d
  semantic  the frame's ground-truth labels (labels_of: each visible agent's '<object> <predicate>' - a cyclist only "on the
            sidewalk" / "on the crosswalk" - plus "empty sidewalk" / "empty intersection" when empty, plus "mountains in
            the distance");
            the labels are concatenated into ONE sentence per frame, sentence(labels): "I see a cyclist on the sidewalk,
            an empty intersection and mountains in the distance."; every distinct sentence - of any frame, plus one
            per config semantic.extra_labels label ("I see an empty sidewalk.") - is embedded ONCE with
            e5-mistral-7b-instruct (8-bit, last token, aesop_recreate's E5MistralEmbedder): 4096-d, unit norm, in
            data/_sentences.npz {sentences, labels (JSON lists), emb}
  vision    DINOv2-base patch tokens of the whole frame resized to 224 x 294 (DinoPatchEncoder): 16 x 21 = 336 patches
            x 768, float16 (scored per patch in score.py)

Per group npz: idx, labels (one JSON list per frame), desc (the labels joined, for display), physics, vision.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
MAIN = HERE.parent
sys.path[:0] = [str(HERE), str(MAIN), str(MAIN / "aesop_recreate")]

import physics as P  # noqa: E402

from cfg import CACHES, CFG, DATA  # noqa: E402

SEM = CFG["semantic"]
SENTENCES_FILE = DATA / "_sentences.npz"


def _crosswalk_dist(x: float, y: float) -> float:
    """Distance (m) from a ground point to the nearest crosswalk (0 inside one), from data/_geometry.json."""
    p, best = np.array([x, y]), np.inf
    for q in (np.array(c) for c in json.loads((DATA / "_geometry.json").read_text())["crosswalks"]):
        n, inside = len(q), False
        for i in range(n):
            (x1, y1), (x2, y2) = q[i], q[(i + 1) % n]
            if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
                inside = not inside
            ab = q[(i + 1) % n] - q[i]
            t = float(np.clip((p - q[i]) @ ab / (ab @ ab), 0.0, 1.0))
            best = min(best, 0.0 if inside else float(np.linalg.norm(p - (q[i] + t * ab))))
        if inside:
            return 0.0
    return best


def labels_of(rec: dict) -> list[str]:
    """The frame's ground-truth labels:
    - '<object> <predicate>' of every agent with >= visible_px pixels.
      A cyclist's predicate is one of semantic.cyclist_predicates: "on the crosswalk" if it is on a crosswalk or
      within crosswalk_near_m of one, else "on the sidewalk". Other agents keep the map's predicate.
      A plain pedestrian's object is its walker's noun (semantic.walker_nouns: man / woman / child, from CARLA's gender
      and age attributes); the traffic conductor keeps its object.
    - "empty sidewalk" if no labelled agent is on the sidewalk.
    - "empty intersection" if none is on a crosswalk or in the intersection (the crosswalks count as part of it).
    - semantic.always_labels ("mountains in the distance")."""
    last = {o["id"]: o for o in rec["history"][-1]["others"]}
    labels, preds = [], set()
    for i, a in rec["agents"].items():
        o = last.get(int(i))
        if o is None or o.get("px", 0) < CFG["scene"]["visible_px"]:
            continue
        pred = o["pred"]
        if a["type"] == "cyclist":
            near = pred == "on the crosswalk" or _crosswalk_dist(o["x"], o["y"]) <= SEM["crosswalk_near_m"]
            pred = SEM["cyclist_predicates"][1] if near else SEM["cyclist_predicates"][0]
        obj = a["obj"]
        if obj == "pedestrian":                                                      # man / woman / child, from CARLA's
            obj = SEM["walker_nouns"].get(a.get("bp"), obj)                              # gender / age of the walker
        labels.append(f"{obj} {pred}")
        preds.add(pred)
    if "on the sidewalk" not in preds:
        labels.append(SEM["empty_sidewalk_label"])
    if not preds & {"on the crosswalk", "in the intersection"}:
        labels.append(SEM["empty_label"])
    return labels + list(SEM["always_labels"])


def _with_article(label: str) -> str:
    return label if label.split()[0] in SEM["no_article"] else ("an " if label[0] in "aeiou" else "a ") + label


def sentence(labels: list[str]) -> str:
    """The text a frame's labels are embedded as, concatenated in order:
    'I see an empty sidewalk.' / 'I see a cyclist on the sidewalk and mountains in the distance.' /
    'I see a pedestrian on the crosswalk, an empty sidewalk and mountains in the distance.'"""
    parts = [_with_article(l) for l in labels]
    return "I see " + (parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]) + "."


def states(cache: str, group: str) -> list[dict]:
    d = DATA / cache / group / "states"
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.exists() else []


def types_of(rec: dict) -> dict[int, str]:
    return {int(i): a["type"] for i, a in rec["agents"].items()}


def load_sentence_table() -> dict:
    """{sentences: [...], labels: [JSON list, ...], emb: (S, D)} or empty."""
    if not SENTENCES_FILE.exists():
        return {"sentences": [], "labels": [], "emb": None}
    z = np.load(SENTENCES_FILE)
    return {"sentences": z["sentences"].tolist(), "labels": z["labels"].tolist(), "emb": z["emb"]}


def update_sentence_table(label_lists: list[list[str]], force: bool) -> None:
    """Embed the sentence of every label list not yet in the table (all of them with force), one at a time (no padding:
    a sentence always gets the same vector)."""
    table = {"sentences": [], "labels": [], "emb": None} if force else load_sentence_table()
    by_sentence = {sentence(ls): ls for ls in label_lists}
    missing = [t for t in by_sentence if t not in table["sentences"]]
    if not missing:
        print(f"sentence table: {len(table['sentences'])} sentences, none missing", flush=True)
        return
    from encoders.text_embedders import E5MistralEmbedder

    text = E5MistralEmbedder()
    new = np.concatenate([text.embed([t]) for t in missing]).astype(np.float32)
    emb = new if table["emb"] is None else np.concatenate([table["emb"], new])
    np.savez(SENTENCES_FILE, sentences=np.array(table["sentences"] + missing),
             labels=np.array(table["labels"] + [json.dumps(by_sentence[t]) for t in missing]), emb=emb)
    print(f"sentence table: {len(table['sentences']) + len(missing)} sentences (+{len(missing)}: {missing})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="recompute --reps for every group (and re-embed every sentence)")
    ap.add_argument("--reps", nargs="+", default=["physics", "vision", "semantic"])
    args = ap.parse_args()
    all_lists = [[l] for l in SEM["extra_labels"]]
    todo = []
    for cache, groups in CACHES.items():
        for group in groups:
            f = DATA / cache / "_emb" / f"{group}.npz"
            recs = states(cache, group)
            all_lists += [labels_of(r) for r in recs]
            have = dict(np.load(f)) if f.exists() else {}
            stale = "idx" not in have or len(have["idx"]) != len(recs)          # re-recorded: start over
            if recs and (stale or args.force or "labels" not in have or any(r not in have for r in ("physics", "vision"))):
                todo.append((cache, group, f, recs, {} if stale else have))
    if "semantic" in args.reps:
        update_sentence_table(all_lists, args.force)
    print(f"{len(todo)} groups to encode", flush=True)
    phys = P.PhysicsEncoder(history=CFG["history"])
    vis = None
    for cache, group, f, recs, out in todo:
        f.parent.mkdir(parents=True, exist_ok=True)
        idx = np.array([r["plan"]["idx"] for r in recs])
        labels = [labels_of(r) for r in recs]
        out.pop("semantic", None)                                  # the old per-frame sentence vectors
        out.update(idx=idx, labels=np.array([json.dumps(l) for l in labels]), desc=np.array([" · ".join(l) for l in labels]))
        if (args.force and "physics" in args.reps) or "physics" not in out:
            out["physics"] = np.stack([phys.encode(r["history"], types_of(r))[0] for r in recs]).astype(np.float32)
        if (args.force and "vision" in args.reps) or "vision" not in out:
            if vis is None:
                from encoders.vision_encoder import DinoPatchEncoder

                vis = DinoPatchEncoder(image_size=(224, 294))
            v = []
            for i in range(0, len(recs), 32):
                imgs = [Image.open(DATA / cache / group / "frames" / f"{j:04d}.jpg").convert("RGB") for j in idx[i : i + 32]]
                v.append(vis.encode_batch(imgs))
            out["vision"] = np.concatenate(v).astype(np.float16)
        np.savez(f, **out)
        print(cache, group, {k: out[k].shape for k in ("labels", "physics", "vision")}, flush=True)


if __name__ == "__main__":
    main()
