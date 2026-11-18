"""The semantic24 corpus: one point per scored frame (every P.STRIDE-th tick) of every run, its concepts, the prompt
built from them, and that prompt's vector from each text model.

    .venv/bin/python aesop_recreate/scripts/build_corpus.py                                     # every run
    .venv/bin/python aesop_recreate/scripts/build_corpus.py --runs t2_haybale --embedders minilm

Nominal pool runs are encoded from their ground truth (concepts_gt_raw.jsonl with gt_overrides.json applied: the
dashboard's corrections); their raw OWL-ViT concepts are kept for information only. Test runs are encoded twice: from
their raw OWL-ViT concepts (concepts_owl.jsonl, the detector's view) and from their ground truth; score.py scores both
against the ground-truth cache. The prompt is sim/concepts.describe(): the de-duplicated, sorted concepts in AESOP's
template. Per run, <run>/corpus/:
  frames.jsonl          {t, raw: [OWL-ViT concepts], gt: [ground-truth concepts], prompts: {source: prompt}}, one line per point
  <embedder>@<src>.npy  (points, d) float32, L2-normalised, row i = line i of frames.jsonl; src gt (every run), owl (test runs)
Vectors are cached per prompt in <dataset>/_corpus_cache/<embedder>.npz, so after ground-truth corrections only the
prompts that changed are embedded again.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent      # aesop_recreate/
sys.path[:0] = [str(ROOT), str(ROOT.parent)]       # sim/ and encoders/ merge with multimodal-ood-detection's
from encoders.text_embedders import TEXT_EMBEDDERS, load_text_embedder  # noqa: E402
from sim import concepts as K, semantic24 as P  # noqa: E402


def runs_of(data: Path, names: list[str] | None) -> list[P.Plan]:
    """The dataset's recorded runs with concepts extracted."""
    return [p for p in P.plan()
            if (data / p.name / "concepts_gt_raw.jsonl").exists() and (data / p.name / "concepts_owl.jsonl").exists()
            and (not names or p.name in names)]


def points(data: Path, p: P.Plan, overrides: dict) -> list[dict]:
    """The run's corpus points: every P.STRIDE-th tick."""
    gt = {r["t"]: K.effective_concepts(r["instances"], r["t"], overrides.get(p.name, {}))
          for r in map(json.loads, (data / p.name / "concepts_gt_raw.jsonl").read_text().splitlines())}
    raw = {r["t"]: sorted({c["concept"] for c in r["concepts"]})
           for r in map(json.loads, (data / p.name / "concepts_owl.jsonl").read_text().splitlines())}
    out = []
    for t in range(0, len(gt), P.STRIDE):
        prompts = {"gt": K.describe(gt.get(t, []))}
        if not p.pools:   # test runs: the detector's view too
            prompts["owl"] = K.describe(raw.get(t, []))
        out.append({"t": t, "raw": raw.get(t, []), "gt": gt.get(t, []), "prompts": prompts})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=P.DATASET)
    ap.add_argument("--runs", help="comma-separated run names (default: every run with concepts)")
    ap.add_argument("--embedders", default=",".join(TEXT_EMBEDDERS))
    args = ap.parse_args()
    data = ROOT / "data" / args.dataset
    ov = data / "gt_overrides.json"
    overrides = json.loads(ov.read_text()) if ov.exists() else {}
    runs = runs_of(data, args.runs.split(",") if args.runs else None)
    corpus = {}
    for p in runs:
        corpus[p.name] = points(data, p, overrides)
        (data / p.name / "corpus").mkdir(exist_ok=True)
        with open(data / p.name / "corpus" / "frames.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in corpus[p.name])
    prompts = sorted({x for pts in corpus.values() for r in pts for x in r["prompts"].values()})
    print(f"{len(runs)} runs, {sum(map(len, corpus.values()))} points, {len(prompts)} distinct prompts", flush=True)
    cache_dir = data / "_corpus_cache"
    cache_dir.mkdir(exist_ok=True)
    for e in args.embedders.split(","):
        path = cache_dir / f"{e}.npz"
        cache = {}
        if path.exists():
            z = np.load(path, allow_pickle=False)
            cache = dict(zip(z["prompts"].tolist(), z["vectors"]))
        new = [t for t in prompts if t not in cache]
        if new:
            emb = load_text_embedder(e)
            for i in range(0, len(new), 64):   # in chunks: one batch of thousands of prompts runs out of GPU memory
                cache.update(zip(new[i : i + 64], emb.embed(new[i : i + 64]).astype(np.float32)))
            del emb
            keys = sorted(cache)
            np.savez(path, prompts=np.array(keys), vectors=np.stack([cache[k] for k in keys]))
        for name, pts in corpus.items():
            for src in pts[0]["prompts"]:
                np.save(data / name / "corpus" / f"{e}@{src}.npy", np.stack([cache[r["prompts"][src]] for r in pts]).astype(np.float32))
        print(f"{e}: {len(new)} new prompts embedded, {len(prompts) - len(new)} from the cache", flush=True)


if __name__ == "__main__":
    main()
