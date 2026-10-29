"""Concepts and embeddings for the semantic24 runs, one stage at a time: concept encoders (encoders/concept_encoders.py)
-> text embedders (encoders/text_embedders.py), and the end-to-end encoders (encoders/end_to_end_encoders.py).

    .venv/bin/python aesop_recreate/scripts/run_concepts.py owl      # OWL-ViT concepts, every frame -> concepts_owl.jsonl
    .venv/bin/python aesop_recreate/scripts/run_concepts.py qwen     # Qwen closed / open concepts, every STRIDE-th frame -> concepts_qwen.jsonl
    .venv/bin/python aesop_recreate/scripts/run_concepts.py embed    # descriptions -> embeddings/<source>@<embedder><suffix>.npz
    .venv/bin/python aesop_recreate/scripts/run_concepts.py image    # CLIP and DINOv2 patch tokens -> embeddings/{clip_image,vision_patch}<suffix>.npz
    ... <stage> --runs t2_umbrella,t2_emergency                      # only some runs

Sources for embed: gt (concepts_gt_raw.jsonl with data/semantic24/gt_overrides.json applied), owl, qwen_closed,
qwen_open, and on request gt_qwen (gt plus Qwen's own corrections, gt_qwen_overrides.json); embedders: e5_mistral, minilm, mpnet, bert_large. Everything is scored at ticks 0, STRIDE, 2 STRIDE, ...
Nominal runs (any reference pool) get the '_nominal' suffix, test runs none, as multimodal-ood-detection's
run_metrics.py expects. Each npz also keeps the descriptions it embedded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent      # aesop_recreate/
sys.path[:0] = [str(ROOT), str(ROOT.parent)]       # sim/ and encoders/ merge with multimodal-ood-detection's
from encoders.end_to_end_encoders import END_TO_END_ENCODERS  # noqa: E402
from encoders.text_embedders import TEXT_EMBEDDERS  # noqa: E402
from sim import concepts as K, semantic24 as P  # noqa: E402

OUT = ROOT / "data" / "semantic24"
SOURCES = ("gt", "owl", "qwen_closed", "qwen_open")


def suffix(p: P.Plan) -> str:
    return "_nominal" if p.pools else ""


def frames(name: str, ticks) -> list[Path]:
    return [OUT / name / "frames" / f"{t:06d}.jpg" for t in ticks]


def ticks_of(name: str) -> list[int]:
    return list(range(0, len(list((OUT / name / "frames").glob("*.jpg"))), P.STRIDE))


def load_overrides(name: str = "gt_overrides.json") -> dict:
    path = OUT / name
    return json.loads(path.read_text()) if path.exists() else {}


def concepts_of(name: str, source: str, overrides: dict) -> dict[int, list[str]]:
    """{tick: concepts} of one source for a run."""
    d = OUT / name
    if source in ("gt", "gt_qwen"):
        rows = [json.loads(l) for l in (d / "concepts_gt_raw.jsonl").read_text().splitlines()]
        gt = {r["t"]: K.effective_concepts(r["instances"], r["t"], overrides.get(name, {})) for r in rows}
        if source == "gt_qwen":
            q = load_overrides("gt_qwen_overrides.json").get(name, {})
            gt = {t: K.qwen_concepts(c, t, q) for t, c in gt.items()}
        return gt
    if source == "owl":
        return {r["t"]: [c["concept"] for c in r["concepts"]] for r in map(json.loads, (d / "concepts_owl.jsonl").read_text().splitlines())}
    key = source.split("_", 1)[1]
    return {r["t"]: r[key] for r in map(json.loads, (d / "concepts_qwen.jsonl").read_text().splitlines())}


def stage_owl(runs: list[P.Plan], batch: int = 16) -> None:
    from encoders.concept_encoders import OwlConceptEncoder

    enc = OwlConceptEncoder()
    for p in runs:
        paths = sorted((OUT / p.name / "frames").glob("*.jpg"))
        with open(OUT / p.name / "concepts_owl.jsonl", "w") as f:
            for i in tqdm(range(0, len(paths), batch), desc=f"{p.name}/owl"):
                rows = enc.encode_batch([Image.open(x).convert("RGB") for x in paths[i : i + batch]])
                for j, r in enumerate(rows):
                    f.write(json.dumps({"t": i + j, "concepts": r}) + "\n")


def stage_qwen(runs: list[P.Plan], batch: int = 6) -> None:
    from encoders.concept_encoders import QwenConceptEncoder

    enc = QwenConceptEncoder()
    for p in runs:
        ticks = ticks_of(p.name)
        with open(OUT / p.name / "concepts_qwen.jsonl", "w") as f:
            for i in tqdm(range(0, len(ticks), batch), desc=f"{p.name}/qwen"):
                tb = ticks[i : i + batch]
                for t, r in zip(tb, enc.encode_batch([Image.open(x).convert("RGB") for x in frames(p.name, tb)])):
                    f.write(json.dumps({"t": t, **r}) + "\n")


def stage_embed(runs: list[P.Plan], embedders: list[str], sources: list[str]) -> None:
    import torch

    overrides = load_overrides()
    texts = {}   # (run, source) -> (ticks, descriptions)
    for p in runs:
        for s in sources:
            ticks = ticks_of(p.name)
            c = concepts_of(p.name, s, overrides)
            texts[p.name, s] = (ticks, [K.describe(c.get(t, [])) for t in ticks])
    for e in embedders:
        from encoders.text_embedders import load_text_embedder

        emb = load_text_embedder(e)
        for p in tqdm(runs, desc=f"embed/{e}"):
            (OUT / p.name / "embeddings").mkdir(exist_ok=True)
            for s in sources:
                ticks, desc = texts[p.name, s]
                np.savez(OUT / p.name / "embeddings" / f"{s}@{e}{suffix(p)}.npz", embeddings=emb.embed(desc).astype(np.float32),
                         ticks=np.array(ticks), descriptions=np.array(desc), text_embedder=np.array([getattr(emb, "model_name", e)]))
        del emb
        torch.cuda.empty_cache()


def stage_image(runs: list[P.Plan], batch: int = 24) -> None:
    from encoders.end_to_end_encoders import load_end_to_end

    encs = {name: load_end_to_end(name) for name in END_TO_END_ENCODERS}
    for p in runs:
        (OUT / p.name / "embeddings").mkdir(exist_ok=True)
        ticks = ticks_of(p.name)
        out = {name: [] for name in encs}
        for i in tqdm(range(0, len(ticks), batch), desc=f"{p.name}/image"):
            imgs = [Image.open(x).convert("RGB") for x in frames(p.name, ticks[i : i + batch])]
            for name, enc in encs.items():
                out[name].append(enc.encode_batch(imgs))
        for name, parts in out.items():
            np.savez(OUT / p.name / "embeddings" / f"{name}{suffix(p)}.npz", embeddings=np.concatenate(parts), ticks=np.array(ticks))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["owl", "qwen", "embed", "image"])
    ap.add_argument("--runs", help="comma-separated run names (default: every recorded run)")
    ap.add_argument("--embedders", default=",".join(TEXT_EMBEDDERS))
    ap.add_argument("--sources", default=",".join(SOURCES))
    args = ap.parse_args()
    runs = [p for p in P.plan() if (OUT / p.name / "camera.jsonl").exists() and (not args.runs or p.name in args.runs.split(","))]
    if args.stage == "owl":
        stage_owl(runs)
    elif args.stage == "qwen":
        stage_qwen(runs)
    elif args.stage == "embed":
        stage_embed(runs, args.embedders.split(","), args.sources.split(","))
    else:
        stage_image(runs)


if __name__ == "__main__":
    main()
