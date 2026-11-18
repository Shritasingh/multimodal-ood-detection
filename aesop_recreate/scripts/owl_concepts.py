"""Raw OWL-ViT concepts for every recorded frame (encoders/concept_encoders.OwlConceptEncoder): the detector's view of
the test runs, which build_corpus.py encodes and score.py scores against the ground-truth cache.

    .venv/bin/python aesop_recreate/scripts/owl_concepts.py                       # every recorded run
    .venv/bin/python aesop_recreate/scripts/owl_concepts.py --runs t2_haybale     # some runs

Writes data/semantic24/<run>/concepts_owl.jsonl: {"t", "concepts": [{concept, score, box}, ...]} per tick.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent      # aesop_recreate/
sys.path[:0] = [str(ROOT), str(ROOT.parent)]       # sim/ and encoders/ merge with multimodal-ood-detection's
from sim import semantic24 as P  # noqa: E402

OUT = ROOT / "data" / P.DATASET


def owl(runs: list[P.Plan], batch: int = 16) -> None:
    from encoders.concept_encoders import OwlConceptEncoder

    enc = OwlConceptEncoder()
    for p in runs:
        paths = sorted((OUT / p.name / "frames").glob("*.jpg"))
        with open(OUT / p.name / "concepts_owl.jsonl", "w") as f:
            for i in tqdm(range(0, len(paths), batch), desc=f"{p.name}/owl"):
                rows = enc.encode_batch([Image.open(x).convert("RGB") for x in paths[i : i + batch]])
                for j, r in enumerate(rows):
                    f.write(json.dumps({"t": i + j, "concepts": r}) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", help="comma-separated run names (default: every recorded run)")
    args = ap.parse_args()
    owl([p for p in P.plan() if (OUT / p.name / "camera.jsonl").exists() and (not args.runs or p.name in args.runs.split(","))])


if __name__ == "__main__":
    main()
