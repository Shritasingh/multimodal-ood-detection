"""Export the runs_aesop dashboard as a static site (GitHub Pages or any static host): no Python server.

Writes to --out (e.g. <gh-pages worktree>/runs_aesop):
    index.html                               the dashboard page in static mode (read-only: ground-truth edits are not saved)
    data/<pool>/index.json                   what /api/index?pool=<pool> returns
    data/<pool>/labels/<run>.json            /api/labels/<run>   (frames point at frames/<run>; gt / owl of those frames)
    data/<pool>/scores/<run>.json            /api/scores/<run>
    data/<pool>/experiment.json, embmap.json /api/experiment, /api/embmap
    frames/<run>/<tick>.jpg                  every STRIDE-th frame (the scored points), re-encoded at --quality
All paths are relative, so the site works from a sub-path.

    .venv/bin/python aesop_recreate/dashboard/export_static.py --out ../mmood-pages/runs_aesop
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

import serve as SV


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(SV.finite(obj), separators=(",", ":")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quality", type=int, default=80, help="JPEG quality of the frames")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    data = SV.Data()
    runs = set()
    for pool in SV.POOLS:
        idx = data.index(pool)
        write_json(out / "data" / pool / "index.json", idx)
        for r in idx["runs"]:
            lab = data.labels(r["name"], pool)
            lab["frames"] = f"frames/{r['name']}"
            for k in ("gt", "owl"):     # only the exported frames' labels
                lab[k] = {t: v for t, v in lab[k].items() if t % SV.P.STRIDE == 0}
            write_json(out / "data" / pool / "labels" / f"{r['name']}.json", lab)
            write_json(out / "data" / pool / "scores" / f"{r['name']}.json", data.scores(r["name"], pool))
            runs.add((r["name"], r["n"]))
        for name, f in (("experiment", "experiment"), ("embmap", "embed_map")):
            src = data.dir / "_results" / f"{f}_{pool}.json"
            dst = out / "data" / pool / f"{name}.json"
            shutil.copyfile(src, dst) if src.exists() else write_json(dst, {})
        print(f"{pool}: {len(idx['runs'])} runs", flush=True)
    for name, n in sorted(runs):
        for t in range(0, n, SV.P.STRIDE):
            dst = out / "frames" / name / f"{t:06d}.jpg"
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                Image.open(data.dir / name / "frames" / f"{t:06d}.jpg").save(dst, "JPEG", quality=args.quality)
        print(f"  frames/{name}: {len(range(0, n, SV.P.STRIDE))}", flush=True)
    page = (SV.HERE / "index.html").read_text().replace("<script>\n", "<script>window.STATIC_SITE = true;</script>\n<script>\n", 1)
    (out / "index.html").write_text(page)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"wrote {out} ({size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
