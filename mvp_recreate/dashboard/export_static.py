"""Export the snaps_mvp dashboard as a static site (GitHub Pages or any static host): no Python server.

Writes to --out (e.g. <gh-pages worktree>/snaps_mvp):
    index.html                                   the dashboard page in static mode
    api/index.json                               what /api/index returns
    api/frame/<dataset>/<group>/<pos>.json       /api/frame/<dataset>/<group>/<pos>
    data/<dataset>/<group>/frames/<idx>.jpg      the snapshots, copied as they are
All paths are relative, so the site works from a sub-path.

    .venv/bin/python mvp_recreate/dashboard/export_static.py --out ../mmood-pages/snaps_mvp
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import serve as SV


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    data, err = SV.Holder().get()
    if data is None:
        raise SystemExit(err)
    (out / "api").mkdir(parents=True, exist_ok=True)
    (out / "api" / "index.json").write_bytes(data.index_bytes)
    for cache, C in data.caches.items():
        for g, G in C["groups"].items():
            for pos, idx in enumerate(G["idx"]):
                dst = out / "api" / "frame" / cache / g / f"{pos}.json"
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(json.dumps(data.frame(cache, g, pos)))
                img = out / "data" / cache / G["dir"] / "frames" / f"{int(idx):04d}.jpg"
                img.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(SV.DATA / cache / G["dir"] / "frames" / img.name, img)
            print(f"{cache}/{g}: {len(G['idx'])} frames", flush=True)
    page = (SV.HERE / "index.html").read_text().replace("<script>\n", "<script>window.STATIC_SITE = true;</script>\n<script>\n", 1)
    (out / "index.html").write_text(page)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"wrote {out} ({size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
