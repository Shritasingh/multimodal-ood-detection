"""The snaps_mvp dashboard: every snapshot of both datasets, scored by score.py.

    .venv/bin/python mvp_recreate/dashboard/serve.py            # http://127.0.0.1:8013
    .venv/bin/python mvp_recreate/dashboard/serve.py --port 8014

Images  2 x 2: the frame (DINOv2 patch-distance heatmap, k=1 to the nominal patches, and ground-truth boxes) | the kNN
        nominal neighbours per representation (frame, colour-coded concepts, top-down physics) / z-scored plots of every
        frame of the dataset with each representation's dashed conformal threshold | the concepts, the semantic
        sentence and the top-down physics view
Scores  each anomaly's score, z, tau, above-tau and the share of nominal frames below it (results/metrics.json)

    GET /api/index                         datasets, groups, every frame's z-scores and thresholds, results, geometry
    GET /api/frame/<dataset>/<group>/<pos> one snapshot: history, roles, scores, patch map, kNN (with their history)
    GET /data/<dataset>/<group>/frames/<idx>.jpg

The server never serves a mix of old and new data: it reloads when score.py has rewritten the scores, and answers
409 when frames on disk are newer than the scores (re-recorded, not yet encoded / scored).
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import physics as P  # noqa: E402
import score as SC  # noqa: E402
from cfg import CACHES, CFG, DATA, OWN, REPS, RESULTS, SHIFTED  # noqa: E402

DASH = CFG["dashboard"]
LABELS = {"nominal": "nominal (leave-one-out)", "wrongway": "wrong-way cyclist (physics anomaly)",
          "droplet": "blurred blobs on the lens (vision anomaly)", "conductor": "traffic conductor (semantic anomaly)"}
GRID = (16, 21)


def frames_mtime() -> float:
    return max(f.stat().st_mtime for f in DATA.glob("*/*/frames/*.jpg"))


def scores_mtime() -> float:
    return min([(DATA / c / "_scores.npz").stat().st_mtime for c in CACHES if (DATA / c / "_scores.npz").exists()]
               + [(RESULTS / "metrics.json").stat().st_mtime])


def slots(rec: dict) -> dict[str, str | None]:
    """agent id -> its physics-encoder role (slot) at the frame."""
    types = {int(i): a["type"] for i, a in rec["agents"].items()}
    return {str(i): v["role"] for i, v in P.assign(rec["history"][-1], types).items()}


class Data:
    def __init__(self):
        self.loaded_scores = scores_mtime()
        self.metrics = json.loads((RESULTS / "metrics.json").read_text())
        self.version = int(frames_mtime())          # image URLs carry it: no stale browser images
        self.caches = {}
        for cache, cm in self.metrics["caches"].items():
            z = np.load(DATA / cache / "_scores.npz")
            sc = {k: z[k] for k in z.files}
            G = {}
            for g in [g for g in CACHES[cache] if f"{g}/idx" in sc]:
                idx = sc[f"{g}/idx"]
                recs = [json.loads((DATA / cache / g / "states" / f"{i:04d}.json").read_text()) for i in idx]
                G[g] = {"dir": g, "idx": idx, "recs": recs, "sc": {k.split("/", 1)[1]: v for k, v in sc.items() if k.startswith(g + "/")}}
            self.caches[cache] = {"groups": G, "stats": cm["stats"], "n": cm["n"], "sem_sentences": sc["cache/semantic_sentences"],
                                  "sem_labels": [json.loads(l) for l in sc["cache/semantic_labels"]],
                                  "sem_count": sc["cache/semantic_count"], "sem_added": sc["cache/semantic_added"], "phys_entries": cm.get("physics_entries"),
                                  "vmax": float(np.quantile(G["nominal"]["sc"]["vision_map"], DASH["vmax_quantile"])) * DASH["vmax_factor"]}
            print(f"{cache}: " + ", ".join(f"{g} {len(v['idx'])}" for g, v in G.items()), flush=True)
        self.index_bytes = json.dumps(self.index()).encode()

    def z(self, cache: str, r: str, x):
        s = self.caches[cache]["stats"][r]
        return (np.asarray(x, dtype=np.float64) - s["mean"]) / s["std"]

    def index(self) -> dict:
        out = {}
        for cache, C in self.caches.items():
            groups = []
            for g, G in C["groups"].items():
                sc = G["sc"]
                groups.append({"name": g, "label": LABELS[g], "n": len(G["idx"]), "idx": G["idx"].tolist(),
                               "s": {r: np.round(sc[r].astype(float), 6).tolist() for r in REPS},
                               "tau": {r: np.round(sc[f"{r}_tau"].astype(float), 6).tolist() for r in REPS},
                               "above": {r: (sc[r] > sc[f"{r}_tau"]).astype(int).tolist() for r in REPS},
                               "desc": sc["desc"].tolist()})
            out[cache] = {"groups": groups, "stats": C["stats"], "n": C["n"], "vmax": C["vmax"], "phys_entries": C["phys_entries"],
                          "sem_cache": [{"sentence": str(t), "labels": l, "frames": int(c), "added": bool(a)}
                                        for t, l, c, a in zip(C["sem_sentences"], C["sem_labels"], C["sem_count"], C["sem_added"])]}
        return {"caches": out, "reps": list(REPS), "colors": SC.COLORS, "metrics": self.metrics, "camera": CFG["scene"]["camera"],
                "visible_px": CFG["scene"]["visible_px"], "roles": list(P.ROLES), "history": CFG["history"], "grid": list(GRID),
                "top_frac": SC.TOP_FRAC, "knn_patches": SC.KNN_PATCHES, "k": SC.K, "shifted": SHIFTED, "own": OWN,
                "heat_opacity": DASH["heat_opacity"],
                "geometry": json.loads((DATA / "_geometry.json").read_text()), "version": self.version}

    def frame(self, cache: str, g: str, pos: int) -> dict | None:
        C = self.caches.get(cache)
        if C is None or g not in C["groups"] or not 0 <= pos < len(C["groups"][g]["idx"]):
            return None
        G, nom = C["groups"][g], C["groups"]["nominal"]
        sc, rec = G["sc"], G["recs"][pos]
        scores = {r: {"s": float(sc[r][pos]), "tau": float(sc[f"{r}_tau"][pos]), "z": float(self.z(cache, r, sc[r][pos])),
                      "tau_z": float(self.z(cache, r, sc[f"{r}_tau"][pos]))} for r in REPS}
        knn = {}
        for r in REPS:
            if r == "physics_euclid":   # no panel of its own: the physics panel shows the plain Euclidean neighbours
                continue
            rows = []
            if r == "semantic":     # the cache sentences nearest to this frame's sentence
                for j, d in zip(sc["semantic_knn"][pos].tolist(), sc["semantic_knn_d"][pos].tolist()):
                    if j >= 0:
                        rows.append({"sentence": str(C["sem_sentences"][j]), "labels": C["sem_labels"][j],
                                     "d": None if not np.isfinite(d) else d,
                                     "frames": int(C["sem_count"][j]), "added": bool(C["sem_added"][j])})
                knn[r] = rows
                continue
            counts = (sc["vision_knn_n"] if r == "vision" else sc["physics_knn_n"])[pos].tolist()
            for j, d, cnt in zip(sc[f"{r}_knn"][pos].tolist(), sc[f"{r}_knn_d"][pos].tolist(), counts):
                if j < 0:
                    continue
                nr = nom["recs"][j]
                pairs = None        # anomaly frames: which of this frame's outlined patches matched which patch of the neighbour
                if r == "vision" and g != "nominal" and "vision_knn_src" in sc:
                    pairs = [[int(a), int(b)] for a, f, b in sc["vision_knn_src"][pos].tolist() if f == j and a >= 0]
                rows.append({"pos": j, "idx": int(nom["idx"][j]), "d": d, "patches": cnt if r == "vision" else None, "pairs": pairs,
                             "same": cnt if r == "physics" else None, "concepts": nr["concepts"], "desc": str(nom["sc"]["desc"][j]),
                             "s": {q: float(nom["sc"][q][j]) for q in REPS},
                             "rec": {"history": nr["history"][-CFG["history"]:], "agents": nr["agents"]}, "slot": slots(nr)})
            knn[r] = rows
        last = {o["id"]: o for o in rec["history"][-1]["others"]}
        return {"cache": cache, "group": g, "pos": pos, "idx": int(G["idx"][pos]), "dir": G["dir"],
                "rec": {**rec, "history": rec["history"][-CFG["history"]:]}, "slot": slots(rec),
                "last": {str(i): {"px": o.get("px", 0), "pred": o.get("pred"), "box": o.get("box")} for i, o in last.items()},
                "desc": str(sc["desc"][pos]), "scores": scores, "map": np.round(sc["vision_map"][pos], 4).tolist(), "knn": knn,
                "same_desc": int((nom["sc"]["desc"] == sc["desc"][pos]).sum()),
                "sentence": SC.sentence(str(sc["desc"][pos]).split(" · "))}


class Holder:
    """The loaded data, reloaded when score.py has rewritten the scores; refuses to serve when frames are newer."""

    def __init__(self):
        self.lock = threading.Lock()
        self.data = Data()

    def get(self) -> tuple[Data | None, str]:
        with self.lock:
            sm = scores_mtime()
            if frames_mtime() > sm + 1:
                return None, ("the frames on disk are newer than the scores: re-run encode.py and score.py "
                              "(the dashboard reloads by itself once they are done)")
            if sm > self.data.loaded_scores + 1:
                print("scores changed on disk: reloading", flush=True)
                self.data = Data()
            return self.data, ""


def make_handler(holder: Holder):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, body: bytes, ctype: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            u = urlparse(self.path)
            parts = u.path.strip("/").split("/")
            if u.path in ("/", "/index.html"):
                return self.send((HERE / "index.html").read_bytes(), "text/html")
            if parts[0] in ("api", "data"):
                data, err = holder.get()
                if data is None:
                    return self.send(json.dumps({"error": err}).encode(), "application/json", status=409)
                if u.path == "/api/index":
                    return self.send(data.index_bytes, "application/json")
                if len(parts) == 5 and parts[:2] == ["api", "frame"]:
                    r = data.frame(parts[2], parts[3], int(parts[4]))
                    if r is not None:
                        return self.send(json.dumps(r).encode(), "application/json")
                if parts[0] == "data":
                    path = (ROOT / u.path.lstrip("/")).resolve()
                    if path.is_file() and DATA in path.parents and path.suffix == ".jpg":
                        return self.send(path.read_bytes(), "image/jpeg")
            self.send(b'{"error": "not found"}', "application/json", status=404)

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=DASH["port"])
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    holder = Holder()
    print(f"snaps_mvp dashboard: http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), make_handler(holder)).serve_forever()


if __name__ == "__main__":
    main()
