"""The runs_aesop dashboard: Images (per frame: labels, boxes, scores, kNN neighbours; ground-truth correction),
Scores (whole-run scores, conformal thresholds, Elhafsi metrics) and Experiment (setup, pool / corpus / threshold sweeps).

    .venv/bin/python aesop_recreate/dashboard/serve.py                 # http://127.0.0.1:8010
    .venv/bin/python aesop_recreate/dashboard/serve.py --port 8012

It serves the semantic24 dataset (sim/semantic24.DATASET). It reads
aesop_recreate/data/semantic24/: each run's frames, concepts_gt_raw.jsonl, concepts_owl.jsonl and, once built,
corpus/ (build_corpus.py: frames.jsonl, scores_<pool>.csv, knn_<pool>.json) and _results/score_<pool>.json
(score.py), _results/experiment_<pool>.json (experiment.py). Ground-truth corrections are written to <dataset>/gt_overrides.json (the format of
sim/concepts.apply_overrides); re-run build_corpus.py and score.py afterwards.

    GET  /api/index?pool=P             runs of the pool (+ the test runs), windows, results
    GET  /api/labels/<run>?pool=P      frames, ground truth, OWL-ViT, corrections, corpus, neighbours
    POST /api/labels/<run>             the run's ground-truth corrections
    GET  /api/scores/<run>?pool=P      the run's per-point scores (scores_<pool>.csv)
    GET  /api/experiment?pool=P        the Experiment tab's results (experiment_<pool>.json), {} if not computed
    GET  /api/embmap?pool=P            the Experiment tab's embedding-space view (embed_map_<pool>.json, embed_map.py), {} if not computed
    GET  /data/<path>                  files under aesop_recreate/data/ (frames)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent      # aesop_recreate/
sys.path[:0] = [str(ROOT), str(ROOT.parent)]
from sim import concepts as K, semantic24 as P  # noqa: E402

HERE = Path(__file__).resolve().parent
DATASET = P.DATASET
POOLS = ("withheld", "included")
EMBEDDERS = {"e5_mistral": ["e5-Mistral", "#c98500"], "minilm": ["MiniLM", "#3987e5"], "mpnet": ["MPNet", "#d95926"],
             "bert_large": ["BERT-large", "#199e70"]}   # the colours of scripts/score_videos.py


class Data:
    def __init__(self, dataset: str = DATASET):
        self.name, self.dir = dataset, ROOT / "data" / dataset
        self.plans = {p.name: p for p in P.plan()}

    def recorded(self, name: str) -> bool:
        return (self.dir / name / "camera.jsonl").exists()

    def windows(self) -> dict:
        cfg = ROOT / "config" / f"data_{self.name}.json"
        return {r["run_name"]: {"windows": r.get("windows", [[r["onset_tick"], r["offset_tick"]]]), "kind": r["anomaly_kind"]}
                for r in (json.loads(cfg.read_text())["anomaly_runs"] if cfg.exists() else [])}

    def results(self, pool: str) -> dict | None:
        path = self.dir / "_results" / f"score_{pool}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def overrides(self) -> dict:
        path = self.dir / "gt_overrides.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def index(self, pool: str) -> dict:
        wins = self.windows()
        runs = [{"name": n, "group": "pool" if pool in p.pools else "test", "kind": p.kind,
                 "windows": wins.get(n, {}).get("windows", []), "n": len(list((self.dir / n / "frames").glob("*.jpg")))}
                for n, p in self.plans.items() if (pool in p.pools or not p.pools) and self.recorded(n)]
        return {"dataset": self.name, "pools": POOLS, "pool": pool, "runs": runs, "embedders": EMBEDDERS,
                "stride": P.STRIDE, "results": self.results(pool)}

    def corpus(self, name: str) -> list[dict]:
        path = self.dir / name / "corpus" / "frames.jsonl"
        return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []

    def scores(self, name: str, pool: str) -> dict:
        """{tick: [...], "<embedder>/<metric>": [...]} of scores_<pool>.csv ({} before score.py)."""
        path = self.dir / name / "corpus" / f"scores_{pool}.csv"
        if not path.exists():
            return {}
        rows = list(csv.DictReader(path.open()))
        return {k: [int(r[k]) if k == "tick" else float(r[k]) for r in rows] for k in (rows[0] if rows else {})}

    def prompt_counts(self, pool: str) -> dict:
        """prompt -> how many of the pool's points (the ground-truth cache) have it."""
        out: dict = {}
        for n, p in self.plans.items():
            if pool in p.pools:
                for r in self.corpus(n):
                    out[r["prompts"]["gt"]] = out.get(r["prompts"]["gt"], 0) + 1
        return out

    def labels(self, name: str, pool: str) -> dict | None:
        if name not in self.plans or not self.recorded(name):
            return None
        d = self.dir / name
        read = lambda f: [json.loads(l) for l in (d / f).read_text().splitlines()] if (d / f).exists() else []  # noqa: E731
        corpus = self.corpus(name)
        knn_path = d / "corpus" / f"knn_{pool}.json"
        knn = json.loads(knn_path.read_text()) if knn_path.exists() else {}
        # every neighbour's stored labels and its own leave-one-out cosine_k5 score per embedder
        refs = {(r, t) for per_src in knn.values() for per_t in per_src.values() for nb in per_t.values() for r, t, _ in nb}
        nbr_labels, nbr_scores = {}, {e: {} for e in EMBEDDERS}
        for r in {r for r, _ in refs}:
            rows, sc = {x["t"]: x for x in self.corpus(r)}, self.scores(r, pool)
            at = {t: i for i, t in enumerate(sc.get("tick", []))}
            for rr, t in refs:
                if rr == r:
                    nbr_labels[f"{r}|{t}"] = rows.get(t, {}).get("gt", [])
                    for e in EMBEDDERS:
                        if f"gt@{e}/cosine_k5" in sc and t in at:
                            nbr_scores[e][f"{r}|{t}"] = sc[f"gt@{e}/cosine_k5"][at[t]]
        counts = self.prompt_counts(pool) if corpus else {}
        return {"frames": f"/data/{self.name}/{name}/frames", "n": len(list((d / "frames").glob("*.jpg"))),
                "group": "pool" if pool in self.plans[name].pools else "test",
                "gt": {r["t"]: r["instances"] for r in read("concepts_gt_raw.jsonl")},
                "owl": {r["t"]: r["concepts"] for r in read("concepts_owl.jsonl")},
                "overrides": self.overrides().get(name, {}),
                "corpus": {r["t"]: {**r, "same_prompt": {src: counts.get(x, 0) for src, x in r["prompts"].items()}} for r in corpus},
                "knn": knn, "nbr_labels": nbr_labels, "nbr_scores": nbr_scores,
                "objects": K.OBJECTS, "predicates": K.PREDICATES, "scenery": sorted(K.SCENERY)}

    def save(self, name: str, run_overrides: dict) -> bool:
        if name not in self.plans:
            return False
        ov = self.overrides()
        ov[name] = run_overrides
        (self.dir / "gt_overrides.json").write_text(json.dumps(ov, indent=1, sort_keys=True))
        return True


def finite(x):
    if isinstance(x, dict):
        return {k: finite(v) for k, v in x.items()}
    if isinstance(x, list):
        return [finite(v) for v in x]
    return None if isinstance(x, float) and not math.isfinite(x) else x


def make_handler(data: Data):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send_json(self, obj, status: int = 200) -> None:
            body = json.dumps(finite(obj)).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_file(self, path: Path, cache: bool = True) -> None:
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=300" if cache else "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            pool = q.get("pool", POOLS[0])
            parts = u.path.strip("/").split("/")
            if u.path in ("/", "/index.html"):
                return self.send_file(HERE / "index.html", cache=False)
            if u.path == "/api/index":
                return self.send_json(data.index(pool))
            if u.path in ("/api/experiment", "/api/embmap"):
                path = data.dir / "_results" / f"{'experiment' if u.path.endswith('experiment') else 'embed_map'}_{pool}.json"
                return self.send_file(path, cache=False) if path.exists() and pool in POOLS else self.send_json({})
            if len(parts) == 3 and parts[:2] == ["api", "labels"]:
                r = data.labels(parts[2], pool)
                return self.send_json(r) if r is not None else self.send_json({"error": "no such run"}, 404)
            if len(parts) == 3 and parts[:2] == ["api", "scores"]:
                return self.send_json(data.scores(parts[2], pool))
            if parts[0] == "data":
                path = (ROOT / u.path.lstrip("/")).resolve()
                if path.is_file() and (ROOT / "data") in path.parents:
                    return self.send_file(path)
            self.send_json({"error": "not found"}, 404)

        def do_POST(self):
            parts = urlparse(self.path).path.strip("/").split("/")
            if len(parts) == 3 and parts[:2] == ["api", "labels"]:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                return self.send_json({"ok": data.save(parts[2], body)})
            self.send_json({"error": "not found"}, 404)

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    data = Data()
    print(f"{DATASET} dashboard: http://{args.host}:{args.port}  ({data.dir})", flush=True)
    ThreadingHTTPServer((args.host, args.port), make_handler(data)).serve_forever()


if __name__ == "__main__":
    main()
