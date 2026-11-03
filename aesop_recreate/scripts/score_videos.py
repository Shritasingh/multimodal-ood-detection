"""Score videos of the semantic24 test runs: the camera next to the concept scores, one frame per tick (like the
runs_190 dashboard videos, but only the camera and score plots).

    .venv/bin/python aesop_recreate/scripts/score_videos.py                       # the 4 test runs, included-routes pool
    .venv/bin/python aesop_recreate/scripts/score_videos.py --pool withheld --runs t2_haybale

    left          camera (with the tick and, inside the anomaly window, what was injected), Qwen open below it
    right         OWL-ViT ground truth, OWL-ViT raw, Qwen closed, top to bottom
Each plot: one line per text embedder, its raw score (cosine_k5 against the whole pool; Qwen's descriptions against the
ground truth's nominal cache) with its own threshold tau as a dashed line of the same colour, tau as in
evaluate_semantic24.py's results file (95th percentile of the pool's run-level leave-one-out scores). The embedders' scores
differ by up to ~10x, so the y axis is logarithmic (values are unchanged). Every anomaly window (one per injection in the
long runs) is shaded. One video frame per scored tick (every STRIDE-th), at FPS / STRIDE frames a second: real time.
Writes aesop_recreate/data/semantic24_long/<run>/<run>_overlay_<output_stem>.webm (one per pool), which the dashboard's Videos
tab plays for the matching dataset.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent      # aesop_recreate/
MAIN = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(MAIN), str(MAIN / "scripts"), str(MAIN / "analysis")]
import common as C  # noqa: E402
import evaluate_semantic24 as E  # noqa: E402
from sim import semantic24 as P  # noqa: E402
import run_metrics as rm  # noqa: E402

CONFIGS = {"included": "aesop_recreate/config/encoders_semantic24_long_incl.json", "withheld": "aesop_recreate/config/encoders_semantic24_long.json"}
FN = "cosine_k5"
SOURCES = [("gt", "OWL-ViT ground truth"), ("owl", "OWL-ViT raw"), ("qwen_closed_gtc", "Qwen closed  (vs ground-truth cache)"),
           ("qwen_open_gtc", "Qwen open  (vs ground-truth cache)")]
EMB = [("minilm", "MiniLM", "#3987e5"), ("mpnet", "MPNet", "#d95926"), ("bert_large", "BERT-large", "#199e70"), ("e5_mistral", "e5-Mistral", "#c98500")]
KIND = {"haybale": "hay bales on the sidewalk", "pedestrian": "person walks into the ego lane", "emergency": "ambulance, police car, warning sign"}
W, H = 1920, 1080
CAM = (960, 720)                                   # camera (frames are 800 x 600)
# panel boxes (x, y, w, h): three concept embedders in the right column, the fourth under the camera
BOXES = {"gt": (960, 0, 960, 360), "owl": (960, 360, 960, 360), "qwen_closed_gtc": (960, 720, 960, 360), "qwen_open_gtc": (0, 720, 960, 360)}
FLOOR = 1e-4                                       # log axis: a score of 0 (an exact nominal duplicate) is drawn here


def plot_panel(title: str, size: tuple[int, int], ticks: np.ndarray, series: list, windows: list, n: int) -> tuple[np.ndarray, callable, tuple[int, int]]:
    """A static plot image, tick -> x pixel and the plot area's (top, bottom) pixel rows, for drawing the cursor.
    series: [(label, colour, scores, tau)]."""
    fig, ax = plt.subplots(figsize=(size[0] / 100, size[1] / 100), dpi=100, facecolor=C.BG)
    wide = size[0] > 900                            # wide: legend in one row; narrow: two rows
    fig.subplots_adjust(left=0.075 if wide else 0.12, right=0.98, top=0.8 if wide else 0.78, bottom=0.15)
    ax.set_facecolor(C.SURFACE)
    for w in windows:
        ax.axvspan(*w, color=C.TEXT, alpha=0.10, lw=0)
    for label, colour, v, tau in series:
        ax.plot(ticks, np.maximum(v, FLOOR), color=colour, lw=1.6, label=f"{label}  τ {tau:.3g}")
        ax.axhline(tau, color=colour, lw=1.1, ls=(0, (5, 4)))
    ax.set_yscale("log")
    ax.set_xlim(0, n)
    lo = max(FLOOR, min(np.percentile(np.maximum(v, FLOOR), 2) for _, _, v, _ in series) * 0.7)
    hi = max(max(v.max(), tau) for _, _, v, tau in series) * 1.4
    ax.set_ylim(lo, hi)
    ax.set_title(title, color=C.TEXT, fontsize=12, loc="left", pad=24 if wide else 40)
    ax.set_ylabel("score (log scale)", color=C.DIM, fontsize=9)
    ax.set_xlabel("tick", color=C.DIM, fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(colors=C.DIM, labelsize=8, which="both")
    ax.grid(color=C.GRIDC, lw=0.6, which="major")
    for s in ax.spines.values():
        s.set_color(C.GRIDC)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=4 if wide else 2, fontsize=8, frameon=False,
              labelcolor=C.TEXT, handlelength=1.5, columnspacing=1.2)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    (x0, ya), (x1, yb) = ax.transData.transform([(0, lo), (n, hi)])
    h = img.shape[0]
    plt.close(fig)
    return img, lambda t: int(round(x0 + (x1 - x0) * t / n)), (int(h - yb), int(h - ya))   # matplotlib y runs upwards


def camera_panel(frame: np.ndarray, run: str, t: int, kind: str | None, active: bool, pool: str) -> np.ndarray:
    panel, texts = cv2.resize(frame, CAM, interpolation=cv2.INTER_AREA), []
    panel[:34] = (panel[:34] * 0.35).astype(np.uint8)
    C.put(texts, 10, 8, f"{run}  ·  tick {t}  ·  {t / C.FPS:.1f} s", C.TEXT, "head")
    panel[-26:] = (panel[-26:] * 0.35).astype(np.uint8)
    C.put(texts, 10, CAM[1] - 20, f"{pool} pool · {FN} · dashed = each embedder's τ · log scale", C.TEXT, "small")
    if active and kind:
        label = f"ANOMALY ACTIVE · {KIND.get(kind, kind)}"
        cv2.rectangle(panel, (10, 42), (30 + int(C.font("head").getlength(label)), 68), C.rgb(C.ALERT), -1)
        C.put(texts, 20, 46, label, C.BG, "head")
    return C.finish(panel, texts)


def make_video(runs: E.Runs, run: str, pool: str, tau: dict) -> None:
    d = rm.resolve_path(runs.path[run]).parent          # .../semantic24/<run>
    stem = rm.load_config(rm.resolve_path(CONFIGS[pool]))["output_stem"]
    rows = list(csv.DictReader(open(d / "embeddings" / f"{stem}.csv")))
    ticks = np.array([int(r["tick"]) for r in rows])
    windows = runs.window.get(run, [])
    kind = run.split("_", 1)[1] if windows else None
    n = len(list((d / "frames").glob("*.jpg")))
    panels = {}
    for src, title in SOURCES:
        series = [(lab, col, np.array([float(r[f"{src}@{e}/{FN}"]) for r in rows]), tau[f"{src}@{e}/{FN}"]) for e, lab, col in EMB]
        panels[src] = plot_panel(title, BOXES[src][2:], ticks, series, windows, n)
    out = d / f"{run}_overlay_{stem}.webm"
    part = out.with_suffix(".part.webm")
    writer = cv2.VideoWriter(str(part), cv2.VideoWriter_fourcc(*"VP90"), C.FPS / P.STRIDE, (W, H))
    for t in tqdm(range(0, n, P.STRIDE), desc=run):
        canvas = np.full((H, W, 3), C.rgb(C.BG), np.uint8)
        frame = cv2.cvtColor(cv2.imread(str(d / "frames" / f"{t:06d}.jpg")), cv2.COLOR_BGR2RGB)
        canvas[: CAM[1], : CAM[0]] = camera_panel(frame, run, t, kind, any(on <= t < off for on, off in windows), pool)
        for src, (img, xpix, (top, bottom)) in panels.items():
            x, y, w, h = BOXES[src]
            p = img[:h, :w].copy()
            cv2.line(p, (xpix(t), top), (xpix(t), bottom), C.rgb(C.TEXT), 1)
            canvas[y : y + h, x : x + w] = p
        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    writer.release()
    part.replace(out)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", choices=list(CONFIGS), default="included")
    ap.add_argument("--runs", help="comma-separated test runs (default: all 4)")
    args = ap.parse_args()
    cfg = rm.load_config(rm.resolve_path(CONFIGS[args.pool]))
    runs = E.Runs(cfg)
    full = json.loads((E.OUT / f"{cfg['output_stem']}_eval.json").read_text())["full"]
    tau = {f"{src}@{e}/{FN}": full[f"{src}@{e}/{FN}"]["tau"] for src, _ in SOURCES for e, _, _ in EMB}
    for run in (args.runs.split(",") if args.runs else runs.test):
        make_video(runs, run, args.pool, tau)


if __name__ == "__main__":
    main()
