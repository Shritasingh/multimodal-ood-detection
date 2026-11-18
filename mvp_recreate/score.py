"""Score the snapshots per dataset: one OOD score per representation and a frame-level conformal threshold.

    .venv/bin/python mvp_recreate/score.py [--q 0.95]      # -> data/<dataset>/_scores.npz, results/

Scores (per frame, against the nominal cache of the same dataset; config.json: scoring, semantic):
  physics   mean standardised Euclidean distance to the k nearest cache entries. The cache holds each distinct nominal
            physics vector ONCE (vectors within duplicate_tol, max abs difference, are one entry, which remembers how
            many nominal frames share it); every dimension is scaled by the cache's mean / std (std floored at sd_floor)
  physics_euclid  the same over the same cache, but plain (unscaled) Euclidean distance: metres, m/s, cos / sin as they are
  semantic  each frame is ONE sentence, its labels concatenated (encode.sentence: "I see a ..., an ... and ...").
            The cache is the set of distinct sentences: those of the nominal frames (each once, remembering how many
            nominal frames have it) plus one single-label sentence per semantic.extra_labels label, embedded in
            encode.py (data/_sentences.npz). Score: the mean cosine distance of the frame's sentence to its semantic_k
            nearest cache sentences (< exact counts as 0)
  vision    DINOv2 patches, k = 1: each patch's cosine distance to its nearest nominal patch (any frame, any position);
            the frame score is the mean of its worst vision_top_frac patches

Leave-one-out: a nominal frame is scored without ITS copy of the cache: an entry (physics vector / sentence) it shares with
other nominal frames stays (their copies remain), one only it supplied is removed; the extra-label sentences always stay. So a
frame scores the same whether or not it is in the nominal set. A test frame made from a nominal one (the droplet frame)
is scored without its source frame's copy in the same way.
Threshold (one parameter q): tau = the conformal q-quantile of the nominal leave-one-out scores (the ceil((n+1) q)-th
smallest); a nominal frame's own tau uses the other nominal frames' scores only. Plotted: z = (s - mean) / std with the
mean / std of the nominal leave-one-out scores. Each anomaly group has one frame: reported are its score, z, tau,
whether it is above tau, and the share of nominal frames scoring below it.

kNN neighbours (the dashboard): physics: the k nearest cache entries by PLAIN Euclidean distance, i.e. physics_euclid's
neighbours (a nominal frame each, with how many nominal frames share that vector) and their distances; the standardised
score's own k nearest are saved as physics_std_knn / _d / _n; semantic: the k nearest cache sentences;
vision: the
frame's vision_knn_patches worst patches (the ones the dashboard outlines) each have a nearest nominal patch; the
neighbours are the nominal frames those come from, most of them first (largest overlap; ties: smaller mean distance),
with the number of patches each supplied and those patches' mean distance; vision_knn_src keeps, per outlined patch,
(this frame's patch, the nominal frame, that frame's patch) of its match (the dashboard outlines them on anomaly frames).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cfg import CACHES, CFG, DATA, OWN, REPS, RESULTS  # noqa: E402
from encode import SENTENCES_FILE, sentence  # noqa: E402

SC, SEM = CFG["scoring"], CFG["semantic"]
K, SD_FLOOR, EXACT, TOP_FRAC, KNN_PATCHES = SC["k"], SC["sd_floor"], SC["exact"], SC["vision_top_frac"], SC["vision_knn_patches"]
DUP_TOL, SEM_K = SC["duplicate_tol"], SC["semantic_k"]
OUT = RESULTS
COLORS = {"physics": "#2a78d6", "physics_euclid": "#9b7be0", "semantic": "#eb6834", "vision": "#1baf7a"}


def load(cache: str, group: str) -> dict | None:
    f = DATA / cache / "_emb" / f"{group}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    out = {k: z[k] for k in z.files}
    out["label_lists"] = [json.loads(s) for s in out["labels"]]
    out["sentences"] = [sentence(ls) for ls in out["label_lists"]]
    return out


def load_sentence_table() -> dict[str, np.ndarray]:
    """sentence -> its unit-norm embedding (encode.py)."""
    z = np.load(SENTENCES_FILE)
    emb = z["emb"].astype(np.float64)
    return dict(zip(z["sentences"].tolist(), emb / np.linalg.norm(emb, axis=1, keepdims=True)))


def distinct(X: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """(entry id per row, first row of each entry): rows within tol (max abs difference) of an earlier entry's first
    row share its id."""
    uid, firsts = np.full(len(X), -1), []
    for i in range(len(X)):
        if uid[i] < 0:
            same = np.nonzero((uid < 0) & (np.abs(X - X[i]).max(1) <= tol))[0]
            uid[same] = len(firsts)
            firsts.append(i)
    return uid, np.array(firsts)


def conformal_tau(scores: np.ndarray, q: float) -> float:
    s = np.sort(np.asarray(scores))
    k = math.ceil((len(s) + 1) * q)
    return float(s[min(k, len(s)) - 1])


class Cache:
    """The nominal reference of one dataset: its distinct physics vectors, its distinct sentences (+ the extra labels'), and
    the patches of every nominal frame."""

    def __init__(self, nominal: dict, table: dict[str, np.ndarray], device: str | None = None):
        import torch

        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # physics: each distinct vector once, with how many nominal frames share it
        ph = nominal["physics"].astype(np.float64)
        self.p_uid, self.p_first = distinct(ph, DUP_TOL)
        self.p_count = np.bincount(self.p_uid)
        u = ph[self.p_first]
        self.mu, self.sd = u.mean(0), np.maximum(u.std(0), SD_FLOOR)
        self.phys = (u - self.mu) / self.sd
        self.raw = u
        # semantic: each distinct sentence once; how many nominal frames have it; the extra labels' sentences always stay
        extra = [sentence([l]) for l in SEM["extra_labels"]]
        lists = dict(zip(nominal["sentences"], nominal["label_lists"])) | {t: [l] for t, l in zip(extra, SEM["extra_labels"])}
        self.s_sentences = list(dict.fromkeys(nominal["sentences"] + extra))
        self.s_labels = [lists[t] for t in self.s_sentences]
        self.s_count = np.array([nominal["sentences"].count(t) for t in self.s_sentences])
        self.s_fixed = np.array([t in extra for t in self.s_sentences])
        self.s_pos = {t: i for i, t in enumerate(self.s_sentences)}
        self.n_sentences = nominal["sentences"]
        self.table = table
        self.s_emb = np.stack([table[t] for t in self.s_sentences])
        v = nominal["vision"]
        self.P = v.shape[1]
        self.dt = torch.float16 if self.dev == "cuda" else torch.float32
        self.bank = torch.nn.functional.normalize(torch.from_numpy(v.reshape(-1, v.shape[-1]).astype(np.float32)).to(self.dev), dim=-1).to(self.dt)

    def physics(self, x: np.ndarray, excl: np.ndarray, plain: bool = False):
        """(scores, the k nearest entries as nominal frames (each entry's first frame), their distances, how many
        nominal frames share each). plain: unscaled Euclidean distance instead of the standardised one."""
        x = x.astype(np.float64)
        if plain:
            ref = self.raw
        else:
            x, ref = (x - self.mu) / self.sd, self.phys
        d = np.sqrt(np.maximum((x ** 2).sum(1)[:, None] + (ref ** 2).sum(1)[None] - 2 * x @ ref.T, 0.0))
        rows = np.nonzero(excl >= 0)[0]
        u = self.p_uid[excl[rows]]
        only = self.p_count[u] == 1                       # the left-out frame was the entry's only copy
        d[rows[only], u[only]] = np.inf
        idx = np.argsort(d, axis=1, kind="stable")[:, :K]
        dk = np.take_along_axis(d, idx, axis=1)
        return dk.mean(1), self.p_first[idx], dk, self.p_count[idx]

    def semantic(self, sentences: list[str], excl: np.ndarray):
        """(scores, the k nearest cache sentences of each frame: indices (K), distances)."""
        scores, kn, kd = [], np.full((len(sentences), K), -1), np.full((len(sentences), K), np.nan)
        for i, t in enumerate(sentences):
            d = 1.0 - self.s_emb @ self.table[t]
            d[d < EXACT] = 0.0
            if excl[i] >= 0:                                  # its copy, if it was the sentence's only nominal frame
                k = self.s_pos[self.n_sentences[excl[i]]]
                if self.s_count[k] == 1 and not self.s_fixed[k]:
                    d[k] = np.inf
            order = np.argsort(d, kind="stable")
            scores.append(float(d[order[:SEM_K]].mean()))
            kn[i, :len(order[:K])], kd[i, :len(order[:K])] = order[:K], d[order[:K]]
        return np.array(scores), kn, kd

    def vision(self, x: np.ndarray, excl: np.ndarray):
        """(scores, maps (F, P) patch distances, nearest nominal frame per patch (F, P), that frame's patch (F, P))."""
        t = self.torch
        n_worst = max(1, int(round(TOP_FRAC * self.P)))
        scores, maps, nnf, nnp = [], [], [], []
        with t.no_grad():
            for i in range(len(x)):
                q = t.nn.functional.normalize(t.from_numpy(x[i].astype(np.float32)).to(self.dev), dim=-1).to(self.dt)
                sim = q @ self.bank.T
                if excl[i] >= 0:
                    sim[:, excl[i] * self.P:(excl[i] + 1) * self.P] = -2.0
                best, arg = sim.max(dim=1)
                d = (1.0 - best.float()).clamp(min=0.0)
                scores.append(float(d.topk(n_worst).values.mean()))
                maps.append(d.cpu().numpy())
                nnf.append((arg // self.P).cpu().numpy())
                nnp.append((arg % self.P).cpu().numpy())
        return np.array(scores), np.stack(maps), np.stack(nnf), np.stack(nnp)

    @staticmethod
    def vision_knn(maps: np.ndarray, nnf: np.ndarray, nnp: np.ndarray):
        """The nominal frames supplying the nearest patches of each frame's KNN_PATCHES worst patches: most patches first
        (ties: smaller mean distance); (frames, mean distance, patch count), each (F, K), -1 / nan / 0 padded; and
        src (F, KNN_PATCHES, 3): per worst patch (this frame's patch, the nominal frame, that frame's patch) of its match."""
        kn, kd, kc = np.full((len(maps), K), -1), np.full((len(maps), K), np.nan), np.zeros((len(maps), K), int)
        src = np.full((len(maps), KNN_PATCHES, 3), -1)
        for i in range(len(maps)):
            worst = np.argsort(-maps[i])[:KNN_PATCHES]
            src[i, :len(worst)] = np.stack([worst, nnf[i][worst], nnp[i][worst]], 1)
            frames, counts = np.unique(nnf[i][worst], return_counts=True)
            mean_d = np.array([maps[i][worst][nnf[i][worst] == f].mean() for f in frames])
            order = np.lexsort((mean_d, -counts))[:K]
            kn[i, :len(order)], kd[i, :len(order)], kc[i, :len(order)] = frames[order], mean_d[order], counts[order]
        return kn, kd, kc, src

    def score(self, emb: dict, excl: np.ndarray) -> dict:
        """excl[i]: the nominal frame whose copy query i is scored without (itself / its source), -1 for none."""
        out = {}
        out["physics"], out["physics_std_knn"], out["physics_std_knn_d"], out["physics_std_knn_n"] = self.physics(emb["physics"], excl)
        # the dashboard's physics kNN: the plain Euclidean neighbours (physics_euclid's)
        out["physics_euclid"], out["physics_knn"], out["physics_knn_d"], out["physics_knn_n"] = self.physics(emb["physics"], excl, plain=True)
        out["semantic"], out["semantic_knn"], out["semantic_knn_d"] = self.semantic(emb["sentences"], excl)
        out["vision"], out["vision_map"], nnf, nnp = self.vision(emb["vision"], excl)
        out["vision_knn"], out["vision_knn_d"], out["vision_knn_n"], out["vision_knn_src"] = self.vision_knn(out["vision_map"], nnf, nnp)
        return out


def source_exclusions(cache: str, group: str, e: dict, nominal: dict) -> np.ndarray:
    """For frames made from a nominal frame (plan.src), that frame's position in the nominal group; -1 otherwise."""
    excl = np.full(len(e["idx"]), -1)
    for i, j in enumerate(e["idx"]):
        src = json.loads((DATA / cache / group / "states" / f"{int(j):04d}.json").read_text())["plan"].get("src")
        if src and src["cache"] == cache and src["group"] == "nominal":
            excl[i] = int(np.nonzero(nominal["idx"] == src["idx"])[0][0])
    return excl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=float, default=SC["q"], help="conformal quantile of the nominal leave-one-out scores")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = {"q": args.q, "k": K, "semantic_k": SEM_K, "top_frac": TOP_FRAC, "knn_patches": KNN_PATCHES, "caches": {}}
    table = load_sentence_table()
    lines = ["# snaps_mvp results", "", f"Conformal threshold q = {args.q}; z = (s - mean) / std of the dataset's nominal leave-one-out scores.", ""]
    for cache, groups in CACHES.items():
        nominal = load(cache, "nominal")
        if nominal is None:
            continue
        n = len(nominal["idx"])
        c = Cache(nominal, table)
        res, emb_groups = {"nominal": c.score(nominal, np.arange(n))}, {"nominal": nominal}
        for g in groups[1:]:
            e = load(cache, g)
            if e is not None:
                res[g], emb_groups[g] = c.score(e, source_exclusions(cache, g, e, nominal)), e
        nom = res["nominal"]
        stats = {r: {"mean": float(nom[r].mean()), "std": float(nom[r].std()) or 1.0, "tau": conformal_tau(nom[r], args.q)} for r in REPS}
        for r in REPS:      # each nominal frame's threshold from the other nominal frames' scores
            nom[f"{r}_tau"] = np.array([conformal_tau(np.delete(nom[r], i), args.q) for i in range(n)])
            for g, gr in res.items():
                if g != "nominal":
                    gr[f"{r}_tau"] = np.full(len(gr[r]), stats[r]["tau"])
        out = {"cache/semantic_sentences": np.array(c.s_sentences), "cache/semantic_labels": np.array([json.dumps(l) for l in c.s_labels]),
               "cache/semantic_count": c.s_count, "cache/semantic_added": c.s_fixed}
        for g, gr in res.items():
            out.update({f"{g}/{k}": v for k, v in gr.items()})
            out[f"{g}/idx"], out[f"{g}/desc"] = emb_groups[g]["idx"], emb_groups[g]["desc"]
        np.savez_compressed(DATA / cache / "_scores.npz", **out)

        cm = {"n": n, "physics_entries": int(len(c.p_first)), "semantic_sentences": c.s_sentences, "stats": stats, "groups": {}}
        lines += [f"## {cache}: {n} nominal frames ({len(c.p_first)} distinct physics vectors; {len(c.s_sentences)} cache sentences)", "",
                  "Semantic cache (nominal frames having each sentence; + = an extra label):", "",
                  *[f"- {cnt}{' +' if fx else ''}: {t}" for t, cnt, fx in zip(c.s_sentences, c.s_count, c.s_fixed)], "",
                  "| group | rep | score | tau | z | tau z | above tau | nominal below |", "|---|---|---|---|---|---|---|---|"]
        for g, gr in res.items():
            gm = {"n": int(len(gr["physics"]))}
            for r in REPS:
                st = stats[r]
                above = gr[r] > gr[f"{r}_tau"]
                gm[r] = {"above": float(above.mean())}
                if g == "nominal":
                    lines.append(f"| nominal (false alarms, leave-one-out) | {r} | | {st['tau']:.4f} | | {(st['tau'] - st['mean']) / st['std']:.2f} | {above.mean():.3f} | |")
                    continue
                s0 = float(gr[r][0])
                gm[r].update(score=s0, z=(s0 - st["mean"]) / st["std"], tau=st["tau"], tau_z=(st["tau"] - st["mean"]) / st["std"],
                             below=float((nom[r] < s0).mean()))
                own = " **(own)**" if OWN.get(g) == r else ""
                lines.append(f"| {g} | {r}{own} | {s0:.4f} | {st['tau']:.4f} | {gm[r]['z']:.2f} | {gm[r]['tau_z']:.2f} | "
                             f"{'**yes**' if above[0] else 'no'} | {gm[r]['below']:.3f} |")
            cm["groups"][g] = gm
        metrics["caches"][cache] = cm
        lines.append("")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (OUT / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
