"""Score identity results against ground-truth labels made with scripts/gt_label.py.

Ground truth: gt/<video>_gt.csv  (frame, gt_id, x1, y1, x2, y2)
  gt_id >= 1  a real cow           gt_id == 0  not a cow (false positive object)
  gt_id == -1 ignored (fragment shorter than 0.5 s / unlabeled) - predictions on it are removed for every method
Predictions: outputs/ablation/<video>/<variant>.csv (frame, tid, ...) and, if present, an online run
  (tracks_live.csv given with --online <video>=<path>).

Metrics (boxes matched per frame with IoU >= 0.5, Hungarian)
  IDF1, IDP, IDR  identity F1 / precision / recall (Ristani et al., 2016)
  IDSW            identity switches (CLEAR MOT): a GT cow matched to a different predicted id than last time
  HOTA, DetA, AssA  at alpha = 0.5 (Luiten et al., 2021)
Because the GT boxes come from the same detector, DetA mostly reflects false positives / dropped boxes;
the association metrics (IDF1, IDSW, AssA) are the ones that compare the methods.

Usage: python scripts/eval_gt.py [cow03 cow09 ...] [--online cow03=outputs/live/.../tracks_live.csv]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
GT_DIR = ROOT / "gt"
ABL_DIR = ROOT / "outputs" / "ablation"
ORDER = ["raw", "stitch2", "herd_motion", "herd_hsv", "herd_reid_ns", "herd_reid", "full"]


def iou(a, b):
    """a: (n,4) b: (m,4) → (n,m)"""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def load_pred(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "tid" not in df.columns:
        for c in ("id", "cow_id", "slot", "animal_id"):
            if c in df.columns:
                df = df.rename(columns={c: "tid"}); break
    if "interp" in df.columns:
        df = df[df.interp == 0]
    df = df[df.tid.notna() & (df.tid >= 0)]
    return df[["frame", "tid", "x1", "y1", "x2", "y2"]]


def evaluate(gt: pd.DataFrame, pr: pd.DataFrame, thr: float = 0.5) -> dict:
    f0, f1 = gt.frame.min(), gt.frame.max()
    pr = pr[(pr.frame >= f0) & (pr.frame <= f1)]
    gtf = {f: g for f, g in gt.groupby("frame")}
    prf = {f: g for f, g in pr.groupby("frame")}
    pairs = defaultdict(int)                     # (g, p) → matched frames
    gcount, pcount = defaultdict(int), defaultdict(int)
    matches = []                                 # (frame, g, p)
    n_fp = 0
    for f in sorted(set(gtf) | set(prf)):
        g = gtf.get(f); p = prf.get(f)
        gb = g[["x1", "y1", "x2", "y2"]].to_numpy() if g is not None else np.zeros((0, 4))
        gid = g.gt_id.to_numpy() if g is not None else np.zeros(0, int)
        pb = p[["x1", "y1", "x2", "y2"]].to_numpy() if p is not None else np.zeros((0, 4))
        pid = p.tid.to_numpy() if p is not None else np.zeros(0, int)
        if len(pb) and len(gb):
            # drop predictions on ignored GT boxes
            ign = gid == -1
            if ign.any():
                keep = ~(iou(pb, gb[ign]).max(1) >= thr)
                pb, pid = pb[keep], pid[keep]
        cow = gid >= 1
        gb_c, gid_c = gb[cow], gid[cow]
        for x in gid_c:
            gcount[x] += 1
        for x in pid:
            pcount[x] += 1
        if len(pb) and len(gb_c):
            m = iou(gb_c, pb)
            r, c = linear_sum_assignment(-m)
            ok = m[r, c] >= thr
            for i, j in zip(r[ok], c[ok]):
                pairs[(gid_c[i], pid[j])] += 1
                matches.append((f, gid_c[i], pid[j]))
    n_gt = sum(gcount.values()); n_pr = sum(pcount.values()); tp = len(matches)
    # IDF1: one-to-one GT id ↔ predicted id maximising co-matched frames
    gids = sorted(gcount); pids = sorted(pcount)
    if gids and pids:
        M = np.zeros((len(gids), len(pids)))
        gi = {x: i for i, x in enumerate(gids)}; pi = {x: i for i, x in enumerate(pids)}
        for (g, p), n in pairs.items():
            M[gi[g], pi[p]] = n
        r, c = linear_sum_assignment(-M)
        idtp = M[r, c].sum()
    else:
        idtp = 0
    idfp, idfn = n_pr - idtp, n_gt - idtp
    # IDSW
    last, idsw = {}, 0
    for f, g, p in matches:
        if g in last and last[g] != p:
            idsw += 1
        last[g] = p
    # HOTA at alpha = 0.5
    deta = tp / max(n_gt + n_pr - tp, 1)
    assa = 0.0
    for f, g, p in matches:
        tpa = pairs[(g, p)]
        assa += tpa / (gcount[g] + pcount[p] - tpa)
    assa = assa / max(tp, 1)
    return dict(IDF1=round(2 * idtp / max(2 * idtp + idfp + idfn, 1), 4), IDP=round(idtp / max(n_pr, 1), 4),
                IDR=round(idtp / max(n_gt, 1), 4), IDSW=idsw, HOTA=round(float(np.sqrt(deta * assa)), 4),
                DetA=round(deta, 4), AssA=round(assa, 4), gt_ids=len(gids), pred_ids=len(pids),
                gt_boxes=n_gt, pred_boxes=n_pr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="*")
    ap.add_argument("--online", nargs="*", default=[], help="video=path/to/tracks_live.csv")
    ap.add_argument("--glob", default="*.csv", help="prediction files inside outputs/ablation/<video>/")
    a = ap.parse_args()
    online = dict(x.split("=", 1) for x in a.online)
    names = a.videos or sorted(p.name[:-7] for p in GT_DIR.glob("*_gt.csv"))
    rows = []
    for v in names:
        gt = pd.read_csv(GT_DIR / f"{v}_gt.csv")
        if (gt.gt_id >= 1).sum() == 0:
            print(f"{v}: no labeled cows yet"); continue
        files = sorted((ABL_DIR / v).glob(a.glob), key=lambda p: (ORDER.index(p.stem) if p.stem in ORDER else 99, p.stem))
        if v in online:
            files.append(Path(online[v]))
        for p in files:
            r = evaluate(gt, load_pred(p))
            name = "online" if v in online and p == Path(online[v]) else p.stem
            rows.append(dict(video=v, method=name, **r))
    df = pd.DataFrame(rows)
    if df.empty:
        print("nothing to evaluate"); return
    print(df.to_string(index=False))
    out = GT_DIR / "eval_results.csv"
    df.to_csv(out, index=False)
    print("saved", out)


if __name__ == "__main__":
    main()
