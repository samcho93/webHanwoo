"""Ablation study of the offline identity pipeline, reusing cached detection/tracking and ReID embeddings.

For every video in outputs/<name>/ (needs cache_track.npz, tracks_raw.csv, cache_reid.npz) it rebuilds the
final identities under several variants and saves each as outputs/ablation/<name>/<variant>.csv
(frame, tid, x1, y1, x2, y2) so that scripts/eval_gt.py can later score them against ground-truth labels.

Variants
  raw          BoT-SORT output as is
  stitch2      legacy two-stage tracklet linking (colour histogram + motion), no herd constraint
  herd_motion  fixed-herd assignment with motion cost only (appearance weight 0)
  herd_hsv     fixed-herd assignment with an HSV colour histogram as the appearance feature
  herd_reid_ns fixed-herd assignment with the self-supervised ReID, without swap splitting
  herd_reid    + ReID swap splitting                 (= proposed, without detection zones)
  full         + detection zones / static false-positive exclusion (differs only where a zone file exists)

Proxy metrics (no ground truth needed)
  ids        number of final identities
  excl       share of detections not assigned to any identity
  coverage   mean fraction of the video's frames in which an identity is observed (higher = less fragmentation)
  frag       identities per estimated animal (ids / N)

Usage: python scripts/ablation.py [cow01 cow03 ...] [--wa 0.3 0.5 0.7 0.9]
  --wa runs the appearance-weight sensitivity sweep of the proposed variant instead.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hanwoo.config import OUTPUT_DIR, PipelineConfig  # noqa: E402
from hanwoo.identity import assign_identities, attach_embeddings, estimate_herd_size, split_swaps  # noqa: E402
from hanwoo.stitching import clean_short, fit_perspective, stitch_tracklets  # noqa: E402
from hanwoo.zones import ZoneFilter, load_zones, zones_path_for  # noqa: E402

VIDEO_DIR = Path(r"E:\Video\cow")
ABL_DIR = OUTPUT_DIR / "ablation"
COLS = ["frame", "tid", "x1", "y1", "x2", "y2"]


def load_cache(name: str):
    d = OUTPUT_DIR / name
    z = np.load(d / "cache_track.npz", allow_pickle=True)
    raw = pd.read_csv(d / "tracks_raw.csv")
    feats = {(int(f), int(t)): v for (f, t), v in zip(z["keys"], z["vals"].astype(np.float32))}
    meta = z["meta"].item()
    r = np.load(d / "cache_reid.npz", allow_pickle=True)
    emb_df = pd.DataFrame(r["meta"], columns=["frame", "tid", "conf", "occ"]).astype(dict(frame=int, tid=int))
    emb = r["emb"].astype(np.float32)
    return raw, feats, meta, emb_df, emb


def hsv_embeddings(feats: dict, emb_df: pd.DataFrame):
    """HSV histogram 'embeddings' at the same (frame, tid) samples as the ReID cache (Hellinger → cosine)."""
    keep, vecs = [], []
    for i, (f, t) in enumerate(zip(emb_df.frame.to_numpy(), emb_df.tid.to_numpy())):
        v = feats.get((int(f), int(t)))
        if v is not None:
            v = np.sqrt(np.maximum(v, 0))
            n = np.linalg.norm(v)
            if n > 0:
                keep.append(i); vecs.append(v / n)
    return emb_df.iloc[keep].reset_index(drop=True), np.stack(vecs).astype(np.float32)


def herd(clean, feats, emb_df, emb, fps, persp, n_herd, cfg, split=True):
    stitched, _, _ = stitch_tracklets(clean, dict(feats), fps, cfg, persp, stages=("short",))
    embs = attach_embeddings(stitched, emb_df, emb)
    if split:
        stitched, embs, _ = split_swaps(stitched, embs, fps, cfg)
    out, stats, _ = assign_identities(stitched, embs, fps, persp, n_herd, cfg)
    return out, stats


def metrics(df: pd.DataFrame, n_det: int, n_frames: int, n_herd: int) -> dict:
    ids = df.tid.nunique()
    cov = df.groupby("tid").frame.nunique().mean() / n_frames if ids else 0.0
    return dict(ids=int(ids), excl=round(100 * (1 - len(df) / max(n_det, 1)), 2), coverage=round(float(cov), 3),
                frag=round(ids / max(n_herd, 1), 2))


def run_video(name: str, cfg: PipelineConfig, wa_sweep=None) -> list[dict]:
    t0 = time.time()
    raw, feats, meta, emb_df, emb = load_cache(name)
    fps, n_frames = meta["fps"], meta["n_frames"]
    raw = raw.assign(rtid=raw.tid)
    n_det = len(raw)
    n_herd = cfg.herd_size or estimate_herd_size(raw, n_frames)
    clean = clean_short(raw, cfg)
    persp = fit_perspective(clean)
    out_dir = ABL_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    def save(variant, df, extra=None):
        df[COLS].to_csv(out_dir / f"{variant}.csv", index=False, float_format="%.1f")
        m = dict(video=name, variant=variant, N=n_herd, **metrics(df, n_det, n_frames, n_herd), **(extra or {}))
        rows.append(m)
        print(f"  {variant:13s} {m}")

    if wa_sweep:
        for wa in wa_sweep:
            c = dataclasses.replace(cfg, id_app_weight=wa)
            df, st = herd(clean, feats, emb_df, emb, fps, persp, n_herd, c)
            save(f"wa{wa:.1f}", df, dict(wa=wa))
        return rows

    save("raw", raw)
    st2, _, _ = stitch_tracklets(clean, dict(feats), fps, cfg, persp)
    save("stitch2", st2)
    df, _ = herd(clean, feats, emb_df, emb, fps, persp, n_herd, dataclasses.replace(cfg, id_app_weight=0.0), split=False)
    save("herd_motion", df)
    he_df, he = hsv_embeddings(feats, emb_df)
    df, _ = herd(clean, feats, he_df, he, fps, persp, n_herd, cfg, split=False)
    save("herd_hsv", df)
    df, _ = herd(clean, feats, emb_df, emb, fps, persp, n_herd, cfg, split=False)
    save("herd_reid_ns", df)
    df, _ = herd(clean, feats, emb_df, emb, fps, persp, n_herd, cfg)
    save("herd_reid", df)

    zf = ZoneFilter(load_zones(zones_path_for(VIDEO_DIR / f"{name}.mp4")), meta["width"], meta["height"])
    if zf.active:
        rz = raw[zf.keep_df(raw)].reset_index(drop=True)
        cz = clean_short(rz, cfg)
        n_z = estimate_herd_size(rz, n_frames) if not cfg.herd_size else n_herd
        df, _ = herd(cz, feats, emb_df, emb, fps, fit_perspective(cz), n_z, cfg)
        m_before = len(rows)
        save("full", df, dict(N_full=n_z))
        rows[m_before]["N"] = n_z
    else:
        df.to_csv(out_dir / "full.csv", index=False, columns=COLS, float_format="%.1f")
        rows.append(dict(rows[-1], variant="full"))
    print(f"  {name} done in {time.time() - t0:.0f}s")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="*")
    ap.add_argument("--wa", nargs="*", type=float)
    a = ap.parse_args()
    names = a.videos or sorted(p.parent.name for p in OUTPUT_DIR.glob("*/cache_reid.npz"))
    cfg = PipelineConfig()
    rows = []
    for n in names:
        print(f"=== {n} ===")
        rows += run_video(n, cfg, a.wa)
    df = pd.DataFrame(rows)
    out = ABL_DIR / ("sensitivity_wa.csv" if a.wa else "ablation_proxy.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print("saved", out)


if __name__ == "__main__":
    main()
