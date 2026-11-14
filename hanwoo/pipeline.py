"""Full pipeline: one video → outputs/<name>/ (video, CSV, JSON, dashboard)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .activity import compute_activity, summarize, timeseries
from .config import OUTPUT_DIR, PipelineConfig
from .dashboard import write_video_dashboard
from .detect_track import run_detect_track
from .identity import assign_identities, attach_embeddings, estimate_herd_size, split_swaps
from .render import render_video
from .stitching import clean_short, fit_perspective, relabel_and_interpolate, stitch_tracklets
from .zones import ZoneFilter, load_zones, size_ok, zones_path_for


def process_video(video_path, cfg: PipelineConfig | None = None, out_root: Path = OUTPUT_DIR,
                  reuse: bool = False) -> dict:
    cfg = cfg or PipelineConfig()
    video_path = Path(video_path)
    name = video_path.stem
    out_dir = Path(out_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"=== {name} ===")
    cache = out_dir / "cache_track.npz"
    if reuse and cache.exists():
        # reuse detection/tracking results → for quickly retuning linking/activity parameters
        z = np.load(cache, allow_pickle=True)
        raw = pd.read_csv(out_dir / "tracks_raw.csv")
        feats = {(int(f), int(t)): v for (f, t), v in zip(z["keys"], z["vals"].astype(np.float32))}
        meta = z["meta"].item()
        t_track = float(z["t_track"])
        print("  reuse cached detection/tracking")
    else:
        raw, feats, meta = run_detect_track(video_path, cfg)
        t_track = time.time() - t0
        raw.to_csv(out_dir / "tracks_raw.csv", index=False)
        keys = np.array(list(feats.keys()), dtype=np.int32).reshape(-1, 2)
        vals = np.array(list(feats.values()), dtype=np.float16).reshape(len(keys), -1)
        np.savez_compressed(cache, keys=keys, vals=vals, meta=np.array(meta, dtype=object), t_track=t_track)
    fps = meta["fps"]
    raw_ids = raw.tid.nunique()

    raw = raw.assign(rtid=raw.tid)                 # original tracker id (links ReID embeddings)
    # detection zones (ROI): drop false-positive boxes outside the barn / on blurry background (configs/zones/<video>.json)
    zpath = Path(cfg.zones_file) if cfg.zones_file else zones_path_for(video_path)
    zf = ZoneFilter(load_zones(zpath), meta["width"], meta["height"])
    n_zone = 0
    if zf.active:
        keep = zf.keep_df(raw)
        n_zone = int((~keep).sum())
        raw = raw[keep].reset_index(drop=True)
        print(f"  zones {zpath.name}: removed {n_zone} boxes ({n_zone / max(len(keep), 1):.1%})")
    if cfg.min_size_ratio > 0:
        p0 = fit_perspective(raw)
        keep = np.array([size_ok(b, p0, cfg.min_size_ratio) for b in raw[["x1", "y1", "x2", "y2"]].to_numpy()], bool)
        n_zone += int((~keep).sum())
        raw = raw[keep].reset_index(drop=True)
    clean = clean_short(raw, cfg)
    persp = fit_perspective(clean)
    use_id = cfg.use_identity and Path(cfg.reid_weights).exists()
    id_stats, splits = None, []
    if use_id:
        # (A) fixed herd: link short gaps only → ReID swap splitting → assign to N slots
        stitched, feats, links = stitch_tracklets(clean, feats, fps, cfg, persp, stages=("short",))
        emb_df, emb = load_or_extract_embeddings(video_path, raw, out_dir, cfg, reuse)
        embs = attach_embeddings(stitched, emb_df, emb)
        stitched, embs, splits = split_swaps(stitched, embs, fps, cfg)
        n_herd = cfg.herd_size or estimate_herd_size(raw, meta["n_frames"])
        stitched, id_stats, _ = assign_identities(stitched, embs, fps, persp, n_herd, cfg)
        print(f"  identity: herd N={n_herd}, swaps split {len(splits)}, {id_stats}")
    else:
        # (B) legacy: two-stage tracklet linking
        stitched, feats, links = stitch_tracklets(clean, feats, fps, cfg, persp)
    final, mapping = relabel_and_interpolate(stitched.drop(columns=["rtid"]), fps, cfg)
    act = compute_activity(final, fps, persp, cfg)
    summ = summarize(act, fps)
    ts = timeseries(act)
    print(f"  raw IDs {raw_ids} → after cleaning {clean.tid.nunique()} → final {summ.shape[0]}")

    act.round(4).to_csv(out_dir / "tracks_activity.csv", index=False)
    summ.to_csv(out_dir / "summary.csv", index=False)

    # background image (for dashboard trajectories)
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(48, meta["n_frames"] - 1))
    ok, bg = cap.read()
    cap.release()
    if ok:
        cv2.imwrite(str(out_dir / f"{name}_bg.jpg"), bg)

    t1 = time.time()
    if cfg.write_video:
        render_video(video_path, act, meta, out_dir / f"{name}_tracked.mp4", cfg, title=name)
    t_render = time.time() - t1

    info = dict(model=Path(cfg.weights).name if Path(cfg.weights).exists() else cfg.fallback_weights,
                cow_height_m=cfg.cow_height_m, perspective=dict(a=persp[0], b=persp[1]),
                stitch_links=links, raw_ids=int(raw_ids),
                identity=id_stats, swap_splits=splits,
                reid=Path(cfg.reid_weights).name if use_id else None)
    write_video_dashboard(out_dir, name, summ, ts, act, meta, info)

    result = dict(
        name=name, duration_s=meta["n_frames"] / fps, n_ids=int(summ.shape[0]), raw_ids=int(raw_ids),
        total_distance=float(summ.distance_m.sum()), mean_activity=float(summ.activity_index.mean()),
        proc_fps=meta["n_frames"] / max(t_track, 1e-6), track_time_s=round(t_track, 1), n_frames=meta["n_frames"],
        render_time_s=round(t_render, 1), perspective=dict(a=persp[0], b=persp[1]), n_links=len(links),
        method="reid+herd" if use_id else "stitch2", herd_size=id_stats["herd_size"] if id_stats else None,
        unassigned_det_ratio=id_stats["unassigned_det_ratio"] if id_stats else None, n_splits=len(splits),
        zone_removed=n_zone, zones=zpath.name if zf.active else None,
    )
    (out_dir / "result.json").write_text(json.dumps(dict(result=result, info=info), ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(f"  done: track {t_track:.0f}s, render {t_render:.0f}s")
    return result


def load_or_extract_embeddings(video_path, raw, out_dir, cfg: PipelineConfig, reuse: bool):
    """ReID embeddings (every reid_every frames) - cached in outputs/<name>/cache_reid.npz."""
    from .reid import extract_embeddings, load_reid

    cache = out_dir / "cache_reid.npz"
    if reuse and cache.exists():
        z = np.load(cache, allow_pickle=True)
        if str(z["weights"]) == Path(cfg.reid_weights).name and float(z["mtime"]) == Path(cfg.reid_weights).stat().st_mtime:
            print("  reuse cached ReID embeddings")
            return pd.DataFrame(z["meta"], columns=["frame", "tid", "conf", "occ"]).astype(
                dict(frame=int, tid=int)), z["emb"].astype(np.float32)
    t0 = time.time()
    model = load_reid(cfg.reid_weights)
    emb_df, emb = extract_embeddings(video_path, raw, model, every=cfg.reid_every)
    np.savez_compressed(cache, meta=emb_df.to_numpy(np.float32), emb=emb.astype(np.float16),
                        weights=Path(cfg.reid_weights).name, mtime=Path(cfg.reid_weights).stat().st_mtime)
    print(f"  ReID embeddings {len(emb_df)} crops in {time.time() - t0:.0f}s")
    return emb_df, emb
