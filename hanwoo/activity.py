"""Stage 3: per-animal activity.

- Position: bbox centre smoothed over a 1 s window (removes bbox jitter)
- Perspective correction: px→m using the cow height in px h(y) at screen row y  (m/px = height[m] / h(y))
- Speed: ±0.5 s central difference of the smoothed position  [m/s]
- Local motion (motion energy): mean frame difference inside the bbox → captures feeding / head movement in place
- State: Walk / Active (local activity) / Rest
- Activity index (0-100) = 70%·speed term + 30%·local-motion term
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .stitching import body_px


def compute_activity(df: pd.DataFrame, fps: float, persp, cfg: PipelineConfig) -> pd.DataFrame:
    win = max(3, int(round(cfg.smooth_window_s * fps)) | 1)
    half = max(1, int(round(0.5 * fps)))
    parts = []
    # Activity is computed only from frames with real detections.
    # Interpolated frames (virtual boxes filling missed detections) have no real position, so they are excluded
    # from distance/speed/state and re-attached below as display-only rows carrying the last observed values.
    interp_rows = df[df.interp == 1].copy() if "interp" in df else df.iloc[:0].copy()
    df = df[df.interp == 0] if "interp" in df else df
    df = df.sort_values(["tid", "frame"]).copy()
    # Smooth/differentiate each contiguous observed segment separately so positions before and after a long absence don't mix
    df["seg"] = (df.groupby("tid").frame.diff().fillna(1) > 1).groupby(df.tid).cumsum()
    for (tid, _seg), g in df.groupby(["tid", "seg"]):
        g = g.copy()
        cx = (g.x1 + g.x2) / 2
        cy = (g.y1 + g.y2) / 2
        # two-pass median → mean smoothing (robust to outliers)
        sx = cx.rolling(win, center=True, min_periods=1).median().rolling(win, center=True, min_periods=1).mean()
        sy = cy.rolling(win, center=True, min_periods=1).median().rolling(win, center=True, min_periods=1).mean()
        m_per_px = cfg.cow_height_m / body_px(sy.to_numpy(), persp)
        xm = sx.to_numpy() * m_per_px
        ym = sy.to_numpy() * m_per_px
        n = len(g)
        idx = np.arange(n)
        a = np.clip(idx - half, 0, n - 1)
        b = np.clip(idx + half, 0, n - 1)
        frames = g.frame.to_numpy()
        dt = np.maximum(frames[b] - frames[a], 1) / fps
        # displacement with a single m/px (scale at the current position)
        dx = (sx.to_numpy()[b] - sx.to_numpy()[a]) * m_per_px
        dy = (sy.to_numpy()[b] - sy.to_numpy()[a]) * m_per_px
        speed = np.minimum(np.hypot(dx, dy) / dt, cfg.max_speed)
        motion = g.motion.where(g.interp == 0).ffill().bfill().fillna(0)
        motion_s = motion.rolling(win, center=True, min_periods=1).mean().to_numpy()

        state = np.zeros(n, int)
        state[(speed >= cfg.rest_speed) | (motion_s >= cfg.motion_energy_active)] = 1
        state[speed >= cfg.walk_speed] = 2
        act = 100 * (0.7 * np.clip(speed / 0.5, 0, 1) + 0.3 * np.clip(motion_s / cfg.motion_norm, 0, 1))
        step = np.where(speed >= cfg.rest_speed, speed / fps, 0.0)  # jitter while stationary is not added to distance

        g["cx"], g["cy"] = sx.to_numpy(), sy.to_numpy()
        g["xm"], g["ym"] = xm, ym
        g["speed"] = speed
        g["motion_s"] = motion_s
        g["state"] = state
        g["activity"] = act
        g["step"] = step
        parts.append(g)
    out = pd.concat(parts)
    if len(interp_rows):
        ir = interp_rows.copy()
        ir["cx"], ir["cy"] = (ir.x1 + ir.x2) / 2, (ir.y1 + ir.y2) / 2
        ir["speed"], ir["step"] = 0.0, 0.0
        out = pd.concat([out, ir]).sort_values(["tid", "frame"])
        carry = ["xm", "ym", "motion_s", "state", "activity"]
        out[carry] = out.groupby("tid")[carry].ffill().bfill()
        out["state"] = out.state.astype(int)
    out = out.sort_values(["frame", "tid"]).reset_index(drop=True)
    out["dist_cum"] = out.groupby("tid").step.cumsum()
    out = out.drop(columns=["seg", "step"])
    out["t"] = out.frame / fps
    return out


def summarize(act: pd.DataFrame, fps: float) -> pd.DataFrame:
    rows = []
    for tid, g_all in act.groupby("tid"):
        g = g_all[g_all.interp == 0]          # statistics use observed frames only
        n = len(g)
        rows.append(dict(
            id=int(tid),
            first_s=round(g_all.t.min(), 1), last_s=round(g_all.t.max(), 1),
            tracked_s=round(n / fps, 1),
            observed_ratio=round(n / max(int(g_all.frame.max() - g_all.frame.min()) + 1, 1), 3),
            distance_m=round(float(g_all.dist_cum.iloc[-1]), 2),
            mean_speed=round(float(g.speed.mean()), 3),
            max_speed=round(float(g.speed.quantile(0.98)), 3),
            rest_pct=round(100 * float((g.state == 0).mean()), 1),
            active_pct=round(100 * float((g.state == 1).mean()), 1),
            walk_pct=round(100 * float((g.state == 2).mean()), 1),
            activity_index=round(float(g.activity.mean()), 1),
        ))
    return pd.DataFrame(rows)


def timeseries(act: pd.DataFrame, bin_s: float = 5.0) -> pd.DataFrame:
    """Per-animal mean activity index / speed in bins of bin_s seconds."""
    a = act[act.interp == 0].copy()
    a["bin"] = (a.t // bin_s * bin_s).round(1)
    g = a.groupby(["bin", "tid"]).agg(activity=("activity", "mean"), speed=("speed", "mean")).reset_index()
    return g
