"""3단계: 개체별 활동량 계산.

- 위치: bbox 중심을 1초 창으로 평활화 (bbox 떨림 제거)
- 원근 보정: 화면 y 에서의 소 체고(px) h(y) 로 px→m 변환  (m/px = 체고[m] / h(y))
- 속도: 평활 위치의 ±0.5초 중앙차분  [m/s]
- 국소 움직임(motion energy): bbox 내부 프레임차 평균 → 제자리 섭식·머리 움직임 포착
- 상태: 보행(Walk) / 국소활동(Active) / 휴식(Rest)
- 활동지수(0~100) = 70%·속도항 + 30%·국소움직임항
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
    # 활동량은 '실제로 검출된' 프레임으로만 계산한다.
    # 보간 프레임(검출 누락을 메운 가상 박스)은 위치가 실제가 아니므로 이동거리·속도·상태 계산에서 제외하고,
    # 아래에서 직전 관측값을 이어받는 '표시용' 행으로만 되붙인다.
    interp_rows = df[df.interp == 1].copy() if "interp" in df else df.iloc[:0].copy()
    df = df[df.interp == 0] if "interp" in df else df
    df = df.sort_values(["tid", "frame"]).copy()
    # 관측이 끊긴(프레임 불연속) 구간마다 따로 평활·미분 → 장시간 부재 전후 위치가 섞이지 않게
    df["seg"] = (df.groupby("tid").frame.diff().fillna(1) > 1).groupby(df.tid).cumsum()
    for (tid, _seg), g in df.groupby(["tid", "seg"]):
        g = g.copy()
        cx = (g.x1 + g.x2) / 2
        cy = (g.y1 + g.y2) / 2
        # 중앙값 → 평균 2단 평활화 (outlier 에 강함)
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
        # 같은 m/px 기준으로 변위 계산 (현재 위치 스케일 사용)
        dx = (sx.to_numpy()[b] - sx.to_numpy()[a]) * m_per_px
        dy = (sy.to_numpy()[b] - sy.to_numpy()[a]) * m_per_px
        speed = np.minimum(np.hypot(dx, dy) / dt, cfg.max_speed)
        motion = g.motion.where(g.interp == 0).ffill().bfill().fillna(0)
        motion_s = motion.rolling(win, center=True, min_periods=1).mean().to_numpy()

        state = np.zeros(n, int)
        state[(speed >= cfg.rest_speed) | (motion_s >= cfg.motion_energy_active)] = 1
        state[speed >= cfg.walk_speed] = 2
        act = 100 * (0.7 * np.clip(speed / 0.5, 0, 1) + 0.3 * np.clip(motion_s / cfg.motion_norm, 0, 1))
        step = np.where(speed >= cfg.rest_speed, speed / fps, 0.0)  # 정지 떨림은 거리 누적 제외

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
        g = g_all[g_all.interp == 0]          # 통계는 실제 관측 프레임만
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
    """bin_s 초 단위로 개체별 평균 활동지수 / 이동거리."""
    a = act[act.interp == 0].copy()
    a["bin"] = (a.t // bin_s * bin_s).round(1)
    g = a.groupby(["bin", "tid"]).agg(activity=("activity", "mean"), speed=("speed", "mean")).reset_index()
    return g
