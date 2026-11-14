"""2단계: 오프라인 ID 정합 (Tracklet stitching).

온라인 추적기(BoT-SORT)는 펜스·기둥 가림, 소끼리 겹침 때문에 같은 소에게
여러 개의 ID 를 부여하기 쉽다. 축사는 '닫힌 공간 + 고정 카메라' 이므로
영상 전체를 본 뒤(offline) 끊어진 tracklet 들을 다시 이어붙이면 ID 일관성이 크게 좋아진다.

  1) 짧은 tracklet(노이즈) 제거
  2) 원근 모델  h(y) = a*y + b  (화면 위쪽일수록 소가 작게 보임)
  3) tracklet i 의 '끝'  ↔  tracklet j 의 '시작' 비용 행렬
        cost = (1-w)·거리비용(원근 정규화, 시간에 비례해 허용반경 증가) + w·외형거리(Bhattacharyya)
  4) Hungarian 알고리즘으로 1:1 최적 연결 → 비용이 임계값 이하인 것만 병합 (수렴할 때까지 반복)
  5) ID 재부여(1..N) + 짧은 공백 선형 보간
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .config import PipelineConfig


# --------------------------------------------------------------------------- 원근 모델
def fit_perspective(df: pd.DataFrame):
    """bbox 높이 vs 화면 y 좌표의 선형 관계를 robust 하게 적합.

    가림 때문에 bbox 가 잘린 경우가 많으므로 y 구간별 상위 75% 분위 높이를 사용한다.
    반환: (a, b)  →  h(y) = a*y + b   [px]
    """
    cy = ((df.y1 + df.y2) / 2).to_numpy()
    h = (df.y2 - df.y1).to_numpy()
    if len(cy) < 20:
        return 0.0, float(np.median(h)) if len(h) else 100.0
    bins = np.quantile(cy, np.linspace(0, 1, 11))
    xs, ys = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (cy >= lo) & (cy <= hi)
        if m.sum() >= 5:
            xs.append(np.median(cy[m]))
            ys.append(np.quantile(h[m], 0.75))
    a, b = np.polyfit(xs, ys, 1)
    if a <= 0:  # 비정상 → 상수 모델
        a, b = 0.0, float(np.median(ys))
    return float(a), float(b)


def body_px(y, persp):
    a, b = persp
    return np.maximum(a * np.asarray(y) + b, 20.0)


# --------------------------------------------------------------------------- tracklet 요약
def _summaries(df: pd.DataFrame, feats: dict):
    out = {}
    for tid, g in df.groupby("tid"):
        g = g.sort_values("frame")
        cx = ((g.x1 + g.x2) / 2).to_numpy()
        cy = ((g.y1 + g.y2) / 2).to_numpy()
        k = min(5, len(g))
        hists = [feats[(f, tid)] for f in g.frame.to_numpy() if (f, tid) in feats]
        app = np.mean(hists, axis=0).astype(np.float32) if hists else None
        out[tid] = dict(
            start=int(g.frame.iloc[0]), end=int(g.frame.iloc[-1]),
            p_start=np.array([np.median(cx[:k]), np.median(cy[:k])]),
            p_end=np.array([np.median(cx[-k:]), np.median(cy[-k:])]),
            app=app, n=len(g),
        )
    return out


def _app_dist(a, b):
    if a is None or b is None:
        return 0.5
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


def stitch_stages(cfg: PipelineConfig):
    """(이름, 최대공백s, 기본반경BL, 속도BL/s, 외형가중치, 최대비용) — 1차: 짧은 끊김, 2차: 장시간 재등장"""
    return [
        ("short", cfg.stitch_max_gap_s, cfg.stitch_max_dist_bl, cfg.stitch_speed_bl_s,
         cfg.stitch_app_weight, cfg.stitch_max_cost),
        ("long", cfg.stitch2_max_gap_s, cfg.stitch2_max_dist_bl, cfg.stitch2_speed_bl_s,
         cfg.stitch2_app_weight, cfg.stitch2_max_cost),
    ]


def stitch_tracklets(df: pd.DataFrame, feats: dict, fps: float, cfg: PipelineConfig, persp, stages=("short", "long")):
    """2단계(짧은 끊김 → 장시간 재등장) tracklet 병합. 반환: (df, feats, 병합 로그)

    ReID 기반 고정 개체군 배정(identity.py)을 쓸 때는 1차("short")만 수행하고 장시간 재등장은 배정 단계가 담당한다.
    """
    log = []
    for stage in stitch_stages(cfg):
        if stage[0] not in stages:
            continue
        df, feats, lg = _stitch_stage(df, feats, fps, persp, *stage)
        log += lg
    return df, feats, log


def _stitch_stage(df, feats, fps, persp, stage, max_gap_s, dist_bl, speed_bl_s, app_w, max_cost):
    df = df.copy()
    max_gap = int(max_gap_s * fps)
    log = []
    for _it in range(10):  # 연결이 더 이상 없을 때까지 반복
        S = _summaries(df, feats)
        tids = sorted(S)
        n = len(tids)
        if n < 2:
            break
        C = np.full((n, n), 1e3)
        for i, ti in enumerate(tids):
            si = S[ti]
            for j, tj in enumerate(tids):
                if i == j:
                    continue
                sj = S[tj]
                gap = sj["start"] - si["end"]
                if gap <= 0 or gap > max_gap:
                    continue
                ymid = (si["p_end"][1] + sj["p_start"][1]) / 2
                d_bl = np.linalg.norm(si["p_end"] - sj["p_start"]) / body_px(ymid, persp)
                allowed = dist_bl + speed_bl_s * gap / fps
                if d_bl > allowed:
                    continue
                c_dist = d_bl / allowed
                c_app = _app_dist(si["app"], sj["app"])
                c_gap = 0.15 * gap / max_gap
                C[i, j] = (1 - app_w) * c_dist + app_w * c_app + c_gap
        rows, cols = linear_sum_assignment(C)
        links = [(tids[r], tids[c], C[r, c]) for r, c in zip(rows, cols) if C[r, c] < max_cost]
        if not links:
            break
        # 체인(i→j→k)을 한 번에 처리하기 위해 union-find 형태로 루트 ID 로 치환
        parent = {}
        for a, b, c in links:
            parent[b] = a
            log.append(dict(stage=stage, src=int(b), dst=int(a), cost=round(float(c), 3)))

        def root(t):
            while t in parent:
                t = parent[t]
            return t

        new_tid = df.tid.map(lambda t: root(t))
        # feats 키도 갱신
        feats = {(f, root(t)): v for (f, t), v in feats.items()}
        df["tid"] = new_tid
        # 동일 프레임 중복(이론상 없음) 방지
        df = df.sort_values("conf", ascending=False).drop_duplicates(["frame", "tid"]).sort_values(["frame", "tid"])
    return df.reset_index(drop=True), feats, log


def relabel_and_interpolate(df: pd.DataFrame, fps: float, cfg: PipelineConfig):
    """ID 를 등장 순서대로 1..N 으로 재부여하고, 짧은 공백은 선형 보간."""
    first = df.groupby("tid").agg(f0=("frame", "min"), x=("x1", "first")).sort_values(["f0", "x"])
    mapping = {old: i + 1 for i, old in enumerate(first.index)}
    df = df.copy()
    df["tid"] = df.tid.map(mapping)
    df["interp"] = 0
    max_gap = int(cfg.interp_max_gap_s * fps)
    parts = []
    for tid, g in df.groupby("tid"):
        g = g.set_index("frame").sort_index()
        full = np.arange(g.index.min(), g.index.max() + 1)
        missing = np.setdiff1d(full, g.index)
        if len(missing):
            gg = g.reindex(full)
            # 공백 길이 계산 → max_gap 이하만 보간
            isna = gg.x1.isna().to_numpy()
            run_id = np.cumsum(~isna)
            run_len = pd.Series(isna).groupby(run_id).transform("sum").to_numpy()
            fill = isna & (run_len <= max_gap)
            for c in ["x1", "y1", "x2", "y2"]:
                gg[c] = gg[c].interpolate(limit_area="inside")
            gg.loc[isna, "interp"] = 1
            gg.loc[isna, "conf"] = 0.0
            gg = gg[~isna | fill]
            g = gg
        g["tid"] = tid
        parts.append(g.reset_index().rename(columns={"index": "frame"}))
    out = pd.concat(parts, ignore_index=True).sort_values(["frame", "tid"]).reset_index(drop=True)
    out["frame"] = out.frame.astype(int)
    out["interp"] = out.interp.astype(int)
    return out, mapping


def clean_short(df: pd.DataFrame, cfg: PipelineConfig):
    cnt = df.groupby("tid").frame.transform("count")
    return df[cnt >= cfg.min_track_frames].copy()
