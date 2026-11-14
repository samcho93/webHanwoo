"""3단계(고정 개체군): 축사 소 마릿수 N 이 고정되어 있다는 조건을 이용한 ID 유지.

온라인 추적 + 짧은 끊김 정합 뒤에도 남는 tracklet 들을, 정확히 N 개의 '개체 슬롯' 에 배정한다.

  제약  : 한 개체는 같은 시각에 두 곳에 있을 수 없다 → 시간이 겹치는 tracklet 은 같은 ID 불가
  비용  : 외형(ReID 임베딩 코사인 거리) + 움직임(이전/다음 위치에서 공백 시간 동안 이동 가능한 거리인가)
  절차  :
    1) ID 스왑 분리 — 한 tracklet 안에서 다른 소와 겹친 직후 외형 임베딩이 급변하면 그 지점에서 자른다
    2) 마릿수 N 추정 — 동시 검출 수가 N 이상인 프레임이 1% 이상인 최대 N (또는 사용자 지정)
    3) 탐욕 배정 — 긴(신뢰도 높은) tracklet 부터, 시간이 겹치지 않는 개체 슬롯 중 최소 비용 슬롯에 배정
                   (어느 슬롯에도 못 들어가고 슬롯 수 < N 이면 새 개체)
    4) 반복 개선(ICM) — 짧은 tracklet 부터 하나씩 빼서 '현재 나머지 배정' 기준 최적 슬롯으로 재배정, 수렴까지
  N 개 슬롯 어디에도 넣을 수 없는 tracklet(동시 검출이 N 초과: 중복 박스·오검출)은 제외하고 비율을 보고한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .stitching import body_px


# --------------------------------------------------------------------------- 임베딩 결합
def attach_embeddings(df: pd.DataFrame, emb_df: pd.DataFrame, emb: np.ndarray):
    """df(frame, tid, rtid) 의 각 tracklet 에 ReID 임베딩 시계열을 붙인다.

    emb_df: frame, tid(=추적기 원본 id), conf, occ  /  emb: 같은 순서의 임베딩
    반환: {tid: dict(frame, emb, w)}  (w = 품질 가중치: 겹침이 큰 crop 은 0)
    """
    e = emb_df.rename(columns={"tid": "rtid"}).reset_index().rename(columns={"index": "row"})
    m = df[["frame", "rtid", "tid"]].merge(e, on=["frame", "rtid"], how="inner")
    out = {}
    for tid, g in m.groupby("tid"):
        g = g.sort_values("frame")
        w = np.where(g.occ.to_numpy() < 0.3, g.conf.to_numpy(), 0.0)
        out[tid] = dict(frame=g.frame.to_numpy(), emb=emb[g.row.to_numpy()], w=w, occ=g.occ.to_numpy())
    return out


def _mean_emb(E, w=None):
    if len(E) == 0:
        return None
    w = np.ones(len(E)) if w is None else w
    if w.sum() <= 0:
        w = np.ones(len(E))
    v = (E * w[:, None]).sum(0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


# --------------------------------------------------------------------------- 1) 스왑 분리
def split_swaps(df: pd.DataFrame, embs: dict, fps: float, cfg: PipelineConfig):
    """다른 소와 겹친(occ≥0.2) 순간 전후 window 초의 평균 임베딩이 다르면 tracklet 을 분리."""
    W = int(cfg.split_window_s * fps)
    df = df.copy()
    next_id = int(df.tid.max()) + 1
    log = []
    queue = list(embs)
    while queue:
        tid = queue.pop()
        d = embs.get(tid)
        if d is None or len(d["frame"]) < 8:
            continue
        fr, E, w, occ = d["frame"], d["emb"], d["w"], d["occ"]
        best, best_sim = None, 1.0
        for i in np.where(occ >= 0.2)[0]:
            t = fr[i]
            b = (fr >= t - W) & (fr < t) & (w > 0)
            a = (fr > t) & (fr <= t + W) & (w > 0)
            if b.sum() < 3 or a.sum() < 3 or t - fr[0] < W // 2 or fr[-1] - t < W // 2:
                continue
            sim = float(_mean_emb(E[b], w[b]) @ _mean_emb(E[a], w[a]))
            if sim < best_sim:
                best, best_sim = t, sim
        if best is None or best_sim >= cfg.split_sim_thr:
            continue
        # 분리: best 이후 프레임에 새 id 부여
        m = (df.tid == tid) & (df.frame > best)
        df.loc[m, "tid"] = next_id
        k = fr > best
        embs[next_id] = {key: v[k] for key, v in d.items()}
        embs[tid] = {key: v[~k] for key, v in d.items()}
        log.append(dict(tid=int(tid), new=next_id, frame=int(best), sim=round(best_sim, 3)))
        queue += [tid, next_id]
        next_id += 1
    return df, embs, log


# --------------------------------------------------------------------------- 2) 마릿수 추정
def estimate_herd_size(raw: pd.DataFrame, n_frames: int, min_frac: float = 0.01, conf: float = 0.4) -> int:
    cnt = raw[raw.conf >= conf].groupby("frame").size().reindex(range(n_frames), fill_value=0).to_numpy()
    for k in range(int(cnt.max()), 0, -1):
        if (cnt >= k).mean() >= min_frac:
            return k
    return int(cnt.max())


# --------------------------------------------------------------------------- 3,4) 배정
class _Slot:
    def __init__(self):
        self.members: list[int] = []

    def intervals(self, T, exclude=None):
        return [(T[m]["start"], T[m]["end"], m) for m in self.members if m != exclude]


def _summaries(df, embs):
    T = {}
    for tid, g in df.groupby("tid"):
        g = g.sort_values("frame")
        cx = ((g.x1 + g.x2) / 2).to_numpy(); cy = ((g.y1 + g.y2) / 2).to_numpy()
        k = min(5, len(g))
        d = embs.get(tid)
        e = _mean_emb(d["emb"], d["w"]) if d is not None and len(d["frame"]) else None
        T[tid] = dict(start=int(g.frame.iloc[0]), end=int(g.frame.iloc[-1]), n=len(g),
                      p_start=np.array([np.median(cx[:k]), np.median(cy[:k])]),
                      p_end=np.array([np.median(cx[-k:]), np.median(cy[-k:])]),
                      emb=e, n_emb=0 if d is None else int((d["w"] > 0).sum()))
    return T


def _slot_proto(slot, T, exclude=None):
    es = [(T[m]["emb"], max(T[m]["n_emb"], 1)) for m in slot.members if m != exclude and T[m]["emb"] is not None]
    if not es:
        return None
    return _mean_emb(np.stack([e for e, _ in es]), np.array([n for _, n in es], float))


def _cost(t, slot, T, fps, persp, cfg, tol, exclude=None):
    """tracklet t 를 slot 에 넣는 비용. 불가능하면 None."""
    s = T[t]
    prev = nxt = None
    for a, b, m in slot.intervals(T, exclude):
        if a <= s["end"] - tol and b >= s["start"] + tol:   # 시간 겹침 → 불가
            return None
        if b < s["start"] + tol and (prev is None or b > T[prev]["end"]):
            prev = m
        if a > s["end"] - tol and (nxt is None or a < T[nxt]["start"]):
            nxt = m
    # 움직임 비용: 이전 끝 → t 시작, t 끝 → 다음 시작
    c_mot, n_mot, gaps = 0.0, 0, []
    for p, q in ((prev, t), (t, nxt)):
        if p is None or q is None:
            continue
        gap_s = max(T[q]["start"] - T[p]["end"], 0) / fps
        pa, pb = T[p]["p_end"], T[q]["p_start"]
        d_bl = np.linalg.norm(pa - pb) / body_px((pa[1] + pb[1]) / 2, persp)
        allowed = cfg.id_radius_bl + cfg.id_speed_bl_s * gap_s
        if d_bl > cfg.id_gate * allowed:
            return None
        c_mot += min(d_bl / allowed, cfg.id_gate); n_mot += 1
        gaps.append(gap_s)
    c_mot = c_mot / n_mot if n_mot else 0.5
    # 외형 비용: 시간상 이웃 tracklet(국소) + 슬롯 전체 평균(전역)
    if s["emb"] is None:
        c_app = 0.5
    else:
        loc = [T[m]["emb"] for m in (prev, nxt) if m is not None and T[m]["emb"] is not None]
        glob = _slot_proto(slot, T, exclude=t if exclude is None else exclude)
        sims = []
        if loc:
            sims.append(max(float(s["emb"] @ e) for e in loc))
        if glob is not None:
            sims.append(float(s["emb"] @ glob))
        c_app = 1 - np.mean(sims) if sims else 0.5
    c_gap = min(min(gaps) / 600, 1) if gaps else 0.5
    return cfg.id_app_weight * c_app + (1 - cfg.id_app_weight) * c_mot + 0.05 * c_gap


def assign_identities(df: pd.DataFrame, embs: dict, fps: float, persp, n_herd: int, cfg: PipelineConfig):
    T = _summaries(df, embs)
    tol = int(cfg.id_overlap_tol_s * fps)
    order = sorted(T, key=lambda t: -T[t]["n"])
    slots: list[_Slot] = []
    where: dict[int, int] = {}

    def best_slot(t, exclude=None):
        best, bc = None, np.inf
        for k, sl in enumerate(slots):
            c = _cost(t, sl, T, fps, persp, cfg, tol, exclude)
            if c is not None and c < bc:
                best, bc = k, c
        return best, bc

    # 3) 탐욕 배정 (긴 tracklet 부터)
    for t in order:
        k, c = best_slot(t)
        if k is not None and (c < cfg.id_new_cost or len(slots) >= n_herd):
            slots[k].members.append(t); where[t] = k
        elif len(slots) < n_herd:
            slots.append(_Slot()); slots[-1].members.append(t); where[t] = len(slots) - 1
        # else: 미배정 (N 초과 동시 검출)

    # 4) ICM 반복 개선 (짧은 tracklet 부터 재배정)
    moves_log = []
    for it in range(cfg.id_refine_iters):
        moves = 0
        for t in sorted(T, key=lambda t: T[t]["n"]):
            cur = where.get(t)
            if cur is not None:
                slots[cur].members.remove(t)
            k, c = best_slot(t)
            if k is None:
                if cur is not None:   # 원래 자리로
                    slots[cur].members.append(t)
                continue
            slots[k].members.append(t); where[t] = k
            if k != cur:
                moves += 1
        moves_log.append(moves)
        if moves == 0:
            break

    # 결과: tid → 개체 슬롯 번호
    out = df[df.tid.isin(where)].copy()
    out["tid"] = out.tid.map(lambda t: where[t] + 1000)  # relabel 단계에서 1..N 으로 재부여
    # 허용 오차(tol) 안의 겹침 프레임 → 신뢰도 높은 쪽만 남김
    out = out.sort_values("conf", ascending=False).drop_duplicates(["frame", "tid"]).sort_values(["frame", "tid"])
    n_un = int(df.tid.isin(set(T) - set(where)).sum())
    stats = dict(herd_size=int(n_herd), slots=len(slots), tracklets=len(T), unassigned_tracklets=len(T) - len(where),
                 unassigned_det_ratio=round(n_un / max(len(df), 1), 4), icm_moves=moves_log)
    return out.reset_index(drop=True), stats, {int(t): int(k) for t, k in where.items()}
