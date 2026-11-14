"""Stage 3 (fixed herd): keep IDs stable using the constraint that the barn holds a fixed number N of cows.

Tracklets left after online tracking + short-gap linking are assigned to exactly N 'animal slots'.

  Constraint: one animal cannot be in two places at once → tracklets overlapping in time cannot share an ID
  Cost      : appearance (ReID embedding cosine distance) + motion (is the distance reachable from the previous/next position within the gap?)
  Procedure :
    1) ID-swap splitting - cut a tracklet where its appearance embedding changes abruptly right after overlapping another cow
    2) Herd-size estimate - the largest N such that >= 1% of frames have N or more simultaneous detections (or user-specified)
    3) Greedy assignment - longest (most reliable) tracklets first, into the lowest-cost slot with no time overlap
                   (if it fits no slot and slots < N, it starts a new animal)
    4) Iterative refinement (ICM) - remove tracklets one at a time, shortest first, and reassign each to the best slot given the rest, until convergence
  Tracklets that fit none of the N slots (more than N simultaneous detections: duplicate boxes / false positives) are dropped and their share is reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .stitching import body_px


# --------------------------------------------------------------------------- embedding attachment
def attach_embeddings(df: pd.DataFrame, emb_df: pd.DataFrame, emb: np.ndarray):
    """Attach a ReID embedding time series to each tracklet of df(frame, tid, rtid).

    emb_df: frame, tid (= original tracker id), conf, occ  /  emb: embeddings in the same order
    Returns: {tid: dict(frame, emb, w)}  (w = quality weight: 0 for heavily overlapped crops)
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


# --------------------------------------------------------------------------- 1) swap splitting
def split_swaps(df: pd.DataFrame, embs: dict, fps: float, cfg: PipelineConfig):
    """Split a tracklet when the mean embeddings in the window seconds before/after an overlap with another cow (occ>=0.2) differ."""
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
        # split: assign a new id to frames after best
        m = (df.tid == tid) & (df.frame > best)
        df.loc[m, "tid"] = next_id
        k = fr > best
        embs[next_id] = {key: v[k] for key, v in d.items()}
        embs[tid] = {key: v[~k] for key, v in d.items()}
        log.append(dict(tid=int(tid), new=next_id, frame=int(best), sim=round(best_sim, 3)))
        queue += [tid, next_id]
        next_id += 1
    return df, embs, log


# --------------------------------------------------------------------------- 2) herd-size estimate
def estimate_herd_size(raw: pd.DataFrame, n_frames: int, min_frac: float = 0.01, conf: float = 0.4) -> int:
    cnt = raw[raw.conf >= conf].groupby("frame").size().reindex(range(n_frames), fill_value=0).to_numpy()
    for k in range(int(cnt.max()), 0, -1):
        if (cnt >= k).mean() >= min_frac:
            return k
    return int(cnt.max())


# --------------------------------------------------------------------------- 3,4) assignment
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
    """Cost of putting tracklet t into slot. None if impossible."""
    s = T[t]
    prev = nxt = None
    for a, b, m in slot.intervals(T, exclude):
        if a <= s["end"] - tol and b >= s["start"] + tol:   # overlaps in time → impossible
            return None
        if b < s["start"] + tol and (prev is None or b > T[prev]["end"]):
            prev = m
        if a > s["end"] - tol and (nxt is None or a < T[nxt]["start"]):
            nxt = m
    # motion cost: previous end → t start, t end → next start
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
    # appearance cost: temporally neighbouring tracklets (local) + slot-wide mean (global)
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

    # 3) greedy assignment (longest tracklets first)
    for t in order:
        k, c = best_slot(t)
        if k is not None and (c < cfg.id_new_cost or len(slots) >= n_herd):
            slots[k].members.append(t); where[t] = k
        elif len(slots) < n_herd:
            slots.append(_Slot()); slots[-1].members.append(t); where[t] = len(slots) - 1
        # else: unassigned (more than N simultaneous detections)

    # 4) ICM refinement (reassign, shortest tracklets first)
    moves_log = []
    for it in range(cfg.id_refine_iters):
        moves = 0
        for t in sorted(T, key=lambda t: T[t]["n"]):
            cur = where.get(t)
            if cur is not None:
                slots[cur].members.remove(t)
            k, c = best_slot(t)
            if k is None:
                if cur is not None:   # back to its original slot
                    slots[cur].members.append(t)
                continue
            slots[k].members.append(t); where[t] = k
            if k != cur:
                moves += 1
        moves_log.append(moves)
        if moves == 0:
            break

    # result: tid → animal slot number
    out = df[df.tid.isin(where)].copy()
    out["tid"] = out.tid.map(lambda t: where[t] + 1000)  # renumbered to 1..N in the relabel step
    # overlapping frames within tolerance (tol) → keep only the more confident one
    out = out.sort_values("conf", ascending=False).drop_duplicates(["frame", "tid"]).sort_values(["frame", "tid"])
    n_un = int(df.tid.isin(set(T) - set(where)).sum())
    stats = dict(herd_size=int(n_herd), slots=len(slots), tracklets=len(T), unassigned_tracklets=len(T) - len(where),
                 unassigned_det_ratio=round(n_un / max(len(df), 1), 4), icm_moves=moves_log)
    return out.reset_index(drop=True), stats, {int(t): int(k) for t, k in where.items()}
