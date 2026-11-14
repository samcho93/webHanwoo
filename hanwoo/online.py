"""Real-time (online) herd counting + fixed-population ID maintenance + activity.

The offline approach (identity.py) assigns IDs after seeing the whole video, but a camera stream has no 'future'.
So the work is split into two phases.

  1) Enrollment — the first enroll_s seconds
       · Simultaneous count: max number of detections with confidence ≥ count_conf held for ≥ count_stable_s s  → n_sim
       · Appearance clusters: tracker tracks' mean ReID embeddings grouped by average linkage under the
                       constraint 'tracks seen at the same time are different animals'; number of clusters → n_clu
       · Herd size N : user-specified > (count_mode) rule.  Clusters become the initial appearance galleries of the animal slots.
  2) Tracking
       · Frame-to-frame association is BoT-SORT; N 'animal slots' sit on top of it.
       · A new track collects appearance for confirm_s seconds, then is Hungarian-matched to free (unseen) slots with
         cost = w·appearance distance + (1-w)·motion distance ratio (relative to how far it could move while missing).
       · If a track is unseen for release_s seconds or more, its slot becomes 'missing' and remembers the last position.
       · When two tracks overlap and separate, before/after appearance is compared to correct swapped IDs (swap).
       · If a track's recent appearance is much closer to a 'missing' slot than to its own, it moves to that slot (drift).
       · If all N slots are visible and a confident non-overlapping track stays visible for grow_s more seconds, N is increased by 1 (auto_grow).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

INF = 1e9


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def body_px(y, persp):
    a, b = persp
    return max(a * y + b, 20.0)


# --------------------------------------------------------------------------- state objects
@dataclass
class Track:
    """One tracker (BoT-SORT) track (= a continuous segment the tracker believes is the same animal)."""
    key: int
    t_first: float
    t_last: float
    box: tuple = (0, 0, 0, 0)
    conf: float = 0.0
    embs: deque = field(default_factory=lambda: deque(maxlen=300))  # (t, emb)
    slot: int | None = None          # assigned animal slot index (None = waiting)
    last_slot: int | None = None     # last assigned slot (preferred if the same track returns)
    overlap: dict = field(default_factory=dict)   # other track key → last time they overlapped
    t_unassigned: float | None = None              # time it became 'assignable' while waiting
    t_check: float = 0.0                           # time of the last drift check

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2

    def mean_emb(self, t_from=-INF):
        E = [e for t, e in self.embs if t >= t_from]
        return (_norm(np.mean(E, 0)), len(E)) if E else (None, 0)


@dataclass
class Slot:
    """Animal slot = one animal in the pen.  ID is sid (1..N)."""
    sid: int
    anchors: list = field(default_factory=list)                     # appearance at enrollment (fixed)
    recent: deque = field(default_factory=lambda: deque(maxlen=60))  # appearance updated during tracking
    track: int | None = None          # currently bound track key
    pos: tuple | None = None          # last observed centre
    t_last: float = -INF
    proto: np.ndarray | None = None

    def update_proto(self):
        E = list(self.anchors) + list(self.recent)
        self.proto = _norm(np.mean(E, 0)) if E else None

    def add(self, e):
        if self.proto is not None and float(e @ self.proto) < 0.2:   # keep clearly different appearances out of the gallery
            return
        self.recent.append(e)
        self.update_proto()


# --------------------------------------------------------------------------- activity (causal)
class SlotActivity:
    """Position smoothing → speed → state/activity index/cumulative distance from observed frames only (real time, past data only)."""

    def __init__(self):
        self.hist = deque()     # (t, cx, cy, motion)
        self.dist = 0.0
        self.t_prev = None
        self.speed = 0.0
        self.state = 0
        self.activity = 0.0
        self.t_obs = 0.0        # total observed time
        self.state_time = [0.0, 0.0, 0.0]
        self.act_sum = 0.0

    def update(self, t, cx, cy, motion, persp, cfg):
        if self.hist and t - self.hist[-1][0] > cfg.interp_max_gap_s:
            self.hist.clear()                       # long gap starts a new segment (avoids fake travel distance)
            self.t_prev = None
        self.hist.append((t, cx, cy, motion))
        while self.hist and t - self.hist[0][0] > 2 * cfg.smooth_window_s:
            self.hist.popleft()
        H = np.array(self.hist)
        w = cfg.smooth_window_s
        cur = H[H[:, 0] >= t - w / 2]               # median of the last 0.5 s = current position
        old = H[H[:, 0] <= t - w]                   # around 1 s ago
        sx, sy = np.median(cur[:, 1]), np.median(cur[:, 2])
        mpp = cfg.cow_height_m / body_px(sy, persp)
        if len(old):
            old = old[old[:, 0] >= old[-1, 0] - w / 2]
            ox, oy, ot = np.median(old[:, 1]), np.median(old[:, 2]), np.median(old[:, 0])
            dt = max(t - w / 4 - ot, 1e-3)
            self.speed = min(np.hypot(sx - ox, sy - oy) * mpp / dt, cfg.max_speed)
        else:
            self.speed = 0.0
        motion_s = float(np.mean(H[H[:, 0] >= t - w, 3]))
        st = 0
        if self.speed >= cfg.rest_speed or motion_s >= cfg.motion_energy_active:
            st = 1
        if self.speed >= cfg.walk_speed:
            st = 2
        self.state = st
        self.activity = 100 * (0.7 * min(self.speed / 0.5, 1) + 0.3 * min(motion_s / cfg.motion_norm, 1))
        dt = 0.0 if self.t_prev is None else t - self.t_prev
        if self.speed >= cfg.rest_speed:
            self.dist += self.speed * dt
        self.t_obs += dt
        self.state_time[st] += dt
        self.act_sum += self.activity * dt
        self.t_prev = t
        return sx, sy


# --------------------------------------------------------------------------- main
class OnlineHerd:
    def __init__(self, cfg, log=print):
        self.cfg = cfg
        self.log = log
        self.reset()

    def reset(self, keep_slots=False):
        self.phase = "enroll"
        self.t0 = None
        self.counts = deque()        # (t, confident detection count)
        self.boxes_enroll = []       # for fitting the perspective model (cy, h)
        self.tracks: dict[int, Track] = {}
        self.cooc: dict[tuple, float] = {}   # track pairs seen together during enrollment → overlap time (s)
        if not keep_slots:
            self.slots: list[Slot] = []
            self.acts: list[SlotActivity] = []
        self.persp = (0.0, 100.0)
        self.n_sim = self.n_clu = 0
        self.N = 0
        self.extra_since = None
        self.events = deque(maxlen=200)

    def event(self, t, msg):
        self.events.append((t, msg))
        self.log(f"[{t:7.1f}s] {msg}")

    # ------------------------------------------------------------------ input
    def step(self, t, dets, embs, dt):
        """t: seconds, dets: [(key, box, conf, occ, motion)], embs: {key: emb} (only those extracted this frame)
        Returns: [(key, box, conf, sid or None, label_state)]"""
        cfg = self.cfg
        if self.t0 is None:
            self.t0 = t
        seen = set()
        for key, box, conf, occ, _m in dets:
            tr = self.tracks.get(key)
            if tr is None:
                tr = self.tracks[key] = Track(key, t, t)
            tr.t_last, tr.box, tr.conf = t, box, conf
            seen.add(key)
            if key in embs:
                tr.embs.append((t, embs[key]))
        # record overlaps (occlusion): for swap correction
        keys = [d[0] for d in dets]
        B = np.array([d[1] for d in dets], float).reshape(-1, 4)
        if len(B) > 1:
            iou = _iou_matrix(B)
            ii, jj = np.where(np.triu(iou, 1) >= cfg.swap_iou)
            for i, j in zip(ii, jj):
                self.tracks[keys[i]].overlap[keys[j]] = t
                self.tracks[keys[j]].overlap[keys[i]] = t
        # clean up old tracks
        for k in [k for k, tr in self.tracks.items() if t - tr.t_last > 30]:
            self._unbind_track(k)
            del self.tracks[k]

        if self.phase == "enroll":
            n = sum(1 for d in dets if d[2] >= cfg.count_conf)
            self.counts.append((t, n))
            for _k, b, c, occ, _m in dets:
                if c >= cfg.count_conf and occ < 0.3:
                    self.boxes_enroll.append(((b[1] + b[3]) / 2, b[3] - b[1]))
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    p = (min(keys[i], keys[j]), max(keys[i], keys[j]))
                    self.cooc[p] = self.cooc.get(p, 0.0) + dt
            if t - self.t0 >= cfg.enroll_s:
                self._finish_enroll(t, seen)
            return [(d[0], d[1], d[2], None, "enroll") for d in dets]

        self._track_step(t, dets, seen, dt)
        out = []
        for key, box, conf, _occ, _m in dets:
            tr = self.tracks[key]
            out.append((key, box, conf, tr.slot, "ok" if tr.slot is not None else "pending"))
        return out

    # ------------------------------------------------------------------ 1) enrollment → herd count
    def _finish_enroll(self, t, seen):
        cfg = self.cfg
        # perspective model h(y) = a·y + b
        if len(self.boxes_enroll) >= 20:
            P = np.array(self.boxes_enroll)
            bins = np.quantile(P[:, 0], np.linspace(0, 1, 11))
            xs, ys = [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                m = (P[:, 0] >= lo) & (P[:, 0] <= hi)
                if m.sum() >= 5:
                    xs.append(np.median(P[m, 0])); ys.append(np.quantile(P[m, 1], 0.75))
            if len(xs) >= 2:
                a, b = np.polyfit(xs, ys, 1)
                self.persp = (float(a), float(b)) if a > 0 else (0.0, float(np.median(ys)))
        # (a) stable simultaneous detection count
        C = np.array(self.counts)
        n_sim = 0
        for k in range(int(C[:, 1].max()) if len(C) else 0, 0, -1):
            run, best, t_start = False, 0.0, 0.0
            for tt, c in C:
                if c >= k and not run:
                    run, t_start = True, tt
                elif c < k and run:
                    run = False
                if run:
                    best = max(best, tt - t_start)
            if best >= cfg.count_stable_s:
                n_sim = k
                break
        # (b) appearance clusters (cannot-link: tracks seen at the same time)
        clusters = self._cluster_tracks()
        n_clu = sum(1 for c in clusters if c["dur"] >= cfg.count_min_track_s)
        self.n_sim, self.n_clu = n_sim, n_clu
        if cfg.herd_size > 0:
            N = cfg.herd_size
        elif cfg.count_mode == "cluster":
            N = n_clu
        elif cfg.count_mode == "max":
            N = max(n_sim, n_clu)
        else:
            N = n_sim
        N = max(N, 1)
        self.N = N
        # initialise slots: N clusters, longest-seen first
        clusters.sort(key=lambda c: -c["dur"])
        self.slots, self.acts = [], []
        for i in range(N):
            s = Slot(i + 1)
            if i < len(clusters):
                c = clusters[i]
                s.anchors = c["embs"][-cfg.gallery_anchor:]
                last = max(c["tracks"], key=lambda k: self.tracks[k].t_last)
                s.pos, s.t_last = self.tracks[last].center, self.tracks[last].t_last
                if last in seen and t - self.tracks[last].t_last < 1e-6:
                    s.track = last
                    self.tracks[last].slot = self.tracks[last].last_slot = i
            s.update_proto()
            self.slots.append(s)
            self.acts.append(SlotActivity())
        self.phase = "track"
        self.event(t, f"Enrollment done: simultaneous {n_sim}, appearance clusters {n_clu} → N={N}"
                      f"{' (user-specified)' if cfg.herd_size > 0 else ''}")

    def _cluster_tracks(self):
        cfg = self.cfg
        items = []
        for k, tr in self.tracks.items():
            e, n = tr.mean_emb()
            dur = tr.t_last - tr.t_first
            if e is not None and n >= 2:
                items.append(dict(tracks={k}, embs=[x for _, x in tr.embs], mean=e, dur=dur))
        tol = cfg.cooc_tol_s

        def cannot(a, b):
            for x in a["tracks"]:
                for y in b["tracks"]:
                    if self.cooc.get((min(x, y), max(x, y)), 0.0) > tol:
                        return True
            return False

        while len(items) > 1:
            best, bi, bj = cfg.cluster_sim, -1, -1
            M = np.array([c["mean"] for c in items])
            S = M @ M.T
            np.fill_diagonal(S, -1)
            order = np.dstack(np.unravel_index(np.argsort(-S, axis=None), S.shape))[0]
            for i, j in order:
                if i >= j:
                    continue
                if S[i, j] < best:
                    break
                if not cannot(items[i], items[j]):
                    bi, bj = i, j
                    break
            if bi < 0:
                break
            a, b = items[bi], items[bj]
            embs = a["embs"] + b["embs"]
            items[bi] = dict(tracks=a["tracks"] | b["tracks"], embs=embs, mean=_norm(np.mean(embs, 0)),
                             dur=a["dur"] + b["dur"])
            del items[bj]
        return items

    # ------------------------------------------------------------------ 2) tracking
    def _unbind_track(self, key):
        tr = self.tracks.get(key)
        if tr is not None and tr.slot is not None:
            s = self.slots[tr.slot]
            if s.track == key:
                s.track = None
            tr.slot = None

    def _bind(self, tr: Track, si: int, t, why=""):
        s = self.slots[si]
        if s.track is not None and s.track != tr.key:
            self._unbind_track(s.track)
        self._unbind_track(tr.key)
        if si < len(self.acts):              # restart position history when switching to another track (avoids fake teleport speed)
            self.acts[si].hist.clear()
            self.acts[si].t_prev = None
        s.track = tr.key
        tr.slot = tr.last_slot = si
        tr.t_unassigned = None
        if why:
            self.event(t, f"ID {s.sid}: {why}")

    def _allowed_px(self, s: Slot, t, y):
        cfg = self.cfg
        gap = max(t - s.t_last, 0.0)
        return (cfg.id_radius_bl + cfg.id_speed_bl_s * gap) * body_px(y, self.persp)

    def _cost(self, tr: Track, s: Slot, t, emb):
        cfg = self.cfg
        cx, cy = tr.center
        if s.pos is None:
            m = 0.5
        else:
            d = np.hypot(cx - s.pos[0], cy - s.pos[1])
            m = d / self._allowed_px(s, t, cy)
            if m > cfg.id_gate:
                return INF
            m = min(m, 1.0)
        app = 0.5 if (emb is None or s.proto is None) else (1 - float(emb @ s.proto)) / 1.0
        w = cfg.id_app_weight if emb is not None and s.proto is not None else 0.0
        return w * app + (1 - w) * m

    def _track_step(self, t, dets, seen, dt):
        cfg = self.cfg
        # (1) unseen tracks → release slot (slot remembers last position)
        for s in self.slots:
            if s.track is not None:
                tr = self.tracks.get(s.track)
                if tr is None or t - tr.t_last > cfg.release_s:
                    if tr is not None:
                        tr.slot = None
                    s.track = None
        # (2) same track returned and its previous slot is free → restore it
        for k in seen:
            tr = self.tracks[k]
            if tr.slot is None and tr.last_slot is not None and tr.last_slot < len(self.slots) \
                    and self.slots[tr.last_slot].track is None:
                s = self.slots[tr.last_slot]
                if s.pos is None or np.hypot(tr.center[0] - s.pos[0], tr.center[1] - s.pos[1]) \
                        <= cfg.id_gate * self._allowed_px(s, t, tr.center[1]):
                    self._bind(tr, tr.last_slot, t)
        # (3) track pairs that overlapped then separated → swap check
        self._check_swaps(t, seen)
        # (4) drift: if a bound track's recent appearance is much closer to another 'missing' slot, move it
        self._check_drift(t, seen)
        # (5) waiting tracks → assign to free slots (Hungarian)
        ready = []
        for k in seen:
            tr = self.tracks[k]
            if tr.slot is not None or tr.conf < cfg.count_conf * 0.8:
                continue
            e, n = tr.mean_emb()
            age = t - tr.t_first
            if (age >= cfg.confirm_s and n >= 2) or age >= 3 * cfg.confirm_s:
                ready.append((tr, e))
        free = [i for i, s in enumerate(self.slots) if s.track is None]
        if ready and free:
            Cm = np.full((len(ready), len(free)), INF)
            for a, (tr, e) in enumerate(ready):
                for b, si in enumerate(free):
                    Cm[a, b] = self._cost(tr, self.slots[si], t, e)
            Cf = np.where(Cm >= INF, 1e6, Cm)
            r, c = linear_sum_assignment(Cf)
            for a, b in zip(r, c):
                if Cm[a, b] < INF and Cm[a, b] <= cfg.id_max_cost:
                    tr, si = ready[a][0], free[b]
                    gap = t - self.slots[si].t_last
                    self._bind(tr, si, t, f"re-appeared (after {gap:.1f}s, cost {Cm[a, b]:.2f})" if gap > 3 else "")
        # (6) confident tracks that fit no slot → auto-increase N
        for tr, _e in ready:
            if tr.slot is None and tr.t_unassigned is None:
                tr.t_unassigned = t
        if cfg.auto_grow and cfg.herd_size <= 0:
            extras = [tr for tr, _ in ready if tr.slot is None and tr.t_unassigned is not None
                      and t - tr.t_unassigned >= cfg.grow_s and tr.conf >= cfg.count_conf
                      and not any(t - v < 1.0 for v in tr.overlap.values())]
            if extras and all(s.track is not None for s in self.slots):
                tr = max(extras, key=lambda x: t - x.t_first)
                s = Slot(len(self.slots) + 1)
                self.slots.append(s)
                self.acts.append(SlotActivity())
                self.N = len(self.slots)
                self._bind(tr, len(self.slots) - 1, t, f"new animal → herd size increased to N={self.N}")
        # (7) update bound slots: position, gallery, activity
        motion = {d[0]: (d[3], d[4]) for d in dets}
        for si, s in enumerate(self.slots):
            if s.track is None or s.track not in seen:
                continue
            tr = self.tracks[s.track]
            cx, cy = tr.center
            s.pos, s.t_last = (cx, cy), t
            occ, mot = motion[s.track]
            if tr.embs and tr.embs[-1][0] == t and occ < 0.3 and tr.conf >= cfg.count_conf \
                    and not any(t - v < cfg.swap_post_s for v in tr.overlap.values()):
                s.add(tr.embs[-1][1])
            self.acts[si].update(t, cx, cy, mot, self.persp, cfg)

    def _check_swaps(self, t, seen):
        cfg = self.cfg
        done = set()
        for k in seen:
            a = self.tracks[k]
            for o, t_ov in list(a.overlap.items()):
                if (o, k) in done or o not in self.tracks:
                    continue
                b = self.tracks[o]
                if t - t_ov < cfg.swap_post_s:
                    continue                      # separated only recently → collect more appearance
                del a.overlap[o]
                b.overlap.pop(k, None)
                done.add((k, o))
                if a.slot is None or b.slot is None:
                    continue
                ea, na = a.mean_emb(t_ov)
                eb, nb = b.mean_emb(t_ov)
                sa, sb = self.slots[a.slot], self.slots[b.slot]
                if na < 2 or nb < 2 or sa.proto is None or sb.proto is None:
                    continue
                keep = float(ea @ sa.proto + eb @ sb.proto)
                swap = float(ea @ sb.proto + eb @ sa.proto)
                if swap - keep > cfg.swap_margin:
                    ia, ib = a.slot, b.slot
                    self._unbind_track(a.key); self._unbind_track(b.key)
                    self._bind(a, ib, t); self._bind(b, ia, t)
                    self.event(t, f"ID {sa.sid} ↔ ID {sb.sid} swap corrected (appearance {keep:.2f}→{swap:.2f})")

    def _check_drift(self, t, seen):
        cfg = self.cfg
        free = [i for i, s in enumerate(self.slots) if s.track is None and s.proto is not None]
        if not free:
            return
        for k in seen:
            tr = self.tracks[k]
            if tr.slot is None or t - tr.t_check < cfg.drift_every_s:
                continue
            tr.t_check = t
            e, n = tr.mean_emb(t - cfg.drift_window_s)
            own = self.slots[tr.slot]
            if n < 4 or own.proto is None:
                continue
            s_own = float(e @ own.proto)
            best, bs = None, s_own + cfg.drift_margin
            for si in free:
                s = self.slots[si]
                if self._cost(tr, s, t, None) >= INF:
                    continue
                v = float(e @ s.proto)
                if v > bs:
                    best, bs = si, v
            if best is not None:
                old = own.sid
                self._bind(tr, best, t, f"ID {old} → ID {self.slots[best].sid} appearance correction ({s_own:.2f}→{bs:.2f})")
                free = [i for i, s in enumerate(self.slots) if s.track is None and s.proto is not None]

    # ------------------------------------------------------------------ queries
    def visible_ids(self):
        return [s.sid for s in self.slots if s.track is not None]


def _iou_matrix(B):
    x1 = np.maximum(B[:, None, 0], B[None, :, 0]); y1 = np.maximum(B[:, None, 1], B[None, :, 1])
    x2 = np.minimum(B[:, None, 2], B[None, :, 2]); y2 = np.minimum(B[:, None, 3], B[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    iou = inter / (area[:, None] + area[None, :] - inter + 1e-6)
    np.fill_diagonal(iou, 0)
    return iou
