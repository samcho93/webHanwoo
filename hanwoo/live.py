"""Live engine: input (file / RTSP / webcam) → YOLO + BoT-SORT → ReID → OnlineHerd → display/output.

Used by both the GUI (live_app.py) and headless runs.
All parameters live in a single LiveConfig and can be changed at runtime with engine.update(dict).
  · applied immediately : detection conf/iou/imgsz, ID assignment cost/gate, activity thresholds …
  · tracker restart     : BoT-SORT settings (IDs are kept by the animal slots, so they do not break)
  · recount            : enrollment-related (enroll_s, count_*, herd_size) → engine.recount()
"""
from __future__ import annotations

import csv
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from .config import MODELS_DIR, OUTPUT_DIR, ROOT, STATE_NAMES
from .online import OnlineHerd, _iou_matrix
from .render import STATE_COLORS, _draw_panel, _label, id_color
from .zones import (ZONES_DIR, StaticDetector, ZoneFilter, add_exclusions, load_zones, size_ok, zone_stem,
                    zones_path_for)

TRACKER_KEYS = ("track_high_thresh", "track_low_thresh", "new_track_thresh", "track_buffer", "match_thresh")
RECOUNT_KEYS = ("enroll_s", "count_conf", "count_stable_s", "count_min_track_s", "count_mode", "herd_size",
                "cluster_sim", "cooc_tol_s")


@dataclass
class LiveConfig:
    # Input
    source: str = r"E:\Video\cow\cow09.mp4"   # file path / rtsp://... / webcam index (0)
    rtsp_tcp: bool = True                     # RTSP over TCP (robust to packet loss)
    realtime: bool = True                     # play files at original speed (off = max speed)
    loop_file: bool = False
    # Detection
    weights: str = str(MODELS_DIR / "hanwoo_yolo11s.pt")
    imgsz: int = 960
    conf: float = 0.25
    iou: float = 0.6
    device: str = "0"
    # Tracker (BoT-SORT) — change restarts the tracker
    track_high_thresh: float = 0.35
    track_low_thresh: float = 0.10
    new_track_thresh: float = 0.45
    track_buffer: int = 120
    match_thresh: float = 0.85
    # ReID
    reid_weights: str = str(MODELS_DIR / "hanwoo_reid.pt")
    reid_every: int = 4
    # Enrollment (herd count) — change requires 'Recount'
    enroll_s: float = 60.0                    # initial seconds spent counting the herd and enrolling appearances
    count_conf: float = 0.4                   # min confidence to be included in the count
    count_stable_s: float = 2.0               # simultaneous detection count must hold at least this long
    count_min_track_s: float = 2.0            # min track length for an appearance cluster to count
    count_mode: str = "max"                   # sim (simultaneous) / cluster (appearance clusters) / max (larger of the two)
    herd_size: int = 0                        # 0 = auto, >0 = user-specified herd size
    cluster_sim: float = 0.60                 # merge tracks into one animal only above this similarity and if never seen together
    cooc_tol_s: float = 0.5
    # ID assignment (applied immediately)
    confirm_s: float = 1.0                    # appearance collection time before a new track's ID is confirmed
    release_s: float = 1.0                    # slot becomes 'missing' after being unseen this long
    id_app_weight: float = 0.6
    id_radius_bl: float = 2.0
    id_speed_bl_s: float = 0.5
    id_gate: float = 1.5
    id_max_cost: float = 0.75
    gallery_anchor: int = 40
    swap_iou: float = 0.2
    swap_post_s: float = 1.5
    swap_margin: float = 0.10
    drift_every_s: float = 2.0
    drift_window_s: float = 3.0
    drift_margin: float = 0.25
    auto_grow: bool = True
    grow_s: float = 5.0
    # Activity (applied immediately)
    cow_height_m: float = 1.30
    smooth_window_s: float = 1.0
    rest_speed: float = 0.05
    walk_speed: float = 0.20
    motion_energy_active: float = 2.5
    motion_norm: float = 5.0
    max_speed: float = 2.0
    interp_max_gap_s: float = 0.5
    # Display / output
    trail_s: float = 3.0
    show_pending: bool = True
    show_zones: bool = True
    # Detection zones (remove false positives outside the pen / in blurred background)
    zones_file: str = ""                      # empty = configs/zones/<video name>.json → default.json automatically
    min_size_ratio: float = 0.0               # min box height / expected body height (0=off, 0.3–0.5 recommended)
    static_min_s: float = 120.0               # small box fixed in place at least this long = background-object false-positive candidate
    static_max_ratio: float = 0.5             # max candidate size ratio (protects lying cattle)
    static_auto: bool = False                 # add candidates to the exclude zones without confirmation
    panel_width: int = 380
    save_video: bool = False
    save_csv: bool = True
    out_root: str = str(OUTPUT_DIR / "live")

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def is_stream(src: str) -> bool:
    s = str(src).strip().lower()
    return s.isdigit() or s.startswith(("rtsp://", "rtsps://", "http://", "https://", "udp://", "tcp://", "rtmp://"))


# --------------------------------------------------------------------------- Input
class FrameSource:
    """File: every frame in order (at original speed if realtime).
    Stream: a separate thread reads continuously and hands over only the 'latest frame' (no lag build-up when processing is slow); reconnects on disconnect."""

    def __init__(self, cfg: LiveConfig, log=print):
        self.cfg, self.log = cfg, log
        self.src = str(cfg.source).strip()
        self.stream = is_stream(self.src)
        self.cap = None
        self.fps = 24.0
        self.n_frames = 0
        self.idx = 0
        self.dropped = 0
        self._latest = None
        self._stop = threading.Event()
        self._cv = threading.Condition()
        self.connected = False
        self._open()
        if self.stream:
            self._th = threading.Thread(target=self._reader, daemon=True)
            self._th.start()
        self.t_start = time.time()

    def _open(self):
        if self.stream and self.src.lower().startswith("rtsp"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp" if self.cfg.rtsp_tcp else ""
        src = int(self.src) if self.src.isdigit() else self.src
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG) if not isinstance(src, int) else cv2.VideoCapture(src)
        if not self.cap.isOpened():
            self.connected = False
            if not self.stream:
                raise IOError(f"Cannot open video: {self.src}")
            return False
        f = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = f if f and 1 <= f <= 120 else 24.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.stream else 0
        self.connected = True
        return True

    def _reader(self):
        backoff = 1.0
        while not self._stop.is_set():
            if not self.connected:
                self.log(f"Connecting to stream: {self.src}")
                if not self._open():
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 5)
                    continue
                self.log("Stream connected")
                backoff = 1.0
            ok, fr = self.cap.read()
            if not ok:
                self.log("Stream disconnected → reconnecting")
                self.connected = False
                self.cap.release()
                continue
            with self._cv:
                if self._latest is not None:
                    self.dropped += 1
                self.idx += 1
                self._latest = (self.idx, time.time() - self.t_start, fr)
                self._cv.notify()

    def read(self, timeout=5.0):
        """Returns (frame_no, t_seconds, frame) or None (end)."""
        if self.stream:
            with self._cv:
                if self._latest is None:
                    self._cv.wait(timeout)
                item, self._latest = self._latest, None
            return item if item is not None else ("wait", None, None)
        ok, fr = self.cap.read()
        if not ok:
            if self.cfg.loop_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, fr = self.cap.read()
            if not ok:
                return None
        self.idx += 1
        return self.idx - 1, (self.idx - 1) / self.fps, fr

    def close(self):
        self._stop.set()
        if self.cap is not None:
            self.cap.release()


# --------------------------------------------------------------------------- Engine
class LiveEngine:
    def __init__(self, cfg: LiveConfig, log=print):
        self.cfg = cfg
        self.log = log
        self._pending = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._recount = False
        self.latest = None          # (canvas_bgr, status dict)
        self.status = {}
        self.thread = None
        self.error = None
        self._yolo = None
        self._reid = None
        self._gen = 0
        self._zf = None
        self.raw_frame = None
        self.n_filtered = 0
        self.static_candidates = []
        self._apply_static = False

    # ---------------- Control
    def start(self):
        self._stop.clear()
        self.thread = threading.Thread(target=self._run_safe, daemon=True)
        self.thread.start()

    def stop(self):
        self._stop.set()
        if self.thread:
            self.thread.join(timeout=10)

    def pause(self, on: bool):
        (self._pause.set if on else self._pause.clear)()

    def recount(self):
        self._recount = True

    def update(self, changes: dict):
        with self._lock:
            self._pending.update(changes)

    def running(self):
        return self.thread is not None and self.thread.is_alive()

    # ---------------- Models
    def _load_models(self):
        from ultralytics import YOLO

        from .reid import load_reid

        cfg = self.cfg
        w = cfg.weights if Path(cfg.weights).exists() else "yolo11x.pt"
        self._yolo = YOLO(w)
        self._classes = None if len(self._yolo.names) == 1 else [17, 19]
        self.log(f"Detection model: {Path(w).name}")
        dev = "cuda" if cfg.device != "cpu" else "cpu"
        try:
            import torch
            if not torch.cuda.is_available():
                dev = "cpu"
        except Exception:
            dev = "cpu"
        self._dev = dev
        if Path(cfg.reid_weights).exists():
            self._reid = load_reid(cfg.reid_weights, device=dev)
            self.log(f"ReID model: {Path(cfg.reid_weights).name} ({dev})")
        else:
            self._reid = None
            self.log("No ReID model → IDs assigned from position/motion only (lower accuracy)")
        self._write_tracker_yaml()

    def _write_tracker_yaml(self):
        # Read the base config file (key: value format) and override only the tracker parameters
        lines = []
        for ln in (ROOT / "configs" / "hanwoo_botsort.yaml").read_text(encoding="utf-8").splitlines():
            k = ln.split(":", 1)[0].strip()
            if k in TRACKER_KEYS:
                ln = f"{k}: {getattr(self.cfg, k)}"
            lines.append(ln)
        p = OUTPUT_DIR / "live" / "_tracker_live.yaml"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._tracker_yaml = str(p)

    def _reset_tracker(self):
        self._write_tracker_yaml()
        self._yolo.predictor = None          # new tracker is created on the next track() call
        self._gen += 1                       # avoid track key collisions (key = gen*1e6 + id)
        self.log("Tracker restarted (animal IDs are kept by their slots)")

    # ---------------- Run
    def _run_safe(self):
        try:
            self._run()
        except Exception as e:  # shown in the GUI
            import traceback
            self.error = f"{e}\n{traceback.format_exc()}"
            self.log(f"[Error] {e}")

    def _apply_pending(self, herd: OnlineHerd):
        with self._lock:
            ch, self._pending = self._pending, {}
        if not ch:
            return
        tracker = False
        for k, v in ch.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)
                tracker |= k in TRACKER_KEYS
        if tracker:
            self._reset_tracker()
        if "static_apply" in ch:
            self._apply_static = True
        if "zones_file" in ch or "zones_reload" in ch:
            self._zf = None                  # reload zones on the next frame
        if "herd_size" in ch and herd.phase == "track" and self.cfg.herd_size > 0 \
                and self.cfg.herd_size != len(herd.slots):
            self._recount = True

    def zones_path(self):
        return Path(self.cfg.zones_file) if self.cfg.zones_file else zones_path_for(self.cfg.source)

    def _load_zones(self, W, H):
        p = self.zones_path()
        self._zf = ZoneFilter(load_zones(p), W, H)
        self._zf_size = (W, H)
        if self._zf.active:
            self.log(f"Detection zones applied: {p.name} (include {len(self._zf.inc)}, exclude {len(self._zf.exc)})")

    def _static_step(self, herd, boxes, dt, t, W, H):
        """Detect background-object false positives (small boxes fixed in place for long) → notify candidates / add exclude zones."""
        cfg = self.cfg
        st = self._static
        st.min_s, st.max_ratio = cfg.static_min_s, cfg.static_max_ratio
        st.add(boxes, dt)
        if herd.phase != "track" or (t - self._static_t < 10 and not self._apply_static):
            return
        self._static_t = t
        cands = [c for c in st.candidates(herd.persp, W, H) if self._zf.keep(c["box"])]   # only those not yet excluded
        for c in cands:
            k = tuple(round(v) for v in c["box"])
            if k not in self._static_seen:
                self._static_seen.add(k)
                herd.event(t, f"Static false-positive candidate: box {k} fixed for {c['seconds']:.0f}s, size ratio {c['ratio']}"
                              + ("" if cfg.static_auto else " → apply with [Exclude static false positives]"))
        self.static_candidates = cands
        if cands and (cfg.static_auto or self._apply_static):
            path = self.zones_path() or (ZONES_DIR / f"{zone_stem(cfg.source)}.json")
            add_exclusions(path, [c["rect"] for c in cands])
            if not cfg.zones_file and zones_path_for(cfg.source) != path:
                cfg.zones_file = str(path)
            self._zf = None
            self.static_candidates = []
            herd.event(t, f"Added {len(cands)} static false positives to the exclude zones ({path.name})")
        self._apply_static = False

    def _run(self):
        from .reid import CROP, crop_box, embed_crops

        self._zf, self._zf_size, self.n_filtered = None, None, 0
        self._static = StaticDetector(min_s=self.cfg.static_min_s, max_ratio=self.cfg.static_max_ratio)
        self._static_t, self._static_seen, self.static_candidates = 0.0, set(), []
        cfg = self.cfg
        self.log(f"Opening input: {cfg.source}")
        src = FrameSource(cfg, log=self.log)
        if self._yolo is None:
            self._load_models()
        herd = OnlineHerd(cfg, log=self.log)
        name = Path(cfg.source).stem if not src.stream else "stream"
        out_dir = Path(cfg.out_root) / f"{name}_{datetime.now():%Y%m%d_%H%M%S}"
        writer = csvf = None
        if cfg.save_csv or cfg.save_video:
            out_dir.mkdir(parents=True, exist_ok=True)
            cfg.save(out_dir / "config.json")
        if cfg.save_csv:
            csvf = open(out_dir / "tracks_live.csv", "w", newline="", encoding="utf-8")
            writer = csv.writer(csvf)
            writer.writerow(["frame", "t", "id", "x1", "y1", "x2", "y2", "conf", "state", "speed", "dist_m", "activity"])
        self.out_dir = out_dir
        vw = None
        trails = {}
        prev_gray = None
        t_prev = None
        fps_ema = None
        t_wall0 = time.time()
        n_proc = 0
        last_frame_no = -1
        while not self._stop.is_set():
            if self._pause.is_set():
                time.sleep(0.05)
                continue
            self._apply_pending(herd)
            if self._recount:
                self._recount = False
                herd.reset()
                trails.clear()
                self.log("Recounting herd (restarting enrollment)")
            item = src.read()
            if item is None:
                self.log("End of video")
                break
            fno, t, frame = item
            if fno == "wait":
                self.status.update(connected=src.connected)
                continue
            tic = time.time()
            H, W = frame.shape[:2]
            self.raw_frame = frame
            # 1) detection + tracking
            res = self._yolo.track(frame, persist=True, tracker=self._tracker_yaml, imgsz=cfg.imgsz, conf=cfg.conf,
                                   iou=cfg.iou, classes=self._classes, agnostic_nms=True,
                                   device=cfg.device if self._dev == "cuda" else "cpu", verbose=False)[0]
            gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            diff = cv2.absdiff(gray, prev_gray) if prev_gray is not None and prev_gray.shape == gray.shape else None
            prev_gray = gray
            dets = []
            if res.boxes is not None and res.boxes.id is not None:
                xyxy = res.boxes.xyxy.cpu().numpy()
                ids = res.boxes.id.cpu().numpy().astype(int)
                confs = res.boxes.conf.cpu().numpy()
                self.last_boxes = xyxy.copy()       # reference for the zone editor (before filtering)
                # detection zone (ROI) / size filter: remove false positives outside the pen / in blurred background
                if self._zf is None or self._zf_size != (W, H):
                    self._load_zones(W, H)
                keep = np.array([self._zf.keep(b) and (herd.phase != "track" or
                                                       size_ok(b, herd.persp, cfg.min_size_ratio)) for b in xyxy], bool)
                self.n_filtered += int((~keep).sum())
                xyxy, ids, confs = xyxy[keep], ids[keep], confs[keep]
                occ = _iou_matrix(xyxy).max(1) if len(xyxy) > 1 else np.zeros(len(xyxy))
                for b, i, c, o in zip(xyxy, ids, confs, occ):
                    x1, y1, x2, y2 = np.clip(b, 0, [W, H, W, H]).astype(int)
                    mot = float(diff[y1:y2, x1:x2].mean()) if diff is not None and x2 > x1 and y2 > y1 else 0.0
                    dets.append((self._gen * 1_000_000 + int(i), tuple(map(float, b)), float(c), float(o), mot))
            # 2) ReID (every reid_every frames, only for lightly occluded boxes)
            embs = {}
            if self._reid is not None and n_proc % max(1, cfg.reid_every) == 0:
                sel = [d for d in dets if d[3] < 0.3 and d[2] >= 0.3]
                if sel:
                    crops = np.stack([crop_box(frame, d[1], CROP) for d in sel])
                    E = embed_crops(self._reid, crops, device=self._dev)
                    embs = {d[0]: e for d, e in zip(sel, E)}
            dt = 0.0 if t_prev is None else t - t_prev
            t_prev = t
            self._static_step(herd, [d[1] for d in dets], dt, t, W, H)
            # 3) herd count / IDs
            out = herd.step(t, dets, embs, dt)
            # 4) drawing + output
            canvas, rows = self._draw(frame, out, herd, trails, t, src, fps_ema)
            if writer is not None:
                for r in rows:
                    writer.writerow([fno, f"{t:.3f}", r.tid, *[f"{v:.1f}" for v in r.box], f"{r.conf:.3f}",
                                     STATE_NAMES[r.state], f"{r.speed:.3f}", f"{r.dist_cum:.2f}", f"{r.activity:.1f}"])
            if cfg.save_video:
                if vw is None:
                    vw = _VideoWriter(out_dir / f"{name}_live.mp4", canvas.shape[1], canvas.shape[0], src.fps)
                vw.write(canvas)
            n_proc += 1
            proc = time.time() - tic
            fps_ema = 1 / max(proc, 1e-3) if fps_ema is None else 0.9 * fps_ema + 0.1 / max(proc, 1e-3)
            self.status = self._status(herd, src, fps_ema, t, fno, rows)
            self.latest = (canvas, self.status)
            last_frame_no = fno
            # match original speed for files in realtime mode
            if not src.stream and cfg.realtime:
                lag = t - (time.time() - t_wall0)
                if lag > 0:
                    time.sleep(min(lag, 0.5))
                elif lag < -1.0:
                    t_wall0 = time.time() - t
        src.close()
        if vw is not None:
            vw.close()
        if csvf is not None:
            csvf.close()
            self._write_summary(herd, out_dir)
        self.status["finished"] = True
        self.status["frames"] = last_frame_no + 1
        self.status["wall_s"] = time.time() - t_wall0
        self.log(f"Finished. Results: {out_dir}" if (cfg.save_csv or cfg.save_video) else "Finished")

    # ---------------- Display
    def _draw(self, frame, out, herd: OnlineHerd, trails, t, src, fps_ema):
        cfg = self.cfg
        img = self._zf.draw(frame.copy()) if cfg.show_zones and self._zf is not None else frame.copy()
        rows = []
        for key, box, conf, si, st in out:
            x1, y1, x2, y2 = map(int, box)
            if st == "enroll":
                cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 1, cv2.LINE_AA)
                if conf >= cfg.count_conf:
                    cv2.circle(img, (x1 + 6, y1 + 6), 4, (0, 220, 255), -1)
                continue
            if si is None:
                if cfg.show_pending:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (120, 120, 120), 1, cv2.LINE_4)
                    _label(img, "?", (x1, max(y1, 14)), (230, 230, 230), 0.45, 1, (90, 90, 90))
                continue
            s = herd.slots[si]
            a = herd.acts[si]
            col = id_color(s.sid)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            owner, tr = trails.get(s.sid, (None, []))
            if owner != key or (tr and t - tr[-1][0] > cfg.interp_max_gap_s):
                tr = []                      # restart the trail if switched to another track or unseen for long
            trails[s.sid] = (key, tr)
            tr.append((t, cx, cy))
            while tr and t - tr[0][0] > cfg.trail_s:
                tr.pop(0)
            for k in range(1, len(tr)):
                cv2.line(img, tr[k - 1][1:], tr[k][1:], col, 2, cv2.LINE_AA)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
            _label(img, f"ID {s.sid}", (x1, max(y1, 16)), (255, 255, 255), 0.55, 1, col)
            cv2.circle(img, (x2 - 7, y1 + 8), 5, STATE_COLORS[a.state], -1, cv2.LINE_AA)
            rows.append(SimpleNamespace(tid=s.sid, box=box, conf=conf, state=a.state, speed=a.speed,
                                        dist_cum=a.dist, activity=a.activity))
        # top banner
        if herd.phase == "enroll":
            n_now = sum(1 for o in out if o[2] >= cfg.count_conf)
            txt = f"COUNTING  {t - (herd.t0 or t):4.0f}/{cfg.enroll_s:.0f}s   now {n_now}"
        else:
            txt = f"N={len(herd.slots)}  visible {len(rows)}  missing {len(herd.slots) - len(herd.visible_ids())}"
        _label(img, txt, (8, 24), (255, 255, 255), 0.6, 1, (40, 40, 40))
        H, W = img.shape[:2]
        OH = max(H, 480)
        canvas = np.zeros((OH, W + cfg.panel_width, 3), np.uint8)
        canvas[:H, :W] = img
        title = Path(cfg.source).stem if not src.stream else "LIVE"
        _draw_panel(canvas, W, cfg.panel_width, OH, int(round(t * src.fps)), src.fps, rows, title)
        return canvas, rows

    def _status(self, herd: OnlineHerd, src, fps_ema, t, fno, rows):
        ids = []
        vis = {r.tid: r for r in rows}
        for si, s in enumerate(herd.slots):
            a = herd.acts[si]
            ids.append(dict(id=s.sid, visible=s.sid in vis, state=a.state, speed=a.speed, dist=a.dist,
                            activity=a.activity, missing_s=0.0 if s.sid in vis else max(0.0, t - s.t_last),
                            mean_activity=a.act_sum / a.t_obs if a.t_obs > 0 else 0.0))
        return dict(phase=herd.phase, t=t, frame=fno, enroll_left=max(0.0, self.cfg.enroll_s - (t - (herd.t0 or t))),
                    n_sim=herd.n_sim, n_clu=herd.n_clu, N=len(herd.slots), visible=len(rows), fps=fps_ema or 0.0,
                    src_fps=src.fps, n_frames=src.n_frames, stream=src.stream, connected=src.connected,
                    dropped=src.dropped, filtered=getattr(self, "n_filtered", 0),
                    static=len(self.static_candidates), ids=ids, events=list(herd.events)[-50:])

    def _write_summary(self, herd: OnlineHerd, out_dir: Path):
        rows = []
        for si, s in enumerate(herd.slots):
            a = herd.acts[si]
            T = max(a.t_obs, 1e-6)
            rows.append(dict(id=s.sid, observed_s=round(a.t_obs, 1), distance_m=round(a.dist, 2),
                             activity_index=round(a.act_sum / T, 1),
                             rest_pct=round(100 * a.state_time[0] / T, 1), active_pct=round(100 * a.state_time[1] / T, 1),
                             walk_pct=round(100 * a.state_time[2] / T, 1)))
        with open(out_dir / "summary_live.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["id"])
            w.writeheader()
            w.writerows(rows)
        (out_dir / "events.log").write_text("\n".join(f"{t:.1f}\t{m}" for t, m in herd.events), encoding="utf-8")
        (out_dir / "count.json").write_text(json.dumps(dict(n_sim=herd.n_sim, n_clu=herd.n_clu, N=len(herd.slots)),
                                                       ensure_ascii=False), encoding="utf-8")


class _VideoWriter:
    """H.264 if ffmpeg is available, otherwise OpenCV mp4v."""

    def __init__(self, path, w, h, fps):
        self.proc = self.vw = None
        if shutil.which("ffmpeg"):
            cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                   "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            self.vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, img):
        if self.proc:
            self.proc.stdin.write(img.tobytes())
        else:
            self.vw.write(img)

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.vw.release()
