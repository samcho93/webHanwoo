"""Stage 1: detection (YOLO) + online tracking (BoT-SORT) + per-frame feature collection.

Returns a single pandas DataFrame (frame × object):
  frame, tid, x1,y1,x2,y2, conf, motion  + appearance histograms (separate dict)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import PipelineConfig

# Classes treated as 'cow' with a COCO-pretrained model (horse covers cows misclassified as horses; sheep is excluded because it fires on hay bales)
COCO_ANIMAL_CLASSES = [17, 19]


def load_model(cfg: PipelineConfig):
    from ultralytics import YOLO

    w = cfg.weights if Path(cfg.weights).exists() else cfg.fallback_weights
    model = YOLO(w)
    # single-class (hanwoo) model: use all classes; COCO (80 classes) model: cows/horses only
    classes = None if len(model.names) == 1 else COCO_ANIMAL_CLASSES
    print(f"[detector] {w}  ({'fine-tuned hanwoo' if classes is None else 'COCO pretrained'})")
    return model, classes


def appearance_hist(frame_hsv: np.ndarray, box) -> np.ndarray:
    """H-S 2D histogram of the bbox centre region (minimises fence/background influence) - appearance feature."""
    x1, y1, x2, y2 = [int(v) for v in box]
    w, h = x2 - x1, y2 - y1
    cx1, cy1 = x1 + w // 5, y1 + h // 5
    cx2, cy2 = x2 - w // 5, y2 - h // 5
    roi = frame_hsv[max(cy1, 0):max(cy2, 1), max(cx1, 0):max(cx2, 1)]
    if roi.size == 0:
        return np.zeros(16 * 8, np.float32)
    hist = cv2.calcHist([roi], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten()
    s = hist.sum()
    return (hist / s).astype(np.float32) if s > 0 else hist.astype(np.float32)


def run_detect_track(video_path: str | Path, cfg: PipelineConfig, verbose_every: int = 500):
    model, classes = load_model(cfg)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rows, feats = [], {}
    prev_gray = None
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok or (cfg.max_frames and fi >= cfg.max_frames):
            break
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        res = model.track(
            frame, persist=True, tracker=cfg.tracker_cfg, imgsz=cfg.imgsz,
            conf=cfg.conf, iou=cfg.iou, classes=classes, agnostic_nms=True,
            device=cfg.device, verbose=False,
        )[0]
        if res.boxes is not None and res.boxes.id is not None:
            xyxy = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            diff = cv2.absdiff(gray, prev_gray) if prev_gray is not None else None
            for b, t, c in zip(xyxy, ids, confs):
                x1, y1, x2, y2 = np.clip(b, 0, [W, H, W, H]).astype(int)
                motion = 0.0
                if diff is not None and x2 > x1 and y2 > y1:
                    motion = float(diff[y1:y2, x1:x2].mean())
                rows.append((fi, int(t), *map(float, b), float(c), motion))
                feats[(fi, int(t))] = appearance_hist(hsv, b)
        prev_gray = gray
        fi += 1
        if verbose_every and fi % verbose_every == 0:
            print(f"  [track] {fi}/{n_total} frames")
    cap.release()

    df = pd.DataFrame(rows, columns=["frame", "tid", "x1", "y1", "x2", "y2", "conf", "motion"])
    meta = dict(fps=fps, width=W, height=H, n_frames=fi, video=str(video_path))
    return df, feats, meta
