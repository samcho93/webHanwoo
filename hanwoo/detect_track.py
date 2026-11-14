"""1단계: 검출(YOLO) + 온라인 추적(BoT-SORT) + 프레임별 특징 수집.

결과는 pandas DataFrame 한 장(프레임 × 객체)으로 반환한다.
  frame, tid, x1,y1,x2,y2, conf, motion  + 외형 히스토그램(별도 dict)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import PipelineConfig

# COCO 사전학습 모델 사용 시 '소'로 간주할 클래스 (소가 말로 오분류되는 경우 보완, sheep 은 곤포 오검출로 제외)
COCO_ANIMAL_CLASSES = [17, 19]


def load_model(cfg: PipelineConfig):
    from ultralytics import YOLO

    w = cfg.weights if Path(cfg.weights).exists() else cfg.fallback_weights
    model = YOLO(w)
    # 클래스 1개(hanwoo) 학습 모델이면 전체 사용, COCO(80 클래스) 모델이면 소/말만 사용
    classes = None if len(model.names) == 1 else COCO_ANIMAL_CLASSES
    print(f"[detector] {w}  ({'fine-tuned hanwoo' if classes is None else 'COCO pretrained'})")
    return model, classes


def appearance_hist(frame_hsv: np.ndarray, box) -> np.ndarray:
    """bbox 중앙부(펜스·배경 영향 최소화)의 H-S 2D 히스토그램 (외형 특징)."""
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
