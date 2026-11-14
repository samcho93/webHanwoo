"""Detection zone (ROI) handling: remove false-positive boxes from outside the pen / blurred background.

Besides the pen of interest, the camera view also shows background beyond the fence, neighbouring pens and blurred distant areas.
Small boxes from those areas inflate the herd count and steal IDs. They are filtered in two ways.

  1) Zones — polygons in normalised screen coordinates (0–1).
       include : only boxes whose foot position is inside are used (empty = whole frame)
       exclude : boxes whose foot position is inside are dropped (multiple allowed)
     A box's 'foot position' = bottom centre (the ground point the animal stands on → less sensitive to perspective)
  2) Size ratio — drop the box if box height / expected body height at that position (perspective model h(y)) < min_size_ratio
     (0 = off. Occluded real cattle also appear small, so zones come first and the size filter is secondary)

Zone file: configs/zones/<video name>.json  (falls back to configs/zones/default.json; no filter if neither exists)
  {"include": [[[x,y],...]], "exclude": [[[x,y],...], ...]}
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .config import ROOT

ZONES_DIR = ROOT / "configs" / "zones"


def zone_stem(source) -> str:
    """Zone file name: file name for video files, camera_<host> for streams, webcam<index> for webcams."""
    s = str(source).strip()
    if s.isdigit():
        return f"webcam{s}"
    if "://" in s:
        host = s.split("://", 1)[1].split("/", 1)[0].split("@")[-1].split(":")[0]
        return "camera_" + "".join(c if c.isalnum() else "_" for c in host)
    return Path(s).stem


def zones_path_for(source: str | Path) -> Path | None:
    stem = zone_stem(source)
    for p in (ZONES_DIR / f"{stem}.json", ZONES_DIR / "default.json"):
        if p.exists():
            return p
    return None


def load_zones(path) -> dict:
    if not path or not Path(path).exists():
        return dict(include=[], exclude=[])
    z = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(include=z.get("include", []), exclude=z.get("exclude", []))


def save_zones(path, zones: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(dict(include=zones.get("include", []), exclude=zones.get("exclude", [])),
                                     indent=1), encoding="utf-8")


class ZoneFilter:
    def __init__(self, zones: dict, W: int, H: int):
        sc = np.array([W, H], np.float32)
        self.inc = [(np.array(p, np.float32) * sc).reshape(-1, 1, 2) for p in zones.get("include", []) if len(p) >= 3]
        self.exc = [(np.array(p, np.float32) * sc).reshape(-1, 1, 2) for p in zones.get("exclude", []) if len(p) >= 3]

    @property
    def active(self):
        return bool(self.inc or self.exc)

    def keep(self, box) -> bool:
        x1, y1, x2, y2 = box
        pt = (float((x1 + x2) / 2), float(y2))
        if self.inc and not any(cv2.pointPolygonTest(p, pt, False) >= 0 for p in self.inc):
            return False
        return not any(cv2.pointPolygonTest(p, pt, False) >= 0 for p in self.exc)

    def keep_df(self, df) -> np.ndarray:
        if not self.active:
            return np.ones(len(df), bool)
        return np.array([self.keep(b) for b in df[["x1", "y1", "x2", "y2"]].to_numpy()], bool)

    def draw(self, img, alpha=0.25):
        if not self.active:
            return img
        ov = img.copy()
        for p in self.exc:
            cv2.fillPoly(ov, [p.astype(np.int32)], (0, 0, 200))
        img = cv2.addWeighted(ov, alpha, img, 1 - alpha, 0)
        for p in self.exc:
            cv2.polylines(img, [p.astype(np.int32)], True, (0, 0, 255), 1, cv2.LINE_AA)
        for p in self.inc:
            cv2.polylines(img, [p.astype(np.int32)], True, (0, 220, 0), 1, cv2.LINE_AA)
        return img


def size_ok(box, persp, min_ratio: float) -> bool:
    if min_ratio <= 0:
        return True
    x1, y1, x2, y2 = box
    a, b = persp
    return (y2 - y1) >= min_ratio * max(a * (y1 + y2) / 2 + b, 20.0)


# --------------------------------------------------------------------------- automatic static false-positive detection
class StaticDetector:
    """Small boxes that stay in the same place 'without moving' for a long time = background-object false-positive candidates.

    Cattle lying down for long eventually stand up or change posture, but objects like troughs, posts and feed bins
    are detected at the same coordinates and size throughout the video. Box coordinates are binned into grid px cells and the accumulated time is counted;
      positions with accumulated ≥ min_s seconds  and  size ratio (box height / expected body height there) < max_ratio
    are reported as candidates.  (The 'small' condition is used too so lying cattle are not removed)
    Candidates can be turned into small rectangular exclude zones around the foot position and added to the zone file.
    """

    def __init__(self, grid: int = 6, min_s: float = 120.0, max_ratio: float = 0.5):
        self.grid, self.min_s, self.max_ratio = grid, min_s, max_ratio
        self.acc = {}       # key → [accumulated seconds, box]

    def add(self, boxes, dt: float):
        for b in boxes:
            k = tuple(int(v) // self.grid for v in b)
            a = self.acc.get(k)
            if a is None:
                self.acc[k] = [dt, tuple(float(v) for v in b)]
            else:
                a[0] += dt

    def candidates(self, persp, W: int, H: int):
        out = []
        for _k, (sec, b) in sorted(self.acc.items(), key=lambda kv: -kv[1][0]):
            if sec < self.min_s:
                break
            x1, y1, x2, y2 = b
            exp = max(persp[0] * (y1 + y2) / 2 + persp[1], 20.0)
            ratio = (y2 - y1) / exp
            if ratio >= self.max_ratio:
                continue
            if any(_iou(b, c["box"]) > 0.5 for c in out):     # drop duplicates from adjacent grid cells
                continue
            out.append(dict(box=b, seconds=round(sec, 1), ratio=round(ratio, 2), rect=foot_rect(b, W, H)))
        return out


def foot_rect(box, W, H):
    """Small rectangle around the box foot position (bottom centre), as a normalised polygon."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, fy = (x1 + x2) / 2, y2
    rx, ry = max(w * 0.6, 8), max(h * 0.4, 6)
    xa, xb, ya, yb = (cx - rx) / W, (cx + rx) / W, (fy - ry) / H, (fy + ry) / H
    r = lambda v: round(min(max(v, 0.0), 1.0), 4)
    return [[r(xa), r(ya)], [r(xb), r(ya)], [r(xb), r(yb)], [r(xa), r(yb)]]


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def add_exclusions(path, rects):
    z = load_zones(path)
    z["exclude"].extend(rects)
    save_zones(path, z)
    return z
