"""검출 영역(ROI) 처리: 축사 밖·흐린 배경에서 생기는 오검출 박스 제거.

카메라 화면에는 관심 축사 말고도 울타리 너머 배경, 옆 칸, 흐린 먼 영역이 같이 찍힌다.
그곳에서 생기는 작은 박스는 마릿수를 부풀리고 ID 를 빼앗는다. 두 가지로 거른다.

  1) 영역(zone) — 화면 비율 좌표(0~1)의 다각형.
       include : 이 안에 발 위치가 있는 박스만 사용 (비워 두면 화면 전체)
       exclude : 이 안에 발 위치가 있는 박스는 버림 (여러 개 가능)
     박스의 '발 위치' = 하단 중앙 (소가 실제로 서 있는 바닥 지점 → 원근에 덜 민감)
  2) 크기 비율 — 박스 높이 / 그 위치에서 기대되는 소 체고(원근 모델 h(y)) 가 min_size_ratio 미만이면 버림
     (0 = 끔. 가려진 진짜 소도 작게 잡히므로 영역 지정이 우선, 크기 필터는 보조)

영역 파일: configs/zones/<영상이름>.json  (없으면 configs/zones/default.json, 둘 다 없으면 필터 없음)
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
    """영역 파일 이름: 영상 파일은 파일 이름, 스트림은 camera_<호스트>, 웹캠은 webcam<번호>."""
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


# --------------------------------------------------------------------------- 고정 오검출 자동 탐지
class StaticDetector:
    """같은 자리에 오래 '움직이지 않고' 남아 있는 작은 박스 = 배경 물체 오검출 후보.

    소는 오래 누워 있어도 결국 일어나거나 자세를 바꾸지만, 구유·기둥·사료통 같은 물체는
    영상 내내 같은 좌표에 같은 크기로 검출된다. 박스 좌표를 grid px 단위로 묶어 누적 시간을 세고,
      누적 ≥ min_s 초  그리고  크기비(박스 높이 / 그 위치의 기대 체고) < max_ratio
    인 위치를 후보로 낸다.  (누운 소를 지우지 않도록 '작다' 는 조건을 함께 쓴다)
    후보는 발 위치 주변의 작은 사각형 제외 영역으로 바꿔 영역 파일에 추가할 수 있다.
    """

    def __init__(self, grid: int = 6, min_s: float = 120.0, max_ratio: float = 0.5):
        self.grid, self.min_s, self.max_ratio = grid, min_s, max_ratio
        self.acc = {}       # key → [누적초, 박스]

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
            if any(_iou(b, c["box"]) > 0.5 for c in out):     # 인접 격자 중복 제거
                continue
            out.append(dict(box=b, seconds=round(sec, 1), ratio=round(ratio, 2), rect=foot_rect(b, W, H)))
        return out


def foot_rect(box, W, H):
    """박스 발 위치(하단 중앙) 주변의 작은 사각형 (정규화 다각형)."""
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
