"""4단계: 결과 동영상 렌더링 (ID 박스 + 이동궤적 + 실시간 미니 대시보드 패널).

브라우저 재생 호환을 위해 ffmpeg 파이프로 H.264(yuv420p, faststart) 인코딩.
"""
from __future__ import annotations

import colorsys
import subprocess
from collections import defaultdict, deque

import cv2
import numpy as np
import pandas as pd

from .config import STATE_NAMES, PipelineConfig

STATE_COLORS = {0: (160, 160, 160), 1: (0, 200, 255), 2: (60, 220, 60)}  # BGR


def id_color(tid: int):
    h = (tid * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def _label(img, text, org, color, scale=0.5, thick=1, bg=(0, 0, 0)):
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + bl - 1), bg, -1)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def render_video(video_path, act: pd.DataFrame, meta: dict, out_path, cfg: PipelineConfig, title=""):
    fps, W, H = meta["fps"], meta["width"], meta["height"]
    PW = cfg.panel_width
    OW, OH = W + PW, max(H, 480)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    by_frame = {f: g for f, g in act.groupby("frame")}
    trails = defaultdict(lambda: deque(maxlen=int(cfg.trail_len_s * fps)))
    last_seen = {}
    cap = cv2.VideoCapture(str(video_path))
    fi = 0
    n = meta["n_frames"]
    while fi < n:
        ok, frame = cap.read()
        if not ok:
            break
        canvas = np.zeros((OH, OW, 3), np.uint8)
        g = by_frame.get(fi)
        visible = []
        if g is not None:
            for r in g.itertuples():
                tid = int(r.tid)
                col = id_color(tid)
                x1, y1, x2, y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
                thick = 1 if r.interp else 2
                if r.interp:  # 보간(가림) 구간은 점선 느낌으로 얇게
                    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 1, cv2.LINE_4)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), col, thick, cv2.LINE_AA)
                if not r.interp:   # 궤적은 실제 검출 위치만 (보간 위치로 경로를 그리지 않음)
                    trails[tid].append((int(r.cx), int(r.cy)))
                    last_seen[tid] = fi
                pts = list(trails[tid])
                for k in range(1, len(pts)):
                    cv2.line(frame, pts[k - 1], pts[k], col, 2, cv2.LINE_AA)
                _label(frame, f"ID {tid}", (x1, max(y1, 16)), (255, 255, 255), 0.55, 1, col)
                st = int(r.state)
                cv2.circle(frame, (x2 - 7, y1 + 8), 5, STATE_COLORS[st], -1, cv2.LINE_AA)
                visible.append(r)
        for tid in [t for t, f in last_seen.items() if fi - f > int(0.5 * fps)]:
            trails[tid].clear()
        canvas[:H, :W] = frame
        _draw_panel(canvas, W, PW, OH, fi, fps, visible, title)
        proc.stdin.write(canvas.tobytes())
        fi += 1
    cap.release()
    proc.stdin.close()
    proc.wait()


def _draw_panel(canvas, x0, PW, OH, fi, fps, visible, title):
    canvas[:, x0:] = (28, 24, 22)
    cv2.putText(canvas, "HANWOO ACTIVITY MONITOR", (x0 + 12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    t = fi / fps
    cv2.putText(canvas, f"{title}  t={int(t // 60):02d}:{t % 60:05.2f}", (x0 + 12, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1, cv2.LINE_AA)
    n_state = [sum(1 for r in visible if r.state == s) for s in (0, 1, 2)]
    cv2.putText(canvas, f"Cows: {len(visible)}   Rest {n_state[0]}  Active {n_state[1]}  Walk {n_state[2]}",
                (x0 + 12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    y = 96
    hdr = ["ID", "State", "m/s", "Dist(m)", "Activity"]
    xs = [12, 52, 120, 170, 240]
    for h, xx in zip(hdr, xs):
        cv2.putText(canvas, h, (x0 + xx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.line(canvas, (x0 + 10, y + 6), (x0 + PW - 10, y + 6), (80, 80, 80), 1)
    y += 24
    for r in sorted(visible, key=lambda r: r.tid):
        if y > OH - 10:
            break
        col = id_color(int(r.tid))
        cv2.rectangle(canvas, (x0 + 10, y - 12), (x0 + 16, y + 2), col, -1)
        cv2.putText(canvas, f"{int(r.tid)}", (x0 + 22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        st = int(r.state)
        cv2.putText(canvas, STATE_NAMES[st], (x0 + xs[1], y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, STATE_COLORS[st], 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{r.speed:.2f}", (x0 + xs[2], y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{r.dist_cum:.1f}", (x0 + xs[3], y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
        bw = int(min(r.activity, 100) / 100 * 110)
        cv2.rectangle(canvas, (x0 + xs[4], y - 10), (x0 + xs[4] + 110, y), (60, 60, 60), -1)
        cv2.rectangle(canvas, (x0 + xs[4], y - 10), (x0 + xs[4] + bw, y), col, -1)
        y += 22
    cv2.putText(canvas, "dot on box: grey=Rest  orange=Active  green=Walk", (x0 + 12, OH - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1, cv2.LINE_AA)
