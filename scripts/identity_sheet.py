"""ID 일관성 육안 검증용 '콘택트 시트': 각 ID 행에 영상 전 구간에서 고르게 뽑은 crop 을 나열.

한 행이 처음부터 끝까지 같은 소로 보이면 ID 가 유지된 것이다.
사용:  python scripts/identity_sheet.py outputs/cow03 --cols 12
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hanwoo.reid import crop_box  # noqa: E402
from hanwoo.render import id_color  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--video", default=None)
    ap.add_argument("--cols", type=int, default=12)
    ap.add_argument("--size", type=int, default=80)
    ap.add_argument("--dst", default=None)
    a = ap.parse_args()
    od = Path(a.out_dir)
    video = a.video or f"E:/Video/cow/{od.name}.mp4"
    act = pd.read_csv(od / "tracks_activity.csv")
    act = act[act.interp == 0]
    fps = 24.0
    nmax = act.frame.max()
    edges = np.linspace(0, nmax, a.cols + 1)
    picks = []  # (row, col, frame, box)
    ids = sorted(act.tid.unique())
    for r, tid in enumerate(ids):
        g = act[act.tid == tid]
        for c in range(a.cols):
            m = g[(g.frame >= edges[c]) & (g.frame < edges[c + 1])]
            if len(m):
                x = m.iloc[len(m) // 2]
                picks.append((r, c, int(x.frame), (x.x1, x.y1, x.x2, x.y2)))
    S = a.size
    lab_w = 70
    head = 22
    sheet = np.full((head + len(ids) * S, lab_w + a.cols * S, 3), 255, np.uint8)
    for c in range(a.cols):
        t = (edges[c] + edges[c + 1]) / 2 / fps
        cv2.putText(sheet, f"{int(t // 60)}:{int(t % 60):02d}", (lab_w + c * S + 18, 15), 0, 0.42, (60, 60, 60), 1, cv2.LINE_AA)
    for r, tid in enumerate(ids):
        col = id_color(int(tid))
        y = head + r * S
        cv2.rectangle(sheet, (4, y + 20), (22, y + S - 20), col, -1)
        cv2.putText(sheet, f"ID{tid}", (26, y + S // 2 + 5), 0, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
    # 순차 디코딩 (임의 탐색(seek)은 일부 코덱에서 프레임을 놓친다)
    cap = cv2.VideoCapture(video)
    by_f = {}
    for p in picks:
        by_f.setdefault(p[2], []).append(p)
    fi, last = 0, max(by_f)
    while fi <= last:
        if fi in by_f:
            ok, frame = cap.read()
            if not ok:
                break
            for r, c, _f, box in by_f[fi]:
                sheet[head + r * S:head + (r + 1) * S, lab_w + c * S:lab_w + (c + 1) * S] = crop_box(frame, box, S)
        elif not cap.grab():
            break
        fi += 1
    cap.release()
    dst = a.dst or str(od / "identity_sheet.jpg")
    cv2.imwrite(dst, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(dst, sheet.shape)


if __name__ == "__main__":
    main()
