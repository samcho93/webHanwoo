"""Build ReID training data: collect cow crops using tracking results as 'self-supervised labels'.

Labelling principle (no human labelling)
  - crops within the same tracklet (or linked ID)  → same cow (positive)
  - different tracklets present at the same time  → necessarily different cows (negative)

  mode "id"       : short videos → final IDs of the offline linking result (tracks_activity.csv) as labels
  mode "tracklet" : long videos  → pre-linking tracklets (tracks_raw.csv) as labels (one cow may span several tracklets, so
                                training compares only tracklets that coexist in time)

Clean crops only: conf ≥ 0.5, no interpolated frames, IoU < 0.15 with other cows (no overlap), height ≥ 24px

Usage:  python scripts/prepare_reid.py E:/Video/cow
Output: data/reid/<video>.npz  (imgs uint8 N×128×128×3, label, frame, mode)
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hanwoo.config import DATA_DIR, OUTPUT_DIR  # noqa: E402
from hanwoo.reid import CROP, crop_box, max_iou_others  # noqa: E402


def load_labels(name: str, long_video: bool, min_tracklet_s: float, fps: float):
    if long_video:
        df = pd.read_csv(OUTPUT_DIR / name / "tracks_raw.csv")
        n = df.groupby("tid").frame.transform("count")
        df = df[n >= min_tracklet_s * fps].copy()
        mode = "tracklet"
    else:
        df = pd.read_csv(OUTPUT_DIR / name / "tracks_activity.csv")
        df = df[df.interp == 0].copy()
        mode = "id"
    return df[["frame", "tid", "x1", "y1", "x2", "y2", "conf"]].reset_index(drop=True), mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", help="video folder")
    ap.add_argument("--out", default=str(DATA_DIR / "reid"))
    ap.add_argument("--long-s", type=float, default=300, help="use tracklet mode for videos at least this long (s)")
    ap.add_argument("--crops-per-s", type=float, default=4, help="crops per second per ID (short videos)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    for vp in sorted(Path(a.videos).glob("*.mp4")):
        name = vp.stem
        cap = cv2.VideoCapture(str(vp))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        long_video = dur >= a.long_s
        df, mode = load_labels(name, long_video, 5.0, fps)
        # long videos: 1 crop per second; short videos: crops_per_s
        every = int(round(fps)) if long_video else max(1, int(round(fps / a.crops_per_s)))
        if dur > 100 and not long_video:
            every *= 2
        df = df[df.frame % every == 0].reset_index(drop=True)
        df["occ"] = max_iou_others(df)
        df = df[(df.conf >= 0.5) & (df.occ < 0.15) & ((df.y2 - df.y1) >= 24)].reset_index(drop=True)
        by_frame = df.groupby("frame").indices
        imgs = np.zeros((len(df), CROP, CROP, 3), np.uint8)
        last = df.frame.max()
        fi = 0
        while fi <= last:
            if fi in by_frame:
                ok, frame = cap.read()
                if not ok:
                    break
                for i in by_frame[fi]:
                    r = df.iloc[i]
                    imgs[i] = crop_box(frame, (r.x1, r.y1, r.x2, r.y2))
            elif not cap.grab():
                break
            fi += 1
        cap.release()
        np.savez(out / f"{name}.npz", imgs=imgs, label=df.tid.to_numpy(np.int32),
                 frame=df.frame.to_numpy(np.int32), mode=mode, fps=fps)
        print(f"{name}: {mode:8s} every {every:2d}f  crops {len(df):6d}  ids {df.tid.nunique()}")

    # preview grid (for the course): 6 crops per ID
    S = np.load(out / "cow04.npz")
    rows = []
    for lab in np.unique(S["label"])[:8]:
        idx = np.where(S["label"] == lab)[0]
        pick = idx[np.linspace(0, len(idx) - 1, 8).astype(int)]
        rows.append(np.hstack([S["imgs"][i] for i in pick]))
    cv2.imwrite(str(out / "preview_cow04.jpg"), np.vstack(rows))


if __name__ == "__main__":
    main()
