"""ReID 학습 데이터 생성: 추적 결과를 '자기지도 라벨'로 사용해 소 crop 을 모은다.

라벨 원리 (사람 라벨링 없음)
  - 같은 tracklet(또는 정합된 ID) 안의 crop  → 같은 소 (positive)
  - 같은 시각에 동시에 존재하는 다른 tracklet → 반드시 다른 소 (negative)

  mode "id"       : 짧은 영상 → 오프라인 정합 결과(tracks_activity.csv)의 최종 ID 를 라벨로 사용
  mode "tracklet" : 긴 영상   → 정합 전 tracklet(tracks_raw.csv) 을 라벨로 사용 (같은 소가 여러 tracklet 일 수 있으므로
                                학습 시 '동시에 존재하는 tracklet 끼리만' 비교)

깨끗한 crop 만 사용: conf ≥ 0.5, 보간 프레임 제외, 다른 소와 IoU < 0.15 (겹침 없음), 높이 ≥ 24px

사용:  python scripts/prepare_reid.py E:/Video/cow
출력:  data/reid/<영상>.npz  (imgs uint8 N×128×128×3, label, frame, mode)
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
    ap.add_argument("videos", help="영상 폴더")
    ap.add_argument("--out", default=str(DATA_DIR / "reid"))
    ap.add_argument("--long-s", type=float, default=300, help="이 길이(초) 이상이면 tracklet 모드")
    ap.add_argument("--crops-per-s", type=float, default=4, help="ID 당 초당 crop 수 (짧은 영상)")
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
        # 긴 영상은 1초당 1장, 짧은 영상은 crops_per_s 장
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

    # 미리보기 그리드 (강좌용): ID 별 crop 6장
    S = np.load(out / "cow04.npz")
    rows = []
    for lab in np.unique(S["label"])[:8]:
        idx = np.where(S["label"] == lab)[0]
        pick = idx[np.linspace(0, len(idx) - 1, 8).astype(int)]
        rows.append(np.hstack([S["imgs"][i] for i in pick]))
    cv2.imwrite(str(out / "preview_cow04.jpg"), np.vstack(rows))


if __name__ == "__main__":
    main()
