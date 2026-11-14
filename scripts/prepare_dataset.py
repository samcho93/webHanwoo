"""학습 데이터 자동 구축 (Pseudo-labeling / Knowledge distillation).

대형 COCO 모델(teacher, YOLO11x + TTA, 고해상도)로 축사 영상 프레임을 자동 라벨링하여
'hanwoo' 단일 클래스 YOLO 데이터셋을 만든다. 이후 경량 모델(student, YOLO11s)을 이 데이터로
미세조정(fine-tune)하면, 축사 환경(펜스 가림·원거리 소·소/말 혼동)에 특화되면서도 빠른 검출기를 얻는다.

  - 영상 단위로 train / val 분리 (같은 영상 프레임이 양쪽에 섞이면 성능이 과대평가됨)
  - 프레임 샘플링 간격: 짧은 영상 1초, 긴 영상 3초 (중복 프레임 최소화)
  - teacher 신뢰도 >= hi(0.4) 는 정답 라벨, lo~hi 사이의 '애매한' 박스는 회색으로 칠해 무시영역(ignore region)
    처리 → 학습 시 정답도 배경도 아닌 영역이 되어 노이즈 라벨을 막는다.
  - 사용 클래스: cow(19), horse(17). sheep 은 흰색 사료 곤포(베일)를 오검출하므로 제외

usage: python scripts/prepare_dataset.py E:/Video/cow --val cow02 cow09
"""
import argparse
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hanwoo.config import DATA_DIR  # noqa: E402

ANIMALS = [17, 19]  # horse, cow → '한우' 후보 (sheep 은 곤포 오검출이 많아 제외)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dir")
    ap.add_argument("--val", nargs="+", default=["cow02", "cow09"])
    ap.add_argument("--teacher", default="yolo11x.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--hi", type=float, default=0.40)
    ap.add_argument("--lo", type=float, default=0.15)
    ap.add_argument("--out", default=str(DATA_DIR / "hanwoo_ds"))
    a = ap.parse_args()

    from ultralytics import YOLO
    teacher = YOLO(a.teacher)
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    for s in ("train", "val"):
        (out / "images" / s).mkdir(parents=True)
        (out / "labels" / s).mkdir(parents=True)

    stats = {"train": [0, 0, 0], "val": [0, 0, 0]}  # frames, boxes, ignored boxes
    for vp in sorted(Path(a.video_dir).glob("*.mp4")):
        split = "val" if vp.stem in a.val else "train"
        cap = cv2.VideoCapture(str(vp))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        every = int(fps * (3 if n / fps > 300 else 1))
        fi = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if fi % every == 0:
                _, frame = cap.retrieve()
                r = teacher.predict(frame, imgsz=a.imgsz, conf=a.lo, classes=ANIMALS, agnostic_nms=True,
                                    augment=True, verbose=False)[0]
                confs = r.boxes.conf.cpu().numpy()
                amb = (confs >= a.lo) & (confs < a.hi)
                for x1, y1, x2, y2 in r.boxes.xyxy.cpu().numpy()[amb].astype(int):
                    frame[y1:y2, x1:x2] = (114, 114, 114)  # ignore region (YOLO letterbox 회색)
                stats[split][2] += int(amb.sum())
                if True:
                    xywhn = r.boxes.xywhn.cpu().numpy()[confs >= a.hi]
                    stem = f"{vp.stem}_{fi:06d}"
                    cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), frame)
                    with open(out / "labels" / split / f"{stem}.txt", "w") as f:
                        for x, y, w, h in xywhn:
                            f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                    stats[split][0] += 1
                    stats[split][1] += len(xywhn)
            fi += 1
        cap.release()
        print(f"{vp.name} [{split}] done → {stats}")

    (out / "hanwoo.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: hanwoo\n", encoding="utf-8")
    print("dataset:", out, stats)


if __name__ == "__main__":
    main()
