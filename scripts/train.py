"""Fine-tune the student model (YOLO11s).

usage: python scripts/train.py --epochs 60
Output: models/hanwoo_yolo11s.pt  (copy of runs/hanwoo/train/weights/best.pt)
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/hanwoo_ds/hanwoo.yaml"))
    ap.add_argument("--model", default=str(ROOT / "models/yolo11s.pt"))  # start from COCO-pretrained weights (transfer learning)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.model)
    model.train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=0, workers=4,
        project=str(ROOT / "runs/hanwoo"), name="train", exist_ok=True,
        patience=20, cos_lr=True, close_mosaic=10,
        # augmentation for fixed-camera barn video: horizontal flip yes, vertical flip no, brightness (HSV) changes for day/night
        fliplr=0.5, flipud=0.0, hsv_v=0.4, degrees=0.0, scale=0.4, mosaic=1.0, mixup=0.0,
        plots=True,
    )
    best = ROOT / "runs/hanwoo/train/weights/best.pt"
    dst = ROOT / "models/hanwoo_yolo11s.pt"
    shutil.copy(best, dst)
    print("saved:", dst)
    metrics = YOLO(str(dst)).val(data=a.data, imgsz=a.imgsz, device=0, plots=False)
    print("val mAP50:", metrics.box.map50, "mAP50-95:", metrics.box.map)


if __name__ == "__main__":
    main()
