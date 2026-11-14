"""Usage
  python run_activity.py E:/Video/cow                # process every mp4 in the folder
  python run_activity.py E:/Video/cow/cow01.mp4      # single video
  python run_activity.py E:/Video/cow --reuse        # reuse detection/tracking, retune only linking/activity parameters
  python run_activity.py E:/Video/cow --reuse --herd-size 15   # set the herd size of the barn
  python run_activity.py new_farm.mp4 --weights models/hanwoo_yolo11s.pt --cow-height 1.25
"""
import argparse
import json
from pathlib import Path

from hanwoo.config import OUTPUT_DIR, PipelineConfig
from hanwoo.dashboard import write_index
from hanwoo.pipeline import process_video


def main():
    ap = argparse.ArgumentParser(description="Hanwoo individual ID tracking and activity measurement")
    ap.add_argument("source", help="video file or folder")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out", default=str(OUTPUT_DIR))
    ap.add_argument("--cow-height", type=float, default=None, help="reference cow height (m) for perspective correction")
    ap.add_argument("--herd-size", type=int, default=0, help="herd size N in the barn (0 = estimate automatically)")
    ap.add_argument("--no-reid", action="store_true", help="use only two-stage linking instead of ReID fixed-herd assignment")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="reuse previous detection/tracking results (cache); recompute only linking and activity")
    a = ap.parse_args()

    cfg = PipelineConfig()
    if a.weights:
        cfg.weights = a.weights
    if a.cow_height:
        cfg.cow_height_m = a.cow_height
    cfg.max_frames = a.max_frames
    cfg.herd_size = a.herd_size
    cfg.use_identity = not a.no_reid
    cfg.write_video = not a.no_video

    src = Path(a.source)
    videos = sorted(src.glob("*.mp4")) if src.is_dir() else [src]
    out = Path(a.out)
    results = []
    for v in videos:
        results.append(process_video(v, cfg, out, reuse=a.reuse))
    # refresh index including earlier results
    allres = []
    for p in sorted(out.glob("*/result.json")):
        allres.append(json.loads(p.read_text(encoding="utf-8"))["result"])
    write_index(out, allres)
    print("index:", out / "index.html")


if __name__ == "__main__":
    main()
