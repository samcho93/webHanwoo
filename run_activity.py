"""사용법
  python run_activity.py E:/Video/cow                # 폴더 내 모든 mp4 처리
  python run_activity.py E:/Video/cow/cow01.mp4      # 단일 영상
  python run_activity.py E:/Video/cow --reuse        # 검출·추적 재사용, 정합/활동량 파라미터만 재튜닝
  python run_activity.py E:/Video/cow --reuse --herd-size 15   # 축사 마릿수 지정
  python run_activity.py new_farm.mp4 --weights models/hanwoo_yolo11s.pt --cow-height 1.25
"""
import argparse
import json
from pathlib import Path

from hanwoo.config import OUTPUT_DIR, PipelineConfig
from hanwoo.dashboard import write_index
from hanwoo.pipeline import process_video


def main():
    ap = argparse.ArgumentParser(description="한우 개체 ID 추적 및 활동량 측정")
    ap.add_argument("source", help="동영상 파일 또는 폴더")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out", default=str(OUTPUT_DIR))
    ap.add_argument("--cow-height", type=float, default=None, help="원근 보정 기준 체고(m)")
    ap.add_argument("--herd-size", type=int, default=0, help="축사 마릿수 N (0=자동 추정)")
    ap.add_argument("--no-reid", action="store_true", help="ReID 고정 개체군 배정 대신 2단계 정합만 사용")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="이전 검출·추적 결과(cache) 재사용, 정합·활동량만 재계산")
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
    # 기존 결과까지 포함해 index 갱신
    allres = []
    for p in sorted(out.glob("*/result.json")):
        allres.append(json.loads(p.read_text(encoding="utf-8"))["result"])
    write_index(out, allres)
    print("index:", out / "index.html")


if __name__ == "__main__":
    main()
