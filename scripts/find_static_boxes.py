"""고정 오검출(배경 물체) 자동 탐지 → 제외 영역 제안/적용.

같은 좌표에 오래(기본 120초 이상) 남아 있는 '작은' 박스는 구유·사료통 같은 배경 물체를 소로 잘못 본 것이다.
사용:
  python scripts/find_static_boxes.py outputs/cow03                 # 후보만 출력
  python scripts/find_static_boxes.py outputs/cow03 --apply         # configs/zones/cow03.json 에 제외 영역 추가
  python scripts/find_static_boxes.py outputs/live/cow03_2026...    # 실시간 결과도 가능
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hanwoo.stitching import fit_perspective  # noqa: E402
from hanwoo.zones import ZONES_DIR, StaticDetector, add_exclusions, zone_stem  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--min-s", type=float, default=120.0, help="같은 자리 누적 시간(초) 하한")
    ap.add_argument("--max-ratio", type=float, default=0.5, help="박스 높이/기대 체고 상한 (누운 소 보호)")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--size", default="854x480", help="영상 크기 WxH")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    od = Path(a.out_dir)
    if (od / "tracks_live.csv").exists():
        d = pd.read_csv(od / "tracks_live.csv")
        source = json.loads((od / "config.json").read_text(encoding="utf-8"))["source"]
    else:
        d = pd.read_csv(od / "tracks_raw.csv")
        source = od.name
    W, H = map(int, a.size.lower().split("x"))
    sd = StaticDetector(min_s=a.min_s, max_ratio=a.max_ratio)
    for _f, g in d.groupby("frame"):
        sd.add(g[["x1", "y1", "x2", "y2"]].to_numpy(), 1 / a.fps)
    cands = sd.candidates(fit_perspective(d), W, H)
    if not cands:
        print("고정 오검출 후보 없음")
        return
    for c in cands:
        print(f"후보: 박스 {tuple(round(v) for v in c['box'])}  {c['seconds']:.0f}초 고정  크기비 {c['ratio']}")
    if a.apply:
        path = ZONES_DIR / f"{zone_stem(source)}.json"
        z = add_exclusions(path, [c["rect"] for c in cands])
        print(f"→ {path} 제외 영역 {len(z['exclude'])}개")


if __name__ == "__main__":
    main()
