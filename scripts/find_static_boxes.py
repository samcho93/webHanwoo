"""Automatic detection of static false positives (background objects) → propose/apply exclusion zones.

Small boxes that stay at the same coordinates for a long time (default >= 120 s) are background objects such as feed troughs mistaken for cows.
Usage:
  python scripts/find_static_boxes.py outputs/cow03                 # print candidates only
  python scripts/find_static_boxes.py outputs/cow03 --apply         # add exclusion zones to configs/zones/cow03.json
  python scripts/find_static_boxes.py outputs/live/cow03_2026...    # works on real-time results too
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
    ap.add_argument("--min-s", type=float, default=120.0, help="minimum accumulated time at the same spot (s)")
    ap.add_argument("--max-ratio", type=float, default=0.5, help="max box height / expected cow height (protects lying cows)")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--size", default="854x480", help="video size WxH")
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
        print("No static false-positive candidates")
        return
    for c in cands:
        print(f"Candidate: box {tuple(round(v) for v in c['box'])}  static for {c['seconds']:.0f} s  size ratio {c['ratio']}")
    if a.apply:
        path = ZONES_DIR / f"{zone_stem(source)}.json"
        z = add_exclusions(path, [c["rect"] for c in cands])
        print(f"→ {path}: {len(z['exclude'])} exclusion zones")


if __name__ == "__main__":
    main()
