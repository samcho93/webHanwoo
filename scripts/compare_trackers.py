"""Tracker comparison experiment - which combination keeps IDs most stable in this barn?

There are no ground-truth ID labels, so the following proxy metrics are used.
  herd   : 95th percentile of detections per frame ≈ estimated true head count
  ids    : final number of IDs (ideally equal to herd)
  ratio  : ids / herd  (closer to 1.0 is better; larger means IDs are fragmented)
  mean_len_s : mean tracked time per ID
  full   : number of IDs lasting >= 90% of the video (larger is better)

usage: python scripts/compare_trackers.py E:/Video/cow/cow09.mp4 E:/Video/cow/cow04.mp4
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hanwoo.config import PipelineConfig  # noqa: E402
from hanwoo.detect_track import run_detect_track  # noqa: E402
from hanwoo.stitching import clean_short, fit_perspective, relabel_and_interpolate, stitch_tracklets  # noqa: E402

VARIANTS = [
    ("ByteTrack (default)", "bytetrack.yaml", False),
    ("BoT-SORT (default)", "botsort.yaml", False),
    ("BoT-SORT barn-tuned", str(ROOT / "configs/hanwoo_botsort.yaml"), False),
    ("BoT-SORT barn-tuned + offline linking", str(ROOT / "configs/hanwoo_botsort.yaml"), True),
]


def metrics(df, n_frames, fps, herd):
    g = df.groupby("tid").frame.agg(["min", "max", "count"])
    span = (g["max"] - g["min"] + 1) / fps
    return dict(ids=int(len(g)), ratio=round(len(g) / herd, 2), mean_len_s=round(float(span.mean()), 1),
                full=int((span >= 0.9 * n_frames / fps).sum()))


def main():
    out = []
    for vp in sys.argv[1:]:
        cache = {}
        for name, trk, stitch in VARIANTS:
            cfg = PipelineConfig()
            cfg.tracker_cfg = trk
            if trk not in cache:
                cache[trk] = run_detect_track(vp, cfg, verbose_every=0)
            raw, feats, meta = cache[trk]
            fps = meta["fps"]
            herd = max(1, int(raw.groupby("frame").size().quantile(0.95)))
            df = clean_short(raw, cfg)
            if stitch:
                persp = fit_perspective(df)
                df, _, _ = stitch_tracklets(df, feats, fps, cfg, persp)
                df, _ = relabel_and_interpolate(df, fps, cfg)
            m = dict(video=Path(vp).stem, variant=name, herd=herd, **metrics(df, meta["n_frames"], fps, herd))
            print(m)
            out.append(m)
    p = ROOT / "outputs/tracker_comparison.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", p)


if __name__ == "__main__":
    main()
