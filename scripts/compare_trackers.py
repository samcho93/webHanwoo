"""추적 알고리즘 비교 실험 — 어떤 조합이 이 축사에서 ID 를 가장 안정적으로 유지하는가?

정답 ID 라벨(GT)이 없으므로 다음 '대리 지표'를 사용한다.
  herd   : 프레임별 검출 수의 95% 분위 ≈ 실제 두수 추정
  ids    : 최종 ID 개수 (이상적으로 herd 와 같아야 함)
  ratio  : ids / herd  (1.0 에 가까울수록 좋음, 클수록 ID 가 쪼개짐)
  mean_len_s : ID 당 평균 추적 시간
  full   : 영상의 90% 이상 지속된 ID 수 (클수록 좋음)

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
    ("ByteTrack (기본)", "bytetrack.yaml", False),
    ("BoT-SORT (기본)", "botsort.yaml", False),
    ("BoT-SORT 축사튜닝", str(ROOT / "configs/hanwoo_botsort.yaml"), False),
    ("BoT-SORT 축사튜닝 + 오프라인 정합", str(ROOT / "configs/hanwoo_botsort.yaml"), True),
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
