"""Generate figures for the course → site_src/assets/img/
usage: python scripts/make_figures.py
"""
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hanwoo.config import PipelineConfig  # noqa: E402
from hanwoo.render import id_color  # noqa: E402
from hanwoo.stitching import body_px, clean_short, fit_perspective  # noqa: E402

IMG = ROOT / "site_src/assets/img"
VID = Path("E:/Video/cow")
plt.rcParams["font.family"] = "DejaVu Sans"


def frame(name, t):
    cap = cv2.VideoCapture(str(VID / f"{name}.mp4"))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, f = cap.read()
    cap.release()
    return f


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO
    for n in ("cow01", "cow04"):
        cv2.imwrite(str(IMG / f"frame_{n}.jpg"), frame(n, 10))
    f08 = frame("cow08", 10)
    for tag, w, sz, aug, cls in [("yolo11s", "models/yolo11s.pt", 960, False, [16, 17, 18, 19]),
                                 ("yolo11x", "models/yolo11x.pt", 1280, True, [16, 17, 18, 19]),
                                 ("finetuned", "models/hanwoo_yolo11s.pt", 960, False, None)]:
        r = YOLO(str(ROOT / w)).predict(f08, imgsz=sz, conf=0.1, classes=cls, agnostic_nms=True,
                                       augment=aug, verbose=False)[0]
        cv2.imwrite(str(IMG / f"det_{tag}.jpg"), r.plot(line_width=1, font_size=8))

    # dataset sample: an image containing ignore regions
    ds = ROOT / "data/hanwoo_ds"
    best = None
    for p in sorted((ds / "images/train").glob("cow04_*.jpg")):
        im = cv2.imread(str(p))
        gray = (np.abs(im.astype(int) - 114).sum(2) < 6).mean()
        if 0.01 < gray < 0.08:
            best = p
            break
    best = best or next((ds / "images/train").glob("*.jpg"))
    im = cv2.imread(str(best))
    H, W = im.shape[:2]
    for line in open(ds / "labels/train" / (best.stem + ".txt")):
        _, x, y, w, h = map(float, line.split())
        cv2.rectangle(im, (int((x - w / 2) * W), int((y - h / 2) * H)), (int((x + w / 2) * W), int((y + h / 2) * H)),
                      (60, 220, 60), 2)
    cv2.imwrite(str(IMG / "ds_sample.jpg"), im)

    # training curves
    for src, dst in [("results.png", "train_results.png")]:
        p = ROOT / "runs/hanwoo/train" / src
        if p.exists():
            shutil.copy(p, IMG / dst)

    # result snapshot
    vid = ROOT / "outputs/cow04/cow04_tracked.mp4"
    if vid.exists():
        cap = cv2.VideoCapture(str(vid))
        cap.set(cv2.CAP_PROP_POS_MSEC, 30000)
        ok, f = cap.read()
        if ok:
            cv2.imwrite(str(IMG / "snap_tracked.jpg"), f)

    plot_figures()


def plot_figures(name="cow09"):
    """Perspective model + tracklet timeline plots (from existing outputs/<name>/ CSVs only)."""
    raw = pd.read_csv(ROOT / f"outputs/{name}/tracks_raw.csv")
    fin = pd.read_csv(ROOT / f"outputs/{name}/tracks_activity.csv")
    cfg = PipelineConfig()
    clean = clean_short(raw, cfg)
    persp = fit_perspective(clean)
    cy = (clean.y1 + clean.y2) / 2
    h = clean.y2 - clean.y1
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=130)
    ax.scatter(cy[::5], h[::5], s=3, alpha=.25, c="#b5562b", label="detection boxes")
    ys = np.linspace(cy.min(), cy.max(), 50)
    ax.plot(ys, body_px(ys, persp), c="#2f7a54", lw=2.5, label=f"h(y) = {persp[0]:.2f}·y + {persp[1]:.1f}")
    ax.set_xlabel("bbox centre y (px)  ← far · near →")
    ax.set_ylabel("bbox height (px)")
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(IMG / "perspective.png")
    plt.close(fig)

    fps = 24.0
    fig, axs = plt.subplots(2, 1, figsize=(9, 6.5), dpi=130, sharex=True)
    for ax, df, title in [(axs[0], clean, f"Online tracking (BoT-SORT) — {clean.tid.nunique()} IDs"),
                          (axs[1], fin, f"After offline linking — {fin.tid.nunique()} IDs")]:
        g = df.groupby("tid").frame.agg(["min", "max"]).sort_values("min")
        for k, (tid, r) in enumerate(g.iterrows()):
            b, gg, rr = id_color(int(tid))
            ax.plot([r["min"] / fps, r["max"] / fps], [k, k], lw=4, c=(rr / 255, gg / 255, b / 255),
                    solid_capstyle="butt")
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylabel("tracklet")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.3)
    axs[1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(IMG / "stitch_timeline.png")
    plt.close(fig)
    print("figures →", IMG)


if __name__ == "__main__":
    main()
