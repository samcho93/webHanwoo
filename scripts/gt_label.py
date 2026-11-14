"""Ground-truth identity labeling tool (tracklet level).

Labels which real cow each raw BoT-SORT tracklet belongs to, inside a time segment of a video.
The result is used by scripts/eval_gt.py to compute IDF1 / ID switches / AssA for every method variant.

Unit of labeling: raw tracker tracklets (outputs/<video>/tracks_raw.csv) of at least 12 frames that appear in the
segment. Shorter fragments are marked 'ignore' and excluded from evaluation for every method.
The tool deliberately shows no suggestion from the proposed method, so the labels do not inherit its decisions.

Usage:  python scripts/gt_label.py cow03 --start 0 --end 300
        python scripts/gt_label.py cow09            (whole video)

Guideline: label every animal inside the pen being analysed with a cow number. Background objects, and cows
in other pens outside the analysed area (the area the detection zones keep), are 'Not a cow (F)'. If a tracklet
jumps from one cow to another, move to the first frame of the second cow and split it (S). When unsure whether two
tracklets are the same cow, play the video around the gap (⏮ start shows 2 s before the tracklet begins).

Keys:   Space  next unlabeled tracklet       Enter  assign to the cow selected in the list
        N      new cow                       F      not a cow (false positive)
        S      split the tracklet at the current frame   U  clear label
        ← / →  -1 / +1 s     , / .  -1 / +1 frame     P  play / pause     Ctrl+S  save
Labels are saved to gt/<video>.json (autosave on every change) and gt/<video>_gt.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox,
                               QPushButton, QSlider, QVBoxLayout, QWidget)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
GT_DIR = ROOT / "gt"
VIDEO_DIR = Path(r"E:\Video\cow")
MIN_FRAMES = 12
PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
           (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
           (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180)]


def color(label):
    if label is None:
        return (160, 160, 160)
    if label == 0:
        return (90, 90, 90)
    r, g, b = PALETTE[(label - 1) % len(PALETTE)]
    return (b, g, r)


class Data:
    def __init__(self, video: str, start_s: float | None, end_s: float | None):
        self.video = video
        self.path = VIDEO_DIR / f"{video}.mp4"
        self.cap = cv2.VideoCapture(str(self.path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 24.0
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.f0 = int((start_s or 0) * self.fps)
        self.f1 = min(n - 1, int(end_s * self.fps) if end_s else n - 1)
        raw = pd.read_csv(OUT / video / "tracks_raw.csv")
        raw = raw[(raw.frame >= self.f0) & (raw.frame <= self.f1)]
        self.raw = raw
        self.by_frame = {f: g for f, g in raw.groupby("frame")}
        self.json_path = GT_DIR / f"{video}.json"
        if self.json_path.exists():
            d = json.loads(self.json_path.read_text(encoding="utf-8"))
            if (d["f0"], d["f1"]) != (self.f0, self.f1):
                raise SystemExit(f"{self.json_path} was made for frames {d['f0']}-{d['f1']}; use the same segment")
            self.pieces = d["pieces"]
        else:
            self.pieces = []
            for tid, g in raw.groupby("tid"):
                p = dict(tid=int(tid), a=int(g.frame.min()), b=int(g.frame.max()), label=None)
                if len(g) < MIN_FRAMES:
                    p["label"] = -1           # ignore
                self.pieces.append(p)
        self.pieces.sort(key=lambda p: (p["a"], p["tid"]))
        self._frame_cache: dict[int, np.ndarray] = {}

    # ---------------------------------------------------------------- pieces
    def piece_of(self, tid, frame):
        for i, p in enumerate(self.pieces):
            if p["tid"] == tid and p["a"] <= frame <= p["b"]:
                return i
        return None

    def labels(self):
        return sorted({p["label"] for p in self.pieces if p["label"] and p["label"] > 0})

    def cow_pieces(self, label):
        return [p for p in self.pieces if p["label"] == label]

    def overlaps(self, i, label):
        p = self.pieces[i]
        for q in self.cow_pieces(label):
            if q is not p and q["a"] <= p["b"] and p["a"] <= q["b"]:
                ov = min(p["b"], q["b"]) - max(p["a"], q["a"]) + 1
                if ov > 2:
                    return ov
        return 0

    def split(self, i, frame):
        p = self.pieces[i]
        if not (p["a"] < frame <= p["b"]):
            return False
        q = dict(tid=p["tid"], a=frame, b=p["b"], label=None)
        p["b"] = frame - 1
        self.pieces.insert(i + 1, q)
        for x in (p, q):
            n = int(((self.raw.tid == x["tid"]) & (self.raw.frame >= x["a"]) & (self.raw.frame <= x["b"])).sum())
            if n < MIN_FRAMES:
                x["label"] = -1
        self.pieces.sort(key=lambda p: (p["a"], p["tid"]))
        return True

    def boxes(self, i):
        p = self.pieces[i]
        r = self.raw
        return r[(r.tid == p["tid"]) & (r.frame >= p["a"]) & (r.frame <= p["b"])]

    # ---------------------------------------------------------------- io
    def frame(self, f):
        if f in self._frame_cache:
            return self._frame_cache[f]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = self.cap.read()
        if not ok:
            img = np.zeros((480, 854, 3), np.uint8)
        if len(self._frame_cache) > 300:
            self._frame_cache.clear()
        self._frame_cache[f] = img
        return img

    def crop(self, i, where=0.5, size=96):
        b = self.boxes(i)
        if b.empty:
            return np.zeros((size, size, 3), np.uint8)
        row = b.iloc[min(len(b) - 1, int(where * (len(b) - 1)))]
        img = self.frame(int(row.frame))
        x1, y1, x2, y2 = [int(max(0, v)) for v in (row.x1, row.y1, row.x2, row.y2)]
        c = img[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)]
        h, w = c.shape[:2]
        s = size / max(h, w)
        c = cv2.resize(c, (max(1, int(w * s)), max(1, int(h * s))))
        out = np.full((size, size, 3), 40, np.uint8)
        out[:c.shape[0], :c.shape[1]] = c
        return out

    def save(self):
        GT_DIR.mkdir(exist_ok=True)
        self.json_path.write_text(json.dumps(dict(video=self.video, f0=self.f0, f1=self.f1, fps=self.fps,
                                                  pieces=self.pieces), indent=1), encoding="utf-8")
        rows = []
        for i, p in enumerate(self.pieces):
            b = self.boxes(i)
            lab = p["label"]
            gid = -1 if lab is None or lab < 0 else lab        # -1 ignore / unlabeled, 0 not a cow
            for r in b.itertuples():
                rows.append((r.frame, gid, r.x1, r.y1, r.x2, r.y2))
        pd.DataFrame(rows, columns=["frame", "gt_id", "x1", "y1", "x2", "y2"]).sort_values("frame").to_csv(
            GT_DIR / f"{self.video}_gt.csv", index=False, float_format="%.1f")


def to_pixmap(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy())


class Labeler(QWidget):
    def __init__(self, d: Data):
        super().__init__()
        self.d = d
        self.cur = d.f0
        self.sel = None
        self.setWindowTitle(f"GT labeling — {d.video}  frames {d.f0}-{d.f1}")
        self.view = QLabel(); self.view.setMinimumSize(854, 480); self.view.setAlignment(Qt.AlignCenter)
        self.view.mousePressEvent = self.on_click
        self.slider = QSlider(Qt.Horizontal); self.slider.setRange(d.f0, d.f1)
        self.slider.valueChanged.connect(self.seek)
        self.time = QLabel()
        self.info = QLabel(); self.info.setWordWrap(True)
        self.thumbs = QLabel()
        self.cows = QListWidget(); self.cows.setIconSize(QPixmap(64, 64).size())
        self.cows.itemDoubleClicked.connect(lambda *_: self.assign())
        self.progress = QLabel()

        btn = lambda t, f: (lambda b: (b.clicked.connect(f), b)[1])(QPushButton(t))
        left = QVBoxLayout(); left.addWidget(self.view, 1)
        row = QHBoxLayout()
        for t, f in [("⏮ start", self.goto_start), ("−1 s", lambda: self.seek(self.cur - int(d.fps))),
                     ("◀ frame", lambda: self.seek(self.cur - 1)), ("▶ / ❚❚", self.toggle_play),
                     ("frame ▶", lambda: self.seek(self.cur + 1)), ("+1 s", lambda: self.seek(self.cur + int(d.fps))),
                     ("end ⏭", self.goto_end)]:
            row.addWidget(btn(t, f))
        left.addWidget(self.slider); left.addLayout(row); left.addWidget(self.time)

        right = QVBoxLayout()
        right.addWidget(QLabel("<b>Selected tracklet</b> (click a box, or Space for the next unlabeled one)"))
        right.addWidget(self.info); right.addWidget(self.thumbs)
        right.addWidget(QLabel("<b>Cows</b> — last seen before the selected tracklet; ⚠ = visible at the same time"))
        right.addWidget(self.cows, 1)
        g = QHBoxLayout()
        for t, f in [("Assign (Enter)", self.assign), ("New cow (N)", self.new_cow), ("Not a cow (F)", self.not_cow)]:
            g.addWidget(btn(t, f))
        right.addLayout(g)
        g = QHBoxLayout()
        for t, f in [("Split here (S)", self.split), ("Clear (U)", self.clear), ("Next (Space)", self.next_unlabeled)]:
            g.addWidget(btn(t, f))
        right.addLayout(g)
        right.addWidget(self.progress)
        lay = QHBoxLayout(self); lay.addLayout(left, 3); lay.addLayout(right, 2)

        for key, f in [("Space", self.next_unlabeled), ("Return", self.assign), ("Enter", self.assign),
                       ("N", self.new_cow), ("F", self.not_cow), ("S", self.split), ("U", self.clear),
                       ("Left", lambda: self.seek(self.cur - int(d.fps))), ("Right", lambda: self.seek(self.cur + int(d.fps))),
                       (",", lambda: self.seek(self.cur - 1)), (".", lambda: self.seek(self.cur + 1)),
                       ("P", self.toggle_play), ("Ctrl+S", self.save)]:
            QShortcut(QKeySequence(key), self, f)
        self.timer = QTimer(self); self.timer.timeout.connect(lambda: self.seek(self.cur + 1))
        self.next_unlabeled()

    # ---------------------------------------------------------------- navigation
    def seek(self, f):
        f = int(min(max(f, self.d.f0), self.d.f1))
        self.cur = f
        if self.slider.value() != f:
            self.slider.blockSignals(True); self.slider.setValue(f); self.slider.blockSignals(False)
        self.draw()

    def toggle_play(self):
        self.timer.stop() if self.timer.isActive() else self.timer.start(int(1000 / self.d.fps))

    def goto_start(self):
        if self.sel is not None:
            self.seek(self.d.pieces[self.sel]["a"] - int(2 * self.d.fps))

    def goto_end(self):
        if self.sel is not None:
            self.seek(self.d.pieces[self.sel]["b"])

    def select(self, i, seek=True):
        self.sel = i
        if seek and i is not None:
            self.seek(self.d.pieces[i]["a"])
        self.refresh()

    def next_unlabeled(self):
        start = (self.sel + 1) if self.sel is not None else 0
        n = len(self.d.pieces)
        for k in range(n):
            i = (start + k) % n
            if self.d.pieces[i]["label"] is None:
                return self.select(i)
        self.select(self.sel, seek=False)
        QMessageBox.information(self, "Done", "Every tracklet in the segment is labeled.")

    def on_click(self, ev):
        pm = self.view.pixmap()
        if pm is None:
            return
        sx = (ev.position().x() - (self.view.width() - pm.width()) / 2) / self._scale
        sy = (ev.position().y() - (self.view.height() - pm.height()) / 2) / self._scale
        g = self.d.by_frame.get(self.cur)
        if g is None:
            return
        hit = g[(g.x1 <= sx) & (sx <= g.x2) & (g.y1 <= sy) & (sy <= g.y2)]
        if len(hit):
            hit = hit.assign(area=(hit.x2 - hit.x1) * (hit.y2 - hit.y1)).sort_values("area")
            i = self.d.piece_of(int(hit.iloc[0].tid), self.cur)
            if i is not None:
                self.select(i, seek=False)

    # ---------------------------------------------------------------- labeling
    def changed(self):
        self.d.save()
        self.refresh()

    def assign(self):
        it = self.cows.currentItem()
        if self.sel is None or it is None:
            return
        lab = it.data(Qt.UserRole)
        ov = self.d.overlaps(self.sel, lab)
        if ov and QMessageBox.question(self, "Overlap", f"Cow {lab} is visible at the same time for {ov} frames. "
                                       "Assign anyway?") != QMessageBox.Yes:
            return
        self.d.pieces[self.sel]["label"] = lab
        self.changed(); self.next_unlabeled()

    def new_cow(self):
        if self.sel is None:
            return
        labs = self.d.labels()
        self.d.pieces[self.sel]["label"] = (max(labs) + 1) if labs else 1
        self.changed(); self.next_unlabeled()

    def not_cow(self):
        if self.sel is not None:
            self.d.pieces[self.sel]["label"] = 0
            self.changed(); self.next_unlabeled()

    def clear(self):
        if self.sel is not None:
            self.d.pieces[self.sel]["label"] = None
            self.changed()

    def split(self):
        if self.sel is not None and self.d.split(self.sel, self.cur):
            p = self.d.pieces[self.sel]
            self.sel = self.d.pieces.index(p)
            self.changed()

    def save(self):
        self.d.save()
        self.progress.setText(self.progress.text() + "  · saved")

    # ---------------------------------------------------------------- drawing
    def refresh(self):
        d = self.d
        todo = sum(p["label"] is None for p in d.pieces)
        ign = sum(p["label"] == -1 for p in d.pieces)
        n = len(d.pieces) - ign
        self.progress.setText(f"{n - todo}/{n} tracklets labeled ({ign} short ones ignored) · {len(d.labels())} cows · "
                              f"saved to {d.json_path.name}")
        self.cows.clear()
        if self.sel is None:
            self.draw(); return
        p = d.pieces[self.sel]
        lab = {None: "unlabeled", 0: "not a cow", -1: "ignored (too short)"}.get(p["label"], f"cow {p['label']}")
        self.info.setText(f"tracker id {p['tid']} · frames {p['a']}–{p['b']} "
                          f"({(p['a']) / d.fps:.1f}–{p['b'] / d.fps:.1f} s, {(p['b'] - p['a'] + 1) / d.fps:.1f} s) · "
                          f"<b>{lab}</b>")
        strip = np.hstack([d.crop(self.sel, w) for w in (0.0, 0.33, 0.66, 1.0)])
        self.thumbs.setPixmap(to_pixmap(strip))
        items = []
        for c in d.labels():
            ps = d.cow_pieces(c)
            before = [q for q in ps if q["b"] < p["a"]]
            last = max(before, key=lambda q: q["b"]) if before else min(ps, key=lambda q: q["a"])
            ov = d.overlaps(self.sel, c)
            gap = (p["a"] - last["b"]) / d.fps
            text = f"cow {c}   last seen {last['b'] / d.fps:.1f} s ({gap:+.1f} s)" + ("   ⚠ overlap" if ov else "")
            it = QListWidgetItem(text)
            it.setIcon(to_pixmap(d.crop(d.pieces.index(last), 1.0, 64)))
            it.setData(Qt.UserRole, c)
            r, g, b = color(c)[::-1]
            it.setForeground(QColor(150, 150, 150) if ov else QColor(0, 0, 0))
            it.setBackground(QColor(r, g, b, 60))
            items.append((ov > 0, abs(gap), it))
        for _, _, it in sorted(items, key=lambda x: (x[0], x[1])):
            self.cows.addItem(it)
        if self.cows.count():
            self.cows.setCurrentRow(0)
        self.draw()

    def draw(self):
        d = self.d
        img = d.frame(self.cur).copy()
        g = d.by_frame.get(self.cur)
        if g is not None:
            for r in g.itertuples():
                i = d.piece_of(int(r.tid), self.cur)
                lab = d.pieces[i]["label"] if i is not None else -1
                if lab == -1:
                    continue
                c = color(lab)
                th = 3 if i == self.sel else 1
                if i == self.sel:
                    c = (0, 255, 255)
                cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), c, th)
                txt = "?" if lab is None else ("FP" if lab == 0 else str(lab))
                cv2.putText(img, txt, (int(r.x1) + 2, int(r.y1) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        # last known box of each cow before the current frame (dashed hint of where it went)
        if self.sel is not None:
            p = d.pieces[self.sel]
            for c in d.labels():
                before = [q for q in d.cow_pieces(c) if q["b"] < self.cur and self.cur - q["b"] < 60 * d.fps]
                if before and not d.overlaps(self.sel, c):
                    q = max(before, key=lambda q: q["b"])
                    b = d.boxes(d.pieces.index(q)).iloc[-1]
                    cv2.rectangle(img, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color(c), 1, cv2.LINE_4)
                    cv2.putText(img, f"{c}'", (int(b.x1) + 2, int(b.y2) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                color(c), 1)
        pm = to_pixmap(img)
        self._scale = min(self.view.width() / pm.width(), self.view.height() / pm.height())
        self.view.setPixmap(pm.scaled(int(pm.width() * self._scale), int(pm.height() * self._scale),
                                      Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.time.setText(f"frame {self.cur}   t = {self.cur / d.fps:.2f} s   "
                          "(boxes: grey ? = unlabeled, number = cow, FP = not a cow, yellow = selected, "
                          "thin n' = cow's last known position)")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.draw()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=None, help="segment start (s)")
    ap.add_argument("--end", type=float, default=None, help="segment end (s)")
    a = ap.parse_args()
    app = QApplication(sys.argv)
    w = Labeler(Data(a.video, a.start, a.end))
    w.resize(1500, 820)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
