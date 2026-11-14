"""검출 영역 편집 창: 영상 위에 다각형을 클릭해서 그린다.

  왼쪽 클릭 : 점 추가        오른쪽 클릭 / [다각형 닫기] : 현재 다각형 완성
  Backspace : 마지막 점 삭제  [선택 삭제] : 목록에서 고른 다각형 삭제
  제외 영역(빨강) 안에 박스 발 위치(하단 중앙)가 있으면 버리고,
  포함 영역(초록)을 하나라도 그리면 그 안의 박스만 사용한다.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (QDialog, QFileDialog, QHBoxLayout, QLabel, QListWidget, QPushButton, QRadioButton,
                               QVBoxLayout, QWidget)

from .zones import load_zones, save_zones


class Canvas(QWidget):
    def __init__(self, frame_bgr, zones, parent=None):
        super().__init__(parent)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self.img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.W, self.H = w, h
        self.zones = zones
        self.cur = []          # 그리는 중인 점 (정규화 좌표)
        self.mode = "exclude"
        self.boxes = []        # 참고용 검출 박스 (정규화)
        self.setMinimumSize(640, 360)
        self.setMouseTracking(True)
        self.on_change = lambda: None

    def _rect(self):
        s = min(self.width() / self.W, self.height() / self.H)
        dw, dh = self.W * s, self.H * s
        return (self.width() - dw) / 2, (self.height() - dh) / 2, dw, dh

    def _to_norm(self, p):
        x0, y0, dw, dh = self._rect()
        return [min(max((p.x() - x0) / dw, 0), 1), min(max((p.y() - y0) / dh, 0), 1)]

    def _to_px(self, q):
        x0, y0, dw, dh = self._rect()
        return QPointF(x0 + q[0] * dw, y0 + q[1] * dh)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.cur.append([round(v, 4) for v in self._to_norm(e.position())])
        elif e.button() == Qt.RightButton:
            self.close_poly()
        self.update()

    def close_poly(self):
        if len(self.cur) >= 3:
            self.zones[self.mode].append(self.cur)
            self.on_change()
        self.cur = []
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        x0, y0, dw, dh = self._rect()
        p.drawImage(int(x0), int(y0), self.img.scaled(int(dw), int(dh), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        p.setPen(QPen(QColor(255, 255, 0), 1))
        for b in self.boxes:
            a, c = self._to_px((b[0], b[1])), self._to_px((b[2], b[3]))
            p.drawRect(int(a.x()), int(a.y()), int(c.x() - a.x()), int(c.y() - a.y()))
            p.drawEllipse(self._to_px(((b[0] + b[2]) / 2, b[3])), 2, 2)
        for kind, col in (("exclude", QColor(255, 40, 40)), ("include", QColor(40, 220, 40))):
            for poly in self.zones[kind]:
                fill = QColor(col)
                fill.setAlpha(70)
                p.setBrush(QBrush(fill))
                p.setPen(QPen(col, 2))
                p.drawPolygon(QPolygonF([self._to_px(q) for q in poly]))
        if self.cur:
            col = QColor(255, 40, 40) if self.mode == "exclude" else QColor(40, 220, 40)
            p.setPen(QPen(col, 2, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            pts = [self._to_px(q) for q in self.cur]
            for a, b in zip(pts, pts[1:]):
                p.drawLine(a, b)
            for q in pts:
                p.drawEllipse(q, 3, 3)
        p.end()


class ZoneDialog(QDialog):
    def __init__(self, frame_bgr, path: Path, boxes=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"검출 영역 편집 — {path.name}")
        self.resize(1200, 760)
        self.path = Path(path)
        self.zones = load_zones(self.path)
        self.canvas = Canvas(frame_bgr, self.zones)
        H, W = frame_bgr.shape[:2]
        self.canvas.boxes = [(b[0] / W, b[1] / H, b[2] / W, b[3] / H) for b in (boxes or [])]
        self.canvas.on_change = self._refresh

        side = QVBoxLayout()
        help_ = QLabel("왼쪽 클릭: 점 추가\n오른쪽 클릭: 다각형 닫기\nBackspace: 마지막 점 삭제\n\n"
                       "빨강 = 제외 영역 (축사 밖·흐린 배경)\n초록 = 포함 영역 (그리면 이 안만 사용)\n"
                       "판정 기준: 박스 하단 중앙(발 위치)\n노란 박스 = 현재 검출")
        help_.setWordWrap(True)
        side.addWidget(help_)
        self.rb_exc, self.rb_inc = QRadioButton("제외 영역 그리기"), QRadioButton("포함 영역 그리기")
        self.rb_exc.setChecked(True)
        self.rb_exc.toggled.connect(lambda v: setattr(self.canvas, "mode", "exclude" if v else "include"))
        side.addWidget(self.rb_exc)
        side.addWidget(self.rb_inc)
        b_close = QPushButton("다각형 닫기")
        b_close.clicked.connect(self.canvas.close_poly)
        side.addWidget(b_close)
        self.lst = QListWidget()
        side.addWidget(self.lst, 1)
        b_del = QPushButton("선택 삭제")
        b_del.clicked.connect(self._delete)
        b_clear = QPushButton("모두 지우기")
        b_clear.clicked.connect(self._clear)
        side.addWidget(b_del)
        side.addWidget(b_clear)
        h = QHBoxLayout()
        b_save = QPushButton("저장 후 적용")
        b_save.setDefault(True)
        b_save.clicked.connect(self._save)
        b_as = QPushButton("다른 이름으로…")
        b_as.clicked.connect(self._save_as)
        b_cancel = QPushButton("취소")
        b_cancel.clicked.connect(self.reject)
        h.addWidget(b_save)
        h.addWidget(b_as)
        side.addLayout(h)
        side.addWidget(b_cancel)
        sw = QWidget()
        sw.setLayout(side)
        sw.setFixedWidth(260)
        lay = QHBoxLayout(self)
        lay.addWidget(self.canvas, 1)
        lay.addWidget(sw)
        self._refresh()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Backspace and self.canvas.cur:
            self.canvas.cur.pop()
            self.canvas.update()
        else:
            super().keyPressEvent(e)

    def _items(self):
        return [("exclude", i) for i in range(len(self.zones["exclude"]))] + \
               [("include", i) for i in range(len(self.zones["include"]))]

    def _refresh(self):
        self.lst.clear()
        for k, i in self._items():
            self.lst.addItem(f"{'제외' if k == 'exclude' else '포함'} {i + 1}  ({len(self.zones[k][i])}점)")
        self.canvas.update()

    def _delete(self):
        r = self.lst.currentRow()
        if r >= 0:
            k, i = self._items()[r]
            del self.zones[k][i]
            self._refresh()

    def _clear(self):
        self.zones["exclude"].clear()
        self.zones["include"].clear()
        self.canvas.cur = []
        self._refresh()

    def _save(self):
        self.canvas.close_poly()
        save_zones(self.path, self.zones)
        self.accept()

    def _save_as(self):
        p, _ = QFileDialog.getSaveFileName(self, "영역 파일 저장", str(self.path), "JSON (*.json)")
        if p:
            self.path = Path(p)
            self._save()


def grab_frame(source: str, at_s: float = 2.0):
    """편집용 프레임 1장 (파일은 at_s 초 지점)."""
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not str(source).isdigit() and "://" not in str(source):
        cap.set(cv2.CAP_PROP_POS_MSEC, at_s * 1000)
    ok, fr = cap.read()
    cap.release()
    return fr if ok else np.zeros((480, 854, 3), np.uint8)
