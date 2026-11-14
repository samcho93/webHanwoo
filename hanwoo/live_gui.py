"""PySide6 GUI: 입력 선택 · 파라미터 실시간 튜닝 · 영상/상태 표시."""
from __future__ import annotations

import os
import time
from dataclasses import asdict
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow,
                               QMessageBox, QPlainTextEdit, QPushButton, QRadioButton, QScrollArea, QSizePolicy,
                               QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from .config import ROOT, STATE_KO
from .live import RECOUNT_KEYS, TRACKER_KEYS, LiveConfig, LiveEngine

# (키, 이름, 최소, 최대, 간격, 설명)   — 그룹별
PARAMS = {
    "검출 (즉시 반영)": [
        ("conf", "검출 신뢰도 하한", 0.05, 0.95, 0.05, "낮추면 가려진 소도 잡지만 오검출 증가"),
        ("iou", "NMS IoU", 0.3, 0.9, 0.05, "겹친 박스 제거 기준"),
        ("imgsz", "입력 크기(px)", 320, 1600, 32, "클수록 작은 소 검출↑, 속도↓"),
        ("reid_every", "ReID 간격(프레임)", 1, 12, 1, "작을수록 외형 정보↑, 속도↓"),
    ],
    "추적기 BoT-SORT (바꾸면 추적기 재시작)": [
        ("track_high_thresh", "1차 매칭 신뢰도", 0.1, 0.9, 0.05, ""),
        ("track_low_thresh", "2차 매칭 신뢰도", 0.01, 0.5, 0.01, ""),
        ("new_track_thresh", "새 트랙 신뢰도", 0.1, 0.95, 0.05, "높이면 ID 남발 감소"),
        ("track_buffer", "트랙 기억(프레임)", 10, 600, 10, "가려져도 같은 트랙으로 유지할 시간"),
        ("match_thresh", "매칭 임계값", 0.5, 0.99, 0.01, ""),
    ],
    "마릿수 카운트 (바꾸면 '다시 세기')": [
        ("herd_size", "마릿수 N (0=자동)", 0, 200, 1, "알고 있으면 직접 입력하는 것이 가장 정확"),
        ("enroll_s", "등록 시간(초)", 5, 600, 5, "처음 이 시간 동안 세고 외형 등록"),
        ("count_conf", "카운트 신뢰도", 0.1, 0.95, 0.05, ""),
        ("count_stable_s", "유지 시간(초)", 0.2, 30, 0.2, "동시 검출 수가 이만큼 유지돼야 인정"),
        ("cluster_sim", "외형 군집 유사도", 0.2, 0.95, 0.05, "낮추면 군집 수↓"),
        ("count_min_track_s", "군집 최소 길이(초)", 0.5, 30, 0.5, ""),
    ],
    "ID 배정 (즉시 반영)": [
        ("confirm_s", "ID 확정 대기(초)", 0.2, 5, 0.1, "새 트랙 외형 수집 시간 = 표시 지연"),
        ("release_s", "사라짐 판정(초)", 0.2, 10, 0.1, ""),
        ("id_app_weight", "외형 가중치", 0, 1, 0.05, "나머지는 이동거리 비중"),
        ("id_radius_bl", "기본 이동 반경(몸길이)", 0.5, 10, 0.25, ""),
        ("id_speed_bl_s", "초당 추가 반경(몸길이)", 0, 3, 0.05, "사라진 시간에 비례해 커짐"),
        ("id_gate", "이동 게이트 배수", 1, 5, 0.1, ""),
        ("id_max_cost", "배정 최대 비용", 0.2, 1.5, 0.05, "넘으면 '?' 로 대기"),
        ("swap_margin", "뒤바뀜 교정 여유", 0, 1, 0.02, "작을수록 교정을 자주"),
        ("drift_margin", "외형 교정 여유", 0, 1, 0.02, ""),
        ("grow_s", "N 증가 대기(초)", 1, 60, 1, ""),
    ],
    "활동량 (즉시 반영)": [
        ("cow_height_m", "체고(m)", 0.8, 2.0, 0.05, "원근 보정 기준"),
        ("rest_speed", "휴식 속도 상한(m/s)", 0.01, 0.5, 0.01, ""),
        ("walk_speed", "보행 속도 하한(m/s)", 0.05, 1.5, 0.01, ""),
        ("motion_energy_active", "국소활동 움직임", 0.5, 20, 0.5, ""),
        ("smooth_window_s", "평활 창(초)", 0.2, 5, 0.1, ""),
        ("trail_s", "궤적 길이(초)", 0, 30, 0.5, ""),
    ],
}
BOOL_PARAMS = [("auto_grow", "N 자동 증가 (확실한 새 소가 계속 보이면)"),
               ("show_pending", "ID 대기 중인 박스(?) 표시")]


class MainWindow(QMainWindow):
    def __init__(self, cfg: LiveConfig, preset_path=None):
        super().__init__()
        self.cfg = cfg
        self.preset_path = preset_path
        self.engine: LiveEngine | None = None
        self.widgets = {}
        self.log_lines = []
        self.setWindowTitle("한우 실시간 추적 (webHanwoo Live)")
        self.resize(1500, 860)

        self.video = QLabel("입력을 고르고 [시작] 을 누르세요")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(640, 360)
        self.video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.video.setStyleSheet("background:#111;color:#aaa;font-size:16px")

        tabs = QTabWidget()
        tabs.addTab(self._tab_input(), "입력")
        tabs.addTab(self._tab_params(), "파라미터")
        tabs.addTab(self._tab_status(), "상태")
        self.tabs = tabs

        split = QSplitter()
        split.addWidget(self.video)
        split.addWidget(tabs)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setSizes([1050, 450])
        self.setCentralWidget(split)
        self.statusBar().showMessage("준비")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)
        self._last_frame = None

    # ------------------------------------------------------------------ 입력 탭
    def _tab_input(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        g = QGroupBox("입력 소스")
        f = QVBoxLayout(g)
        self.rb_file, self.rb_rtsp, self.rb_cam = QRadioButton("영상 파일"), QRadioButton("RTSP 카메라"), QRadioButton("웹캠")
        bg = QButtonGroup(self)
        for rb in (self.rb_file, self.rb_rtsp, self.rb_cam):
            bg.addButton(rb)
        src = self.cfg.source
        h = QHBoxLayout()
        self.ed_file = QLineEdit(src if not src.lower().startswith("rtsp") and not src.isdigit() else "")
        b = QPushButton("찾기…")
        b.clicked.connect(self._browse)
        h.addWidget(self.ed_file)
        h.addWidget(b)
        self.ed_rtsp = QLineEdit(src if src.lower().startswith("rtsp") else "")
        self.ed_rtsp.setPlaceholderText("rtsp://사용자:비밀번호@192.168.0.10:554/stream1")
        self.sp_cam = QSpinBox()
        self.sp_cam.setRange(0, 9)
        self.cb_tcp = QCheckBox("RTSP 를 TCP 로 받기 (끊김·깨짐 줄임)")
        self.cb_tcp.setChecked(self.cfg.rtsp_tcp)
        f.addWidget(self.rb_file)
        f.addLayout(h)
        f.addWidget(self.rb_rtsp)
        f.addWidget(self.ed_rtsp)
        f.addWidget(self.cb_tcp)
        f.addWidget(self.rb_cam)
        f.addWidget(self.sp_cam)
        (self.rb_rtsp if src.lower().startswith("rtsp") else self.rb_cam if src.isdigit() else self.rb_file).setChecked(True)
        lay.addWidget(g)

        g2 = QGroupBox("실행 옵션")
        f2 = QVBoxLayout(g2)
        self.cb_realtime = QCheckBox("파일을 원래 속도로 재생 (끄면 최대 속도로 처리)")
        self.cb_realtime.setChecked(self.cfg.realtime)
        self.cb_realtime.toggled.connect(lambda v: self._set("realtime", v))
        self.cb_loop = QCheckBox("파일 반복 재생")
        self.cb_loop.setChecked(self.cfg.loop_file)
        self.cb_loop.toggled.connect(lambda v: self._set("loop_file", v))
        self.cb_csv = QCheckBox("CSV 저장 (프레임별 ID·활동량, 요약)")
        self.cb_csv.setChecked(self.cfg.save_csv)
        self.cb_video = QCheckBox("결과 영상 저장 (mp4)")
        self.cb_video.setChecked(self.cfg.save_video)
        for c in (self.cb_realtime, self.cb_loop, self.cb_csv, self.cb_video):
            f2.addWidget(c)
        lay.addWidget(g2)

        h2 = QHBoxLayout()
        self.bt_start = QPushButton("▶ 시작")
        self.bt_pause = QPushButton("❚❚ 일시정지")
        self.bt_stop = QPushButton("■ 정지")
        self.bt_pause.setCheckable(True)
        self.bt_start.clicked.connect(self.start)
        self.bt_pause.toggled.connect(self._pause)
        self.bt_stop.clicked.connect(self.stop)
        for b in (self.bt_start, self.bt_pause, self.bt_stop):
            b.setMinimumHeight(36)
            h2.addWidget(b)
        lay.addLayout(h2)
        self.bt_recount = QPushButton("↻ 마릿수 다시 세기 (등록 단계 재시작)")
        self.bt_recount.clicked.connect(lambda: self.engine and self.engine.recount())
        lay.addWidget(self.bt_recount)
        self.bt_out = QPushButton("결과 폴더 열기")
        self.bt_out.clicked.connect(self._open_out)
        lay.addWidget(self.bt_out)
        lay.addStretch(1)
        self._buttons(False)
        return w

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(self, "영상 선택", str(Path(self.ed_file.text()).parent if self.ed_file.text() else ROOT),
                                           "Video (*.mp4 *.avi *.mkv *.mov *.ts);;All (*.*)")
        if p:
            self.ed_file.setText(p)
            self.rb_file.setChecked(True)

    def _open_out(self):
        d = getattr(self.engine, "out_dir", None) if self.engine else None
        d = d if d and Path(d).exists() else Path(self.cfg.out_root)
        Path(d).mkdir(parents=True, exist_ok=True)
        os.startfile(str(d))

    # ------------------------------------------------------------------ 파라미터 탭
    def _tab_params(self):
        inner = QWidget()
        lay = QVBoxLayout(inner)
        h = QHBoxLayout()
        for text, fn in (("프리셋 불러오기", self._load_preset), ("프리셋 저장", self._save_preset),
                         ("기본값으로", self._defaults)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            h.addWidget(b)
        lay.addLayout(h)
        for group, items in PARAMS.items():
            g = QGroupBox(group)
            form = QFormLayout(g)
            for key, name, lo, hi, step, tip in items:
                val = getattr(self.cfg, key)
                if isinstance(val, int) and not isinstance(val, bool):
                    sp = QSpinBox()
                    sp.setRange(int(lo), int(hi))
                    sp.setSingleStep(int(step))
                    sp.setValue(int(val))
                else:
                    sp = QDoubleSpinBox()
                    sp.setDecimals(3 if step < 0.05 else 2)
                    sp.setRange(lo, hi)
                    sp.setSingleStep(step)
                    sp.setValue(float(val))
                sp.setKeyboardTracking(False)
                sp.valueChanged.connect(lambda v, k=key: self._set(k, v))
                if tip:
                    sp.setToolTip(tip)
                lb = QLabel(name)
                lb.setToolTip(tip)
                form.addRow(lb, sp)
                self.widgets[key] = sp
            if group.startswith("마릿수"):
                cm = QComboBox()
                cm.addItems(["max", "sim", "cluster"])
                cm.setCurrentText(self.cfg.count_mode)
                cm.setToolTip("sim=동시 검출 수, cluster=외형 군집 수, max=둘 중 큰 값")
                cm.currentTextChanged.connect(lambda v: self._set("count_mode", v))
                form.addRow(QLabel("마릿수 결정 방식"), cm)
                self.widgets["count_mode"] = cm
            if group.startswith("ID"):
                for key, name in BOOL_PARAMS:
                    cb = QCheckBox(name)
                    cb.setChecked(getattr(self.cfg, key))
                    cb.toggled.connect(lambda v, k=key: self._set(k, v))
                    form.addRow(cb)
                    self.widgets[key] = cb
            lay.addWidget(g)
        lay.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(inner)
        return sc

    def _set(self, key, val):
        if isinstance(getattr(self.cfg, key), int) and not isinstance(getattr(self.cfg, key), bool):
            val = int(val)
        setattr(self.cfg, key, val)
        if self.engine and self.engine.running():
            self.engine.update({key: val})
            note = " → 추적기 재시작" if key in TRACKER_KEYS else (" → '다시 세기' 를 눌러 적용" if key in RECOUNT_KEYS and key != "herd_size" else "")
            self.statusBar().showMessage(f"{key} = {val}{note}", 5000)

    def _refresh_widgets(self):
        for k, w in self.widgets.items():
            v = getattr(self.cfg, k)
            w.blockSignals(True)
            if isinstance(w, QCheckBox):
                w.setChecked(bool(v))
            elif isinstance(w, QComboBox):
                w.setCurrentText(str(v))
            else:
                w.setValue(v)
            w.blockSignals(False)
            if self.engine and self.engine.running():
                self.engine.update({k: getattr(self.cfg, k)})

    def _load_preset(self):
        p, _ = QFileDialog.getOpenFileName(self, "프리셋 불러오기", str(ROOT / "configs"), "JSON (*.json)")
        if p:
            src = self.cfg.source
            self.cfg.__dict__.update(asdict(LiveConfig.load(p)))
            self.cfg.source = src
            self._refresh_widgets()
            self.statusBar().showMessage(f"불러옴: {p}", 5000)

    def _save_preset(self):
        p, _ = QFileDialog.getSaveFileName(self, "프리셋 저장", str(ROOT / "configs" / "live_my.json"), "JSON (*.json)")
        if p:
            self._collect_source()
            self.cfg.save(p)
            self.statusBar().showMessage(f"저장: {p}", 5000)

    def _defaults(self):
        src = self.cfg.source
        self.cfg.__dict__.update(asdict(LiveConfig()))
        self.cfg.source = src
        self._refresh_widgets()

    # ------------------------------------------------------------------ 상태 탭
    def _tab_status(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.lb_count = QLabel("-")
        f = QFont()
        f.setPointSize(13)
        f.setBold(True)
        self.lb_count.setFont(f)
        self.lb_info = QLabel("-")
        self.lb_info.setWordWrap(True)
        lay.addWidget(self.lb_count)
        lay.addWidget(self.lb_info)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["ID", "상태", "속도 m/s", "이동 m", "활동지수", "사라짐(초)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        lay.addWidget(self.table, 3)
        lay.addWidget(QLabel("이벤트 (ID 교정 · 재등장 · 마릿수 변경)"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        lay.addWidget(self.log, 2)
        return w

    # ------------------------------------------------------------------ 실행
    def _collect_source(self):
        if self.rb_file.isChecked():
            self.cfg.source = self.ed_file.text().strip()
        elif self.rb_rtsp.isChecked():
            self.cfg.source = self.ed_rtsp.text().strip()
        else:
            self.cfg.source = str(self.sp_cam.value())
        self.cfg.rtsp_tcp = self.cb_tcp.isChecked()
        self.cfg.save_csv = self.cb_csv.isChecked()
        self.cfg.save_video = self.cb_video.isChecked()

    def _buttons(self, running):
        self.bt_start.setEnabled(not running)
        self.bt_pause.setEnabled(running)
        self.bt_stop.setEnabled(running)
        self.bt_recount.setEnabled(running)
        for wdg in (self.rb_file, self.rb_rtsp, self.rb_cam, self.ed_file, self.ed_rtsp, self.sp_cam, self.cb_tcp,
                    self.cb_csv, self.cb_video):
            wdg.setEnabled(not running)

    def start(self):
        self._collect_source()
        if not self.cfg.source:
            QMessageBox.warning(self, "입력 없음", "영상 파일이나 RTSP 주소를 입력하세요.")
            return
        if self.rb_file.isChecked() and not Path(self.cfg.source).exists():
            QMessageBox.warning(self, "파일 없음", f"파일을 찾을 수 없습니다:\n{self.cfg.source}")
            return
        old = self.engine
        self.engine = LiveEngine(self.cfg, log=self._log)
        if old is not None and old._yolo is not None:       # 모델 재사용 (로딩 시간 절약)
            self.engine._yolo, self.engine._reid = old._yolo, old._reid
            self.engine._classes, self.engine._dev = old._classes, old._dev
            self.engine._yolo.predictor = None
            self.engine._write_tracker_yaml()
        self.engine.start()
        self._buttons(True)
        self.bt_pause.setChecked(False)
        self.tabs.setCurrentIndex(2)
        self.statusBar().showMessage("모델 로딩 / 입력 연결 중…")

    def _pause(self, on):
        if self.engine:
            self.engine.pause(on)
            self.bt_pause.setText("▶ 계속" if on else "❚❚ 일시정지")

    def stop(self):
        if self.engine:
            self.statusBar().showMessage("정지 중… (결과 저장)")
            QApplication.processEvents()
            self.engine.stop()
        self._buttons(False)

    def _log(self, msg):  # 엔진 스레드에서 호출 → 큐에 쌓고 타이머가 표시
        self.log_lines.append(f"{time.strftime('%H:%M:%S')}  {msg}")

    def _tick(self):
        while self.log_lines:
            self.log.appendPlainText(self.log_lines.pop(0))
        eng = self.engine
        if eng is None:
            return
        if eng.error:
            err, eng.error = eng.error, None
            QMessageBox.critical(self, "오류", err[:2000])
        if not eng.running() and self.bt_stop.isEnabled():
            self._buttons(False)
            self.statusBar().showMessage("종료됨")
        lt = eng.latest
        if lt is None or lt is self._last_frame:
            return
        self._last_frame = lt
        canvas, st = lt
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.video.setPixmap(QPixmap.fromImage(img).scaled(self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self._update_status(st)

    def _update_status(self, st):
        if st["phase"] == "enroll":
            self.lb_count.setText(f"마릿수 세는 중…  남은 {st['enroll_left']:.0f}초")
        else:
            self.lb_count.setText(f"마릿수 N = {st['N']}   보임 {st['visible']}   사라짐 {st['N'] - st['visible']}")
        src = "스트림" + (" (연결됨)" if st["connected"] else " (연결 끊김)") if st["stream"] else \
            f"파일 {st['frame']}/{st['n_frames']} 프레임"
        cnt = f"등록 결과: 동시검출 {st['n_sim']} · 외형군집 {st['n_clu']}" if st["phase"] == "track" else ""
        self.lb_info.setText(f"{src}   t={st['t']:.1f}s   처리 {st['fps']:.1f} fps (원본 {st['src_fps']:.0f} fps)"
                             + (f"   버린 프레임 {st['dropped']}" if st["stream"] else "") + f"\n{cnt}")
        ids = st["ids"]
        self.table.setRowCount(len(ids))
        for r, d in enumerate(ids):
            vals = [str(d["id"]), STATE_KO[d["state"]] if d["visible"] else "안 보임", f"{d['speed']:.2f}",
                    f"{d['dist']:.1f}", f"{d['activity']:.0f}", "" if d["visible"] else f"{d['missing_s']:.0f}"]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setTextAlignment(Qt.AlignCenter)
                if not d["visible"]:
                    it.setForeground(Qt.gray)
                self.table.setItem(r, c, it)
        self.statusBar().showMessage("실행 중" if self.engine.running() else "종료됨")

    def closeEvent(self, e):
        if self.engine and self.engine.running():
            self.engine.stop()
        e.accept()


def run_gui(preset=None, source=None):
    app = QApplication.instance() or QApplication([])
    app.setFont(QFont("Malgun Gothic", 9))
    cfg = LiveConfig.load(preset) if preset else LiveConfig()
    if source:
        cfg.source = source
    win = MainWindow(cfg, preset)
    win.show()
    app.exec()
