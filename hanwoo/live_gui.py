"""PySide6 GUI: input selection · live parameter tuning · video/status display."""
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
from .zones import ZONES_DIR, load_zones, zone_stem, zones_path_for

# (key, label, min, max, step, tooltip)   — per group
PARAMS = {
    "Detection (applied immediately)": [
        ("conf", "Min detection confidence", 0.05, 0.95, 0.05, "Lower catches occluded cattle but adds false positives"),
        ("iou", "NMS IoU", 0.3, 0.9, 0.05, "Threshold for removing overlapping boxes"),
        ("imgsz", "Input size (px)", 320, 1600, 32, "Larger finds small cattle better but runs slower"),
        ("reid_every", "ReID interval (frames)", 1, 12, 1, "Smaller gives more appearance info but runs slower"),
        ("min_size_ratio", "Min size ratio (0=off)", 0, 0.9, 0.05,
         "Box height / expected body height at that position. Removes small false positives in the far background (may also drop occluded cattle; 0.3–0.4 recommended)"),
    ],
    "Tracker BoT-SORT (change restarts tracker)": [
        ("track_high_thresh", "1st-stage match confidence", 0.1, 0.9, 0.05, ""),
        ("track_low_thresh", "2nd-stage match confidence", 0.01, 0.5, 0.01, ""),
        ("new_track_thresh", "New track confidence", 0.1, 0.95, 0.05, "Higher reduces spurious new IDs"),
        ("track_buffer", "Track memory (frames)", 10, 600, 10, "How long a track is kept while occluded"),
        ("match_thresh", "Match threshold", 0.5, 0.99, 0.01, ""),
    ],
    "Herd count (change requires 'Recount')": [
        ("herd_size", "Herd size N (0=auto)", 0, 200, 1, "Entering it directly is most accurate if known"),
        ("enroll_s", "Enrollment time (s)", 5, 600, 5, "Count and enroll appearances during this initial period"),
        ("count_conf", "Count confidence", 0.1, 0.95, 0.05, ""),
        ("count_stable_s", "Hold time (s)", 0.2, 30, 0.2, "Simultaneous detection count must hold this long to be accepted"),
        ("cluster_sim", "Appearance cluster similarity", 0.2, 0.95, 0.05, "Lower gives fewer clusters"),
        ("count_min_track_s", "Min cluster track length (s)", 0.5, 30, 0.5, ""),
    ],
    "ID assignment (applied immediately)": [
        ("confirm_s", "ID confirm delay (s)", 0.2, 5, 0.1, "Appearance collection time for new tracks = display delay"),
        ("release_s", "Missing after (s)", 0.2, 10, 0.1, ""),
        ("id_app_weight", "Appearance weight", 0, 1, 0.05, "The rest is the motion-distance weight"),
        ("id_radius_bl", "Base motion radius (body lengths)", 0.5, 10, 0.25, ""),
        ("id_speed_bl_s", "Extra radius per second (body lengths)", 0, 3, 0.05, "Grows with time missing"),
        ("id_gate", "Motion gate multiplier", 1, 5, 0.1, ""),
        ("id_max_cost", "Max assignment cost", 0.2, 1.5, 0.05, "Above this the box waits as '?'"),
        ("swap_margin", "Swap correction margin", 0, 1, 0.02, "Smaller corrects more often"),
        ("drift_margin", "Appearance correction margin", 0, 1, 0.02, ""),
        ("grow_s", "N increase delay (s)", 1, 60, 1, ""),
    ],
    "Static false-positive detection (background objects)": [
        ("static_min_s", "Static time (s)", 10, 3600, 10, "Boxes staying in place at least this long become candidates"),
        ("static_max_ratio", "Max size ratio", 0.1, 1.0, 0.05, "Box height / expected body height must be below this to be a candidate (protects lying cattle)"),
    ],
    "Activity (applied immediately)": [
        ("cow_height_m", "Body height (m)", 0.8, 2.0, 0.05, "Reference for perspective correction"),
        ("rest_speed", "Max resting speed (m/s)", 0.01, 0.5, 0.01, ""),
        ("walk_speed", "Min walking speed (m/s)", 0.05, 1.5, 0.01, ""),
        ("motion_energy_active", "Local activity motion", 0.5, 20, 0.5, ""),
        ("smooth_window_s", "Smoothing window (s)", 0.2, 5, 0.1, ""),
        ("trail_s", "Trail length (s)", 0, 30, 0.5, ""),
    ],
}
BOOL_PARAMS = [("auto_grow", "Auto-increase N (when a clearly new animal keeps appearing)"),
               ("show_pending", "Show boxes awaiting ID (?)"),
               ("show_zones", "Show detection zones (exclude=red, include=green)"),
               ("static_auto", "Exclude static false-positive candidates automatically without confirmation")]


class MainWindow(QMainWindow):
    def __init__(self, cfg: LiveConfig, preset_path=None):
        super().__init__()
        self.cfg = cfg
        self.preset_path = preset_path
        self.engine: LiveEngine | None = None
        self.widgets = {}
        self.log_lines = []
        self.setWindowTitle("Hanwoo Live Tracking (webHanwoo Live)")
        self.resize(1500, 860)

        self.video = QLabel("Choose an input and press [Start]")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(640, 360)
        self.video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.video.setStyleSheet("background:#111;color:#aaa;font-size:16px")

        tabs = QTabWidget()
        tabs.addTab(self._tab_input(), "Input")
        tabs.addTab(self._tab_params(), "Parameters")
        tabs.addTab(self._tab_status(), "Status")
        self.tabs = tabs

        split = QSplitter()
        split.addWidget(self.video)
        split.addWidget(tabs)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setSizes([1050, 450])
        self.setCentralWidget(split)
        self.statusBar().showMessage("Ready")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)
        self._last_frame = None

    # ------------------------------------------------------------------ Input tab
    def _tab_input(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        g = QGroupBox("Input source")
        f = QVBoxLayout(g)
        self.rb_file, self.rb_rtsp, self.rb_cam = QRadioButton("Video file"), QRadioButton("RTSP camera"), QRadioButton("Webcam")
        bg = QButtonGroup(self)
        for rb in (self.rb_file, self.rb_rtsp, self.rb_cam):
            bg.addButton(rb)
        src = self.cfg.source
        h = QHBoxLayout()
        self.ed_file = QLineEdit(src if not src.lower().startswith("rtsp") and not src.isdigit() else "")
        b = QPushButton("Browse…")
        b.clicked.connect(self._browse)
        h.addWidget(self.ed_file)
        h.addWidget(b)
        self.ed_rtsp = QLineEdit(src if src.lower().startswith("rtsp") else "")
        self.ed_rtsp.setPlaceholderText("rtsp://user:password@192.168.0.10:554/stream1")
        self.sp_cam = QSpinBox()
        self.sp_cam.setRange(0, 9)
        self.cb_tcp = QCheckBox("Receive RTSP over TCP (fewer drops/artifacts)")
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

        gz = QGroupBox("Detection zones (remove false positives outside the pen / in blurred background)")
        fz = QVBoxLayout(gz)
        self.lb_zone = QLabel("-")
        self.lb_zone.setWordWrap(True)
        bz = QPushButton("Edit zones…  (draw polygons)")
        bz.clicked.connect(self._edit_zones)
        self.bt_static = QPushButton("Exclude static false positives (candidates 0)")
        self.bt_static.setToolTip("Add small boxes that stay in one place for a long time (troughs, feed bins and other background objects) to the exclude zones")
        self.bt_static.clicked.connect(self._apply_static)
        self.bt_static.setEnabled(False)
        fz.addWidget(self.lb_zone)
        fz.addWidget(bz)
        fz.addWidget(self.bt_static)
        lay.addWidget(gz)
        for wdg in (self.ed_file, self.ed_rtsp):
            wdg.textChanged.connect(lambda _t: self._zone_label())
        for rb in (self.rb_file, self.rb_rtsp, self.rb_cam):
            rb.toggled.connect(lambda _v: self._zone_label())

        g2 = QGroupBox("Run options")
        f2 = QVBoxLayout(g2)
        self.cb_realtime = QCheckBox("Play files at original speed (off = process as fast as possible)")
        self.cb_realtime.setChecked(self.cfg.realtime)
        self.cb_realtime.toggled.connect(lambda v: self._set("realtime", v))
        self.cb_loop = QCheckBox("Loop file playback")
        self.cb_loop.setChecked(self.cfg.loop_file)
        self.cb_loop.toggled.connect(lambda v: self._set("loop_file", v))
        self.cb_csv = QCheckBox("Save CSV (per-frame IDs/activity, summary)")
        self.cb_csv.setChecked(self.cfg.save_csv)
        self.cb_video = QCheckBox("Save result video (mp4)")
        self.cb_video.setChecked(self.cfg.save_video)
        for c in (self.cb_realtime, self.cb_loop, self.cb_csv, self.cb_video):
            f2.addWidget(c)
        lay.addWidget(g2)

        h2 = QHBoxLayout()
        self.bt_start = QPushButton("▶ Start")
        self.bt_pause = QPushButton("❚❚ Pause")
        self.bt_stop = QPushButton("■ Stop")
        self.bt_pause.setCheckable(True)
        self.bt_start.clicked.connect(self.start)
        self.bt_pause.toggled.connect(self._pause)
        self.bt_stop.clicked.connect(self.stop)
        for b in (self.bt_start, self.bt_pause, self.bt_stop):
            b.setMinimumHeight(36)
            h2.addWidget(b)
        lay.addLayout(h2)
        self.bt_recount = QPushButton("↻ Recount herd (restart enrollment)")
        self.bt_recount.clicked.connect(lambda: self.engine and self.engine.recount())
        lay.addWidget(self.bt_recount)
        self.bt_out = QPushButton("Open output folder")
        self.bt_out.clicked.connect(self._open_out)
        lay.addWidget(self.bt_out)
        lay.addStretch(1)
        self._buttons(False)
        self._zone_label()
        return w

    def _zone_path(self):
        self._collect_source()
        if self.cfg.zones_file:
            return Path(self.cfg.zones_file)
        return zones_path_for(self.cfg.source) or (ZONES_DIR / f"{zone_stem(self.cfg.source)}.json")

    def _zone_label(self):
        if not hasattr(self, "lb_zone"):
            return
        p = self._zone_path()
        z = load_zones(p) if p.exists() else dict(include=[], exclude=[])
        self.lb_zone.setText(f"{p.name}: {len(z['exclude'])} exclude, {len(z['include'])} include"
                             if p.exists() else f"No zones (saved as {p.name})")

    def _apply_static(self):
        if self.engine and self.engine.running():
            self.engine.update({"static_apply": True})
            self.statusBar().showMessage("Adding static false-positive candidates to the exclude zones", 5000)
            QTimer.singleShot(1500, self._zone_label)

    def _edit_zones(self):
        from .zone_editor import ZoneDialog, grab_frame

        path = self._zone_path()
        eng = self.engine
        if eng is not None and eng.running() and eng.raw_frame is not None:
            frame = eng.raw_frame.copy()
            boxes = [tuple(b) for b in getattr(eng, "last_boxes", [])]
        else:
            if not self.cfg.source:
                QMessageBox.warning(self, "No input", "Enter a video file or RTSP address first.")
                return
            frame, boxes = grab_frame(self.cfg.source), []
        dlg = ZoneDialog(frame, path, boxes, self)
        if dlg.exec():
            if dlg.path != path:
                self._set("zones_file", str(dlg.path))
            if self.engine and self.engine.running():
                self.engine.update({"zones_reload": True})
                self.statusBar().showMessage("Detection zones applied (from the next frame)", 5000)
            self._zone_label()

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select video", str(Path(self.ed_file.text()).parent if self.ed_file.text() else ROOT),
                                           "Video (*.mp4 *.avi *.mkv *.mov *.ts);;All (*.*)")
        if p:
            self.ed_file.setText(p)
            self.rb_file.setChecked(True)

    def _open_out(self):
        d = getattr(self.engine, "out_dir", None) if self.engine else None
        d = d if d and Path(d).exists() else Path(self.cfg.out_root)
        Path(d).mkdir(parents=True, exist_ok=True)
        os.startfile(str(d))

    # ------------------------------------------------------------------ Parameters tab
    def _tab_params(self):
        inner = QWidget()
        lay = QVBoxLayout(inner)
        h = QHBoxLayout()
        for text, fn in (("Load preset", self._load_preset), ("Save preset", self._save_preset),
                         ("Reset to defaults", self._defaults)):
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
            if group.startswith("Herd count"):
                cm = QComboBox()
                cm.addItems(["max", "sim", "cluster"])
                cm.setCurrentText(self.cfg.count_mode)
                cm.setToolTip("sim=simultaneous detections, cluster=appearance clusters, max=larger of the two")
                cm.currentTextChanged.connect(lambda v: self._set("count_mode", v))
                form.addRow(QLabel("Herd count mode"), cm)
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
            note = " → tracker restarted" if key in TRACKER_KEYS else (" → press 'Recount' to apply" if key in RECOUNT_KEYS and key != "herd_size" else "")
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
        p, _ = QFileDialog.getOpenFileName(self, "Load preset", str(ROOT / "configs"), "JSON (*.json)")
        if p:
            src = self.cfg.source
            self.cfg.__dict__.update(asdict(LiveConfig.load(p)))
            self.cfg.source = src
            self._refresh_widgets()
            self.statusBar().showMessage(f"Loaded: {p}", 5000)

    def _save_preset(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save preset", str(ROOT / "configs" / "live_my.json"), "JSON (*.json)")
        if p:
            self._collect_source()
            self.cfg.save(p)
            self.statusBar().showMessage(f"Saved: {p}", 5000)

    def _defaults(self):
        src = self.cfg.source
        self.cfg.__dict__.update(asdict(LiveConfig()))
        self.cfg.source = src
        self._refresh_widgets()

    # ------------------------------------------------------------------ Status tab
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
        self.table.setHorizontalHeaderLabels(["ID", "State", "Speed m/s", "Distance m", "Activity index", "Missing (s)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        lay.addWidget(self.table, 3)
        lay.addWidget(QLabel("Events (ID corrections · re-appearances · herd size changes)"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        lay.addWidget(self.log, 2)
        return w

    # ------------------------------------------------------------------ Run
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
            QMessageBox.warning(self, "No input", "Enter a video file or RTSP address.")
            return
        if self.rb_file.isChecked() and not Path(self.cfg.source).exists():
            QMessageBox.warning(self, "File not found", f"Cannot find file:\n{self.cfg.source}")
            return
        old = self.engine
        self.engine = LiveEngine(self.cfg, log=self._log)
        if old is not None and old._yolo is not None:       # reuse models (saves loading time)
            self.engine._yolo, self.engine._reid = old._yolo, old._reid
            self.engine._classes, self.engine._dev = old._classes, old._dev
            self.engine._yolo.predictor = None
            self.engine._write_tracker_yaml()
        self.engine.start()
        self._buttons(True)
        self.bt_pause.setChecked(False)
        self.tabs.setCurrentIndex(2)
        self.statusBar().showMessage("Loading models / connecting to input…")

    def _pause(self, on):
        if self.engine:
            self.engine.pause(on)
            self.bt_pause.setText("▶ Resume" if on else "❚❚ Pause")

    def stop(self):
        if self.engine:
            self.statusBar().showMessage("Stopping… (saving results)")
            QApplication.processEvents()
            self.engine.stop()
        self._buttons(False)

    def _log(self, msg):  # called from the engine thread → queued, shown by the timer
        self.log_lines.append(f"{time.strftime('%H:%M:%S')}  {msg}")

    def _tick(self):
        while self.log_lines:
            self.log.appendPlainText(self.log_lines.pop(0))
        eng = self.engine
        if eng is None:
            return
        if eng.error:
            err, eng.error = eng.error, None
            QMessageBox.critical(self, "Error", err[:2000])
        if not eng.running() and self.bt_stop.isEnabled():
            self._buttons(False)
            self.statusBar().showMessage("Finished")
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
            self.lb_count.setText(f"Counting herd…  {st['enroll_left']:.0f}s left")
        else:
            self.lb_count.setText(f"Herd N = {st['N']}   visible {st['visible']}   missing {st['N'] - st['visible']}")
        src = "Stream" + (" (connected)" if st["connected"] else " (disconnected)") if st["stream"] else \
            f"File frame {st['frame']}/{st['n_frames']}"
        cnt = f"Enrollment: simultaneous {st['n_sim']} · appearance clusters {st['n_clu']}" if st["phase"] == "track" else ""
        self.lb_info.setText(f"{src}   t={st['t']:.1f}s   processing {st['fps']:.1f} fps (source {st['src_fps']:.0f} fps)"
                             + (f"   dropped frames {st['dropped']}" if st["stream"] else "")
                             + f"\nBoxes removed by zone/size filters {st.get('filtered', 0)}   {cnt}")
        n_st = st.get("static", 0)
        self.bt_static.setText(f"Exclude static false positives (candidates {n_st})")
        self.bt_static.setEnabled(n_st > 0)
        ids = st["ids"]
        self.table.setRowCount(len(ids))
        for r, d in enumerate(ids):
            vals = [str(d["id"]), STATE_KO[d["state"]] if d["visible"] else "Not visible", f"{d['speed']:.2f}",
                    f"{d['dist']:.1f}", f"{d['activity']:.0f}", "" if d["visible"] else f"{d['missing_s']:.0f}"]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setTextAlignment(Qt.AlignCenter)
                if not d["visible"]:
                    it.setForeground(Qt.gray)
                self.table.setItem(r, c, it)
        self.statusBar().showMessage("Running" if self.engine.running() else "Finished")

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
