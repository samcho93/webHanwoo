# Hanwoo Individual Tracking & Activity Measurement (webHanwoo)

Assigns a per-animal ID to Hanwoo cattle in fixed-camera barn video and measures distance travelled, speed,
behaviour state and an activity index, producing an ID-annotated video and an HTML dashboard.

```
Detection (fine-tuned YOLO11s) → online tracking (BoT-SORT+ReID) → offline tracklet linking → activity (perspective-corrected) → video + dashboard
```

## Quick start (Windows)
1. Run `setup.bat` → installs the virtual environment, PyTorch (CUDA) and packages
2. Download the model weights from [Releases](../../releases) into `models/`
   (`hanwoo_yolo11s.pt` detection, `hanwoo_reid.pt` individual re-identification)
3. Run `run.bat` → pick from the menu
   - Without the menu: `run.bat all E:\Video\cow 15` (folder, herd size) / `run.bat reuse` / `run.bat serve` / `run.bat build`

## Installation (manual)
```bash
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
FFmpeg must be on PATH. See the course page `site/01_env.html` for details.

## Running
```bash
python run_activity.py E:/Video/cow            # whole folder
python run_activity.py some_video.mp4          # single video
```
Output: `outputs/<video>/{<video>_tracked.mp4, dashboard.html, summary.csv, tracks_activity.csv, result.json}`, overall summary `outputs/index.html`

## Training (for a new barn)
```bash
python scripts/prepare_dataset.py E:/Video/cow --teacher models/yolo11x.pt --val cow02 cow09
python scripts/train.py --epochs 60
```

## ReID (stable IDs for a fixed herd)
```bash
python scripts/prepare_reid.py E:/Video/cow      # tracking results → self-supervised crop dataset
python scripts/train_reid.py --iters 3000         # → models/hanwoo_reid.pt
python run_activity.py E:/Video/cow --reuse --herd-size 15
python scripts/identity_sheet.py outputs/cow03    # contact sheet for checking ID consistency
```
`models/hanwoo_reid.pt` is used automatically when present (disable with `--no-reid`).

## Real-time tracking app (file / RTSP camera)
`live.bat` (or `run.bat` menu item 7) → choose an input in the window and press [Start].
- **Input**: a video file, `rtsp://user:password@IP:554/path`, or a webcam index
- **Herd counting**: during the initial enrollment time (default 60 s), N is set from the number of simultaneous detections plus ReID appearance clustering, and each cow's appearance is enrolled
  (if you know the herd size, entering it directly as herd size N is most accurate)
- **Real-time IDs**: IDs are assigned only within the N animal slots (appearance + distance, Hungarian matching), swaps after overlaps are corrected, and N grows automatically when a clearly new cow stays visible
- **Parameters tab**: detection, ID assignment and activity settings apply immediately; tracker settings restart only the tracker (IDs are kept); counting settings need [Recount herd]
- **Detection zones / static false positives**: draw exclude (red) and include (green) polygons with [Edit zones…]; boxes are filtered by their foot point.
  Saved to `configs/zones/<video name>.json` (cameras: `camera_<IP>.json`) and also used automatically by the offline `run_activity.py`.
  Small boxes that stay in the same place for 120 s or more (feed troughs and other background objects) are reported as candidates and can be added with [Exclude static false positives]
  (for offline results: `python scripts/find_static_boxes.py outputs/cow03 --apply`)
- **Saving**: `outputs/live/<name>_<timestamp>/` receives `tracks_live.csv`, `summary_live.csv`, `events.log`, and optionally the result video
```bash
python live_app.py                                              # GUI
python live_app.py --headless --source rtsp://... --save-video  # no display (server use)
python scripts/identity_sheet.py outputs/live/cow03_...          # check IDs of a real-time result
```

## Evaluation against ground truth
The proxy numbers (ID count, excluded detections) cannot show whether IDs are *correct*, so a few segments are
labeled by hand and every method variant is scored with IDF1 / ID switches / HOTA.
```bash
python scripts/ablation.py                       # variants → outputs/ablation/<video>/<variant>.csv (uses caches)
python scripts/ablation.py --wa 0.3 0.5 0.7 0.9  # appearance-weight sensitivity
python scripts/gt_label.py cow03 --start 0 --end 300   # label which cow each tracklet is → gt/cow03.json, gt/cow03_gt.csv
python scripts/gt_label.py cow09                       # whole video
python scripts/eval_gt.py                        # → gt/eval_results.csv
python scripts/eval_gt.py cow03 --online cow03=outputs/live/cow03_.../tracks_live.csv
```
Labeling is done per raw tracker tracklet (no suggestions from the proposed method are shown, to avoid bias):
`Space` next tracklet, `Enter` assign to the selected cow, `N` new cow, `F` not a cow / outside the analysed pen,
`S` split at the current frame (when a tracklet jumps between cows), `←/→` ±1 s.

## Course site
```bash
python scripts/make_figures.py
python scripts/build_site.py
python -m http.server 8000      # → http://localhost:8000/site/
```
