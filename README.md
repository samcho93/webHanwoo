# 한우 개체 추적 · 활동량 측정 (webHanwoo)

고정 카메라 축사 영상에서 한우 개체별 ID 를 부여하고 이동거리·속도·행동상태·활동지수를 측정,
ID 표시 결과 영상과 HTML 대시보드를 출력합니다.

```
검출(YOLO11s 미세조정) → 온라인 추적(BoT-SORT+ReID) → 오프라인 tracklet 정합 → 활동량(원근보정) → 영상+대시보드
```

## 빠른 시작 (Windows)
1. `setup.bat` 실행 → 가상환경·PyTorch(CUDA)·패키지 설치
2. 모델 가중치를 [Releases](../../releases) 에서 받아 `models/` 에 넣기
   (`hanwoo_yolo11s.pt` 검출, `hanwoo_reid.pt` 개체 재식별)
3. `run.bat` 실행 → 메뉴에서 선택
   - 메뉴 없이: `run.bat all E:\Video\cow 15` (폴더, 마릿수) / `run.bat reuse` / `run.bat serve` / `run.bat build`

## 설치 (수동)
```bash
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
FFmpeg 가 PATH 에 있어야 합니다. 자세한 과정은 강좌 `site/01_env.html`.

## 실행
```bash
python run_activity.py E:/Video/cow            # 폴더 전체
python run_activity.py some_video.mp4          # 단일 영상
```
결과: `outputs/<영상>/{<영상>_tracked.mp4, dashboard.html, summary.csv, tracks_activity.csv, result.json}`, 전체 요약 `outputs/index.html`

## 학습 (새 축사에 적용할 때)
```bash
python scripts/prepare_dataset.py E:/Video/cow --teacher models/yolo11x.pt --val cow02 cow09
python scripts/train.py --epochs 60
```

## ReID (고정 개체군 ID 유지)
```bash
python scripts/prepare_reid.py E:/Video/cow      # 추적 결과 → 자기지도 crop 데이터
python scripts/train_reid.py --iters 3000         # → models/hanwoo_reid.pt
python run_activity.py E:/Video/cow --reuse --herd-size 15
python scripts/identity_sheet.py outputs/cow03    # ID 일관성 확인용 콘택트 시트
```
`models/hanwoo_reid.pt` 가 있으면 자동으로 사용합니다 (`--no-reid` 로 끔).

## 실시간 추적 프로그램 (파일 / RTSP 카메라)
`live.bat` (또는 `run.bat` 메뉴 7) → 창에서 입력을 고르고 [시작].
- **입력**: 영상 파일, `rtsp://사용자:비밀번호@IP:554/경로`, 웹캠 번호
- **마릿수 카운트**: 처음 `등록 시간`(기본 60초) 동안 동시 검출 수 + ReID 외형 군집으로 N 을 정하고 소마다 외형을 등록
  (마릿수를 알면 `마릿수 N` 에 직접 입력하는 것이 가장 정확)
- **실시간 ID**: N 개 개체 슬롯 안에서만 ID 배정 (외형 + 이동거리, 헝가리안), 겹친 뒤 뒤바뀜 교정, 확실한 새 소가 계속 보이면 N 자동 증가
- **파라미터 탭**: 검출·ID 배정·활동량은 즉시 반영, 추적기 설정은 추적기만 재시작(ID 유지), 카운트 설정은 [마릿수 다시 세기]
- **저장**: `outputs/live/<이름>_<시각>/` 에 `tracks_live.csv`, `summary_live.csv`, `events.log`, 선택 시 결과 영상
```bash
python live_app.py                                              # GUI
python live_app.py --headless --source rtsp://... --save-video  # 화면 없이 (서버용)
python scripts/identity_sheet.py outputs/live/cow03_...          # 실시간 결과 ID 확인
```

## 강좌 사이트
```bash
python scripts/make_figures.py
python scripts/build_site.py
python -m http.server 8000      # → http://localhost:8000/site/
```
