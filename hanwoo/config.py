"""전역 설정값 (Hanwoo activity pipeline)."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "data"


@dataclass
class PipelineConfig:
    # ---- 검출 (Detection) ----
    weights: str = str(MODELS_DIR / "hanwoo_yolo11s.pt")  # 학습된 모델 (없으면 fallback 사용)
    fallback_weights: str = "yolo11x.pt"                   # COCO 사전학습 모델
    coco_cow_class: int = 19                                # COCO 'cow' 클래스 번호
    imgsz: int = 960
    conf: float = 0.25
    iou: float = 0.6
    device: str = "0"

    # ---- 추적 (Tracking) ----
    tracker_cfg: str = str(ROOT / "configs" / "hanwoo_botsort.yaml")
    min_track_frames: int = 12          # 이보다 짧은 tracklet 은 노이즈로 제거 (0.5초)

    # ---- 오프라인 ID 정합 (Tracklet stitching) ----
    stitch_max_gap_s: float = 20.0      # 끊긴 구간 최대 허용 시간
    stitch_max_dist_bl: float = 2.5     # 끊긴 위치 사이 최대 거리 (몸길이 단위, 시간에 따라 증가)
    stitch_speed_bl_s: float = 0.6      # 끊긴 동안 이동 가능한 속도 상한(몸길이/초)
    stitch_app_weight: float = 0.5      # 외형 유사도 가중치
    stitch_max_cost: float = 0.85
    # 2차 정합: 장시간(최대 5분) 사라졌다 '같은 자리 근처'에서 다시 나타난 소 (누워 있음·기둥 뒤·군집)
    stitch2_max_gap_s: float = 300.0
    stitch2_max_dist_bl: float = 3.0    # 시간과 무관하게 고정 반경
    stitch2_speed_bl_s: float = 0.0
    stitch2_app_weight: float = 0.6     # 외형 비중을 높임
    stitch2_max_cost: float = 0.55      # 더 엄격한 임계값
    interp_max_gap_s: float = 0.5       # 보간(표시용) 최대 구간 — 순간적인 검출 누락만 메움. 더 긴 공백은
                                        # '보이지 않음'으로 두어 가상 박스·가짜 이동경로가 생기지 않게 한다

    # ---- 고정 개체군 ID 배정 (ReID + 마릿수 제약) ----
    reid_weights: str = str(MODELS_DIR / "hanwoo_reid.pt")  # 없으면 기존 2단계 정합만 사용
    use_identity: bool = True
    herd_size: int = 0                  # 축사 마릿수 N (0 = 동시 검출 수로 자동 추정)
    reid_every: int = 4                 # ReID 임베딩 추출 간격(프레임)
    split_window_s: float = 4.0         # 스왑 검사: 겹침 전후 비교 구간(초)
    split_sim_thr: float = 0.55         # 전후 평균 임베딩 코사인 유사도가 이보다 낮으면 분리
    id_radius_bl: float = 2.0           # 움직임 게이트: 기본 반경(몸길이)
    id_speed_bl_s: float = 0.5          # 움직임 게이트: 공백 1초당 추가 이동 가능 거리(몸길이)
    id_gate: float = 1.5                # 허용거리 × gate 초과 시 배정 불가
    id_app_weight: float = 0.7          # 비용 중 외형 비중
    id_new_cost: float = 0.45           # 슬롯 수 < N 일 때, 이 비용보다 나쁘면 새 개체로 시작
    id_overlap_tol_s: float = 0.5       # tracklet 경계 겹침 허용(초)
    id_refine_iters: int = 5

    # ---- 활동량 (Activity) ----
    cow_height_m: float = 1.30          # 한우 체고(m) - 원근 보정용 기준 길이
    smooth_window_s: float = 1.0        # 위치 평활화 창 (초)
    rest_speed: float = 0.05            # m/s 이하: 정지 (Resting/Standing still)
    walk_speed: float = 0.20            # m/s 이상: 보행 (Walking)
    motion_energy_active: float = 2.5   # 정지 상태여도 bbox 내부 프레임차 평균이 크면 국소활동(머리·섭식)
    motion_norm: float = 5.0            # 활동지수 계산 시 motion energy 정규화 상한
    max_speed: float = 2.0              # 물리적으로 가능한 상한(m/s) - 가림에 의한 bbox 급변 억제

    # ---- 출력 ----
    panel_width: int = 380
    trail_len_s: float = 3.0
    write_video: bool = True
    max_frames: int = 0                 # 0 = 전체


STATE_NAMES = {0: "Rest", 1: "Active", 2: "Walk"}
STATE_KO = {0: "휴식", 1: "국소활동", 2: "보행"}
