"""Global settings (Hanwoo activity pipeline)."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "data"


@dataclass
class PipelineConfig:
    # ---- Detection ----
    weights: str = str(MODELS_DIR / "hanwoo_yolo11s.pt")  # trained model (fallback used if missing)
    fallback_weights: str = "yolo11x.pt"                   # COCO-pretrained model
    coco_cow_class: int = 19                                # COCO 'cow' class index
    imgsz: int = 960
    conf: float = 0.25
    iou: float = 0.6
    device: str = "0"

    # ---- Tracking ----
    tracker_cfg: str = str(ROOT / "configs" / "hanwoo_botsort.yaml")
    min_track_frames: int = 12          # tracklets shorter than this are dropped as noise (0.5 s)

    # ---- Offline ID linking (tracklet stitching) ----
    stitch_max_gap_s: float = 20.0      # max allowed gap duration
    stitch_max_dist_bl: float = 2.5     # max distance between break points (body lengths, grows with time)
    stitch_speed_bl_s: float = 0.6      # max speed possible during the gap (body lengths/s)
    stitch_app_weight: float = 0.5      # appearance similarity weight
    stitch_max_cost: float = 0.85
    # 2nd pass: cows that vanish for a long time (up to 5 min) and reappear near the same spot (lying, behind a pillar, in a cluster)
    stitch2_max_gap_s: float = 300.0
    stitch2_max_dist_bl: float = 3.0    # fixed radius regardless of time
    stitch2_speed_bl_s: float = 0.0
    stitch2_app_weight: float = 0.6     # higher appearance weight
    stitch2_max_cost: float = 0.55      # stricter threshold
    interp_max_gap_s: float = 0.5       # max interpolation (display-only) gap - fills only momentary missed detections. Longer gaps
                                        # stay 'not visible' so no virtual boxes or fake paths are created

    # ---- Detection zones (remove false positives outside the barn / blurry background, hanwoo/zones.py) ----
    zones_file: str = ""                # empty = configs/zones/<video name>.json → default.json automatically
    min_size_ratio: float = 0.0         # lower bound of box height / expected cow height (0 = off)

    # ---- Fixed-herd ID assignment (ReID + herd-size constraint) ----
    reid_weights: str = str(MODELS_DIR / "hanwoo_reid.pt")  # if missing, only the two-stage linking is used
    use_identity: bool = True
    herd_size: int = 0                  # herd size N (0 = estimate from simultaneous detections)
    reid_every: int = 4                 # ReID embedding interval (frames)
    split_window_s: float = 4.0         # swap check: comparison window before/after an overlap (s)
    split_sim_thr: float = 0.55         # split if cosine similarity of before/after mean embeddings is below this
    id_radius_bl: float = 2.0           # motion gate: base radius (body lengths)
    id_speed_bl_s: float = 0.5          # motion gate: extra allowed distance per second of gap (body lengths)
    id_gate: float = 1.5                # no assignment beyond allowed distance × gate
    id_app_weight: float = 0.7          # appearance share of the cost
    id_new_cost: float = 0.45           # while slots < N, start a new animal if the cost is worse than this
    id_overlap_tol_s: float = 0.5       # allowed overlap at tracklet boundaries (s)
    id_refine_iters: int = 5

    # ---- Activity ----
    cow_height_m: float = 1.30          # Hanwoo height (m) - reference length for perspective correction
    smooth_window_s: float = 1.0        # position smoothing window (s)
    rest_speed: float = 0.05            # at or below (m/s): resting / standing still
    walk_speed: float = 0.20            # at or above (m/s): walking
    motion_energy_active: float = 2.5   # even when stationary, a large mean frame difference in the bbox = local activity (head, feeding)
    motion_norm: float = 5.0            # motion-energy normalisation cap for the activity index
    max_speed: float = 2.0              # physically plausible max (m/s) - suppresses bbox jumps caused by occlusion

    # ---- Output ----
    panel_width: int = 380
    trail_len_s: float = 3.0
    write_video: bool = True
    max_frames: int = 0                 # 0 = all


STATE_NAMES = {0: "Rest", 1: "Active", 2: "Walk"}
STATE_KO = {0: "Resting", 1: "Active (local)", 2: "Walking"}  # display labels (name kept for compatibility)
