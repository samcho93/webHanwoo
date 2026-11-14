"""Hanwoo re-identification (ReID) model: cow crop → 256-d L2-normalised embedding.

- Backbone: ImageNet-pretrained ResNet50 with last stride 1 (keeps spatial resolution for small cow crops)
- Head: GAP → BN → Linear(256) → L2 normalisation
- Training: scripts/train_reid.py (tracking results used as self-supervised labels, Supervised-Contrastive loss)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

CROP = 128                       # input size (square letterbox)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class CowReID(nn.Module):
    def __init__(self, dim: int = 256, pretrained: bool = True):
        super().__init__()
        import torchvision

        w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        r = torchvision.models.resnet50(weights=w)
        # last stride 1 (BoT trick): 4x4 → 8x8 feature map at 128px input
        r.layer4[0].conv2.stride = (1, 1)
        r.layer4[0].downsample[0].stride = (1, 1)
        self.backbone = nn.Sequential(*list(r.children())[:-2])
        self.bn = nn.BatchNorm1d(2048)
        self.fc = nn.Linear(2048, dim)

    def forward(self, x):
        f = self.backbone(x)
        f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return F.normalize(self.fc(self.bn(f)), dim=1)


# --------------------------------------------------------------------------- crop
def crop_box(frame: np.ndarray, box, size: int = CROP, pad: float = 0.08) -> np.ndarray:
    """Crop the bbox slightly enlarged and place it, aspect ratio kept, on a size×size grey canvas (letterbox)."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1, x2 = int(max(0, x1 - pad * w)), int(min(W, x2 + pad * w))
    y1, y2 = int(max(0, y1 - pad * h)), int(min(H, y2 + pad * h))
    roi = frame[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)]
    rh, rw = roi.shape[:2]
    s = size / max(rh, rw)
    nw, nh = max(1, int(round(rw * s))), max(1, int(round(rh * s)))
    roi = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size, 3), 114, np.uint8)
    ox, oy = (size - nw) // 2, (size - nh) // 2
    out[oy:oy + nh, ox:ox + nw] = roi
    return out


def to_tensor(crops_bgr: np.ndarray) -> torch.Tensor:
    """(N,H,W,3) uint8 BGR → normalised (N,3,H,W) float tensor"""
    x = crops_bgr[..., ::-1].astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))


def max_iou_others(df: pd.DataFrame) -> np.ndarray:
    """Max IoU of each detection box with other boxes in the same frame (occlusion indicator)."""
    out = np.zeros(len(df))
    for _, idx in df.groupby("frame").indices.items():
        b = df[["x1", "y1", "x2", "y2"]].to_numpy()[idx]
        if len(b) < 2:
            continue
        x1 = np.maximum(b[:, None, 0], b[None, :, 0]); y1 = np.maximum(b[:, None, 1], b[None, :, 1])
        x2 = np.minimum(b[:, None, 2], b[None, :, 2]); y2 = np.minimum(b[:, None, 3], b[None, :, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        iou = inter / (area[:, None] + area[None, :] - inter + 1e-6)
        np.fill_diagonal(iou, 0)
        out[idx] = iou.max(1)
    return out


# --------------------------------------------------------------------------- inference
def load_reid(weights: str | Path, device: str = "cuda") -> CowReID:
    m = CowReID(pretrained=False)
    m.load_state_dict(torch.load(weights, map_location="cpu"))
    return m.to(device).eval()


@torch.no_grad()
def embed_crops(model: CowReID, crops: np.ndarray, device: str = "cuda", bs: int = 256) -> np.ndarray:
    outs = []
    for i in range(0, len(crops), bs):
        x = to_tensor(crops[i:i + bs]).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            e = model(x) + model(torch.flip(x, dims=[3]))  # horizontal-flip TTA
        outs.append(F.normalize(e.float(), dim=1).cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros((0, 256), np.float32)


@torch.no_grad()
def extract_embeddings(video_path, df: pd.DataFrame, model: CowReID, every: int = 4,
                       device: str = "cuda", verbose_every: int = 5000) -> pd.DataFrame:
    """Crop → embed detections df(frame,tid,x1..y2,conf) every `every` frames.

    Returns: DataFrame[frame, tid, occ (max IoU), conf] with emb(N,256) in attrs.
    """
    sel = df[df.frame % every == 0].copy().reset_index(drop=True)
    sel["occ"] = max_iou_others(sel)
    by_frame = sel.groupby("frame").indices
    cap = cv2.VideoCapture(str(video_path))
    crops = np.zeros((len(sel), CROP, CROP, 3), np.uint8)
    fi = 0
    targets = set(by_frame)
    last = max(targets) if targets else -1
    while fi <= last:
        if fi % every == 0 and fi in targets:
            ok, frame = cap.read()
            if not ok:
                break
            for i in by_frame[fi]:
                r = sel.iloc[i]
                crops[i] = crop_box(frame, (r.x1, r.y1, r.x2, r.y2))
        else:
            if not cap.grab():
                break
        fi += 1
        if verbose_every and fi % verbose_every == 0:
            print(f"  [reid] {fi}/{last} frames")
    cap.release()
    emb = embed_crops(model, crops, device)
    return sel[["frame", "tid", "conf", "occ"]], emb
