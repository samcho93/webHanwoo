"""한우 ReID 모델 학습 (자기지도: 추적 결과 → 라벨).

배치 구성 (P×K 샘플링, '같은 영상 + 동시 존재' 제약)
  - 한 배치는 한 영상에서만 뽑는다 → 다른 영상의 같은 소가 negative 로 들어가는 오류 방지
  - mode "tracklet"(긴 영상): 임의 시각 t 에 '동시에 존재하는' tracklet 들만 P 개 선택
        → 같은 순간에 화면에 함께 있으므로 반드시 서로 다른 소
  - 각 ID 에서 K 장 crop (시간적으로 떨어진 장면을 섞어 자세 변화 학습)
손실: Supervised Contrastive (온도 0.1) + Batch-hard Triplet (margin 0.3)

평가 (검증 영상 cow02, cow09 — 학습에서 제외)
  각 ID 의 '앞 절반' 평균 임베딩을 갤러리로, '뒤 절반'의 2초 구간 평균 임베딩을 질의로 →
  같은 영상의 N 마리 중 올바른 소를 1순위로 찾는 비율 (Rank-1). HSV 히스토그램 / ImageNet 특징과 비교.

사용:  python scripts/train_reid.py --iters 4000
       python scripts/train_reid.py --iters 4000 --all     # 검증 영상까지 포함한 최종 모델
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hanwoo.config import DATA_DIR, MODELS_DIR, OUTPUT_DIR  # noqa: E402
from hanwoo.reid import CROP, CowReID, embed_crops, to_tensor  # noqa: E402

VAL = ["cow02", "cow09"]


# --------------------------------------------------------------------------- 데이터
class Video:
    def __init__(self, path: Path):
        z = np.load(path)
        self.name = path.stem
        self.imgs, self.label, self.frame = z["imgs"], z["label"], z["frame"]
        self.mode = str(z["mode"])
        self.ids = np.unique(self.label)
        self.idx = {l: np.where(self.label == l)[0] for l in self.ids}
        self.ids = np.array([l for l in self.ids if len(self.idx[l]) >= 4])
        self.span = {l: (self.frame[self.idx[l]].min(), self.frame[self.idx[l]].max()) for l in self.ids}

    def sample(self, rng, P, K):
        if self.mode == "tracklet":
            t = rng.integers(self.frame.min(), self.frame.max() + 1)
            alive = [l for l in self.ids if self.span[l][0] <= t <= self.span[l][1]]
            cand = np.array(alive) if len(alive) >= 4 else self.ids
        else:
            cand = self.ids
        ids = rng.choice(cand, size=min(P, len(cand)), replace=False)
        sel, lab = [], []
        for l in ids:
            sel += list(rng.choice(self.idx[l], size=K, replace=len(self.idx[l]) < K))
            lab += [l] * K
        return self.imgs[sel], np.array(lab)


def augment(imgs: np.ndarray, rng) -> np.ndarray:
    out = np.empty_like(imgs)
    for i, im in enumerate(imgs):
        if rng.random() < 0.5:
            im = im[:, ::-1]
        # 약한 스케일·이동
        s = rng.uniform(0.85, 1.1)
        tx, ty = rng.uniform(-0.08, 0.08, 2) * CROP
        M = np.array([[s, 0, (1 - s) * CROP / 2 + tx], [0, s, (1 - s) * CROP / 2 + ty]], np.float32)
        im = cv2.warpAffine(np.ascontiguousarray(im), M, (CROP, CROP), borderValue=(114, 114, 114))
        # 밝기·대비 (색상(hue)은 개체 단서이므로 바꾸지 않음)
        a, b = rng.uniform(0.75, 1.25), rng.uniform(-20, 20)
        im = np.clip(im.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        # random erasing: 펜스·다른 소에 의한 가림 모사
        if rng.random() < 0.5:
            eh, ew = int(CROP * rng.uniform(0.15, 0.4)), int(CROP * rng.uniform(0.15, 0.4))
            y, x = rng.integers(0, CROP - eh), rng.integers(0, CROP - ew)
            im[y:y + eh, x:x + ew] = rng.integers(0, 255, 3)
        out[i] = im
    return out


# --------------------------------------------------------------------------- 손실
def supcon(z, y, temp=0.1):
    sim = z @ z.T / temp
    n = len(y)
    eye = torch.eye(n, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e4)
    pos = (y[:, None] == y[None, :]) & ~eye
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return -(logp * pos).sum(1).div(pos.sum(1).clamp(min=1)).mean()


def triplet_hard(z, y, margin=0.3):
    d = torch.cdist(z, z)
    same = y[:, None] == y[None, :]
    hp = (d * same).max(1).values
    hn = d.masked_fill(same, 1e4).min(1).values
    return F.relu(hp - hn + margin).mean()


# --------------------------------------------------------------------------- 평가
def hsv_feat(imgs):
    out = []
    for im in imgs:
        c = im[CROP // 5:-CROP // 5, CROP // 5:-CROP // 5]
        hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten()
        out.append(np.sqrt(h / max(h.sum(), 1)))  # Hellinger 매핑 → 코사인이 Bhattacharyya 와 동치
    return np.array(out, np.float32)


def rank1(feat, label, frame, fps=24.0, win_s=2.0):
    """앞 절반 평균(갤러리) vs 뒤 절반 2초 구간 평균(질의) → Rank-1, 단일 crop Rank-1"""
    mid = np.median(frame)
    ids = [l for l in np.unique(label) if ((label == l) & (frame < mid)).sum() >= 3 and ((label == l) & (frame >= mid)).sum() >= 3]
    G = np.stack([feat[(label == l) & (frame < mid)].mean(0) for l in ids])
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    hit = tot = hit1 = tot1 = 0
    for k, l in enumerate(ids):
        m = (label == l) & (frame >= mid)
        f, fr = feat[m], frame[m]
        for w in np.unique(fr // int(win_s * fps)):
            q = f[fr // int(win_s * fps) == w].mean(0)
            hit += int(np.argmax(G @ q) == k); tot += 1
        p = np.argmax(f @ G.T, 1)
        hit1 += int((p == k).sum()); tot1 += len(p)
    return hit / tot, hit1 / tot1, len(ids)


def evaluate(model, vids, device):
    res = {}
    for v in vids:
        r = {}
        f_h = hsv_feat(v.imgs)
        r["hsv"] = rank1(f_h, v.label, v.frame)
        if model is not None:
            model.eval()
            f_m = embed_crops(model, v.imgs, device)
            r["reid"] = rank1(f_m, v.label, v.frame)
            model.train()
        res[v.name] = r
    return res


@torch.no_grad()
def imagenet_baseline(vids, device):
    import torchvision
    r = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    r.fc = torch.nn.Identity()
    r = r.to(device).eval()
    out = {}
    for v in vids:
        fs = []
        for i in range(0, len(v.imgs), 256):
            fs.append(F.normalize(r(to_tensor(v.imgs[i:i + 256]).to(device)), dim=1).cpu().numpy())
        out[v.name] = rank1(np.concatenate(fs), v.label, v.frame)
    return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_DIR / "reid"))
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--P", type=int, default=12)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3, help="헤드(BN+FC) 학습률")
    ap.add_argument("--lr-backbone", type=float, default=5e-5, help="사전학습 백본 학습률 (작게: 특징 붕괴 방지)")
    ap.add_argument("--all", action="store_true", help="검증 영상까지 학습에 포함 (최종 배포 모델)")
    ap.add_argument("--out", default=str(MODELS_DIR / "hanwoo_reid.pt"))
    a = ap.parse_args()
    dev = "cuda"
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    vids = [Video(p) for p in sorted(Path(a.data).glob("cow*.npz"))]
    val = [v for v in vids if v.name in VAL]
    train = vids if a.all else [v for v in vids if v.name not in VAL]
    print("train:", [(v.name, v.mode, len(v.ids)) for v in train])
    # 샘플링 가중치: crop 수에 비례 (긴 영상이 너무 지배하지 않도록 제곱근)
    w = np.sqrt([len(v.label) for v in train]); w /= w.sum()

    log = dict(baseline_imagenet=imagenet_baseline(val, dev), curve=[])
    print("ImageNet ResNet50:", log["baseline_imagenet"])

    model = CowReID().to(dev).train()
    head = list(model.bn.parameters()) + list(model.fc.parameters())
    opt = torch.optim.AdamW([dict(params=model.backbone.parameters(), lr=a.lr_backbone),
                             dict(params=head, lr=a.lr)], weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr_backbone, a.lr], total_steps=a.iters, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    t0 = time.time()
    run = 0.0
    for it in range(1, a.iters + 1):
        v = train[rng.choice(len(train), p=w)]
        imgs, y = v.sample(rng, a.P, a.K)
        x = to_tensor(augment(imgs, rng)).to(dev, non_blocking=True)
        y = torch.from_numpy(y).to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            z = model(x)
        z = z.float()
        loss = supcon(z, y) + triplet_hard(z, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if it > 1 else loss.item()
        if it % 500 == 0 or it == a.iters:
            ev = evaluate(model, val, dev)
            log["curve"].append(dict(it=it, loss=round(run, 4), eval=ev))
            print(f"it {it:5d} loss {run:.3f}  {time.time() - t0:.0f}s  ", {k: {m: tuple(round(x, 3) for x in r) for m, r in e.items()} for k, e in ev.items()})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), a.out)
    log["final"] = log["curve"][-1]["eval"]
    log["args"] = vars(a)
    tag = "all" if a.all else "holdout"
    (OUTPUT_DIR / f"reid_train_{tag}.json").write_text(json.dumps(log, indent=2, default=float), encoding="utf-8")
    print("saved", a.out)


if __name__ == "__main__":
    main()
