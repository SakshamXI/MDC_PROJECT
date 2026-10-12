"""Train a 98-point WFLW landmark regressor (MobileNetV3 backbone).
Edit the settings below, then run:  python train_landmarks.py
"""
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from wflw_dataset import ATTR_NAMES, OUTER_EYE, WFLW

# ---------------- settings ----------------
WFLW_ROOT = "/path/to/WFLW"      # folder with WFLW_images and WFLW_annotations
EPOCHS = 60
BATCH = 64
LR = 5e-4
IMG_SIZE = 160
WORKERS = 4
OUT_PATH = "landmarks_best.pt"
# ------------------------------------------

N_PTS = 98


def build_model(pretrained=True):
    w = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
    m = mobilenet_v3_large(weights=w)
    m.classifier = nn.Sequential(
        nn.Linear(960, 512), nn.Hardswish(), nn.Dropout(0.2), nn.Linear(512, N_PTS * 2)
    )
    return m


class WingLoss(nn.Module):
    """Wing loss on pixel-scaled coordinates."""

    def __init__(self, w=10.0, eps=2.0):
        super().__init__()
        self.w, self.eps = w, eps
        self.C = w - w * math.log(1 + w / eps)

    def forward(self, pred, target):
        d = (pred - target).abs()
        loss = torch.where(d < self.w, self.w * torch.log(1 + d / self.eps), d - self.C)
        return loss.mean()


def nme(pred, gt):
    """Per-sample NME normalised by outer-eye-corner distance. pred/gt: (B,196)."""
    p, g = pred.view(-1, N_PTS, 2), gt.view(-1, N_PTS, 2)
    iod = (g[:, OUTER_EYE[0]] - g[:, OUTER_EYE[1]]).norm(dim=1)
    return (p - g).norm(dim=2).mean(dim=1) / iod


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    errs, attrs = [], []
    for img, lm, attr in loader:
        pred = model(img.to(dev)).float().cpu()
        errs.append(nme(pred, lm))
        attrs.append(attr)
    errs, attrs = torch.cat(errs), torch.cat(attrs)
    res = {"all": errs.mean().item() * 100}
    for k, name in enumerate(ATTR_NAMES):
        mask = attrs[:, k] == 1
        if mask.any():
            res[name] = errs[mask].mean().item() * 100
    return res


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = WFLW(WFLW_ROOT, "train", IMG_SIZE, train=True)
    te = WFLW(WFLW_ROOT, "test", IMG_SIZE, train=False)
    tl = DataLoader(tr, BATCH, shuffle=True, num_workers=WORKERS, drop_last=True, pin_memory=True)
    vl = DataLoader(te, BATCH, shuffle=False, num_workers=WORKERS)

    model = build_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(tl), pct_start=0.1
    )
    crit = WingLoss()
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    best = 1e9

    for ep in range(EPOCHS):
        model.train()
        run = 0.0
        for img, lm, _ in tl:
            img, lm = img.to(dev, non_blocking=True), lm.to(dev, non_blocking=True)
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                pred = model(img)
            loss = crit(pred.float() * IMG_SIZE, lm * IMG_SIZE)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
        res = evaluate(model, vl, dev)
        print(f"ep {ep + 1:3d} loss {run / len(tl):.3f} | NME% " +
              " ".join(f"{k}={v:.2f}" for k, v in res.items()))
        if res["all"] < best:
            best = res["all"]
            torch.save({"model": model.state_dict(), "size": IMG_SIZE}, OUT_PATH)
    print(f"best NME: {best:.2f}% -> {OUT_PATH}")

    # optional ONNX export of the best model
    ck = torch.load(OUT_PATH, map_location="cpu")
    model = build_model(pretrained=False)
    model.load_state_dict(ck["model"])
    model.eval()
    torch.onnx.export(model, torch.randn(1, 3, IMG_SIZE, IMG_SIZE), "landmarks.onnx",
                      input_names=["input"], output_names=["landmarks"], opset_version=17)
    print("exported landmarks.onnx")


if __name__ == "__main__":
    main()
