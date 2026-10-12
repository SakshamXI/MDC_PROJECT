"""Convert WIDER FACE to YOLO format. Edit the settings below, then run:  python wider2yolo.py

Expected layout of WIDER_DIR:
  WIDER_train/images/<event>/<img>.jpg
  WIDER_val/images/<event>/<img>.jpg
  wider_face_split/wider_face_train_bbx_gt.txt
  wider_face_split/wider_face_val_bbx_gt.txt
"""
import os
import shutil
from pathlib import Path

from PIL import Image

# ---------------- settings ----------------
WIDER_DIR = "/path/to/WIDER"          # folder containing WIDER_train, WIDER_val, wider_face_split
OUT_DIR = "datasets/wider_yolo"
MIN_SIZE = 20                          # drop faces smaller than this (px)
# ------------------------------------------


def parse_annotations(path):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    i = 0
    while i < len(lines):
        name = lines[i]
        n = int(lines[i + 1])
        i += 2
        rows = max(n, 1)  # images with 0 faces still have one dummy line
        boxes = []
        if n > 0:
            for j in range(n):
                v = list(map(int, lines[i + j].split()[:10]))
                boxes.append(v)  # x, y, w, h, blur, expr, illum, invalid, occl, pose
        i += rows
        yield name, boxes


def link_or_copy(src, dst):
    if dst.exists():
        return
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        shutil.copy(src, dst)


def convert(wider, out, split, min_size):
    ann = wider / "wider_face_split" / f"wider_face_{split}_bbx_gt.txt"
    img_root = wider / f"WIDER_{split}" / "images"
    img_out = out / "images" / split
    lbl_out = out / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    kept_imgs = kept_boxes = 0
    for name, boxes in parse_annotations(ann):
        src = img_root / name
        if not src.exists():
            continue
        W, H = Image.open(src).size
        lines = []
        for x, y, w, h, _, _, _, invalid, _, _ in boxes:
            if invalid == 1 or w < min_size or h < min_size:
                continue
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W, x + w), min(H, y + h)
            if x2 <= x1 or y2 <= y1:
                continue
            cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
            bw, bh = (x2 - x1) / W, (y2 - y1) / H
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        if not lines:
            continue
        flat = name.replace("/", "_")
        link_or_copy(src, img_out / flat)
        (lbl_out / (Path(flat).stem + ".txt")).write_text("\n".join(lines))
        kept_imgs += 1
        kept_boxes += len(lines)
    print(f"{split}: {kept_imgs} images, {kept_boxes} faces")


if __name__ == "__main__":
    wider, out = Path(WIDER_DIR), Path(OUT_DIR)
    for split in ("train", "val"):
        convert(wider, out, split, MIN_SIZE)
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: face\n"
    )
    print("wrote", out / "data.yaml")
