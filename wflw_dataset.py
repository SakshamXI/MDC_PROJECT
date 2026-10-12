"""WFLW dataset with square-crop, box/rotation jitter and correct left-right flip.

Expected layout of root:
  WFLW_images/<folder>/<img>.jpg
  WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_{train,test}.txt

Each annotation line: 196 landmark coords, 4 box (x1 y1 x2 y2), 6 attributes, image path.
Attributes: pose, expression, illumination, make-up, occlusion, blur.
"""
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# WFLW left<->right landmark pairs. Points not listed map to themselves.
FLIP_PAIRS = [
    (0, 32), (1, 31), (2, 30), (3, 29), (4, 28), (5, 27), (6, 26), (7, 25),
    (8, 24), (9, 23), (10, 22), (11, 21), (12, 20), (13, 19), (14, 18), (15, 17),
    (33, 46), (34, 45), (35, 44), (36, 43), (37, 42), (38, 50), (39, 49),
    (40, 48), (41, 47),
    (60, 72), (61, 71), (62, 70), (63, 69), (64, 68), (65, 75), (66, 74), (67, 73),
    (55, 59), (56, 58),
    (76, 82), (77, 81), (78, 80), (87, 83), (86, 84),
    (88, 92), (89, 91), (95, 93),
    (96, 97),
]
FLIP_IDX = np.arange(98)
for _a, _b in FLIP_PAIRS:
    FLIP_IDX[_a], FLIP_IDX[_b] = _b, _a

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
ATTR_NAMES = ["pose", "expression", "illumination", "makeup", "occlusion", "blur"]
OUTER_EYE = (60, 72)  # inter-ocular normalisation points


def crop_matrix(cx, cy, size, out_size, angle=0.0):
    """Affine matrix: square of side `size` centred at (cx,cy) -> out_size x out_size."""
    M = cv2.getRotationMatrix2D((cx, cy), angle, out_size / size)
    M[0, 2] += out_size / 2 - cx
    M[1, 2] += out_size / 2 - cy
    return M


class WFLW(Dataset):
    def __init__(self, root, split="train", img_size=192, train=True, expand=1.25):
        root = Path(root)
        ann = root / "WFLW_annotations" / "list_98pt_rect_attr_train_test" / f"list_98pt_rect_attr_{split}.txt"
        self.img_dir = root / "WFLW_images"
        self.S, self.train, self.expand = img_size, train, expand
        self.items = []
        with open(ann) as f:
            for line in f:
                p = line.split()
                if len(p) < 207:
                    continue
                v = np.array(p[:206], np.float32)
                self.items.append((v[:196].reshape(98, 2), v[196:200], v[200:206].astype(int), p[206]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        lm, box, attr, rel = self.items[i]
        img = cv2.imread(str(self.img_dir / rel))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        size = max(x2 - x1, y2 - y1) * self.expand
        angle = 0.0
        if self.train:
            # simulate imperfect detector boxes
            size *= random.uniform(0.85, 1.25)
            cx += random.uniform(-0.08, 0.08) * size
            cy += random.uniform(-0.08, 0.08) * size
            angle = float(np.clip(random.gauss(0, 15), -30, 30))

        M = crop_matrix(cx, cy, size, self.S, angle)
        out = cv2.warpAffine(img, M, (self.S, self.S), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(127, 127, 127))
        pts = np.hstack([lm, np.ones((98, 1), np.float32)]) @ M.T  # (98,2) in crop pixels

        if self.train:
            if random.random() < 0.5:  # horizontal flip + index swap
                out = out[:, ::-1]
                pts = pts.copy()
                pts[:, 0] = self.S - 1 - pts[:, 0]
                pts = pts[FLIP_IDX]
            out = self._photometric(np.ascontiguousarray(out))

        out = (out.astype(np.float32) / 255.0 - MEAN) / STD
        img_t = torch.from_numpy(out.transpose(2, 0, 1).copy())
        lm_t = torch.from_numpy((pts / self.S).astype(np.float32).reshape(-1))
        return img_t, lm_t, torch.from_numpy(attr)

    def _photometric(self, img):
        img = img.astype(np.float32)
        img = img * random.uniform(0.7, 1.3) + random.uniform(-25, 25)
        img = np.clip(img, 0, 255).astype(np.uint8)
        if random.random() < 0.2:
            img = cv2.GaussianBlur(img, (5, 5), random.uniform(0.5, 2.0))
        if random.random() < 0.25:  # random occlusion patch
            S = self.S
            w, h = random.randint(S // 8, S // 3), random.randint(S // 8, S // 3)
            x, y = random.randint(0, S - w), random.randint(0, S - h)
            img[y:y + h, x:x + w] = np.random.randint(0, 255, 3)
        return img
