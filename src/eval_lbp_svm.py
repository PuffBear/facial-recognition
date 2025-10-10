#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classical baseline: LBP histogram features + Linear SVM classifier.
Works on aligned crops (e.g., 112x112 or 160x160).
"""

import argparse
from pathlib import Path
import numpy as np, cv2
from collections import defaultdict
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}

def iter_split(root: Path):
    X, y = [], []
    for cls in sorted(p.name for p in root.iterdir() if p.is_dir()):
        for p in (root/cls).iterdir():
            if p.suffix.lower() in IMG_EXTS:
                X.append(str(p)); y.append(cls)
    return X, y

def lbp_image(gray):
    # 8-neighborhood, radius=1 uniform LBP-like (simple 8-bit code)
    H, W = gray.shape
    codes = np.zeros((H-2, W-2), dtype=np.uint8)
    c = gray[1:-1, 1:-1]
    nbrs = [
        gray[0:-2, 0:-2], gray[0:-2, 1:-1], gray[0:-2, 2:  ],
        gray[1:-1, 0:-2],                     gray[1:-1, 2:  ],
        gray[2:  , 0:-2], gray[2:  , 1:-1], gray[2:  , 2:  ],
    ]
    weights = [1,2,4,8,16,32,64,128]
    for w, nb in zip(weights, nbrs):
        codes |= (nb >= c).astype(np.uint8)*w
    return codes

def lbp_hist_features(img_bgr, grid=(8,8), bins=256):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    codes = lbp_image(gray)
    H, W = codes.shape
    gh, gw = grid
    sh, sw = H//gh, W//gw
    feats = []
    for i in range(gh):
        for j in range(gw):
            patch = codes[i*sh:(i+1)*sh, j*sw:(j+1)*sw]
            hist = np.bincount(patch.ravel(), minlength=bins).astype(np.float32)
            hist = hist / (hist.sum() + 1e-9)
            feats.append(hist)
    return np.concatenate(feats, axis=0)  # (gh*gw*bins,)

def load_feats(paths):
    X = []
    for p in tqdm(paths, desc="features"):
        im = cv2.imread(p)
        X.append(lbp_hist_features(im, grid=(8,8), bins=256))
    return np.vstack(X)

def eval_split(y_true, y_pred, split):
    acc = accuracy_score(y_true, y_pred)
    print("\n=== {} REPORT ===".format(split.upper()))
    print(classification_report(y_true, y_pred, digits=3))
    print("accuracy: {:.3f}".format(acc))
    cm = confusion_matrix(y_true, y_pred, labels=sorted(set(y_true)))
    print("Confusion matrix shape:", cm.shape)
    return acc

def main(data_root):
    root = Path(data_root)
    Xtr_p, Ytr = iter_split(root/"train")
    Xv_p,  Yv  = iter_split(root/"val")
    Xte_p, Yte = iter_split(root/"test")

    Xtr = load_feats(Xtr_p)
    Xv  = load_feats(Xv_p)
    Xte = load_feats(Xte_p)

    clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000)
    clf.fit(Xtr, Ytr)

    Vpred = clf.predict(Xv)
    eval_split(Yv, Vpred, "val")

    Tpred = clf.predict(Xte)
    eval_split(Yte, Tpred, "test")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/aligned")
    args = ap.parse_args()
    main(args.data)
