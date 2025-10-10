#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check overfitting for closed-set face ID with InsightFace embeddings.

Reports:
  - Train (leave-one-out) top-1 accuracy
  - Val/Test top-1 accuracy
  - Near-duplicate rate (test-to-train, same-class cosine >= 0.95)

Usage:
  python -m src.check_overfit --data data/aligned --pack buffalo_l --det-size 224
  # or try pack buffalo_m if available
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score

from insightface.app import FaceAnalysis

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def iter_split(root: Path):
    X, y = [], []
    for cls in sorted(p.name for p in root.iterdir() if p.is_dir()):
        for p in (root/cls).iterdir():
            if p.suffix.lower() in IMG_EXTS:
                X.append(str(p)); y.append(cls)
    return X, y

def best_face(faces):
    if not faces: return None
    return max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))

def embed_paths(app, paths, desc="embed"):
    embs, ok_mask = [], []
    for p in tqdm(paths, desc=desc):
        im = cv2.imread(p)
        if im is None:
            embs.append(None); ok_mask.append(False); continue
        faces = app.get(im)
        f = best_face(faces)
        if f is None or getattr(f, "normed_embedding", None) is None:
            embs.append(None); ok_mask.append(False); continue
        e = f.normed_embedding.astype(np.float32)
        e = e / (np.linalg.norm(e) + 1e-9)
        embs.append(e); ok_mask.append(True)
    return embs, np.array(ok_mask, bool)

def centroids_from_train(train_embs, train_labels):
    by_cls = defaultdict(list)
    for e, c in zip(train_embs, train_labels):
        if e is not None:
            by_cls[c].append(e)
    cents = {}
    sums = {}
    counts = {}
    for c, vecs in by_cls.items():
        M = np.vstack(vecs)
        s = M.sum(axis=0)
        m = s / (len(vecs) + 1e-9)
        cents[c]  = (m / (np.linalg.norm(m) + 1e-9)).astype(np.float32)
        sums[c]   = s.astype(np.float32)
        counts[c] = len(vecs)
    return cents, sums, counts

def predict_cosine(centroids, embs):
    classes = list(centroids.keys())
    C = np.vstack([centroids[c] for c in classes]).T  # [d, K]
    C = C / (np.linalg.norm(C, axis=0, keepdims=True) + 1e-9)
    preds = []
    for e in embs:
        if e is None:
            preds.append(None); continue
        s = e @ C
        preds.append(classes[int(np.argmax(s))])
    return preds

def train_loo_preds(train_embs, train_labels, sums, counts):
    """Per-sample leave-one-out prediction using class sums/counts."""
    classes = sorted(set([c for c in train_labels if c in counts]))
    preds, ytrue = [], []
    # Prebuild matrix of non-LOO centroids on the fly
    for e, c in zip(train_embs, train_labels):
        if e is None or c not in counts or counts[c] <= 1:
            preds.append(None); ytrue.append(c); continue
        # mean excluding this sample
        m = (sums[c] - e) / (counts[c] - 1 + 1e-9)
        m = m / (np.linalg.norm(m) + 1e-9)

        # build class matrix with this class's LOO mean
        cent = []
        cls_ord = []
        for cc in classes:
            if cc == c:
                cent.append(m)
            else:
                # use the full-mean for other classes (recompute once per class would be fine)
                # we don't have their full means here, but we can approximate via sums/counts
                mm = (sums[cc] / (counts[cc] + 1e-9))
                mm = mm / (np.linalg.norm(mm) + 1e-9)
                cent.append(mm.astype(np.float32))
            cls_ord.append(cc)
        C = np.vstack(cent).T  # [d,K]
        C = C / (np.linalg.norm(C, axis=0, keepdims=True) + 1e-9)

        s = e @ C
        preds.append(cls_ord[int(np.argmax(s))]); ytrue.append(c)
    return preds, ytrue

def near_dup_rate(train_embs, train_labels, test_embs, test_labels, thr=0.95):
    # build per-class matrix for train embeddings
    per_cls = defaultdict(list)
    for e, c in zip(train_embs, train_labels):
        if e is not None:
            per_cls[c].append(e)
    # normalize stacks
    for c in list(per_cls.keys()):
        A = np.vstack(per_cls[c]).astype(np.float32)
        A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
        per_cls[c] = A

    cnt, tot = 0, 0
    for e, c in zip(test_embs, test_labels):
        if e is None or c not in per_cls: continue
        A = per_cls[c]  # [n_c, d]
        s = (A @ e).max()  # nearest same-class train img
        tot += 1
        if s >= thr:
            cnt += 1
    rate = (cnt / max(tot,1))
    return rate, cnt, tot

def eval_split(y_true, y_pred, split_name):
    mask = np.array([p is not None for p in y_pred], dtype=bool)
    y_true_f = np.array(y_true)[mask].tolist()
    y_pred_f = np.array(y_pred, dtype=object)[mask].tolist()
    acc = accuracy_score(y_true_f, y_pred_f)
    print(f"\n=== {split_name.upper()} REPORT ===")
    print(classification_report(y_true_f, y_pred_f, digits=3))
    print(f"accuracy: {acc:.3f} on {len(y_true_f)} / {len(y_true)} usable samples")
    return acc

def main(data_root, pack, det_size):
    root = Path(data_root)

    app = FaceAnalysis(name=pack)
    app.prepare(ctx_id=-1, det_size=(det_size, det_size))

    # --- TRAIN ---
    Xtr, Ytr = iter_split(root/"train")
    Etr, OKtr = embed_paths(app, Xtr, desc="embed(train)")
    cents, sums, counts = centroids_from_train(Etr, Ytr)

    # Train LOO
    P_tr_loo, Y_tr_loo = train_loo_preds(Etr, Ytr, sums, counts)
    acc_tr_loo = eval_split(Y_tr_loo, P_tr_loo, "train(LOO)")

    # --- VAL ---
    Xv, Yv = iter_split(root/"val")
    Ev, _ = embed_paths(app, Xv, desc="embed(val)")
    P_v = predict_cosine(cents, Ev)
    acc_v = eval_split(Yv, P_v, "val")

    # --- TEST ---
    Xte, Yte = iter_split(root/"test")
    Ete, _ = embed_paths(app, Xte, desc="embed(test)")
    P_te = predict_cosine(cents, Ete)
    acc_te = eval_split(Yte, P_te, "test")

    # Near-dup rate (test -> train, same-class)
    rate, cnt, tot = near_dup_rate(Etr, Ytr, Ete, Yte, thr=0.95)
    print(f"\nNear-duplicate (embedding) rate on TEST (same-class, cos≥0.95): {cnt}/{tot} = {rate:.3f}")

    print("\nGaps:")
    print(f"  Train(LOO) - Val  = {acc_tr_loo - acc_v:+.3f}")
    print(f"  Train(LOO) - Test = {acc_tr_loo - acc_te:+.3f}")
    print("\nHeuristic: gaps > 0.10 suggest overfitting/leakage or centroid contamination.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/aligned")
    ap.add_argument("--pack", default="buffalo_l", help="e.g., buffalo_l, buffalo_m")
    ap.add_argument("--det-size", type=int, default=224)
    args = ap.parse_args()
    main(args.data, args.pack, args.det_size)
