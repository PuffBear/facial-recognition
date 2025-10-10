#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_and_align.py
Split a per-ID raw dataset into train/val/test and write aligned crops (112x112 by default).

Per image:
  1) Try InsightFace FaceAnalysis (5-point alignment via norm_crop)
  2) Fallback: MTCNN (facenet-pytorch) if available
  3) Fallback: center square crop + resize (to avoid losing samples)

Usage:
  python scripts/prepare_and_align.py \
    --src data/raw_selected \
    --out data/aligned \
    --ids-max 30 \
    --per-id-max 30 \
    --size 112 \
    --train-frac 0.6 --val-frac 0.2
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from insightface.app import FaceAnalysis
from insightface.utils import face_align

# OPTIONAL: MTCNN (safe on py3.13 — will be skipped if not installed)
try:
    from facenet_pytorch import MTCNN
except Exception:
    MTCNN = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def choose_best_face(faces):
    """Pick the detected face with highest det_score (tie-break by area)."""
    if not faces:
        return None
    best = faces[0]
    best_score = float(getattr(best, "det_score", 0.0))
    for f in faces[1:]:
        s = float(getattr(f, "det_score", 0.0))
        if s > best_score:
            best, best_score = f, s
        elif s == best_score:
            bx = f.bbox.astype(int)
            ba = (bx[2] - bx[0]) * (bx[3] - bx[1])
            bbx = best.bbox.astype(int)
            bba = (bbx[2] - bbx[0]) * (bbx[3] - bbx[1])
            if ba > bba:
                best = f
    return best


def imread_bgr(path_str):
    """Robust image read that handles unicode paths."""
    p = str(path_str)
    data = np.fromfile(p, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
    return img


def try_align_insightface(img_bgr, app, size):
    faces = app.get(img_bgr)
    if not faces:
        return None
    f = choose_best_face(faces)
    if f is None or getattr(f, "kps", None) is None:
        return None
    kps = np.array(f.kps, dtype=np.float32)
    crop = face_align.norm_crop(img_bgr, landmark=kps, image_size=size)
    return crop


def try_align_mtcnn(img_bgr, mtcnn, size):
    if mtcnn is None:
        return None
    # MTCNN expects RGB; returns torch tensor [3,H,W] or None
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    out = mtcnn(rgb)
    if out is None:
        return None
    import torch  # local import to avoid hard dependency when missing
    if isinstance(out, torch.Tensor):
        out = out.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        out = (out * 255.0).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    if isinstance(out, (list, tuple)) and len(out):
        img = out[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        img = (img * 255.0).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return None


def center_square_resize(img_bgr, size):
    """Last-resort: center square crop -> resize to (size,size)."""
    h, w = img_bgr.shape[:2]
    s = min(h, w)
    y0 = max(0, (h - s) // 2)
    x0 = max(0, (w - s) // 2)
    crop = img_bgr[y0 : y0 + s, x0 : x0 + s]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def split_list(items, frac_train, frac_val, seed=42):
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    n = len(items)
    n_train = int(frac_train * n)
    n_val = int(frac_val * n)
    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val :]
    return train, val, test


def main(src_root, out_root, ids_max, per_id_max, size, train_frac, val_frac, seed):
    src = Path(src_root)
    out = Path(out_root)
    for sub in ["train", "val", "test"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    # InsightFace detector/recognizer (CPU by default). Larger det_size helps tiny faces.
    app = FaceAnalysis(name="buffalo_l")
    ctx_id = 0 if cv2.cuda.getCudaEnabledDeviceCount() > 0 else -1
    app.prepare(ctx_id=ctx_id, det_size=(1024, 1024))

    # Optional MTCNN (will be None on py3.13 unless you install facenet-pytorch)
    mtcnn = MTCNN(
        image_size=size, margin=20, post_process=False, keep_all=False, device="cpu"
    ) if MTCNN is not None else None

    id_dirs = [p for p in sorted(src.iterdir()) if p.is_dir()]
    if ids_max and ids_max > 0:
        id_dirs = id_dirs[:ids_max]

    kept_id_count, total_saved = 0, 0

    for id_dir in id_dirs:
        imgs = [p for p in id_dir.iterdir() if p.suffix.lower() in IMG_EXTS]

        # shuffle FIRST, then cap
        rng = random.Random(seed)
        rng.shuffle(imgs)
        if per_id_max and per_id_max > 0:
            imgs = imgs[:per_id_max]

        if len(imgs) < 6:
            print(f"[skip] {id_dir.name}: only {len(imgs)} images")
            continue

        train, val, test = split_list(imgs, train_frac, val_frac, seed)
        saved_total_for_id = 0

        for split_name, paths in [("train", train), ("val", val), ("test", test)]:
            (out / split_name / id_dir.name).mkdir(parents=True, exist_ok=True)
            for p in tqdm(paths, desc=f"{id_dir.name}:{split_name}", leave=False):
                img_bgr = imread_bgr(p)
                if img_bgr is None:
                    continue

                # 1) InsightFace
                crop = try_align_insightface(img_bgr, app, size)
                # 2) MTCNN (optional)
                if crop is None:
                    crop = try_align_mtcnn(img_bgr, mtcnn, size)
                # 3) Center-crop fallback
                if crop is None:
                    crop = center_square_resize(img_bgr, size)
                if crop is None:
                    continue

                out_p = out / split_name / id_dir.name / f"{p.stem}.jpg"
                ok = cv2.imwrite(str(out_p), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                if ok:
                    saved_total_for_id += 1
                    total_saved += 1

        if saved_total_for_id == 0:
            print(f"[warn] {id_dir.name}: no aligned crops saved — all detections failed")
            continue

        kept_id_count += 1

    print(
        f"Prepared & aligned crops: {total_saved} across {kept_id_count} identities "
        f"→ {out_root}/train|val|test/<ID>/*.jpg"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="raw identities: data/raw_selected/<id>/*.jpg")
    ap.add_argument("--out", default="data/aligned")
    ap.add_argument("--ids-max", type=int, default=30)
    ap.add_argument("--per-id-max", type=int, default=30)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Map CLI args to main() params
    main(
        src_root=args.src,
        out_root=args.out,
        ids_max=args.ids_max,
        per_id_max=args.per_id_max,
        size=args.size,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )
