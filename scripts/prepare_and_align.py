#!/usr/bin/env python3
import argparse, random
from pathlib import Path
import cv2, numpy as np
from tqdm import tqdm

from insightface.app import FaceAnalysis
from insightface.utils import face_align

def choose_best_face(faces):
    # pick highest detection score
    best = None; best_score = -1
    for f in faces:
        s = getattr(f, "det_score", 0.0)
        if s > best_score:
            best, best_score = f, s
    return best

def align_face_bgr(img_bgr, face, size):
    # 5-point kps → aligned crop
    kps = np.array(face.kps, dtype=np.float32)
    return face_align.norm_crop(img_bgr, landmark=kps, image_size=size)

def split_list(L, frac_train, frac_val, seed=42):
    random.Random(seed).shuffle(L)
    n = len(L); n_train = int(frac_train*n); n_val = int(frac_val*n)
    train = L[:n_train]; val = L[n_train:n_train+n_val]; test = L[n_train+n_val:]
    return train, val, test

def main(src_root, out_root, ids_max, per_id_max, size, train_frac, val_frac, seed):
    src = Path(src_root); out = Path(out_root)
    for sub in ["train","val","test"]:
        (out/sub).mkdir(parents=True, exist_ok=True)

    # set up FaceAnalysis for detection (CPU ok)
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0 if cv2.cuda.getCudaEnabledDeviceCount()>0 else -1, det_size=(640,640))

    id_dirs = [p for p in sorted(src.iterdir()) if p.is_dir()]
    if ids_max: id_dirs = id_dirs[:ids_max]

    kept_ids = 0
    for id_dir in id_dirs:
        imgs = [p for p in sorted(id_dir.glob("*")) if p.suffix.lower() in {".jpg",".jpeg",".png"}]
        if per_id_max: imgs = imgs[:per_id_max]
        if len(imgs) < 6:  # too few to split meaningfully
            continue

        train, val, test = split_list(imgs, train_frac, val_frac, seed)
        if not train or not val or not test:
            continue

        for split, paths in [("train",train),("val",val),("test",test)]:
            (out/split/id_dir.name).mkdir(parents=True, exist_ok=True)
            for p in tqdm(paths, desc=f"{id_dir.name}:{split}", leave=False):
                img_bgr = cv2.imread(str(p))
                if img_bgr is None: continue
                faces = app.get(img_bgr)
                if not faces: continue
                f = choose_best_face(faces)
                crop = align_face_bgr(img_bgr, f, size=size)
                out_p = out/split/id_dir.name/f"{p.stem}.jpg"
                cv2.imwrite(str(out_p), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        kept_ids += 1
    print(f"Prepared & aligned identities: {kept_ids}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="raw identities: data/raw/<id>/*.jpg")
    ap.add_argument("--out", default="data/aligned")
    ap.add_argument("--ids-max", type=int, default=30)
    ap.add_argument("--per-id-max", type=int, default=30)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(**vars(args))
