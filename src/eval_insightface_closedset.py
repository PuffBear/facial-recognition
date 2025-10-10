#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Closed-set eval for InsightFace packs (ArcFace-style).
- If the pack supports FaceAnalysis end-to-end (detector + recognizer), we use it.
- If not (e.g., antelopev2 on some setups), we fall back to recognition-only:
  load the ONNX recognizer from ~/.insightface/models/<pack>/ and embed aligned crops directly.
"""

import argparse, os
from pathlib import Path
import numpy as np, cv2
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- optional FaceAnalysis path ---
FA_AVAILABLE = True
try:
    from insightface.app import FaceAnalysis
except Exception:
    FA_AVAILABLE = False

# recognition-only loader
from insightface.model_zoo import get_model

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
 
def iter_split(root: Path):
    X, y = [], []
    for cls in sorted(p.name for p in root.iterdir() if p.is_dir()):
        for p in (root/cls).iterdir():
            if p.suffix.lower() in IMG_EXTS:
                X.append(str(p)); y.append(cls)
    return X, y

def centroids_from_train(embs, labels):
    by_cls = defaultdict(list)
    for e, c in zip(embs, labels):
        if e is not None: by_cls[c].append(e)
    cents = {}
    for c, vecs in by_cls.items():
        M = np.vstack(vecs)
        m = M.mean(axis=0)
        m = m / (np.linalg.norm(m) + 1e-9)
        cents[c] = m.astype(np.float32)
    return cents

def predict_cosine(cents, embs):
    classes = list(cents.keys())
    C = np.vstack([cents[c] for c in classes]).T  # [d, K]
    C = C / (np.linalg.norm(C, axis=0, keepdims=True) + 1e-9)
    preds = []
    for e in embs:
        if e is None:
            preds.append(None); continue
        s = e @ C
        preds.append(classes[int(np.argmax(s))])
    return preds

# ---------- MODE A: FaceAnalysis path ----------
def best_face(faces):
    if not faces: return None
    return max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))

def embed_with_fa(app, paths):
    embs = []
    for p in tqdm(paths, desc="embed(FA)"):
        im = cv2.imread(p)
        if im is None:
            embs.append(None); continue
        faces = app.get(im)
        f = best_face(faces)
        if f is None or getattr(f, "normed_embedding", None) is None:
            embs.append(None); continue
        e = f.normed_embedding.astype(np.float32)
        e = e / (np.linalg.norm(e) + 1e-9)
        embs.append(e)
    return embs

# ---------- MODE B: recognition-only path ----------
# map packs to recognizer ONNX inside ~/.insightface/models/<pack>/
RECOG_ONNX = {
    "antelopev2": "glintr100.onnx",
    "buffalo_l": "w600k_r50.onnx",
    "buffalo_m": "w600k_r50.onnx",
    # add more if you try other packs
}

def load_recognizer_from_pack(pack: str):
    from insightface.model_zoo import get_model
    pack_dir = Path.home() / ".insightface" / "models" / pack
    if not pack_dir.exists():
        raise FileNotFoundError(f"Pack not found: {pack_dir}")

    # Preferred names first, then heuristic scan
    preferred = ["glintr100.onnx", "w600k_r100.onnx", "w600k_r50.onnx", "iresnet100.onnx"]
    candidates = [pack_dir / n for n in preferred if (pack_dir / n).exists()]
    if not candidates:
        # Heuristic: any .onnx that is NOT a detector/landmark/genderage
        for p in sorted(pack_dir.glob("*.onnx")):
            name = p.name.lower()
            if any(k in name for k in ["det", "landmark", "gender", "age", "2d106", "1k3d", "scrfd"]):
                continue
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(f"No recognizer .onnx found in {pack_dir}")

    last_err = None
    for onnx_path in candidates:
        try:
            rec = get_model(str(onnx_path))
            rec.prepare(ctx_id=-1)  # CPU
            # quick sanity: embed a dummy crop
            dummy = np.zeros((112,112,3), dtype=np.uint8)
            e = None
            for fn in ("get_embedding","get_feat"):
                if hasattr(rec, fn):
                    try:
                        e = getattr(rec, fn)(dummy)
                        break
                    except Exception:
                        pass
            if e is not None:
                print(f"[ok] recognizer: {onnx_path.name}")
                return rec
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to load any recognizer from {pack_dir}. Last error: {last_err}")


def try_embed_recognizer(rec, img):
    # try BGR and RGB, and both get_embedding / get_feat
    for im in (img, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)):
        for fn in ("get_embedding", "get_feat"):
            if hasattr(rec, fn):
                try:
                    e = getattr(rec, fn)(im)
                    if e is None: continue
                    e = np.array(e).astype(np.float32).ravel()
                    e = e / (np.linalg.norm(e) + 1e-9)
                    return e
                except Exception:
                    pass
    return None

def embed_with_recognizer(rec, paths):
    embs = []
    for p in tqdm(paths, desc="embed(rec-only)"):
        im = cv2.imread(p)
        if im is None:
            embs.append(None); continue
        e = try_embed_recognizer(rec, im)
        embs.append(e)
    return embs

# ---------- eval ----------
def eval_split(y_true, y_pred, split_name):
    mask = np.array([p is not None for p in y_pred], dtype=bool)
    y_true_f = np.array(y_true)[mask].tolist()
    y_pred_f = np.array(y_pred, dtype=object)[mask].tolist()
    acc = accuracy_score(y_true_f, y_pred_f)
    print(f"\n=== {split_name.upper()} REPORT ===")
    print(classification_report(y_true_f, y_pred_f, digits=3))
    print(f"accuracy: {acc:.3f} on {len(y_true_f)} / {len(y_true)} usable samples")
    cm = confusion_matrix(y_true_f, y_pred_f, labels=sorted(set(y_true_f)))
    print("Confusion matrix shape:", cm.shape)
    return acc

def main(data_root, pack, det_size):
    root = Path(data_root)

    # TRAIN → embeddings → centroids
    Xtr, Ytr = iter_split(root/"train")

    use_fa = False
    app = None
    if FA_AVAILABLE:
        try:
            app = FaceAnalysis(name=pack)
            app.prepare(ctx_id=-1, det_size=(det_size, det_size))
            # ensure it actually has detection+recognition
            if "detection" in getattr(app, "models", {}) and "recognition" in app.models:
                use_fa = True
        except AssertionError:
            use_fa = False

    if use_fa:
        tr_embs = embed_with_fa(app, Xtr)
    else:
        # recognition-only mode
        rec = load_recognizer_from_pack(pack)
        tr_embs = embed_with_recognizer(rec, Xtr)

    cents = centroids_from_train(tr_embs, Ytr)

    # VAL
    Xv, Yv = iter_split(root/"val")
    if use_fa:
        v_embs = embed_with_fa(app, Xv)
    else:
        v_embs = embed_with_recognizer(rec, Xv)
    v_pred = predict_cosine(cents, v_embs)
    eval_split(Yv, v_pred, "val")

    # TEST
    Xte, Yte = iter_split(root/"test")
    if use_fa:
        te_embs = embed_with_fa(app, Xte)
    else:
        te_embs = embed_with_recognizer(rec, Xte)
    te_pred = predict_cosine(cents, te_embs)
    eval_split(Yte, te_pred, "test")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/aligned")
    ap.add_argument("--pack", default="antelopev2", help="e.g., buffalo_l, antelopev2, buffalo_m")
    ap.add_argument("--det-size", type=int, default=224)
    args = ap.parse_args()
    main(args.data, args.pack, args.det_size)
