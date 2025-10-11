#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark multiple models on aligned data and write a compact table:
runs/benchmarks/summary.csv  (model, split, accuracy, macro_f1, usable, total)

Deep models: InsightFace packs via FaceAnalysis (same pipeline as standalone scripts)
Classical  : LBP + Linear SVM
"""
import argparse
from pathlib import Path
import numpy as np, cv2, csv
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

# ---------------- shared IO ----------------
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}
def iter_split(root: Path):
    X, y = [], []
    for cls in sorted(p.name for p in root.iterdir() if p.is_dir()):
        for p in (root/cls).iterdir():
            if p.suffix.lower() in IMG_EXTS:
                X.append(str(p)); y.append(cls)
    return X, y

# ---------------- Deep via FaceAnalysis ----------------
from insightface.app import FaceAnalysis

def embed_paths_fa(app: FaceAnalysis, paths, desc="embed(FA)"):
    """Embed (aligned) crops with FaceAnalysis recognizer; return list[ndarray|None]."""
    embs = []
    for p in tqdm(paths, desc=desc):
        im = cv2.imread(p)
        if im is None:
            embs.append(None); continue
        faces = app.get(im)            # OK on aligned crops; may still miss a few
        if not faces or getattr(faces[0], "normed_embedding", None) is None:
            embs.append(None); continue
        e = faces[0].normed_embedding.astype(np.float32)
        e /= (np.linalg.norm(e) + 1e-9)
        embs.append(e)
    return embs

def centroids_from_train(embs, labels):
    by = defaultdict(list)
    for e, c in zip(embs, labels):
        if e is not None: by[c].append(e)
    cents = {}
    for c, vecs in by.items():
        M = np.vstack(vecs); m = M.mean(axis=0)
        m /= (np.linalg.norm(m)+1e-9)
        cents[c]=m.astype(np.float32)
    return cents

def predict_cosine(cents, embs):
    classes = list(cents.keys())
    C = np.vstack([cents[c] for c in classes]).T
    C /= (np.linalg.norm(C, axis=0, keepdims=True)+1e-9)
    preds=[]
    for e in embs:
        if e is None: preds.append(None); continue
        preds.append(classes[int(np.argmax(e @ C))])
    return preds

# ---------------- Classical (LBP + Linear SVM) ----------------
def lbp_image(gray):
    H,W = gray.shape
    codes = np.zeros((H-2,W-2), np.uint8)
    c = gray[1:-1,1:-1]
    nbrs = [gray[0:-2,0:-2],gray[0:-2,1:-1],gray[0:-2,2:],
            gray[1:-1,0:-2],                  gray[1:-1,2:],
            gray[2:,0:-2],  gray[2:,1:-1],  gray[2:,2:]]
    weights=[1,2,4,8,16,32,64,128]
    for w,nb in zip(weights,nbrs):
        codes |= (nb>=c).astype(np.uint8)*w
    return codes

def lbp_hist_features(img_bgr, grid=(8,8), bins=256):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    codes = lbp_image(g); H,W = codes.shape
    gh,gw = grid; sh,sw = H//gh, W//gw
    feats=[]
    for i in range(gh):
        for j in range(gw):
            patch = codes[i*sh:(i+1)*sh, j*sw:(j+1)*sw]
            h = np.bincount(patch.ravel(), minlength=bins).astype(np.float32)
            h /= (h.sum()+1e-9)
            feats.append(h)
    return np.concatenate(feats,0)

def feats_paths(paths):
    X=[]
    for p in tqdm(paths, desc="features(LBP)"):
        im = cv2.imread(p)
        X.append(lbp_hist_features(im))
    return np.vstack(X)

# ---------------- Eval helper ----------------
def acc_f1(y_true, y_pred):
    mask = np.array([p is not None for p in y_pred], bool)
    y_t = np.array(y_true)[mask]; y_p = np.array(y_pred, dtype=object)[mask]
    if len(y_t)==0: return 0.0, 0.0, 0, len(y_true)
    return accuracy_score(y_t,y_p), f1_score(y_t,y_p,average="macro"), len(y_t), len(y_true)

# ---------------- Main ----------------
def main(data_root, deep_packs, include_lbp):
    root = Path(data_root)
    out = Path("runs/benchmarks"); out.mkdir(parents=True, exist_ok=True)
    rows=[["model","split","accuracy","macro_f1","usable","total"]]

    # Deep models (FaceAnalysis, same as your standalone evals)
    for pack in deep_packs:
        app = FaceAnalysis(name=pack)
        app.prepare(ctx_id=-1, det_size=(224, 224))  # CPU

        # train → centroids
        Xtr, Ytr = iter_split(root/"train")
        Etr = embed_paths_fa(app, Xtr, desc=f"embed(FA:{pack}, train)")
        cents = centroids_from_train(Etr, Ytr)

        # val
        Xv, Yv = iter_split(root/"val")
        Ev = embed_paths_fa(app, Xv, desc=f"embed(FA:{pack}, val)")
        Pv = predict_cosine(cents, Ev)
        acc, mf1, u, t = acc_f1(Yv, Pv)
        rows.append([f"deep:{pack}","val",f"{acc:.4f}",f"{mf1:.4f}",u,t])

        # test
        Xte, Yte = iter_split(root/"test")
        Ete = embed_paths_fa(app, Xte, desc=f"embed(FA:{pack}, test)")
        Pt = predict_cosine(cents, Ete)
        acc, mf1, u, t = acc_f1(Yte, Pt)
        rows.append([f"deep:{pack}","test",f"{acc:.4f}",f"{mf1:.4f}",u,t])

    # Classical model
    if include_lbp:
        Xtr_p,Ytr = iter_split(root/"train")
        Xv_p,Yv   = iter_split(root/"val")
        Xte_p,Yte = iter_split(root/"test")
        Xtr = feats_paths(Xtr_p); Xv = feats_paths(Xv_p); Xte = feats_paths(Xte_p)
        clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000)
        clf.fit(Xtr, Ytr)
        Vp = clf.predict(Xv);   acc, mf1, u, t = acc_f1(Yv, Vp); rows.append(["classical:lbp_svm","val",f"{acc:.4f}",f"{mf1:.4f}",u,t])
        Tp = clf.predict(Xte);  acc, mf1, u, t = acc_f1(Yte, Tp); rows.append(["classical:lbp_svm","test",f"{acc:.4f}",f"{mf1:.4f}",u,t])

    with open(out/"summary.csv","w",newline="") as f:
        csv.writer(f).writerows(rows)
    print("Wrote", out/"summary.csv")
    for r in rows: print(",".join(map(str,r)))

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/aligned")
    ap.add_argument("--deep-packs", nargs="*", default=["buffalo_l"])  # e.g., add antelopev2
    ap.add_argument("--include-lbp", action="store_true")
    args = ap.parse_args()
    main(args.data, args.deep_packs, args.include_lbp)
