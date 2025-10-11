#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robustness sweeps for face ID (closed-set, centroid classifier) on aligned crops.

- Uses InsightFace recognizer ONNX directly (no detection), so every crop is usable.
- Trains class centroids on CLEAN TRAIN.
- Evaluates CLEAN VAL/TEST and perturbed VAL/TEST for:
  Bucket 1: Lighting (bright, dark, high_contrast, low_contrast)
  Bucket 2: Quality  (noise_10, noise_20, blur_5, blur_9, jpeg_50, jpeg_30)
  Bucket 3: Occlusion (eyes, mouth, center)

Outputs:
  runs/robustness/results.csv with columns:
    bucket,variant,split,accuracy,delta_pct,usable,total

Run:
  python -m src.robust_sweeps --data data/aligned --pack buffalo_l
"""
import argparse, csv
from pathlib import Path
import numpy as np, cv2
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score

# -------- dataset helpers --------
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}

def iter_split(root: Path):
    X, y = [], []
    for cls in sorted(p.name for p in root.iterdir()):
        pdir = root/cls
        if not pdir.is_dir(): continue
        for p in pdir.iterdir():
            if p.suffix.lower() in IMG_EXTS:
                X.append(str(p)); y.append(cls)
    return X, y

# -------- InsightFace recognizer (recognition-only) --------
from insightface.model_zoo import get_model

def load_recognizer_any(pack_dir: Path):
    # prefer common names; else pick the first non-detector .onnx
    preferred = ["w600k_r50.onnx","glintr100.onnx","iresnet100.onnx","w600k_r100.onnx"]
    cands = [pack_dir/n for n in preferred if (pack_dir/n).exists()]
    if not cands:
        for p in sorted(pack_dir.glob("*.onnx")):
            name = p.name.lower()
            if any(k in name for k in ["det","landmark","gender","age","2d106","1k3d","scrfd"]):
                continue
            cands.append(p)
    if not cands:
        raise FileNotFoundError(f"No recognizer .onnx found in {pack_dir}")
    rec = get_model(str(cands[0])); rec.prepare(ctx_id=-1)
    return rec

def try_embed(rec, bgr):
    # try BGR/RGB and get_embedding/get_feat
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    for im in (bgr, rgb):
        for fn in ("get_embedding","get_feat"):
            if hasattr(rec, fn):
                try:
                    e = getattr(rec, fn)(im)
                    if e is not None:
                        e = np.asarray(e, np.float32).ravel()
                        e /= (np.linalg.norm(e) + 1e-9)
                        return e
                except Exception:
                    pass
    return None

def embed_paths(rec, paths, desc="embed"):
    embs, ok = [], []
    for p in tqdm(paths, desc=desc):
        im = cv2.imread(p)
        if im is None:
            embs.append(None); ok.append(False); continue
        e = try_embed(rec, im)
        embs.append(e); ok.append(e is not None)
    return embs, np.array(ok, bool)

# -------- centroid classifier --------
def centroids_from_train(embs, labels):
    by = defaultdict(list)
    for e, c in zip(embs, labels):
        if e is not None: by[c].append(e)
    cents = {}
    for c, vecs in by.items():
        M = np.vstack(vecs); m = M.mean(0)
        m /= (np.linalg.norm(m) + 1e-9)
        cents[c] = m.astype(np.float32)
    return cents

def predict_cosine(cents, embs):
    classes = list(cents.keys())
    C = np.vstack([cents[c] for c in classes]).T
    C /= (np.linalg.norm(C, axis=0, keepdims=True) + 1e-9)
    preds = []
    for e in embs:
        if e is None: preds.append(None); continue
        preds.append(classes[int(np.argmax(e @ C))])
    return preds

def acc_from_preds(y_true, y_pred):
    mask = np.array([p is not None for p in y_pred], bool)
    y_t = np.array(y_true)[mask]; y_p = np.array(y_pred, dtype=object)[mask]
    if len(y_t) == 0: return 0.0, 0, len(y_true)
    return accuracy_score(y_t, y_p), len(y_t), len(y_true)

# -------- perturbations --------
def bc(img, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def gauss_noise(img, sigma=10):
    n = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
    return out

def blur(img, k=5):
    k = k if k % 2 == 1 else k + 1
    return cv2.GaussianBlur(img, (k, k), 0)

def jpeg(img, q=50):
    q = int(np.clip(q, 5, 95))
    enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])[1]
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)

def occlude_band(img, region="eyes"):
    h, w = img.shape[:2]; out = img.copy()
    if region == "eyes":
        y0, y1 = int(0.28*h), int(0.42*h)
        out[y0:y1, :] = 0
    elif region == "mouth":
        y0, y1 = int(0.68*h), int(0.82*h)
        out[y0:y1, :] = 0
    elif region == "center":
        s = int(0.35 * min(h, w))
        y0 = h//2 - s//2; x0 = w//2 - s//2
        out[y0:y0+s, x0:x0+s] = 0
    return out

VARIANTS = {
    ("lighting", "bright"):        lambda im: bc(im, 1.0, +40),
    ("lighting", "dark"):          lambda im: bc(im, 1.0, -40),
    ("lighting", "high_contrast"): lambda im: bc(im, 1.4, 0),
    ("lighting", "low_contrast"):  lambda im: bc(im, 0.7, 0),

    ("quality", "noise_10"):       lambda im: gauss_noise(im, 10),
    ("quality", "noise_20"):       lambda im: gauss_noise(im, 20),
    ("quality", "blur_5"):         lambda im: blur(im, 5),
    ("quality", "blur_9"):         lambda im: blur(im, 9),
    ("quality", "jpeg_50"):        lambda im: jpeg(im, 50),
    ("quality", "jpeg_30"):        lambda im: jpeg(im, 30),

    ("occlusion", "eyes"):         lambda im: occlude_band(im, "eyes"),
    ("occlusion", "mouth"):        lambda im: occlude_band(im, "mouth"),
    ("occlusion", "center"):       lambda im: occlude_band(im, "center"),
}

def embed_with_transform(rec, paths, transform, desc):
    embs, ok = [], []
    for p in tqdm(paths, desc=desc):
        im = cv2.imread(p)
        if im is None:
            embs.append(None); ok.append(False); continue
        pim = transform(im)
        e = try_embed(rec, pim)
        embs.append(e); ok.append(e is not None)
    return embs, np.array(ok, bool)

# -------- main --------
def main(data_root, pack):
    root = Path(data_root)
    out_dir = Path("runs/robustness"); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir/"results.csv"

    # recognizer
    rec_dir = Path.home()/".insightface"/"models"/pack
    rec = load_recognizer_any(rec_dir)

    # TRAIN (clean) → centroids
    Xtr, Ytr = iter_split(root/"train")
    Etr, _ = embed_paths(rec, Xtr, desc="embed(train, clean)")
    cents = centroids_from_train(Etr, Ytr)

    # Baselines (clean)
    Xv,  Yv  = iter_split(root/"val")
    Xte, Yte = iter_split(root/"test")
    Ev,  _   = embed_paths(rec, Xv,  desc="embed(val, clean)")
    Ete, _   = embed_paths(rec, Xte, desc="embed(test, clean)")
    Pv = predict_cosine(cents, Ev);   acc_v, u_v, t_v = acc_from_preds(Yv,  Pv)
    Pt = predict_cosine(cents, Ete);  acc_t, u_t, t_t = acc_from_preds(Yte, Pt)

    # write CSV
    with open(csv_path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["bucket","variant","split","accuracy","delta_pct","usable","total"])
        w.writerow(["baseline","clean","val",  f"{acc_v:.4f}", f"{0.0:.2f}", u_v, t_v])
        w.writerow(["baseline","clean","test", f"{acc_t:.4f}", f"{0.0:.2f}", u_t, t_t])

        # Perturbed sweeps
        for (bucket, variant), fn in VARIANTS.items():
            Evp, _ = embed_with_transform(rec, Xv,  fn, desc=f"embed(val, {bucket}:{variant})")
            Etp, _ = embed_with_transform(rec, Xte, fn, desc=f"embed(test, {bucket}:{variant})")
            Pv  = predict_cosine(cents, Evp); acc_vp, u_vp, t_vp = acc_from_preds(Yv,  Pv)
            Pt  = predict_cosine(cents, Etp); acc_tp, u_tp, t_tp = acc_from_preds(Yte, Pt)
            dv = (acc_vp - acc_v) * 100.0
            dt = (acc_tp - acc_t) * 100.0
            w.writerow([bucket, variant, "val",  f"{acc_vp:.4f}", f"{dv:.2f}", u_vp, t_vp])
            w.writerow([bucket, variant, "test", f"{acc_tp:.4f}", f"{dt:.2f}", u_tp, t_tp])

    print(f"\nSaved: {csv_path}")
    print("Head:")
    with open(csv_path) as f:
        for i, line in zip(range(8), f):
            print(line.strip())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/aligned")
    ap.add_argument("--pack", default="buffalo_l", help="recognizer pack under ~/.insightface/models")
    args = ap.parse_args()
    main(args.data, args.pack)
