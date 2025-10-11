#!/usr/bin/env python3
# Heatmap of cosine drop by sliding a black patch over an aligned face.
import argparse
from pathlib import Path
import numpy as np, cv2
from tqdm import tqdm
from insightface.model_zoo import get_model

def load_recognizer_any(pack_dir: Path):
    for n in ["w600k_r50.onnx","glintr100.onnx","iresnet100.onnx","w600k_r100.onnx"]:
        p = pack_dir/n
        if p.exists():
            rec = get_model(str(p)); rec.prepare(ctx_id=-1); return rec
    for p in sorted(pack_dir.glob("*.onnx")):
        name=p.name.lower()
        if any(k in name for k in ["det","landmark","gender","age","2d106","1k3d","scrfd"]): continue
        rec = get_model(str(p)); rec.prepare(ctx_id=-1); return rec
    raise FileNotFoundError(f"No recognizer in {pack_dir}")

def embed(rec, bgr):
    # normed embedding
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    for im in (bgr,rgb):
        for fn in ("get_embedding","get_feat"):
            if hasattr(rec, fn):
                try:
                    e = getattr(rec, fn)(im)
                    if e is not None:
                        e = np.asarray(e, np.float32).ravel()
                        e /= (np.linalg.norm(e)+1e-9)
                        return e
                except Exception:
                    pass
    return None

def centroid_for_class(rec, train_dir, cls):
    vecs=[]
    for p in Path(train_dir, cls).glob("*.*"):
        im=cv2.imread(str(p)); e=embed(rec, im)
        if e is not None: vecs.append(e)
    if not vecs: return None
    m = np.mean(np.stack(vecs), axis=0)
    m /= (np.linalg.norm(m)+1e-9)
    return m

def occlusion_map(rec, img, center, k=16, stride=8):
    H,W = img.shape[:2]; base = center @ embed(rec, img)
    heat = np.zeros(( (H-k)//stride+1, (W-k)//stride+1 ), np.float32)
    r=0
    for y in range(0, H-k+1, stride):
        c=0
        for x in range(0, W-k+1, stride):
            pert = img.copy()
            pert[y:y+k, x:x+k] = 0
            e = embed(rec, pert)
            if e is None: drop=0.0
            else: drop = max(0.0, base - (center @ e))
            heat[r,c]=drop
            c+=1
        r+=1
    heat = cv2.resize(heat, (W,H), interpolation=cv2.INTER_CUBIC)
    heat = heat / (heat.max()+1e-9)
    return heat

def main(data_root, pack, image_path, k, stride, out_png):
    pack_dir = Path.home()/".insightface"/"models"/pack
    rec = load_recognizer_any(pack_dir)
    img = cv2.imread(image_path)
    cls = Path(image_path).parent.name  # assume path .../<split>/<id>/file.jpg
    center = centroid_for_class(rec, str(Path(data_root)/"train"), cls)
    if center is None: raise RuntimeError(f"No train centroid for {cls}")
    heat = occlusion_map(rec, img, center, k=k, stride=stride)
    hm = cv2.applyColorMap((heat*255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, hm, 0.4, 0)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_png, overlay)
    print("Saved:", out_png)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/aligned")
    ap.add_argument("--pack", default="buffalo_l")
    ap.add_argument("--image", required=True, help="path to an aligned crop (val/test)")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--out", default="runs/explain/occlusion.png")
    args = ap.parse_args()
    main(args.data, args.pack, args.image, args.k, args.stride, args.out)
