#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-group accuracy on TEST for a chosen model:
- deep:<pack>  (InsightFace recognition-only)
- classical:lbp_svm

Input: runs/bias/attributes.csv (id,group_col)
"""
import argparse, pandas as pd
from pathlib import Path
import numpy as np, cv2
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

IMG_EXTS={".jpg",".jpeg",".png",".bmp",".webp"}

def iter_split(root: Path):
    X,y=[],[]
    for cls in sorted(p.name for p in root.iterdir() if p.is_dir()):
        for p in (root/cls).iterdir():
            if p.suffix.lower() in IMG_EXTS:
                X.append(str(p)); y.append(cls)
    return X,y

# --- deep (recognition-only) ---
from insightface.model_zoo import get_model
def load_recognizer_any(pack_dir: Path):
    for n in ["w600k_r50.onnx","glintr100.onnx","iresnet100.onnx","w600k_r100.onnx"]:
        p = pack_dir/n
        if p.exists(): rec=get_model(str(p)); rec.prepare(ctx_id=-1); return rec
    for p in sorted(pack_dir.glob("*.onnx")):
        name=p.name.lower()
        if any(k in name for k in ["det","landmark","gender","age","2d106","1k3d","scrfd"]): continue
        rec=get_model(str(p)); rec.prepare(ctx_id=-1); return rec
    raise FileNotFoundError(f"No recognizer in {pack_dir}")
def embed_rec(rec,bgr):
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    for im in (bgr,rgb):
        for fn in ("get_embedding","get_feat"):
            if hasattr(rec,fn):
                try:
                    e=getattr(rec,fn)(im)
                    if e is not None:
                        e=np.asarray(e,np.float32).ravel()
                        e/= (np.linalg.norm(e)+1e-9); return e
                except: pass
    return None
def centroids(embs, labels):
    from collections import defaultdict
    by=defaultdict(list)
    for e,c in zip(embs,labels):
        if e is not None: by[c].append(e)
    cents={}
    for c,vecs in by.items():
        M=np.vstack(vecs); m=M.mean(0); m/= (np.linalg.norm(m)+1e-9)
        cents[c]=m
    return cents
def pred_cos(cents, embs):
    classes=list(cents.keys())
    C=np.vstack([cents[c] for c in classes]).T
    C/= (np.linalg.norm(C,0,keepdims=True)+1e-9)
    out=[]
    for e in embs:
        if e is None: out.append(None); continue
        out.append(classes[int(np.argmax(e@C))])
    return out

# --- classical LBP ---
def lbp_image(gray):
    H,W=gray.shape; codes=np.zeros((H-2,W-2),np.uint8); c=gray[1:-1,1:-1]
    nbrs=[gray[0:-2,0:-2],gray[0:-2,1:-1],gray[0:-2,2:],gray[1:-1,0:-2],gray[1:-1,2:],gray[2:,0:-2],gray[2:,1:-1],gray[2:,2:]]
    weights=[1,2,4,8,16,32,64,128]
    for w,nb in zip(weights,nbrs): codes |= (nb>=c).astype(np.uint8)*w
    return codes
def lbp_feats(bgr, grid=(8,8), bins=256):
    g=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY); codes=lbp_image(g); H,W=codes.shape
    gh,gw=grid; sh,sw=H//gh,W//gw; feats=[]
    for i in range(gh):
        for j in range(gw):
            patch=codes[i*sh:(i+1)*sh, j*sw:(j+1)*sw]
            h=np.bincount(patch.ravel(), minlength=bins).astype(np.float32); h/= (h.sum()+1e-9)
            feats.append(h)
    return np.concatenate(feats,0)

def main(data_root, model, attr_csv, group_col):
    root=Path(data_root)
    attrs=pd.read_csv(attr_csv).set_index("id")
    # splits
    Xtr,Ytr = iter_split(root/"train")
    Xte,Yte = iter_split(root/"test")

    if model.startswith("deep:"):
        pack=model.split(":",1)[1]
        rec=load_recognizer_any(Path.home()/".insightface/models"/pack)
        # train
        from collections import defaultdict
        Etr=[]
        for p in tqdm(Xtr, desc="embed train"):
            im=cv2.imread(p); Etr.append(embed_rec(rec, im))
        cents=centroids(Etr,Ytr)
        # test
        Ete=[]; 
        for p in tqdm(Xte, desc="embed test"):
            im=cv2.imread(p); Ete.append(embed_rec(rec, im))
        P=pred_cos(cents,Ete)
    elif model=="classical:lbp_svm":
        XtrF=np.vstack([lbp_feats(cv2.imread(p)) for p in tqdm(Xtr, desc="feats train")])
        clf=LinearSVC(C=1.0, class_weight="balanced", max_iter=5000).fit(XtrF, Ytr)
        XteF=np.vstack([lbp_feats(cv2.imread(p)) for p in tqdm(Xte, desc="feats test")])
        P=clf.predict(XteF).tolist()
    else:
        raise ValueError("Unknown model spec")

    # per-group accuracy
    df = []
    for y_true, pred in zip(Yte, P):
        grp = attrs.loc[y_true, group_col] if y_true in attrs.index else "unknown"
        df.append((grp, y_true, pred))
    import pandas as pd
    df = pd.DataFrame(df, columns=["group","true","pred"])
    res = df.groupby("group").apply(lambda g: accuracy_score(g["true"], g["pred"])).reset_index(name="accuracy")

    out_dir = Path("runs/bias"); out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir/"metrics.csv"
    res.to_csv(out_csv, index=False)
    print("Per-group accuracy →", out_csv)
    print(res)

if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/aligned")
    ap.add_argument("--model", default="deep:buffalo_l", help="deep:<pack> or classical:lbp_svm")
    ap.add_argument("--attr", default="runs/bias/attributes.csv")
    ap.add_argument("--group-col", default="gender")
    args = ap.parse_args()
    main(args.data, args.model, args.attr, args.group_col)
