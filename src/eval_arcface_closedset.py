#!/usr/bin/env python3
import argparse, os
from pathlib import Path
import cv2, numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix

from insightface.app import FaceAnalysis

def list_images(root):
    paths, labels = [], []
    for cls in sorted(Path(root).iterdir()):
        if not cls.is_dir(): continue
        for p in cls.glob("*.jpg"):
            paths.append(str(p)); labels.append(cls.name)
    return paths, labels

def embed_images(paths, app):
    embs = []
    for p in tqdm(paths, desc="embeddings", leave=False):
        img = cv2.imread(p)
        if img is None: embs.append(None); continue
        faces = app.get(img)  # already aligned (112x112), single face expected
        if not faces: embs.append(None); continue
        embs.append(faces[0].normed_embedding)  # 512-D L2-normalized
    return np.array([e if e is not None else np.zeros(512) for e in embs])

def build_prototypes(train_embs, train_labels):
    buckets = defaultdict(list)
    for e, y in zip(train_embs, train_labels):
        if e is None or not np.any(e): continue
        buckets[y].append(e)
    ids = sorted(buckets.keys())
    protos = np.stack([np.mean(buckets[k], axis=0) for k in ids], axis=0)  # [C,512]
    # re-L2 normalize prototypes
    protos = protos / (np.linalg.norm(protos, axis=1, keepdims=True)+1e-9)
    return ids, protos

def predict(embs, ids, protos):
    # cosine similarity to class prototypes
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True)+1e-9)
    sims = embs @ protos.T                     # [N,C]
    idx = np.argmax(sims, axis=1)              # best class
    return [ids[i] for i in idx], sims.max(1)

def main(data_root, out_txt):
    # load aligned splits
    train_root = Path(data_root)/"train"
    val_root   = Path(data_root)/"val"
    test_root  = Path(data_root)/"test"

    train_paths, train_y = list_images(train_root)
    val_paths,   val_y   = list_images(val_root)
    test_paths,  test_y  = list_images(test_root)

    # embedder (recognizer on)
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(224,224))  # CPU ok; crops are already faces

    train_E = embed_images(train_paths, app)
    ids, protos = build_prototypes(train_E, train_y)

    # evaluate on val (threshold tuning hook if you add Unknown later)
    val_E = embed_images(val_paths, app)
    val_pred, _ = predict(val_E, ids, protos)
    print("\n=== VAL REPORT ===")
    print(classification_report(val_y, val_pred, digits=3))

    # test
    test_E = embed_images(test_paths, app)
    test_pred, _ = predict(test_E, ids, protos)
    print("\n=== TEST REPORT ===")
    print(classification_report(test_y, test_pred, digits=3))
    cm = confusion_matrix(test_y, test_pred, labels=ids)
    print("Confusion matrix shape:", cm.shape)

    if out_txt:
        Path(out_txt).parent.mkdir(parents=True, exist_ok=True)
        with open(out_txt, "w") as f:
            f.write("IDS:\n"+",".join(ids)+"\n")
            f.write("Confusion (rows=true, cols=pred):\n")
            np.savetxt(f, cm, fmt="%d")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/aligned", help="aligned data root with train/val/test")
    ap.add_argument("--out", default="runs/arcface_base/confusion.txt")
    args = ap.parse_args()
    main(args.data, args.out)
