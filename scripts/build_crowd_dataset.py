#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end crowd dataset builder for your layout:

data/
  celebs_subset/<ID>/*.jpg     # gallery identities (already there)
  crowd/
    images/                    # downloaded group photos (will be created)
    centroids.npz              # gallery centroids (auto)
    draft_gt.json              # auto-suggested labels (auto)
    gt.json                    # final ground truth (after confirmation)
    splits/{val.txt,test.txt}  # split lists (auto)

Steps:
1) Read identity folder names from --gallery-root.
2) Query Wikimedia Commons for CC group/event photos per identity (+ optional extra queries).
3) Download images to --crowd-dir/images (with polite rate limit, metadata log, and MD5 dedupe).
4) Build centroids from gallery images using facenet-pytorch (MTCNN + InceptionResnetV1).
5) Pre-label faces in crowd images with auto identities (cosine to centroids).
6) (Interactive) Ask you to accept/correct each label quickly.
7) Make val/test split files.

Run:
  python scripts/build_crowd_dataset.py \
      --gallery-root data/celebs_subset \
      --crowd-dir data/crowd \
      --limit 25 --per-id-max 20 --val-ratio 0.2

Requires: facenet-pytorch, numpy, pillow, requests, tqdm
"""

import argparse, hashlib, io, json, os, random, re, time
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# -------------------- utils --------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def md5_bytes(b: bytes) -> str:
    h = hashlib.md5(); h.update(b); return h.hexdigest()

def camel_or_snake_to_words(s: str) -> str:
    s = s.replace("_", " ")
    s = re.sub(r'(?<!^)(?=[A-Z])', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def normalize_id_for_query(idname: str) -> str:
    w = camel_or_snake_to_words(idname)
    # handle very short tokens / acronyms like "ANR"
    if len(w) <= 3 and w.isupper():
        return w
    return w

def l2n(t: torch.Tensor) -> torch.Tensor:
    return t / (t.norm(dim=-1, keepdim=True) + 1e-9)

# -------------------- 1) gather identities --------------------

def list_identity_names(gallery_root: Path) -> List[str]:
    ids = [p.name for p in gallery_root.iterdir() if p.is_dir()]
    ids = sorted(ids)
    return ids

# -------------------- 2) Wikimedia Commons search/download --------------------

WIKI_API = "https://commons.wikimedia.org/w/api.php"

def commons_search(query: str, limit: int = 30) -> List[Dict]:
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query + " filetype:bitmap",
        "gsrlimit": limit,
        "prop": "imageinfo|info",
        "inprop": "url",
        "iiprop": "url|mime|size",
        "iiurlwidth": 2048,
        "format": "json"
    }
    r = requests.get(WIKI_API, params=params, timeout=20)
    r.raise_for_status()
    j = r.json()
    out = []
    for p in j.get("query", {}).get("pages", {}).values():
        ii = (p.get("imageinfo") or [{}])[0]
        url = ii.get("thumburl") or ii.get("url")
        if not url: 
            continue
        if not url.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        out.append({
            "title": p.get("title"),
            "url": url,
            "descurl": p.get("fullurl", ""),
        })
    return out

def polite_get(url: str) -> bytes:
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        return r.content

def download_images_to(images_dir: Path, items: List[Dict], download_log: Path, sleep_sec: float = 0.3):
    ensure_dir(images_dir)
    log = []
    if download_log.exists():
        try:
            log = json.loads(download_log.read_text())
        except Exception:
            log = []
    seen = {e.get("md5") for e in log}

    kept = 0
    for it in items:
        try:
            data = polite_get(it["url"])
            digest = md5_bytes(data)
            if digest in seen:
                continue
            img = Image.open(io.BytesIO(data)).convert("RGB")
            # simple heuristic: keep reasonably wide images
            if img.width < 600 or img.height < 400:
                continue
            name = it["url"].split("/")[-1].split("?")[0]
            path = images_dir / name
            # avoid collisions
            if path.exists():
                stem = path.stem
                ext = path.suffix
                i = 1
                while (images_dir / f"{stem}_{i}{ext}").exists():
                    i += 1
                path = images_dir / f"{stem}_{i}{ext}"
            img.save(path, quality=95)
            log.append({
                "file": path.name,
                "url": it["url"],
                "title": it.get("title",""),
                "descurl": it.get("descurl",""),
                "md5": digest
            })
            kept += 1
            time.sleep(sleep_sec)
        except Exception as e:
            # skip silently but keep going
            continue
    download_log.write_text(json.dumps(log, indent=2))
    return kept

# -------------------- 3) build gallery centroids --------------------

@torch.no_grad()
def build_centroids(gallery_root: Path, out_npz: Path, per_id_max: int, device: str):
    dev = torch.device(device)
    mtcnn = MTCNN(keep_all=True, device=dev)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(dev)

    ids, embs = [], []
    id_dirs = [p for p in gallery_root.iterdir() if p.is_dir()]
    for id_dir in tqdm(sorted(id_dirs), desc="Centroids"):
        samples = []
        imgs = sorted([p for p in id_dir.glob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}])[:per_id_max]
        for p in imgs:
            try:
                img = Image.open(p).convert("RGB")
                boxes, probs = mtcnn.detect(img)
                if boxes is None:
                    continue
                k = int(np.argmax(probs))
                face = mtcnn.extract(img, boxes[[k]])  # [1,3,160,160]
                if face is None:
                    continue
                emb = resnet(l2n(face.to(dev))).squeeze(0).cpu()  # [512]
                samples.append(emb)
            except Exception:
                continue
        if len(samples) >= 3:
            ids.append(id_dir.name)
            embs.append(torch.stack(samples).mean(0))
    if not ids:
        raise RuntimeError("No centroids created. Check gallery images and detection.")
    ids = np.array(ids)
    embs = l2n(torch.stack(embs)).cpu().numpy()
    ensure_dir(out_npz.parent)
    np.savez(out_npz, ids=ids, embs=embs)

# -------------------- 4) prelabel faces in crowd images --------------------

@torch.no_grad()
def prelabel(images_dir: Path, centroids_npz: Path, out_json: Path, device: str, thresh: float):
    dev = torch.device(device)
    data = np.load(centroids_npz, allow_pickle=True)
    ids = [str(x) for x in data["ids"]]
    C = torch.tensor(data["embs"], dtype=torch.float32, device=dev)  # [C,512]

    mtcnn = MTCNN(keep_all=True, device=dev)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(dev)

    out = {}
    imgs = [p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}]
    for p in tqdm(sorted(imgs), desc="Prelabel"):
        img = Image.open(p).convert("RGB")
        boxes, probs = mtcnn.detect(img)
        dets = []
        if boxes is not None and len(boxes) > 0:
            faces = mtcnn.extract(img, boxes)  # [N,3,160,160]
            for i in range(faces.shape[0]):
                emb = l2n(resnet(l2n(faces[i:i+1].to(dev)))).squeeze(0)  # [512]
                sims = torch.matmul(emb, C.T)                             # [C]
                topv, topi = torch.topk(sims, k=min(5, C.shape[0]))
                best_score = float(topv[0].item())
                best_id = ids[int(topi[0].item())]
                label = best_id if best_score >= thresh else "Unknown"
                x1,y1,x2,y2 = boxes[i]
                dets.append({
                    "bbox":[float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    "auto_id": best_id, "score": best_score, "label": label,
                    "top5":[(ids[int(ii)], float(v)) for v,ii in zip(topv.tolist(), topi.tolist())]
                })
        out[p.name] = dets
    ensure_dir(out_json.parent)
    out_json.write_text(json.dumps(out, indent=2))

# -------------------- 5) quick interactive fixer --------------------

def interactive_fix(draft_json: Path, out_json: Path):
    draft = json.loads(draft_json.read_text())
    final = {}
    print("\nType the correct identity (exact folder name) or 'u' for Unknown.")
    print("Press Enter to accept the suggestion shown in [brackets].\n")
    for img, dets in draft.items():
        print(f"=== {img} | faces: {len(dets)} ===")
        final[img] = []
        for k, d in enumerate(dets):
            x,y,w,h = d["bbox"]
            suggestion = d["label"]
            top5 = ", ".join([f"{a}:{s:.2f}" for a,s in d.get("top5", [])])
            print(f" face#{k+1} bbox={[int(x),int(y),int(w),int(h)]}  auto=[{suggestion}]  top5={top5}")
            ans = input("  > id? ").strip()
            if ans == "":
                ans = suggestion
            if ans.lower() == "u":
                ans = "Unknown"
            final[img].append({"bbox":[x,y,w,h], "id": ans})
    out_json.write_text(json.dumps(final, indent=2))
    print(f"\nSaved ground-truth → {out_json}")

# -------------------- 6) make splits --------------------

def make_splits(images_dir: Path, out_dir: Path, val_ratio: float):
    imgs = sorted([p.name for p in images_dir.glob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}])
    random.seed(42); random.shuffle(imgs)
    n_val = max(1, int(len(imgs)*val_ratio))
    ensure_dir(out_dir)
    (out_dir/"val.txt").write_text("\n".join(imgs[:n_val])+"\n")
    (out_dir/"test.txt").write_text("\n".join(imgs[n_val:])+"\n")
    return len(imgs[:n_val]), len(imgs[n_val:])

# -------------------- Orchestrator --------------------

def build_queries_for_ids(id_names: List[str]) -> List[str]:
    base_keywords = ["group", "with", "cast", "awards", "premiere", "event"]
    queries = []
    for raw in id_names:
        person = normalize_id_for_query(raw)
        for kw in base_keywords:
            queries.append(f'"{person}" {kw}')
    # Generic open-set queries (no target faces)
    queries += [
        "Filmfare Awards audience",
        "Bollywood celebrities event group",
        "movie premiere crowd india",
        "press conference bollywood group"
    ]
    return queries

def main(args):
    gallery_root = Path(args.gallery_root)
    crowd_dir    = Path(args.crowd_dir)
    images_dir   = crowd_dir / "images"
    ensure_dir(images_dir)

    # 1) identities
    ids = list_identity_names(gallery_root)
    if not ids:
        raise SystemExit("No identities found in gallery.")
    print(f"Found {len(ids)} identities in {gallery_root}.")

    # 2) fetch images
    queries = build_queries_for_ids(ids)
    if args.extra_queries:
        queries += args.extra_queries
    # search + download
    all_items = []
    print("\nSearching Wikimedia Commons...")
    for q in tqdm(queries, desc="Queries"):
        try:
            all_items += commons_search(q, limit=args.limit)
            time.sleep(0.2)
        except Exception:
            continue
    # Download with dedupe + log
    kept = download_images_to(images_dir, all_items, crowd_dir/"download_log.json", sleep_sec=0.25)
    print(f"Downloaded {kept} images to {images_dir}")

    # 3) centroids
    centroids_npz = crowd_dir / "centroids.npz"
    print("\nBuilding gallery centroids...")
    build_centroids(gallery_root, centroids_npz, args.per_id_max, args.device)
    print(f"Saved centroids → {centroids_npz}")

    # 4) prelabel
    draft_json = crowd_dir / "draft_gt.json"
    print("\nPrelabeling faces (auto suggestions)...")
    prelabel(images_dir, centroids_npz, draft_json, args.device, args.auto_thresh)
    print(f"Wrote draft prelabels → {draft_json}")

    # 5) interactive fix
    print("\nNow confirming labels (quick CLI).")
    gt_json = crowd_dir / "gt.json"
    interactive_fix(draft_json, gt_json)

    # 6) splits
    print("\nCreating val/test splits...")
    nval, ntest = make_splits(images_dir, crowd_dir / "splits", args.val_ratio)
    print(f"val images: {nval} | test images: {ntest}")
    print("\nAll done ✅")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery-root", default="data/celebs_subset")
    ap.add_argument("--crowd-dir", default="data/crowd")
    ap.add_argument("--limit", type=int, default=25, help="images per query")
    ap.add_argument("--per-id-max", type=int, default=20, help="gallery imgs per ID for centroid")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--auto_thresh", type=float, default=0.70, help="cosine threshold for auto label")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--extra-queries", nargs="*", help="optional extra search queries")
    args = ap.parse_args()
    main(args)

