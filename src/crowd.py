# src/crowd.py
import json, math
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
from facenet_pytorch import MTCNN  # detector+aligner
# Reuse your deep utilities (names come from your README)
from src.deep import FaceEmbedder, compute_centroids, cosine_predict

# -------------------- IOU + matching --------------------
def iou_xywh(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_w = max(0, min(ax2, bx2) - max(ax, bx))
    inter_h = max(0, min(ay2, by2) - max(ay, by))
    inter = inter_w * inter_h
    union = aw*ah + bw*bh - inter + 1e-9
    return inter / union

def match_dets_to_gt(dets, gts, iou_thr=0.5):
    # dets: [(bbox, pred_id, score)], gts: [(bbox, true_id)]
    matches, used_gt = [], set()
    for i, (dbb, pid, sc) in enumerate(dets):
        best_j, best_iou = -1, 0.0
        for j, (gbb, tid) in enumerate(gts):
            if j in used_gt: 
                continue
            iou = iou_xywh(dbb, gbb)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thr:
            matches.append((i, best_j))
            used_gt.add(best_j)
    return matches, used_gt

# -------------------- drawing --------------------
def draw_boxes(img: Image.Image, dets, out_path):
    im = img.copy()
    draw = ImageDraw.Draw(im)
    for (x,y,w,h), label, score in dets:
        draw.rectangle([x, y, x+w, y+h], outline=(0,255,0), width=2)
        txt = f"{label} ({score:.2f})"
        draw.rectangle([x, y-18, x+8*len(txt), y], fill=(0,255,0))
        draw.text((x+2, y-16), txt, fill=(0,0,0))
    im.save(out_path)

# -------------------- main crowd eval --------------------
def load_gt(gt_path: Path) -> Dict[str, List[Dict]]:
    return json.loads(gt_path.read_text())

def to_xywh(box_xyxy):
    x1,y1,x2,y2 = [float(v) for v in box_xyxy]
    return [x1, y1, max(0.0, x2-x1), max(0.0, y2-y1)]

@torch.no_grad()
def run(
    images_dir: Path,
    gt_json: Path,
    gallery_root: Path,
    out_dir: Path,
    threshold: float = 0.55,   # start here; you can auto-tune on val
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"; fig_dir.mkdir(exist_ok=True)

    # 1) gallery → centroids
    embedder = FaceEmbedder(device=device)      # uses InceptionResnetV1 per your deep.py
    mtcnn = MTCNN(keep_all=True, device=device) # detector+aligner
    ids, centroids = compute_centroids(gallery_root, embedder)  # returns (List[str], np.ndarray [C,D])

    # 2) load ground truth
    gt = load_gt(gt_json)

    per_image = []
    tp_id = fp_id = fn_id = 0
    tp_det = fp_det = fn_det = 0

    for img_path in sorted(Path(images_dir).glob("*")):
        if img_path.suffix.lower() not in {".jpg",".jpeg",".png"}: 
            continue

        img = Image.open(img_path).convert("RGB")

        # Detect faces + aligned crops
        boxes, probs = mtcnn.detect(img)
        dets = []
        if boxes is not None and len(boxes) > 0:
            # get aligned tensors for each box
            face_tensors = mtcnn.extract(img, boxes, save_path=None)  # [N, 3, 160, 160]
            embs = embedder.encode_batch(face_tensors.to(device))     # shape [N, D]; adjust name if needed

            # cosine to centroids
            for k in range(len(embs)):
                pred_id, score = cosine_predict(embs[k].cpu().numpy(), ids, centroids)
                if score < threshold:
                    pred_id = "Unknown"
                dets.append((to_xywh(boxes[k]), pred_id, float(score)))
        else:
            dets = []

        # Evaluate against GT
        gts = [ (ann["bbox"], ann["id"]) for ann in gt.get(img_path.name, []) ]
        matches, used_gt = match_dets_to_gt(dets, gts, iou_thr=0.5)

        # detection metrics
        tp_det += len(matches)
        fp_det += (len(dets) - len(matches))
        fn_det += (len(gts) - len(used_gt))

        # identification on matched pairs
        for i, j in matches:
            _, pred_id, _ = dets[i]
            _, true_id = gts[j]
            if true_id == pred_id:
                tp_id += 1
            else:
                fp_id += 1
        fn_id += 0  # by definition we only score ID where detection matched

        # save visualization
        draw_boxes(img, dets, fig_dir / f"{img_path.stem}_pred.png")

        per_image.append({
            "image": img_path.name,
            "num_gt_faces": len(gts),
            "num_dets": len(dets),
            "num_matched": len(matches)
        })

    # aggregate metrics
    prec_det = tp_det / max(1, tp_det + fp_det)
    rec_det  = tp_det / max(1, tp_det + fn_det)
    f1_det   = 2 * prec_det * rec_det / max(1e-9, (prec_det + rec_det))

    acc_id   = tp_id / max(1, tp_id + fp_id)

    summary = {
        "threshold": threshold,
        "detection": {"TP": tp_det, "FP": fp_det, "FN": fn_det,
                      "precision": prec_det, "recall": rec_det, "f1": f1_det},
        "identification_on_matched": {"correct": tp_id, "incorrect": fp_id, "accuracy": acc_id},
        "notes": "ID metrics computed only on detections that overlap a GT face (IoU>=0.5)."
    }
    (out_dir / "crowd_metrics.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "crowd_per_image.json").write_text(json.dumps(per_image, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--gt-json", type=Path, required=True)
    ap.add_argument("--gallery-root", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("plots/crowd"))
    ap.add_argument("--threshold", type=float, default=0.55)
    args = ap.parse_args()
    run(args.images_dir, args.gt_json, args.gallery_root, args.outdir, args.threshold)
