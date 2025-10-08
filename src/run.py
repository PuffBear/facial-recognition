# src/run.py
import os
import argparse
import json
import numpy as np
from typing import List, Tuple
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # avoid crash on partial files

from src import data_utils as DU
from src import classical as CL
from src import metrics as MT
from src import deep as DP  # MTCNN + FaceNet embeddings
from src import aug as AG


def load_split_gray(pairs: List[Tuple[str, str]], size: int = 160, align_fn=None):
    """Load (path,label) into grayscale arrays + labels, with optional alignment."""
    X, y = [], []
    for p, lab in pairs:
        try:
            pil = Image.open(p).convert("RGB")
        except Exception:
            pil = Image.new("RGB", (size, size), (0, 0, 0))
        if align_fn is not None:
            try:
                pil = align_fn(pil)
            except Exception:
                pil = pil.resize((size, size))
        pil = pil.resize((size, size)).convert("L")
        X.append(np.array(pil, dtype=np.uint8))
        y.append(lab)
    return X, y


def main(a):
    os.makedirs(a.outdir, exist_ok=True)

    # -------- data & splits --------
    id2 = DU.list_id_folders(a.data_root, min_images_per_id=a.min_images_per_id)
    splits = DU.split_idwise(
        id2, train=a.train_per_id, val=a.val_per_id, test=a.test_per_id
    )
    classes = sorted({lab for _, lab in splits["train"]})
    if not classes:
        raise SystemExit("No identities met the min_images_per_id requirement.")

    # Optional aligner (MTCNN) used for classical parity when requested
    aligner = None
    if a.align_classical:
        aligner = DP.FaceEmbedder(size=a.image_size, use_pretrained=True).align

    # -------- load grayscale for classical --------
    Xtr, ytr = load_split_gray(splits["train"], size=a.image_size, align_fn=aligner)
    Xva, yva = load_split_gray(splits["val"], size=a.image_size, align_fn=aligner)
    Xte, yte = load_split_gray(splits["test"], size=a.image_size, align_fn=aligner)

    # ===== Eigenfaces =====
    eig_clf, eig_x = CL.make_eigenfaces(pca_components=a.pca_components)
    eig_clf.fit(np.stack([eig_x(x) for x in Xtr]), ytr)

    def eig_pred(x):
        return eig_clf.predict([eig_x(x)])[0]

    eig_metrics, eig_yhat = MT.evaluate(eig_pred, Xte, yte)
    MT.save_json(eig_metrics, os.path.join(a.outdir, "metrics_eigen.json"))
    MT.plot_confusion(yte, eig_yhat, classes, os.path.join(a.outdir, "cm_eigen.png"))
    print("[Eigenfaces]", eig_metrics)

    # ===== LBP =====
    lbp_clf, lbp_x = CL.make_lbp(radius=a.lbp_radius, points=a.lbp_points)
    lbp_clf.fit(np.stack([lbp_x(x) for x in Xtr]), ytr)

    def lbp_pred(x):
        return lbp_clf.predict([lbp_x(x)])[0]

    lbp_metrics, lbp_yhat = MT.evaluate(lbp_pred, Xte, yte)
    MT.save_json(lbp_metrics, os.path.join(a.outdir, "metrics_lbp.json"))
    MT.plot_confusion(yte, lbp_yhat, classes, os.path.join(a.outdir, "cm_lbp.png"))
    print("[LBP]", lbp_metrics)

    # ===== Deep (optional) =====
    if a.use_deep:
        E = DP.FaceEmbedder(size=a.image_size, use_pretrained=True)

        def embed_path(pth: str):
            """Align + embed with robust fallbacks for bad files."""
            try:
                pil = Image.open(pth).convert("RGB")
            except Exception:
                pil = Image.new("RGB", (a.image_size, a.image_size), (0, 0, 0))
            pil_al = E.align(pil)
            return E.embed(pil_al)

        # Build train embeddings
        tr_emb, tr_lab = [], []
        for path, lab in splits["train"]:
            tr_emb.append(embed_path(path))
            tr_lab.append(lab)
        if not tr_emb:
            raise SystemExit(
                "Deep: no training embeddings — check your splits/min-images-per-id."
            )
        tr_emb = np.stack(tr_emb, axis=0)

        if a.deep_head == "svm":
            head = DP.make_svm_head()
            head.fit(tr_emb, tr_lab)

            def pred_path(pth: str) -> str:
                e = embed_path(pth)
                return head.predict([e])[0]

            cm_classes = classes

        else:
            cents = DP.compute_centroids(tr_emb, tr_lab)

            # Auto-tune threshold on the val split (if available)
            if splits["val"]:
                best_thr, best_f1 = a.deep_threshold, -1.0
                for t in np.arange(0.40, 0.76, 0.02):
                    yv = [lab for _, lab in splits["val"]]
                    yvhat = []
                    for p, _ in splits["val"]:
                        lab, _ = DP.cosine_predict(embed_path(p), cents, threshold=t)
                        yvhat.append(lab)
                    m, _ = MT.evaluate(lambda v: v, yvhat, yv)  # identity predict_fn
                    if m["macro_f1"] > best_f1:
                        best_f1, best_thr = m["macro_f1"], float(t)
                a.deep_threshold = best_thr
                print(
                    f"[Deep] tuned threshold on val: {a.deep_threshold:.2f} (val macro-F1={best_f1:.3f})"
                )

            def pred_path(pth: str) -> str:
                lab, _ = DP.cosine_predict(embed_path(pth), cents, threshold=a.deep_threshold)
                return lab

            cm_classes = classes + ["unknown"]

        # Evaluate on test
        y_true = [lab for _, lab in splits["test"]]
        y_pred = [pred_path(p) for p, _ in splits["test"]]

        deep_metrics, _ = MT.evaluate(lambda v: v, y_pred, y_true)
        MT.save_json(deep_metrics, os.path.join(a.outdir, "metrics_deep.json"))
        MT.plot_confusion(y_true, y_pred, cm_classes, os.path.join(a.outdir, "cm_deep.png"))
        with open(os.path.join(a.outdir, "preds_deep.json"), "w") as f:
            json.dump(
                [{"path": p, "true": t, "pred": h} for (p, t), h in zip(splits["test"], y_pred)],
                f,
                indent=2,
            )
        print("[Deep]", deep_metrics)

    # ===== Bucketed Stress Tests =====
    if a.eval_buckets:
        os.makedirs(a.outdir, exist_ok=True)

        def prepare_gray_from_pil(pil):
            if aligner is not None:
                pil = aligner(pil)
            pil = pil.resize((a.image_size, a.image_size)).convert("L")
            return np.array(pil, dtype=np.uint8)

        # Classical predictors on PIL via grayscale conversion
        def pred_eigen_pil(pil_img):
            gx = prepare_gray_from_pil(pil_img)
            return eig_clf.predict([eig_x(gx)])[0]

        def pred_lbp_pil(pil_img):
            gx = prepare_gray_from_pil(pil_img)
            return lbp_clf.predict([lbp_x(gx)])[0]

        # Deep predictor on PIL if deep was enabled
        deep_enabled = a.use_deep
        if deep_enabled:
            if a.deep_head == "svm":
                def pred_deep_pil(pil_img):
                    emb = E.embed(E.align(pil_img.convert("RGB")))
                    return head.predict([emb])[0]
            else:
                def pred_deep_pil(pil_img):
                    emb = E.embed(E.align(pil_img.convert("RGB")))
                    lab, _ = DP.cosine_predict(emb, cents, threshold=a.deep_threshold)
                    return lab

        def eval_bucket(transform_fn, severities, name_prefix):
            # Evaluate on transformed test set and save per-severity metrics for each model
            results = {"eigen": {}, "lbp": {}}
            if deep_enabled:
                results["deep"] = {}

            for sev_name, kwargs in severities:
                ys, ye, yl, yd = [], [], [], []
                for p, lab in splits["test"]:
                    try:
                        pil = Image.open(p).convert("RGB")
                    except Exception:
                        pil = Image.new("RGB", (a.image_size, a.image_size), (0, 0, 0))
                    pil_t = transform_fn(pil, **kwargs)
                    ys.append(lab)
                    ye.append(pred_eigen_pil(pil_t))
                    yl.append(pred_lbp_pil(pil_t))
                    if deep_enabled:
                        yd.append(pred_deep_pil(pil_t))

                me, _ = MT.evaluate(lambda v: v, ye, ys)
                ml, _ = MT.evaluate(lambda v: v, yl, ys)
                results["eigen"][sev_name] = me
                results["lbp"][sev_name] = ml
                if deep_enabled:
                    md, _ = MT.evaluate(lambda v: v, yd, ys)
                    results["deep"][sev_name] = md

            MT.save_json(results, os.path.join(a.outdir, f"buckets_{name_prefix}.json"))

        # Lighting: brightness and contrast sweeps
        lighting_sev = [
            ("b0.7_c0.8", {"brightness": 0.7, "contrast": 0.8}),
            ("b0.85_c0.9", {"brightness": 0.85, "contrast": 0.9}),
            ("b1.15_c1.1", {"brightness": 1.15, "contrast": 1.1}),
            ("b1.3_c1.2", {"brightness": 1.3, "contrast": 1.2}),
        ]
        eval_bucket(lambda im, **k: AG.adjust_brightness_contrast(im, **k), lighting_sev, "lighting")

        # Image quality: noise, blur, jpeg
        noise_sev = [(f"sigma{int(s)}", {"sigma": float(s)}) for s in [5, 10, 20, 30]]
        eval_bucket(lambda im, **k: AG.add_gaussian_noise(im, **k), noise_sev, "noise")

        blur_sev = [(f"r{r}", {"radius": float(r)}) for r in [1.0, 2.0, 3.0]]
        eval_bucket(lambda im, **k: AG.gaussian_blur(im, **k), blur_sev, "blur")

        jpeg_sev = [(f"q{q}", {"quality": int(q)}) for q in [70, 50, 30, 15]]
        eval_bucket(lambda im, **k: AG.jpeg_compress(im, **k), jpeg_sev, "jpeg")

        # Occlusions: eyes and mouth
        eyes_sev = [(f"eyes_h{h}", {"frac_height": h}) for h in [0.15, 0.20, 0.25]]
        eval_bucket(lambda im, **k: AG.occlude_eyes(im, **k), eyes_sev, "occl_eyes")

        mouth_sev = [(f"mouth_h{h}", {"frac_height": h}) for h in [0.20, 0.25, 0.30]]
        eval_bucket(lambda im, **k: AG.occlude_mouth(im, **k), mouth_sev, "occl_mouth")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--image-size", type=int, default=160)

    ap.add_argument("--train-per-id", type=int, default=5)
    ap.add_argument("--val-per-id", type=int, default=2)
    ap.add_argument("--test-per-id", type=int, default=1)
    ap.add_argument("--min-images-per-id", type=int, default=8)

    ap.add_argument("--pca-components", type=int, default=150)
    ap.add_argument("--lbp-radius", type=int, default=2)
    ap.add_argument("--lbp-points", type=int, default=16)  # typical: 8, 16, 24

    ap.add_argument("--outdir", default="plots")
    ap.add_argument("--align-classical", action="store_true", help="Use MTCNN-aligned crops for classical methods")
    ap.add_argument("--eval-buckets", action="store_true", help="Run lighting/quality/occlusion severity sweeps and save metrics")

    # Deep args
    ap.add_argument("--use-deep", action="store_true")
    ap.add_argument("--deep-head", choices=["cosine", "svm"], default="cosine")
    ap.add_argument("--deep-threshold", type=float, default=0.55)

    args = ap.parse_args()
    main(args)
