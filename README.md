## Facial Recognition Mini-Project

This repo implements a compact facial recognition pipeline with classical and deep approaches, identity-preserving splits, and basic evaluation/plots.

### What’s already implemented
- **Datasets & splits**: Per-identity listing and deterministic train/val/test splits with minimum images per identity.
  - Code: `src/data_utils.py` → `list_id_folders`, `split_idwise`, `load_gray`.
- **Classical approaches (2/2)**:
  - **Eigenfaces (PCA + LinearSVC)**: `src/classical.py::make_eigenfaces`
  - **LBP histograms + LinearSVC**: `src/classical.py::make_lbp`
- **Classical alignment parity (NEW)**:
  - Optional MTCNN-aligned crops for classical methods via `--align-classical` to fairly compare with deep alignment.
- **Deep approach (1/1)**:
  - **Face detection/alignment + embeddings** using `facenet-pytorch` (MTCNN + InceptionResnetV1 pretrained on VGGFace2).
  - Classification heads:
    - **Cosine-to-class-centroids** with optional threshold for “unknown”.
    - **LinearSVC** on embeddings (toggle via `--deep-head`).
  - Code: `src/deep.py` (`FaceEmbedder`, `compute_centroids`, `cosine_predict`, `make_svm_head`).
- **Evaluation & artifacts**:
  - Metrics: accuracy, macro-precision/recall/F1.
  - Confusion matrices saved as PNGs, metrics saved as JSON.
  - Code: `src/metrics.py` (`evaluate`, `plot_confusion`, `save_json`).
- **Unified runner**: `src/run.py` orchestrates splits, classical training/eval, and optional deep eval. Outputs plots/metrics to `plots/`.
- **Evidence of runs**: `plots/cm_*.png`, `plots/metrics_*.json`, `plots/preds_deep.json` exist.
- **Bucket evaluations (NEW)**:
  - Optional severity sweeps for lighting, image quality (noise/blur/JPEG), and occlusions (eyes/mouth). Saves per-severity metrics under `plots/buckets_*.json`.

### What’s still pending (from your breakdown)
- **Detection & alignment for classical**: Currently classical pipelines use center-cropped grayscale; add eye-based alignment or reuse MTCNN crops for parity with deep pipeline.
  - Status: Implemented parity via `--align-classical`; further improvement could be eye-landmark alignment.
- **Five evaluation buckets (beyond raw accuracy)**: lighting, quality (noise/blur/JPEG), occlusions, explainability, bias analysis.
  - `src/aug.py` now includes brightness/contrast, Gaussian noise, blur, JPEG compression, and eye/mouth occlusions.
  - Severity-sweep evaluations implemented and saved as JSON. Plotting utilities for curves/tables still to add.
  - `src/explain.py` and `src/bias.py` remain to be implemented.
- **Crowd recognition test**: `src/crowd.py` exists but not implemented; build full-image face detection → per-face classify → aggregate metrics.
- **Demo (CLI/notebook)**: Simple CLI to run detect→align→recognize on an arbitrary image and visualize bounding boxes/labels with “Unknown” thresholding.
- **Report (4–6 pages)**: Structure outlined in your brief but not in repo; add under `report/`.
- **Bonus: AI-generated faces**: Not started.

### Suggested project structure
Current `src/` already aligns well. To make progress discoverable, prefer the following folders (create on first use):

```
src/
  data_utils.py      # listing/splitting/loading
  classical.py       # eigenfaces/lbp (done)
  deep.py            # embeddings + heads (done)
  aug.py             # lighting/quality/occlusion transforms (implement)
  explain.py         # eigenfaces/LBP maps/Grad-CAM (implement)
  bias.py            # subgroup metrics and plots (implement)
  crowd.py           # crowd recognition pipeline (implement)
  metrics.py         # metrics + confusion (done)
  run.py             # main training/eval entry (done)

plots/               # metrics_*.json, cm_*.png, qualitative panels
experiments/         # configs + logs per run (add when needed)
figures/             # explainability/bias/crowd visuals
report/              # write-up (PDF/TeX/Notebook)
```

### How to run
Install deps (Python 3.10+ recommended):

```bash
pip install -r requirements.txt
```

Run classical baselines (and optionally deep) on a dataset root (each identity = folder):

```bash
python -m src.run \
  --data-root data/celebs_subset \
  --image-size 160 \
  --train-per-id 5 --val-per-id 2 --test-per-id 1 \
  --min-images-per-id 8 \
  --pca-components 150 \
  --lbp-radius 2 --lbp-points 16 \
  --outdir plots \
  --align-classical \
  --use-deep --deep-head cosine --deep-threshold 0.55 \
  --eval-buckets

Bucket outputs will be written to `plots/` as `buckets_lighting.json`, `buckets_noise.json`, `buckets_blur.json`, `buckets_jpeg.json`, `buckets_occl_eyes.json`, and `buckets_occl_mouth.json`. Each JSON contains per-model metrics keyed by severity.

To visualize curves (accuracy vs severity) and Δ% vs clean, you can load these JSONs and plot using your preferred tool. If you’d like, we can add a helper plotting script next.
```

Artifacts will be written to `plots/`:
- `metrics_eigen.json`, `metrics_lbp.json`, `metrics_deep.json`
- `cm_eigen.png`, `cm_lbp.png`, `cm_deep.png`
- `preds_deep.json` (per-image predictions)

### Recommended next steps (priority order)
1) **Classical alignment parity**: Use MTCNN-aligned crops for classical methods to fairly compare vs deep.
2) **Evaluation buckets**: Implement lighting/quality/occlusion transforms in `src/aug.py` + batch evaluators; save per-severity curves/tables.
3) **Explainability**: 
   - Eigenfaces: visualize top PCA components and example reconstructions.
   - LBP: heatmaps overlaid on faces.
   - Deep: Grad-CAM on backbone or linear head.
4) **Bias analysis**: Add lightweight attribute tags or import CelebA attributes; compute per-group accuracy and gaps.
5) **Crowd test**: Full-image detect→align→classify→evaluate precision/recall; export qualitative panels.
6) **Demo CLI**: Single-image inference with boxes, labels, confidence, and “Unknown” thresholding.
7) **Report**: Summarize methods/datasets/results; include bucket plots and qualitative insights.

### Notes & tips
- Use `--deep-head svm` to compare shallow SVM vs cosine centroids on embeddings.
- `--deep-threshold` is auto-tuned on the val split when using cosine; a default is provided.
- Ensure each identity has ≥ `min_images_per_id` and can satisfy train/val/test counts, or it will be skipped.

### License and credits
- Embeddings and detection use `facenet-pytorch` (MTCNN + InceptionResnetV1 pretrained on VGGFace2). Cite appropriately in your report.


