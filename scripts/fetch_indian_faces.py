# scripts/fetch_indian_faces.py
import os, glob, random, shutil
from pathlib import Path
from typing import Optional, List, Tuple

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def download_kaggle_dataset(dataset: str, dest: Path):
    """
    Downloads & unzips a Kaggle dataset using Kaggle API credentials.
    Prereq: ~/.kaggle/kaggle.json (chmod 600). Also be sure you've accepted the dataset's terms on Kaggle.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi
    ensure_dir(dest)
    api = KaggleApi(); api.authenticate()
    print(f"[+] Downloading {dataset} to {dest} ...")
    api.dataset_download_files(dataset, path=str(dest), unzip=True)
    print("[+] Download complete.")

def list_id_folders(root: Path, min_images_per_id: int) -> List[Tuple[str, list]]:
    ids = []
    for person in sorted(root.iterdir()):
        if not person.is_dir(): continue
        imgs = []
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
            imgs += glob.glob(str(person / ext))
        if len(imgs) >= min_images_per_id:
            ids.append((person.name, sorted(imgs)))
    return ids

def make_subset(src_root: Path,
                dst_root: Path,
                min_per_id: int = 8,
                max_per_id: int = 20,
                max_identities: Optional[int] = 50,
                seed: int = 1337):
    """
    Creates a tidy, balanced subset for fast iteration:
      - keep IDs with >= min_per_id images
      - cap per-ID images to max_per_id
      - optionally cap number of identities
    """
    ensure_dir(dst_root)
    rng = random.Random(seed)

    ids = list_id_folders(src_root, min_per_id)
    if not ids:
        raise SystemExit(f"No identities with >= {min_per_id} images found under {src_root}")

    if max_identities:
        rng.shuffle(ids)
        ids = ids[:max_identities]

    print(f"[+] Subsetting from {len(ids)} identities → {dst_root}")
    for pid, imgs in ids:
        rng.shuffle(imgs)
        keep = imgs[:max_per_id]
        outdir = dst_root / pid
        ensure_dir(outdir)
        for i, src in enumerate(keep):
            ext = os.path.splitext(src)[1].lower()
            shutil.copy2(src, outdir / f"{pid}_{i:04d}{ext}")

    # quick stats
    total = sum(len(list((dst_root/p).glob("*"))) for p in os.listdir(dst_root) if (dst_root/p).is_dir())
    print(f"[+] Subset ready: IDs={len(ids)}, images={total}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="aryankashyapnaveen/indian-face-dataset")
    ap.add_argument("--raw-dir", default="data/indian_face_full")
    ap.add_argument("--subset-dir", default="data/celebs_subset")
    ap.add_argument("--min-per-id", type=int, default=8)
    ap.add_argument("--max-per-id", type=int, default=20)
    ap.add_argument("--max-identities", type=int, default=50)  # set 0 to keep all eligible
    ap.add_argument("--skip-download", action="store_true", help="If data already present in raw-dir")
    args = ap.parse_args()

    raw = Path(args.raw_dir); subset = Path(args.subset_dir)
    if not args.skip_download:
        download_kaggle_dataset(args.dataset, raw)
    if args.max_identities <= 0:
        args.max_identities = None
    make_subset(raw, subset, args.min_per_id, args.max_per_id, args.max_identities)
