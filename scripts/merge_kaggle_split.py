# scripts/merge_kaggle_split.py
import os, glob, shutil
from pathlib import Path

ALLOWED = (".jpg",".jpeg",".png",".bmp",".webp")

def count_existing(dst_dir: Path):
    return len([f for f in dst_dir.iterdir() if f.is_file() and f.suffix.lower() in ALLOWED])

def merge_split(root="data/indian_face_full", out="data/indian_face_full_all"):
    root = Path(root); out = Path(out)
    (out).mkdir(parents=True, exist_ok=True)
    for split in ("train","val"):
        split_dir = root / split
        if not split_dir.exists(): 
            continue
        for person in sorted(split_dir.iterdir()):
            if not person.is_dir(): 
                continue
            dst = out / person.name
            dst.mkdir(parents=True, exist_ok=True)
            start = count_existing(dst)  # continue numbering if folder already has files
            idx = start
            for f in sorted(person.iterdir()):
                if f.suffix.lower() not in ALLOWED: 
                    continue
                idx += 1
                shutil.copy2(f, dst / f"{person.name}_{idx:05d}{f.suffix.lower()}")
    print(f"[+] Merged into {out}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/indian_face_full")
    ap.add_argument("--out", default="data/indian_face_full_all")
    args = ap.parse_args()
    merge_split(args.root, args.out)
