#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Materialize a trimmed dataset by hardlinking (or copying) only selected IDs.

Inputs:
  - A text file with one identity name per line (e.g., runs/eda/ids_selected.txt)
  - A source root of per-ID folders (e.g., data/raw/<ID>/*.jpg)

Outputs:
  - A destination root with only those IDs (e.g., data/raw_selected/<ID>/*.jpg)
  - A CSV mapping of src->dst paths for reproducibility.

Usage:
  python scripts/materialize_selected.py \
    --ids runs/eda/ids_selected.txt \
    --src data/raw \
    --dst data/raw_selected \
    --per-id-max 0 \
    --randomize \
    --seed 42
"""

import argparse, os, shutil, csv, random
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}  # adjust if needed

def read_ids(ids_file: Path):
    ids = []
    for line in ids_file.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):  # allow comments
            continue
        ids.append(s)
    return ids

def iter_images(id_dir: Path):
    for p in id_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            yield p

def safe_link_or_copy(src: Path, dst: Path):
    # Create hardlink; if not possible (cross-device, permissions), copy
    try:
        os.link(src, dst)
        return "link"
    except Exception:
        shutil.copy2(src, dst)
        return "copy"

def main(ids_file: str, src_root: str, dst_root: str, per_id_max: int, randomize: bool, seed: int, map_out: str):
    ids_path = Path(ids_file)
    src = Path(src_root)
    dst = Path(dst_root)
    dst.mkdir(parents=True, exist_ok=True)

    ids = read_ids(ids_path)
    if not ids:
        raise SystemExit(f"No IDs found in {ids_path}")

    if randomize:
        random.seed(seed)

    total_kept = 0
    rows = []
    per_id_counts = {}

    for id_name in ids:
        sdir = src / id_name
        if not sdir.exists() or not sdir.is_dir():
            print(f"[warn] missing ID folder in src: {sdir}")
            continue

        files = list(iter_images(sdir))
        if not files:
            print(f"[warn] no images for ID: {id_name}")
            continue

        if randomize:
            random.shuffle(files)
        # cap per ID if requested (>0)
        if per_id_max and per_id_max > 0:
            files = files[:per_id_max]

        ddir = dst / id_name
        ddir.mkdir(parents=True, exist_ok=True)

        kept = 0
        for f in files:
            out = ddir / f.name
            # avoid collisions if a filename already exists
            if out.exists():
                stem, ext = out.stem, out.suffix
                i = 1
                while True:
                    cand = ddir / f"{stem}_{i}{ext}"
                    if not cand.exists():
                        out = cand
                        break
                    i += 1
            mode = safe_link_or_copy(f, out)
            kept += 1
            total_kept += 1
            rows.append((str(f), str(out), mode))

        per_id_counts[id_name] = kept
        print(f"[ok] {id_name}: kept {kept} files → {ddir}")

    # write mapping csv
    map_path = Path(map_out)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    with open(map_path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["src", "dst", "method"])
        w.writerows(rows)

    # summary
    print("\nSummary")
    for k in sorted(per_id_counts):
        print(f"  {k}: {per_id_counts[k]}")
    print(f"Total kept: {total_kept}")
    print(f"Mapping written to: {map_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="runs/eda/ids_selected.txt", help="file with one ID per line")
    ap.add_argument("--src", default="data/raw", help="source root containing per-ID folders")
    ap.add_argument("--dst", default="data/raw_selected", help="destination root for materialized set")
    ap.add_argument("--per-id-max", type=int, default=0, help="cap images per ID (0 = no cap)")
    ap.add_argument("--randomize", action="store_true", help="randomize image order before capping")
    ap.add_argument("--seed", type=int, default=42, help="rng seed when --randomize is set")
    ap.add_argument("--map-out", default="runs/eda/materialize_map.csv", help="CSV mapping of src->dst")
    args = ap.parse_args()

    # pass args to main with the names main() expects
    main(
        ids_file=args.ids,
        src_root=args.src,
        dst_root=args.dst,
        per_id_max=args.per_id_max,
        randomize=args.randomize,
        seed=args.seed,
        map_out=args.map_out,
    )
