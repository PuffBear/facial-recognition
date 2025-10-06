import os
import glob
from typing import List

def find_image_files(root: str, limit: int) -> List[str]:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    matches: List[str] = []
    for pattern in patterns:
        if len(matches) >= limit:
            break
        # Collect up to 'limit' files across patterns
        found = glob.glob(os.path.join(root, "**", pattern), recursive=True)
        for path in found:
            matches.append(path)
            if len(matches) >= limit:
                break
    return matches

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Peek into a dataset directory and list a few images.")
    ap.add_argument("--root", default="data/indian_face_full", help="Dataset root directory")
    ap.add_argument("--limit", type=int, default=10, help="Max number of image paths to print")
    args = ap.parse_args()

    root = args.root
    if not os.path.isdir(root):
        raise SystemExit(f"Root directory not found: {root}")

    entries = os.listdir(root)
    print(f"Top-level entries under {root} (showing up to 30):\n{entries[:30]}\n")

    images = find_image_files(root, args.limit)
    print(f"Found at least {len(images)} image files (limited to {args.limit} shown):")
    for p in images:
        print(" •", p)
