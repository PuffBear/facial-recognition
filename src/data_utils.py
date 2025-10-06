# src/data_utils.py
import os, glob, random
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def list_id_folders(root: str, min_images_per_id: int = 8) -> Dict[str, List[str]]:
    """
    Return {person_id: [image_paths,...]} where each ID has at least min_images_per_id images.
    """
    id2imgs: Dict[str, List[str]] = {}
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue
        paths: List[str] = []
        for ext in IMG_EXTS:
            paths += glob.glob(os.path.join(p, f"*{ext}"))
        if len(paths) >= min_images_per_id:
            id2imgs[name] = sorted(paths)
    return id2imgs

def split_idwise(
    id2imgs: Dict[str, List[str]],
    train: int = 5, val: int = 2, test: int = 1,
    seed: int = 1337
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Per-ID split: shuffle within each identity, then take first 'train', next 'val', next 'test'.
    Returns {"train":[(path,id),...], "val":[...], "test":[...]} and skips IDs that can't satisfy the split.
    """
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for pid, paths in id2imgs.items():
        paths = paths[:]  # copy
        rng.shuffle(paths)
        need = train + val + test
        if len(paths) < need:
            continue
        t = paths[:train]
        v = paths[train:train+val]
        s = paths[train+val:train+val+test]
        splits["train"].extend((p, pid) for p in t)
        splits["val"].extend((p, pid) for p in v)
        splits["test"].extend((p, pid) for p in s)
    return splits

def load_gray(path: str, size: int = 160) -> np.ndarray:
    """
    Open -> RGB -> resize (size,size) -> grayscale; returns uint8 HxW.
    """
    img = Image.open(path).convert("RGB").resize((size, size))
    img = img.convert("L")
    return np.array(img, dtype=np.uint8)
