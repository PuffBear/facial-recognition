#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np, cv2

def lbp_image(gray):
    H,W = gray.shape
    codes = np.zeros((H-2,W-2), np.uint8)
    c = gray[1:-1,1:-1]
    nbrs=[gray[0:-2,0:-2],gray[0:-2,1:-1],gray[0:-2,2:],gray[1:-1,0:-2],gray[1:-1,2:],gray[2:,0:-2],gray[2:,1:-1],gray[2:,2:]]
    weights=[1,2,4,8,16,32,64,128]
    for w,nb in zip(weights,nbrs):
        codes |= (nb>=c).astype(np.uint8)*w
    return codes

def heat_from_entropy(codes, grid=(8,8)):
    H,W=codes.shape; gh,gw=grid; sh,sw=H//gh, W//gw
    heat=np.zeros((gh,gw), np.float32)
    for i in range(gh):
        for j in range(gw):
            patch = codes[i*sh:(i+1)*sh, j*sw:(j+1)*sw]
            hist = np.bincount(patch.ravel(), minlength=256).astype(np.float32)
            p = hist/(hist.sum()+1e-9)
            ent = -(p*(np.log(p+1e-9))).sum()
            heat[i,j]=ent
    heat = cv2.resize(heat, (W,H), interpolation=cv2.INTER_CUBIC)
    heat /= (heat.max()+1e-9)
    return heat

def main(image_path, out_png):
    img = cv2.imread(image_path)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    codes = lbp_image(g)
    heat = heat_from_entropy(codes, grid=(8,8))
    hm = cv2.applyColorMap((heat*255).astype(np.uint8), cv2.COLORMAP_TURBO)
    # pad codes to original size for visualization
    vis_codes = cv2.resize(codes, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    vis_codes = cv2.applyColorMap((vis_codes/1.0).astype(np.uint8), cv2.COLORMAP_OCEAN)
    combo = np.hstack([img, cv2.addWeighted(img,0.6,hm,0.4,0), vis_codes])
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_png, combo)
    print("Saved:", out_png)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="runs/explain/lbp_vis.png")
    args = ap.parse_args()
    main(args.image, args.out)
