#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np
def main(counts_csv, coverage_csv, out_txt, n, per_id_min, min_bbox, cov_steps):
    counts = pd.read_csv(counts_csv).rename(columns={"path":"count"})
    cov    = pd.read_csv(coverage_csv)[["id","coverage","bbox_frac_mean"]]
    df = counts.merge(cov, on="id", how="left").fillna({"coverage":0, "bbox_frac_mean":0})
    def score_fn(d): return 2.0*d["coverage"] + 0.5*np.log1p(d["count"]) + d["bbox_frac_mean"]
    for cov_t in cov_steps:
        cand = df.query("count >= @per_id_min and coverage >= @cov_t and bbox_frac_mean >= @min_bbox").copy()
        if len(cand) >= n:
            cand["score"] = score_fn(cand)
            sel = cand.sort_values("score", ascending=False).head(n)["id"].tolist()
            pd.Series(sel).to_csv(out_txt, index=False, header=False)
            print(f"Selected {len(sel)} IDs (coverage ≥ {cov_t}, bbox_frac_mean ≥ {min_bbox}) -> {out_txt}")
            print(df.set_index("id").loc[sel][["count","coverage","bbox_frac_mean"]].head(10))
            return
    # fallback
    df["score"] = score_fn(df)
    sel = df.sort_values("score", ascending=False).head(n)["id"].tolist()
    pd.Series(sel).to_csv(out_txt, index=False, header=False)
    print(f"Selected {len(sel)} IDs (fallback: top by score) -> {out_txt}")
    print(df.set_index("id").loc[sel][["count","coverage","bbox_frac_mean"]].head(10))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default="runs/eda/class_counts.csv")
    ap.add_argument("--coverage", default="runs/eda/coverage_per_id.csv")
    ap.add_argument("--out", default="runs/eda/ids_selected.txt")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--per-id-min", type=int, default=30)
    ap.add_argument("--min-bbox", type=float, default=0.03)
    ap.add_argument("--cov-steps", nargs="+", type=float, default=[0.90,0.85,0.80,0.75,0.70])
    args = ap.parse_args()
    main(args.counts, args.coverage, args.out, args.n, args.per_id_min, args.min_bbox, args.cov_steps)
