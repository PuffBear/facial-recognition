#!/usr/bin/env python3
import argparse, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def main(csv_path):
    df = pd.read_csv(csv_path)
    base_v = df.query("bucket=='baseline' and variant=='clean' and split=='val'")["accuracy"].astype(float).iloc[0]
    base_t = df.query("bucket=='baseline' and variant=='clean' and split=='test'")["accuracy"].astype(float).iloc[0]
    sub = df.query("bucket!='baseline'")
    sub["acc"] = sub["accuracy"].astype(float)
    sub["delta_val"]  = (sub.apply(lambda r: r["acc"]-base_v if r["split"]=="val"  else None, axis=1))*100
    sub["delta_test"] = (sub.apply(lambda r: r["acc"]-base_t if r["split"]=="test" else None, axis=1))*100
    out = Path(csv_path).with_name("deltas.csv"); sub.to_csv(out, index=False)
    print("Δs →", out)

    # quick bar (test deltas by variant)
    dtest = sub.dropna(subset=["delta_test"]).copy()
    dtest["name"] = dtest["bucket"]+"-"+dtest["variant"]
    dtest = dtest.groupby("name")["delta_test"].mean().sort_values()
    plt.figure()
    dtest.plot(kind="bar", title="Δ% vs clean (TEST)")
    plt.ylabel("Δ accuracy (%)"); plt.tight_layout()
    png = Path(csv_path).with_name("deltas_test.png"); plt.savefig(png, dpi=160)
    print("Plot →", png)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="runs/robustness/results.csv")
    args = ap.parse_args()
    main(args.csv)
