import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import config
from src.hf_features import run_hf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    df = pd.read_csv(config.DATA_DIR / "sample.csv")
    with open(config.DATA_DIR / "geoms.pkl", "rb") as f:
        geoms = pickle.load(f)

    ids = list(df.qm9_idx)[: args.n]
    jobs = [(i, *geoms[i]) for i in ids]

    t0 = time.time()
    with Pool(args.workers) as pool:
        out = list(tqdm(pool.imap(run_hf, jobs, chunksize=8), total=len(jobs)))

    res = pd.DataFrame(out)
    print("converged:", round(res.ok.mean(), 3), "| seconds:", round(time.time() - t0))
    res = res[res.ok].drop(columns="ok")
    res.to_csv(config.DATA_DIR / f"hf_{config.BASIS}.csv", index=False)

    full = df.merge(res, on="qm9_idx")
    full["delta_kcal"] = full.e_ref_kcal - full.e_hf_kcal
    print(full.delta_kcal.describe().round(1))
    print(full[["name", "smiles", "n_atoms", "delta_kcal"]].head(8))


if __name__ == "__main__":
    main()