import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import config
from src.features import atom_counts, hf_features, load_table, morgan_counts
from src.metrics import mae, rmse
from src.models import ResidualMLP
from src.split import scaffold_split


def main():
    config.RESULTS_DIR.mkdir(exist_ok=True)
    full, geoms = load_table()
    y = full.delta_kcal.values
    C = atom_counts(full, geoms)
    S = np.hstack([C, morgan_counts(full.smiles.tolist())]).astype(np.float32)
    SH = np.hstack([S, hf_features(full)]).astype(np.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| molecules:", len(full))

    rows, preds = [], []
    for seed in config.SEEDS:
        tr, va, te = scaffold_split(full.smiles.tolist(), seed)
        splits = {"train": tr, "val": va, "test": te}
        print(f"seed {seed}: train {len(tr)} val {len(va)} test {len(te)}")

        lin = LinearRegression().fit(C[tr], y[tr])
        resid = y - lin.predict(C)

        rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=seed)
        rf.fit(S[tr], resid[tr])

        mlp_s = ResidualMLP(seed=seed, device=device).fit(S[tr], resid[tr], S[va], resid[va])
        mlp_sh = ResidualMLP(seed=seed, device=device).fit(SH[tr], resid[tr], SH[va], resid[va])

        mean_val = float(y[tr].mean())
        models = {
            "mean": lambda i: np.full(len(i), mean_val),
            "linear_counts": lambda i: lin.predict(C[i]),
            "lin+RF(structure)": lambda i: lin.predict(C[i]) + rf.predict(S[i]),
            "lin+MLP(structure)": lambda i: lin.predict(C[i]) + mlp_s.predict(S[i]),
            "lin+MLP(structure+HF)": lambda i: lin.predict(C[i]) + mlp_sh.predict(SH[i]),
        }

        for name, fn in models.items():
            for split_name, idx in splits.items():
                p = fn(idx)
                rows.append({"seed": seed, "model": name, "split": split_name,
                             "mae": mae(y[idx], p), "rmse": rmse(y[idx], p)})
                if split_name == "test":
                    preds.append(pd.DataFrame({
                        "seed": seed, "model": name,
                        "qm9_idx": full.qm9_idx.values[idx],
                        "y": y[idx], "pred": p}))

    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "metrics.csv", index=False)
    pd.concat(preds).to_csv(config.RESULTS_DIR / "predictions.csv", index=False)

    m = pd.DataFrame(rows)
    t = m[m.split == "test"].groupby("model")[["mae", "rmse"]].agg(["mean", "std"]).round(2)
    print("\nTest error (kcal/mol), mean and std over 5 seeds:")
    print(t.sort_values(("mae", "mean")))


if __name__ == "__main__":
    main()