import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import config
from src.features import atom_counts, hf_features, load_table, morgan_counts
from src.metrics import mae
from src.models import ResidualMLP
from src.split import scaffold_split

FRACS = [0.1, 0.25, 0.5, 0.75, 1.0]


def main():
    full, geoms = load_table()
    y = full.delta_kcal.values
    C = atom_counts(full, geoms)
    S = np.hstack([C, morgan_counts(full.smiles.tolist())]).astype(np.float32)
    SH = np.hstack([S, hf_features(full)]).astype(np.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    smiles = full.smiles.tolist()

    rows = []
    for seed in config.SEEDS:
        tr_all, va, te = scaffold_split(smiles, seed)
        order = np.random.RandomState(seed).permutation(tr_all)
        for frac in FRACS:
            tr = order[: int(frac * len(order))]
            lin = LinearRegression().fit(C[tr], y[tr])
            resid = y - lin.predict(C)
            rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=seed)
            rf.fit(S[tr], resid[tr])
            net = ResidualMLP(dropout=0.2, seed=seed, device=device)
            net.fit(SH[tr], resid[tr], SH[va], resid[va])

            preds = {
                "linear_counts": lin.predict(C[te]),
                "lin+RF(structure)": lin.predict(C[te]) + rf.predict(S[te]),
                "lin+MLP(structure+HF)": lin.predict(C[te]) + net.predict(SH[te]),
            }
            for name, p in preds.items():
                rows.append(dict(seed=seed, frac=frac, n_train=len(tr), model=name,
                                 test_mae=mae(y[te], p)))
        print("seed", seed, "done")

    res = pd.DataFrame(rows)
    res.to_csv(config.RESULTS_DIR / "learning_curve.csv", index=False)

    agg = res.groupby(["model", "n_train"]).test_mae.agg(["mean", "std"]).round(2)
    print(agg.unstack("model")["mean"])

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, g in res.groupby("model"):
        s = g.groupby("n_train").test_mae.agg(["mean", "std"])
        ax.errorbar(s.index, s["mean"], yerr=s["std"], marker="o", capsize=3, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("training molecules")
    ax.set_ylabel("test MAE (kcal/mol)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "learning_curve.png", dpi=150)


if __name__ == "__main__":
    main()