import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import config
from src.features import atom_counts, hf_features, load_table, morgan_counts
from src.metrics import mae
from src.models import ResidualMLP
from src.split import scaffold_split

GRID = {
    "base": {},
    "dropout0.2": dict(dropout=0.2),
    "dropout0.4": dict(dropout=0.4),
    "wd1e-2": dict(wd=1e-2),
    "wd1e-1": dict(wd=1e-1),
    "hidden64": dict(hidden=64),
    "dropout0.3+wd1e-2": dict(dropout=0.3, wd=1e-2),
}
N_ENS = 5


def main():
    full, geoms = load_table()
    y = full.delta_kcal.values
    C = atom_counts(full, geoms)
    X = np.hstack([C, morgan_counts(full.smiles.tolist()), hf_features(full)]).astype(np.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    smiles = full.smiles.tolist()

    splits = {}
    for seed in config.SEEDS:
        tr, va, te = scaffold_split(smiles, seed)
        lin = LinearRegression().fit(C[tr], y[tr])
        splits[seed] = (tr, va, te, lin, y - lin.predict(C))

    def score(nets, seed):
        tr, va, te, lin, _ = splits[seed]
        out = {}
        for name, idx in [("train", tr), ("val", va), ("test", te)]:
            p = lin.predict(C[idx]) + np.mean([n.predict(X[idx]) for n in nets], axis=0)
            out[name] = mae(y[idx], p)
        return out

    rows = []
    for seed in config.SEEDS:
        tr, va, te, lin, resid = splits[seed]
        for name, kw in GRID.items():
            net = ResidualMLP(seed=seed, device=device, **kw).fit(X[tr], resid[tr], X[va], resid[va])
            rows.append(dict(seed=seed, config=name, **score([net], seed)))
        print("grid done for seed", seed)

    res = pd.DataFrame(rows)
    summary = res.groupby("config")[["train", "val", "test"]].mean().sort_values("val").round(2)
    print("\nMean MAE over 5 seeds, sorted by validation MAE:")
    print(summary)

    best = summary.index[0]
    print("\nchosen on validation:", best)

    for seed in config.SEEDS:
        tr, va, te, lin, resid = splits[seed]
        nets = [ResidualMLP(seed=seed * 100 + k, device=device, **GRID[best]).fit(
                    X[tr], resid[tr], X[va], resid[va]) for k in range(N_ENS)]
        rows.append(dict(seed=seed, config=f"ensemble{N_ENS} of {best}", **score(nets, seed)))

    res = pd.DataFrame(rows)
    res.to_csv(config.RESULTS_DIR / "regularisation.csv", index=False)

    ens = f"ensemble{N_ENS} of {best}"
    tab = res.pivot_table(index="seed", columns="config", values="test")[["base", best, ens]].round(2)
    print("\nPer-seed test MAE:")
    print(tab)
    print("\nEnsemble summary:")
    print(res[res.config == ens][["train", "val", "test"]].mean().round(2))


if __name__ == "__main__":
    main()