import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sklearn.linear_model import LinearRegression

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import config
from src.features import atom_counts, load_table

BEST = "lin+MLP(structure+HF)"
SIZE_LABELS = ["<=15", "16-18", "19-21", "22+"]


def main():
    full, geoms = load_table()
    C = atom_counts(full, geoms)  # columns: H C N O F
    y = full.delta_kcal.values
    resid = y - LinearRegression().fit(C, y).predict(C)
    heavy = C[:, 1:].sum(axis=1)

    info = pd.DataFrame({
        "qm9_idx": full.qm9_idx.values,
        "n_atoms": full.n_atoms.values,
        "hetero_frac": C[:, 2:].sum(axis=1) / heavy,
        "resid": resid,
        "n_rings": [rdMolDescriptors.CalcNumRings(Chem.MolFromSmiles(s)) for s in full.smiles],
    })
    info["size_bin"] = pd.cut(info.n_atoms, [0, 15, 18, 21, 30], labels=SIZE_LABELS)
    info["hetero_bin"] = pd.qcut(info.hetero_frac.rank(method="first"), 3,
                                 labels=["low", "mid", "high"])

   
    t1 = info.groupby("size_bin", observed=True).agg(
        n_molecules=("qm9_idx", "size"),
        leftover_std=("resid", "std"),
        hetero_frac=("hetero_frac", "mean"),
        mean_rings=("n_rings", "mean"),
    ).round(2)
    print("1. Sample composition by size (leftover_std = spread of what the linear model misses):\n", t1)

   
    pred = pd.read_csv(config.RESULTS_DIR / "predictions.csv")
    pred = pred[pred.model.isin(["linear_counts", BEST])].copy()
    pred["abs_err"] = (pred.y - pred.pred).abs()
    pred = pred.merge(info, on="qm9_idx")

    t2 = pred.pivot_table(index="size_bin", columns="model", values="abs_err",
                          aggfunc="mean", observed=True).round(2)
    t2["best_over_linear"] = (t2[BEST] / t2["linear_counts"]).round(2)
    print("\n2. Test MAE by size, and the best model's error as a fraction of the linear model's:\n", t2)

    b = pred[pred.model == BEST]

    
    t3 = b.pivot_table(index="size_bin", columns="hetero_bin", values="abs_err",
                       aggfunc="mean", observed=True).round(2)
    n3 = b.pivot_table(index="size_bin", columns="hetero_bin", values="abs_err",
                       aggfunc="size", observed=True)
    print("\n3. Best-model test MAE by size (rows) and heteroatom fraction (columns):\n", t3)
    print("   test rows in each cell:\n", n3)

    
    b = b.assign(ring_bin=pd.cut(b.n_rings, [0, 1, 2, 3, 10], labels=["1", "2", "3", "4+"]))
    t4 = b.groupby("ring_bin", observed=True).abs_err.agg(["mean", "size"]).round(2)
    print("\n4. Best-model test MAE by number of rings:\n", t4)

    t1.to_csv(config.RESULTS_DIR / "diag_composition.csv")
    t2.to_csv(config.RESULTS_DIR / "diag_size_error.csv")
    t3.to_csv(config.RESULTS_DIR / "diag_size_by_hetero.csv")


if __name__ == "__main__":
    main()