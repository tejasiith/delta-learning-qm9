import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import config
from src.features import load_table

MODELS = ["linear_counts", "lin+RF(structure)", "lin+MLP(structure)", "lin+MLP(structure+HF)"]
BEST = "lin+MLP(structure+HF)"


def main():
    full, geoms = load_table()
    info = full[["qm9_idx", "smiles", "n_atoms"]].copy()
    for z, el in [(7, "N"), (8, "O"), (9, "F")]:
        info["has_" + el] = [int((geoms[int(i)][0] == z).any()) for i in info.qm9_idx]

    pred = pd.read_csv(config.RESULTS_DIR / "predictions.csv").merge(info, on="qm9_idx")
    pred = pred[pred.model.isin(MODELS)].copy()
    pred["err"] = pred.pred - pred.y
    pred["abs_err"] = pred.err.abs()

    # 1. paired per-seed differences
    per_seed = pred.groupby(["seed", "model"]).abs_err.mean().unstack()[MODELS]
    print("Per-seed test MAE:\n", per_seed.round(2), "\n")
    pairs = [("lin+MLP(structure)", "lin+RF(structure)"),
             ("lin+MLP(structure+HF)", "lin+MLP(structure)"),
             ("lin+MLP(structure+HF)", "linear_counts")]
    print("Paired differences (first minus second; negative = first is better):")
    for a, b in pairs:
        d = per_seed[a] - per_seed[b]
        print(f"  {a} - {b}: mean {d.mean():.2f} | per seed {d.round(2).tolist()} | first better in {(d < 0).sum()}/5 seeds")

    # 2. by molecule size
    pred["size_bin"] = pd.cut(pred.n_atoms, [0, 15, 18, 21, 30], labels=["<=15", "16-18", "19-21", "22+"])
    by_size = pred.pivot_table(index="size_bin", columns="model", values="abs_err",
                               aggfunc="mean", observed=True)[MODELS].round(2)
    by_size["n_test_rows"] = pred[pred.model == BEST].groupby("size_bin", observed=True).size()
    print("\nMAE by number of atoms:\n", by_size)
    by_size.to_csv(config.RESULTS_DIR / "error_by_size.csv")

    # 3. by element presence
    rows = []
    for el in ["N", "O", "F"]:
        for flag in [0, 1]:
            sub = pred[pred["has_" + el] == flag]
            row = {"element": el, "present": flag,
                   "n_test_rows": int((sub.model == BEST).sum())}
            row.update(sub.groupby("model").abs_err.mean().round(2).to_dict())
            rows.append(row)
    by_el = pd.DataFrame(rows)[["element", "present", "n_test_rows"] + MODELS]
    print("\nMAE by element presence:\n", by_el.to_string(index=False))
    by_el.to_csv(config.RESULTS_DIR / "error_by_element.csv", index=False)

    # 4. worst cases for the best model
    worst = (pred[pred.model == BEST].groupby("qm9_idx")
             .agg(mean_abs_err=("abs_err", "mean"), mean_signed_err=("err", "mean"), n_tests=("abs_err", "size"))
             .join(info.set_index("qm9_idx")).sort_values("mean_abs_err", ascending=False).head(15))
    worst["smiles"] = worst.smiles.str.slice(0, 45)
    print("\nWorst molecules for", BEST, ":\n", worst.round(1).to_string())
    worst.to_csv(config.RESULTS_DIR / "worst_cases.csv")


if __name__ == "__main__":
    main()