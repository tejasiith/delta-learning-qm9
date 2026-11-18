import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import config

ORDER = ["mean", "linear_counts", "lin+RF(structure)", "lin+MLP(structure)", "lin+MLP(structure+HF)"]
NICE = {
    "mean": "Training-set mean",
    "linear_counts": "Linear, atom counts",
    "lin+RF(structure)": "Linear + random forest (structure)",
    "lin+MLP(structure)": "Linear + MLP (structure)",
    "lin+MLP(structure+HF)": "Linear + MLP (structure + HF)",
}


def main():
    m = pd.read_csv(config.RESULTS_DIR / "metrics.csv")
    test = m[m.split == "test"]
    train = m[m.split == "train"]

    g = test.groupby("model")
    tr = train.groupby("model").mae.mean()
    lines = ["| Model | Test MAE | Test RMSE | Train MAE |", "|---|---|---|---|"]
    for k in ORDER:
        lines.append(
            f"| {NICE[k]} | {g.mae.mean()[k]:.2f} ± {g.mae.std()[k]:.2f} "
            f"| {g.rmse.mean()[k]:.2f} ± {g.rmse.std()[k]:.2f} | {tr[k]:.2f} |"
        )

    per_seed = test.pivot_table(index="seed", columns="model", values="mae")
    pairs = [("lin+MLP(structure)", "lin+RF(structure)"),
             ("lin+MLP(structure+HF)", "lin+MLP(structure)"),
             ("lin+MLP(structure+HF)", "linear_counts")]
    lines += ["", "Paired differences in test MAE (first minus second, negative = first is better):", ""]
    for a, b in pairs:
        d = per_seed[a] - per_seed[b]
        lines.append(f"- {NICE[a]} minus {NICE[b]}: mean {d.mean():.2f}, "
                     f"first better in {(d < 0).sum()} of {len(d)} seeds")

    text = "\n".join(lines)
    (config.RESULTS_DIR / "summary.md").write_text(text + "\n")
    print(text)

    # plot (the mean baseline is left out, it would flatten everything else)
    keys = ORDER[1:]
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, k in enumerate(keys):
        ax.bar(i, g.mae.mean()[k], yerr=g.mae.std()[k], capsize=4, alpha=0.6)
        ax.scatter([i] * len(per_seed), per_seed[k], color="black", s=12, zorder=3)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([NICE[k].replace(" (", "\n(") for k in keys], fontsize=7)
    ax.set_ylabel("Test MAE (kcal/mol)")
    ax.set_title("Scaffold-split test error, 5 seeds (dots = single seeds)")
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "test_mae.png", dpi=150)
    print("saved results/summary.md and results/test_mae.png")


if __name__ == "__main__":
    main()