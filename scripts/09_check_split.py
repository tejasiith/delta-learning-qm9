import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import config
from src.features import load_table
from src.split import murcko, scaffold_split


def main():
    full, _ = load_table()
    smiles = full.smiles.tolist()
    n = len(smiles)
    n_acyclic = sum(1 for s in smiles if murcko(s) == "")

    for seed in config.SEEDS:
        tr, va, te = scaffold_split(smiles, seed)
        parts = {"train": tr, "val": va, "test": te}
        scaf = {k: {murcko(smiles[i]) for i in idx} for k, idx in parts.items()}

        assert len(tr) + len(va) + len(te) == n, "molecules lost or duplicated"
        assert len(set(tr) | set(va) | set(te)) == n, "a molecule is in two splits"
        assert not scaf["train"] & scaf["val"], "scaffold shared by train and val"
        assert not scaf["train"] & scaf["test"], "scaffold shared by train and test"
        assert not scaf["val"] & scaf["test"], "scaffold shared by val and test"
        assert "" not in scaf["val"] and "" not in scaf["test"], "acyclic molecule outside train"

        again = scaffold_split(smiles, seed)
        assert all(np.array_equal(a, b) for a, b in zip((tr, va, te), again)), "split is not repeatable"

        print(f"seed {seed}: ok | train {len(tr)}, val {len(va)}, test {len(te)} | "
              f"scaffolds train/val/test {len(scaf['train'])}/{len(scaf['val'])}/{len(scaf['test'])}")
    print("acyclic molecules (always in train):", n_acyclic)


if __name__ == "__main__":
    main()