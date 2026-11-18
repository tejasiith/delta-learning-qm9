import numpy as np
import pandas as pd
from torch_geometric.datasets import QM9
from . import config

ZPVE, U0 = 6, 7


def load_qm9():
    return QM9(root=str(config.DATA_DIR / "qm9"))


def load_sample(n=config.N_MOLECULES, seed=config.SAMPLE_SEED):
    ds = load_qm9()
    idx = np.random.RandomState(seed).choice(len(ds), size=n, replace=False)
    rows, geoms = [], {}
    for i in idx:
        d = ds[int(i)]
        elec_ev = float(d.y[0, U0] - d.y[0, ZPVE])
        rows.append({
            "qm9_idx": int(i),
            "name": d.name,
            "smiles": d.smiles,
            "n_atoms": int(d.z.numel()),
            "e_ref_kcal": elec_ev * config.EV_TO_KCAL,
        })
        geoms[int(i)] = (d.z.numpy().astype(int), d.pos.numpy())
    return pd.DataFrame(rows), geoms