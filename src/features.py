import pickle

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from . import config

ELEMENTS = [1, 6, 7, 8, 9]  # H C N O F


def load_table():
    df = pd.read_csv(config.DATA_DIR / "sample.csv")
    hf = pd.read_csv(config.DATA_DIR / f"hf_{config.BASIS}.csv")
    full = df.merge(hf, on="qm9_idx").reset_index(drop=True)
    ok = [Chem.MolFromSmiles(s) is not None for s in full.smiles]
    print(f"dropped {len(ok) - sum(ok)} of {len(ok)} molecules RDKit could not parse")
    full = full[ok].reset_index(drop=True)
    full["delta_kcal"] = full.e_ref_kcal - full.e_hf_kcal
    with open(config.DATA_DIR / "geoms.pkl", "rb") as f:
        geoms = pickle.load(f)
    return full, geoms


def atom_counts(full, geoms):
    rows = []
    for i in full.qm9_idx:
        z = geoms[int(i)][0]
        rows.append([int((z == e).sum()) for e in ELEMENTS])
    return np.array(rows, dtype=np.float32)


def morgan_counts(smiles, radius=2, n_bits=1024):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for k, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            out[k] = gen.GetCountFingerprintAsNumPy(mol)
    return out


def hf_features(full):
    gap = full.lumo - full.homo
    return np.c_[full.homo, full.lumo, gap, full.dipole].astype(np.float32)