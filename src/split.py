from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from . import config


def murcko(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


def scaffold_split(smiles, seed, frac=config.SPLIT_FRAC):
    """Whole scaffold groups go to one split. Acyclic molecules (no scaffold) go to train."""
    groups = defaultdict(list)
    for i, s in enumerate(smiles):
        groups[murcko(s)].append(i)
    train = list(groups.pop("", []))

    keys = sorted(groups)
    np.random.RandomState(seed).shuffle(keys)

    n = len(smiles)
    n_val, n_test = frac[1] * n, frac[2] * n
    val, test = [], []
    for k in keys:
        g = groups[k]
        if len(test) + len(g) <= n_test:
            test += g
        elif len(val) + len(g) <= n_val:
            val += g
        else:
            train += g
    return np.array(train), np.array(val), np.array(test)