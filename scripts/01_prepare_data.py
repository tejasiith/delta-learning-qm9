import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import config
from src.qm9_data import load_qm9, load_sample

config.DATA_DIR.mkdir(exist_ok=True)

ds = load_qm9()
first = ds[0]
print(len(ds), "molecules |", first.name, first.smiles, "| U0 (eV):", round(float(first.y[0, 7]), 2))

df, geoms = load_sample()
df.to_csv(config.DATA_DIR / "sample.csv", index=False)
with open(config.DATA_DIR / "geoms.pkl", "wb") as f:
    pickle.dump(geoms, f)

print(df.shape)
print(df.n_atoms.describe().round(1))
print(df.smiles.head())