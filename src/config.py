from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

N_MOLECULES = 4000
BASIS = "sto-3g"
SAMPLE_SEED = 0
SEEDS = [0, 1, 2, 3, 4]
SPLIT_FRAC = (0.8, 0.1, 0.1)

HARTREE_TO_KCAL = 627.5095
EV_TO_KCAL = 23.0605