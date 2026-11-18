import numpy as np
from pyscf import gto, scf
from . import config

SYMBOLS = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}


def run_hf(job):
    idx, z, pos = job
    atoms = [(SYMBOLS[int(a)], tuple(p)) for a, p in zip(z, pos)]
    try:
        mol = gto.M(atom=atoms, basis=config.BASIS, unit="Angstrom", verbose=0)
        mf = scf.RHF(mol)
        mf.max_cycle = 100
        e = mf.kernel()
        if not mf.converged:
            return {"qm9_idx": idx, "ok": False}
        occ = mf.mo_occ
        return {
            "qm9_idx": idx,
            "ok": True,
            "e_hf_kcal": e * config.HARTREE_TO_KCAL,
            "homo": float(mf.mo_energy[occ > 0].max()),
            "lumo": float(mf.mo_energy[occ == 0].min()),
            "dipole": float(np.linalg.norm(mf.dip_moment(verbose=0))),
        }
    except Exception:
        return {"qm9_idx": idx, "ok": False}