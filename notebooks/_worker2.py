"""Multiprocessing worker for new ChEMBL SMG hash computation."""

from __future__ import annotations

import sys
import types

# Stub rdmolfiles to avoid blocked-DLL error on Windows WDAC
if "rdkit.Chem.rdmolfiles" not in sys.modules:
    sys.modules["rdkit.Chem.rdmolfiles"] = types.ModuleType(
        "rdkit.Chem.rdmolfiles"
    )

from rdkit import RDLogger
from rdkit.Chem import AddHs, MolFromInchi
from stereomolgraph.rdmol2graph import RDMol2StereoMolGraph

RDLogger.DisableLog("rdApp.*")

_CONVERTER = RDMol2StereoMolGraph(
    resonance=True, stereo_complete=True, lone_pair_stereo=False
)


def compute_hash(
    row: tuple[str, str, str],
) -> tuple[str, str, int | None, str | None]:
    """(chembl_id, inchi, smiles) -> (chembl_id, inchi, smg_hash, error)."""
    chembl_id, inchi, smiles = row
    try:
        mol = MolFromInchi(inchi, sanitize=True)
        if mol is None:
            return chembl_id, inchi, None, "parse failed"
        mol = AddHs(mol)
        smg = _CONVERTER(mol)
        smg.freeze()
        return chembl_id, inchi, hash(smg), None
    except Exception as exc:
        return chembl_id, inchi, None, str(exc)
