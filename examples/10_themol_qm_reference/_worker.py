"""Worker module for parallel molecule processing in spawned subprocesses.

Imported by ProcessPoolExecutor workers — must live in a directory on sys.path.
The ForceField is loaded ONCE per worker process via the initializer, not per call.
"""
from __future__ import annotations

import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey

_ff = None  # set once per worker by init()


def init(ff_name: str, src_path: str) -> None:
    """ProcessPoolExecutor initializer: load ForceField once per worker."""
    global _ff
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from openff.toolkit import ForceField
    _ff = ForceField(ff_name)


def process(cmiles: str, coords: list[list[float]], mol_idx: int) -> list[dict]:
    """Return all bond and angle instance dicts for one molecule."""
    from openff.toolkit import Molecule, Topology

    rows: list[dict] = []
    try:
        off_mol = Molecule.from_mapped_smiles(cmiles, allow_undefined_stereo=True)
        rdmol = off_mol.to_rdkit()
    except Exception:
        return rows

    c = np.asarray(coords)
    if off_mol.n_atoms != c.shape[0]:
        return rows

    try:
        mol_forces = _ff.label_molecules(Topology.from_molecules([off_mol]))[0]
    except Exception:
        return rows

    smiles = Chem.MolToSmiles(rdmol, canonical=True)
    try:
        inchi_key = MolToInchiKey(rdmol) or ''
    except Exception:
        inchi_key = ''

    base = dict(mol_idx=mol_idx, inchi_key=inchi_key, smiles=smiles, cmiles=cmiles)

    for atom_key, param in mol_forces.get('Bonds', {}).items():
        i, j = atom_key
        val = float(np.linalg.norm(c[i] - c[j]))
        rows.append({**base, 'param_type': 'bond', 'atom_key': f'{i},{j}',
                     'param_id': param.id, 'smirks': param.smirks, 'qm_value': val})

    for atom_key, param in mol_forces.get('Angles', {}).items():
        i, j, k = atom_key
        v1, v2 = c[i] - c[j], c[k] - c[j]
        n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        if n1 > 1e-10 and n2 > 1e-10:
            cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
            rows.append({**base, 'param_type': 'angle', 'atom_key': f'{i},{j},{k}',
                         'param_id': param.id, 'smirks': param.smirks,
                         'qm_value': float(np.degrees(np.arccos(cos_a)))})

    return rows
