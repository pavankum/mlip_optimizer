"""Extract per-instance parameter tables from in-process comparison objects.

Operates on the live Python objects produced by ``evaluate_against_qm`` +
``_build_ff_param_lookups`` — no PDF parsing, no SMILES re-canonicalization.

The central guarantee: every ``InstanceRecord.atom_key`` triple and the
``rdmol`` stored alongside it share ONE consistent atom ordering derived from
the QCArchive CMILES.  The same index ``j`` refers to the same physical atom
in the QM geometry, in the FF parameter labeling, and in RDKit.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class InstanceRecord:
    """One angle/bond/torsion instance of a given FF parameter.

    One record per unique (molecule, atom-triple), not per conformer.
    ``qm_values`` contains one float per conformer; use ``qm_mean`` as
    the representative value for clustering.

    Attributes
    ----------
    mol_idx : int
        Index into the parallel ``records`` / ``qm_results`` lists.
    inchi_key : str
    smiles : str
        Canonical (unmapped) SMILES for display.
    cmiles : str
        Canonical mapped SMILES; used to reconstruct ``rdmol`` in CLI
        scripts via ``Molecule.from_mapped_smiles(cmiles).to_rdkit()``.
    rdmol : rdkit.Chem.Mol or None
        RDKit molecule with the same atom ordering as ``cmiles``.
        ``None`` when constructed from a JSON record (use ``load_instances``
        with ``reconstruct_rdmol=True`` to populate it).
    atom_key : tuple[int, ...]
        Atom indices (i, j) for bonds or (i, j, k) for angles or
        (i, j, k, m) for torsions, all in the cmiles ordering.
    qm_values : tuple[float, ...]
        QM reference value for each conformer (Å for bonds, ° for angles).
    qm_mean : float
        Mean across conformers; used as the representative for clustering.
    mm_values : dict[str, list[float]]
        MM-optimized values per potential per conformer.  May be empty when
        ``potential_names`` was not supplied to ``extract_param_instances``.
    errors : dict[str, list[float]]
        Absolute QM–MM difference per potential per conformer.
    param_id : str
        OpenFF parameter ID, e.g. ``'a20'``.
    smirks : str
        Parent SMIRKS pattern for this parameter.
    """

    mol_idx: int
    inchi_key: str
    smiles: str
    cmiles: str
    rdmol: Any  # rdkit.Chem.Mol | None — Any to avoid hard import at module level
    atom_key: tuple
    qm_values: tuple
    qm_mean: float
    mm_values: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    param_id: str = ""
    smirks: str = ""

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict (rdmol is dropped)."""
        return {
            "mol_idx": self.mol_idx,
            "inchi_key": self.inchi_key,
            "smiles": self.smiles,
            "cmiles": self.cmiles,
            "atom_key": list(self.atom_key),
            "qm_values": list(self.qm_values),
            "qm_mean": self.qm_mean,
            "mm_values": {k: list(v) for k, v in self.mm_values.items()},
            "errors": {k: list(v) for k, v in self.errors.items()},
            "param_id": self.param_id,
            "smirks": self.smirks,
        }

    @classmethod
    def from_dict(cls, d: dict, *, reconstruct_rdmol: bool = True) -> "InstanceRecord":
        """Deserialize from a JSON dict produced by :meth:`to_dict`.

        Parameters
        ----------
        reconstruct_rdmol : bool
            If ``True`` (default), rebuild ``rdmol`` from ``cmiles`` via
            ``openff.toolkit.Molecule.from_mapped_smiles``.  Set ``False``
            for fast batch loading when featurization is not needed.
        """
        rdmol = None
        if reconstruct_rdmol and d.get("cmiles"):
            try:
                from openff.toolkit import Molecule as OFFMol
                rdmol = OFFMol.from_mapped_smiles(
                    d["cmiles"], allow_undefined_stereo=True
                ).to_rdkit()
            except Exception:
                pass
        return cls(
            mol_idx=d["mol_idx"],
            inchi_key=d.get("inchi_key", ""),
            smiles=d.get("smiles", ""),
            cmiles=d.get("cmiles", ""),
            rdmol=rdmol,
            atom_key=tuple(d["atom_key"]),
            qm_values=tuple(d["qm_values"]),
            qm_mean=float(d.get("qm_mean", np.mean(d["qm_values"]))),
            mm_values={k: list(v) for k, v in d.get("mm_values", {}).items()},
            errors={k: list(v) for k, v in d.get("errors", {}).items()},
            param_id=d.get("param_id", ""),
            smirks=d.get("smirks", ""),
        )


# ---------------------------------------------------------------------------
# Attribute maps per parameter type
# ---------------------------------------------------------------------------

_ATTR_MAP = {
    "angle":   ("angle_ref_values",   "angle_diffs",   "angle_values"),
    "bond":    ("bond_ref_values",     "bond_diffs",    "bond_values"),
    "torsion": ("torsion_ref_values",  "torsion_diffs", "torsion_values"),
}


def extract_param_instances(
    records,
    qm_results,
    ff_param_lookups: list[dict],
    param_id: str,
    param_type: str = "angle",
    potential_names: list[str] | None = None,
) -> list[InstanceRecord]:
    """Build a flat per-triple instance table for one FF parameter.

    Iterates over all molecules and all atom-key entries in
    ``qm_comp.<param_type>_ref_values``, resolves the parameter ID from the
    precomputed FF lookup, and emits one :class:`InstanceRecord` per unique
    (molecule, triple) pair that belongs to *param_id*.

    Parameters
    ----------
    records : list[MoleculeRecord]
        From ``load_records()`` — carries ``.molecule``, ``.smiles``,
        ``.cmiles``, ``.inchi_key``.
    qm_results : list[QMComparisonResult]
        Parallel to *records*; from ``evaluate_against_qm()``.
    ff_param_lookups : list[dict]
        Parallel to *records*; from ``_build_ff_param_lookups(records, ff_name)``.
        Each dict maps ``atom_index_tuple → (param_id, smirks)``.
    param_id : str
        Target parameter, e.g. ``'a20'`` or ``'b64'``.
    param_type : str
        ``'angle'``, ``'bond'``, or ``'torsion'``.
    potential_names : list[str] or None
        If given, also populate ``mm_values`` and ``errors`` for these
        potentials (requires ``qm_comp.per_potential`` data).

    Returns
    -------
    list[InstanceRecord]
        One record per (molecule, triple).  The atom ordering is the
        CMILES-derived OpenFF ordering, consistent with ``ff_param_lookups``,
        ``angle_ref_values``, and ``rdmol`` from ``to_rdkit()``.
    """
    if param_type not in _ATTR_MAP:
        raise ValueError(f"param_type must be one of {list(_ATTR_MAP)}; got {param_type!r}")

    ref_attr, diff_attr, val_attr = _ATTR_MAP[param_type]
    pot_names: list[str] = potential_names or []

    instances: list[InstanceRecord] = []

    for mol_idx, (rec, qm_comp, ff_lut) in enumerate(
        zip(records, qm_results, ff_param_lookups)
    ):
        if qm_comp is None or not ff_lut:
            continue

        ref_values_map: dict = getattr(qm_comp, ref_attr, {})
        if not ref_values_map:
            continue

        # Build {atom_key: (param_id, smirks)} for this molecule
        # (both fwd and rev are stored in ff_lut already)
        try:
            rdmol = rec.molecule.to_rdkit()
        except Exception:
            rdmol = None

        # Per-conformer diff/value dicts for each potential, keyed by atom_key
        # {pot: {atom_key: [v_conf0, v_conf1, ...]}}
        pot_diff_maps: dict[str, dict] = {}
        pot_val_maps: dict[str, dict] = {}
        if pot_names:
            for pot in pot_names:
                conf_list = qm_comp.per_potential.get(pot, [])
                dmap: dict = {}
                vmap: dict = {}
                for m in conf_list:
                    if m.opt_failed:
                        continue
                    for key, diff in getattr(m, diff_attr, {}).items():
                        if not math.isnan(diff):
                            dmap.setdefault(key, []).append(diff)
                    for key, val in getattr(m, val_attr, {}).items():
                        if not math.isnan(val):
                            vmap.setdefault(key, []).append(val)
                pot_diff_maps[pot] = dmap
                pot_val_maps[pot] = vmap

        for atom_key, ref_vals in ref_values_map.items():
            # Resolve FF parameter for this atom-key triple
            entry = ff_lut.get(atom_key) or ff_lut.get(atom_key[::-1])
            if entry is None or entry[0] != param_id:
                continue

            pid, smirks = entry
            clean_vals = tuple(v for v in ref_vals if not math.isnan(v))
            if not clean_vals:
                continue

            qm_mean = float(np.mean(clean_vals))

            mm_vals: dict[str, list[float]] = {}
            errs: dict[str, list[float]] = {}
            for pot in pot_names:
                dmap = pot_diff_maps.get(pot, {})
                vmap = pot_val_maps.get(pot, {})
                mm_v = vmap.get(atom_key) or vmap.get(atom_key[::-1]) or []
                diff_v = dmap.get(atom_key) or dmap.get(atom_key[::-1]) or []
                if mm_v:
                    mm_vals[pot] = mm_v
                if diff_v:
                    errs[pot] = diff_v

            instances.append(
                InstanceRecord(
                    mol_idx=mol_idx,
                    inchi_key=rec.inchi_key,
                    smiles=rec.smiles,
                    cmiles=rec.cmiles,
                    rdmol=rdmol,
                    atom_key=atom_key,
                    qm_values=clean_vals,
                    qm_mean=qm_mean,
                    mm_values=mm_vals,
                    errors=errs,
                    param_id=pid,
                    smirks=smirks,
                )
            )

    return instances


# ---------------------------------------------------------------------------
# JSON I/O helpers for CLI scripts
# ---------------------------------------------------------------------------


def save_instances(instances: list[InstanceRecord], path: str | Path) -> None:
    """Write instances to a JSON file (rdmol not serialized)."""
    path = Path(path)
    with open(path, "w") as fh:
        json.dump([r.to_dict() for r in instances], fh, indent=2)


def append_instances(instances: list[InstanceRecord], path: str | Path) -> None:
    """Append instances as JSON Lines to a file (creates file if needed).

    Each instance is written as one JSON object per line (JSONL format).
    Use :func:`load_instances` to read back — it auto-detects JSON vs JSONL.
    """
    path = Path(path)
    with open(path, "a") as fh:
        for r in instances:
            fh.write(json.dumps(r.to_dict()) + "\n")


def load_instances(
    path: str | Path,
    *,
    reconstruct_rdmol: bool = True,
) -> list[InstanceRecord]:
    """Load instances from a file written by :func:`save_instances` or
    :func:`append_instances`.

    Auto-detects format: a file starting with ``[`` is a JSON array
    (written by ``save_instances``); otherwise it is treated as JSON Lines
    (one record per line, written by ``append_instances``).

    Parameters
    ----------
    reconstruct_rdmol : bool
        Rebuild ``rdmol`` from ``cmiles`` via OpenFF toolkit.  Requires
        ``openff-toolkit`` to be installed.  Set ``False`` for fast
        loading when only QM values are needed (e.g. for stats).
    """
    with open(path) as fh:
        first = fh.read(1)
        fh.seek(0)
        if first == "[":
            data = json.load(fh)
        else:
            data = [json.loads(line) for line in fh if line.strip()]
    return [InstanceRecord.from_dict(d, reconstruct_rdmol=reconstruct_rdmol) for d in data]
