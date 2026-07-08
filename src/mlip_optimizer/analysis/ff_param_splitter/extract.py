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
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Elements excluded from all ff-param-splitter analysis.
# Si: 'Si' substring is unambiguous in SMILES (no conflict with S or other symbols).
#     '\[si' catches aromatic silicon (silole-type rings), which canonical SMILES
#     writes lowercase and the bare 'Si' branch misses.
# B:  '\[B(?!r)' matches [B:, [BH:, [B@@H: etc. but not [Br: (bromine).
#     '\[b' catches aromatic boron.
_EXCLUDE_RE = re.compile(r"Si|\[si|\[B(?!r)|\[b")


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
    source: str = ""
    """Provenance tag: ``"qca"`` (small set, has MM values/errors — set
    automatically by :func:`extract_param_instances`) or ``"themol"`` (large
    QM-only export, no MM counterpart — set by :func:`load_combined_instances`
    when loading a THEMol parquet). Empty string for legacy records written
    before this field existed."""

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
            "source": self.source,
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
            source=d.get("source", ""),
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
                    source="qca",
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


def _instances_from_arrow_table(table, path: Path, *, reconstruct_rdmol: bool) -> list["InstanceRecord"]:
    """Build InstanceRecords from an in-memory Arrow table with THEMol Step 6
    columns, excluding Si/B-containing molecules. Shared by :func:`load_instances`
    (whole-file read) and :func:`sample_themol_prefix` (bounded-prefix read).

    ``rdmol`` is reconstructed once per DISTINCT cmiles and shared (by
    reference) across every row for that molecule — THEMol row density varies
    a lot by parameter (a18a: ~1.4 rows/molecule; a1: ~17-19 rows/molecule),
    and reconstructing the identical rdmol on every row is pure waste (~17x
    more toolkit calls than necessary for a1-like parameters). Safe to share:
    every downstream reader (featurize_instance, validate_hierarchy) only
    reads from rdmol, never mutates it.
    """
    mol_idxs   = table["mol_idx"].to_pylist()
    inchi_keys = table["inchi_key"].to_pylist()
    smiles_col = table["smiles"].to_pylist()
    cmiles_col = table["cmiles"].to_pylist()
    atom_keys  = table["atom_key"].to_pylist()    # list of lists
    qm_vals    = table["qm_values"].to_pylist()   # list of lists
    qm_means   = table["qm_mean"].to_pylist()
    param_ids  = (
        table["param_id"].to_pylist()
        if "param_id" in table.schema.names
        else [path.stem.split("_")[0]] * len(mol_idxs)
    )
    smirks_col = table["smirks"].to_pylist()

    keep = [not _EXCLUDE_RE.search(cm or "") for cm in cmiles_col]
    if not all(keep):
        mol_idxs   = [v for v, k in zip(mol_idxs,   keep) if k]
        inchi_keys = [v for v, k in zip(inchi_keys,  keep) if k]
        smiles_col = [v for v, k in zip(smiles_col,  keep) if k]
        cmiles_col = [v for v, k in zip(cmiles_col,  keep) if k]
        atom_keys  = [v for v, k in zip(atom_keys,   keep) if k]
        qm_vals    = [v for v, k in zip(qm_vals,     keep) if k]
        qm_means   = [v for v, k in zip(qm_means,    keep) if k]
        param_ids  = [v for v, k in zip(param_ids,   keep) if k]
        smirks_col = [v for v, k in zip(smirks_col,  keep) if k]

    rdmol_cache: dict[str, Any] = {}
    instances: list[InstanceRecord] = []
    for mol_idx, ik, sm, cm, ak, qv, qm, pid, sk in zip(
        mol_idxs, inchi_keys, smiles_col, cmiles_col,
        atom_keys, qm_vals, qm_means, param_ids, smirks_col,
    ):
        rdmol = None
        if reconstruct_rdmol and cm:
            if cm in rdmol_cache:
                rdmol = rdmol_cache[cm]
            else:
                try:
                    from openff.toolkit import Molecule as OFFMol
                    rdmol = OFFMol.from_mapped_smiles(
                        cm, allow_undefined_stereo=True
                    ).to_rdkit()
                except Exception:
                    pass
                rdmol_cache[cm] = rdmol
        instances.append(InstanceRecord(
            mol_idx=int(mol_idx),
            inchi_key=ik or "",
            smiles=sm or "",
            cmiles=cm or "",
            rdmol=rdmol,
            atom_key=tuple(int(x) for x in ak),
            qm_values=tuple(float(x) for x in qv),
            qm_mean=float(qm),
            mm_values={},
            errors={},
            param_id=pid or "",
            smirks=sk or "",
            source="themol",
        ))
    return instances


def sample_themol_prefix(
    path: str | Path,
    *,
    max_molecules: int = 10000,
    seed: int = 42,
    reconstruct_rdmol: bool = True,
    batch_size: int = 100_000,
    max_batches: int = 20,
) -> tuple[list["InstanceRecord"], dict]:
    """Sample up to *max_molecules* distinct molecules from the LEADING rows
    of a THEMol parquet, without scanning the rest of the file.

    Reads row-group batches from the front of the file (default 100K rows
    each), accumulating them, until either *max_molecules* distinct
    (Si/B-excluded) molecules have been seen or *max_batches* have been read
    — whichever comes first — then samples down to *max_molecules* and builds
    InstanceRecords from that already-read prefix. No second pass, no
    full-file scan.

    This trades a true whole-file uniform sample for speed/memory: some
    THEMol exports are 40M+ rows (a1: 46.9M rows / 2.5M molecules), and even
    just SCANNING every row to enumerate the full mol_idx population — before
    any rdmol reconstruction — takes minutes and (for the row-level filtered
    reload) risked OOM (measured 26GB peak RSS reading+filtering the full a1
    file for a 10K-molecule sample). Reading a bounded prefix is enough when
    *max_molecules* is modest relative to the file's row density, which holds
    for every THEMol export observed so far (row density ranges roughly
    1.5-19 rows/molecule; a 100K-row prefix already contains 5K-70K distinct
    molecules for the files measured).

    Returns
    -------
    (instances, stats)
        *stats* keys: ``total_rows_in_file`` (free — from parquet metadata,
        no scan), ``prefix_rows_scanned``, ``prefix_matched_mols`` (distinct
        molecules found in the scanned prefix, post Si/B exclusion),
        ``sampled_mols``, ``sampled_n``.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    pf = pq.ParquetFile(path)
    total_rows_in_file = pf.metadata.num_rows

    kept_batches = []
    seen_mols: set[int] = set()
    rows_scanned = 0
    for i, batch in enumerate(pf.iter_batches(batch_size=batch_size)):
        kept_batches.append(batch)
        rows_scanned += batch.num_rows
        for mi, cm in zip(batch.column("mol_idx").to_pylist(), batch.column("cmiles").to_pylist()):
            if not _EXCLUDE_RE.search(cm or ""):
                seen_mols.add(mi)
        if len(seen_mols) >= max_molecules or i + 1 >= max_batches:
            break

    prefix_table = pa.Table.from_batches(kept_batches) if kept_batches else pa.Table.from_batches([], schema=pf.schema_arrow)

    rng = random.Random(seed)
    sorted_seen = sorted(seen_mols)
    sampled_mols = set(rng.sample(sorted_seen, min(max_molecules, len(sorted_seen))))

    if sampled_mols:
        import pyarrow.compute as pc
        mask = pc.is_in(prefix_table["mol_idx"], value_set=pa.array(sorted(sampled_mols)))
        prefix_table = prefix_table.filter(mask)
    else:
        prefix_table = prefix_table.slice(0, 0)

    instances = _instances_from_arrow_table(prefix_table, path, reconstruct_rdmol=reconstruct_rdmol)

    return instances, {
        "total_rows_in_file": total_rows_in_file,
        "prefix_rows_scanned": rows_scanned,
        "prefix_matched_mols": len(sorted_seen),
        "sampled_mols": len(sampled_mols),
        "sampled_n": len(instances),
    }


def load_instances(
    path: str | Path,
    *,
    reconstruct_rdmol: bool = True,
    mol_idx_filter: set[int] | None = None,
) -> list[InstanceRecord]:
    """Load instances from a file written by :func:`save_instances`,
    :func:`append_instances`, or the THEMol Step 6 parquet export.

    Auto-detects format by file extension and content:

    - ``.parquet``: columnar format with zstd compression and dictionary-encoded
      string columns; most compact and fastest for large datasets.
    - ``.json`` / ``.jsonl`` starting with ``[``: JSON array (``save_instances``).
    - ``.jsonl`` flat: one :class:`InstanceRecord` dict per line (``append_instances``).
    - ``.jsonl`` grouped: one molecule per line with a nested ``instances`` list
      (grouped export format from Step 4/6).

    Parameters
    ----------
    reconstruct_rdmol : bool
        Rebuild ``rdmol`` from ``cmiles`` via OpenFF toolkit.  Requires
        ``openff-toolkit`` to be installed.  Set ``False`` for fast batch loading
        when only QM values are needed (e.g. clustering, stats).
    mol_idx_filter : set[int] or None
        If given, only rows whose ``mol_idx`` is in this set are loaded — a
        simple post-hoc filter, fine for moderate-sized files. For very large
        THEMol exports (10s of millions of rows), prefer
        :func:`sample_themol_prefix`, which never reads past the prefix
        needed to find enough molecules, instead of reading/filtering the
        whole file.
    """
    path = Path(path)

    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        if mol_idx_filter is not None:
            import pyarrow as pa
            import pyarrow.compute as pc
            table = table.filter(pc.is_in(table["mol_idx"], value_set=pa.array(sorted(mol_idx_filter))))

        return _instances_from_arrow_table(table, path, reconstruct_rdmol=reconstruct_rdmol)

    with open(path) as fh:
        first = fh.read(1)
        fh.seek(0)
        if first == "[":
            # Legacy JSON array (save_instances)
            data = json.load(fh)
        else:
            # JSONL — two sub-formats:
            #   flat:    one InstanceRecord dict per line  (old)
            #   grouped: one molecule per line with nested "instances" list (new,
            #            eliminates repetition of mol-level fields across instances)
            data = []
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "instances" in obj:
                    for inst in obj["instances"]:
                        data.append({
                            "mol_idx":   obj["mol_idx"],
                            "inchi_key": obj.get("inchi_key", ""),
                            "smiles":    obj.get("smiles", ""),
                            "cmiles":    obj.get("cmiles", ""),
                            "atom_key":  inst["atom_key"],
                            "qm_values": inst["qm_values"],
                            "qm_mean":   inst["qm_mean"],
                            "mm_values": inst.get("mm_values", {}),
                            "errors":    inst.get("errors", {}),
                            "param_id":  obj.get("param_id", ""),
                            "smirks":    obj.get("smirks", ""),
                        })
                else:
                    data.append(obj)
        data = [d for d in data if not _EXCLUDE_RE.search(d.get("cmiles", ""))]
        if mol_idx_filter is not None:
            data = [d for d in data if d.get("mol_idx") in mol_idx_filter]
    return [InstanceRecord.from_dict(d, reconstruct_rdmol=reconstruct_rdmol) for d in data]


# ---------------------------------------------------------------------------
# Combined QCA + THEMol extraction
# ---------------------------------------------------------------------------


def load_combined_instances(
    *,
    param_id: str,
    param_type: str = "angle",
    data_file: str | Path | None = None,
    opt_dir: str | Path | None = None,
    themol_file: str | Path | None = None,
    ff: str = "openff-2.3.0",
    potential_names: list[str] | None = None,
    max_molecules: int = 10000,
    seed: int = 42,
    reconstruct_rdmol: bool = True,
    use_cache: bool = True,
) -> tuple[list[InstanceRecord], dict]:
    """Load instances for *param_id* from QCA and/or THEMol and combine them.

    QCA and THEMol are complementary, not alternative, sources:

    - **QCA** (*data_file* + *opt_dir*): the SMALL dataset (thousands of
      records, not hundreds of thousands) read from the optimized-geometry
      SDFs. Has BOTH QM and MM geometry, so it carries the actual MM-vs-QM
      error signal (``mm_values``/``errors``). It is NOT parameter-specific —
      matching against *param_id* must run across the full record set first
      (``extract_param_instances``) — but that's cheap since QCA is small.
      Every matched instance is used; QCA is NEVER sampled/capped.
    - **THEMol** (*themol_file*): the LARGE, QM-only export (can be 900K+
      distinct molecules, tens of millions of rows). Already parameter-specific
      — every row already carries *param_id* — but has no MM counterpart.
      Sampled down to *max_molecules* (fixed *seed*) from a bounded PREFIX of
      the file (see :func:`sample_themol_prefix`) — large exports are never
      scanned in full just to pick a small sample.

    At least one of (*data_file* + *opt_dir*) or *themol_file* must be given.
    Every returned :class:`InstanceRecord` is tagged ``.source`` (``"qca"`` or
    ``"themol"``) so downstream steps can separate "has MM error data" from
    "QM-only, broader coverage" — see ``auto_split.py``, which validates
    alpha-gap/χ² QCA-only but clusters/describes on the combined set.

    Returns
    -------
    (instances, stats)
        *instances* is QCA instances (if any) followed by THEMol instances
        (if any). *stats* is ``{"qca": {...} | None, "themol": {...} | None}``:

        - ``qca``: ``total_records``, ``matched_n``, ``matched_mols`` (no
          sampled_* keys — QCA is used in full, never capped).
        - ``themol``: ``total_rows_in_file`` (free, from parquet metadata),
          ``prefix_rows_scanned``, ``prefix_matched_mols`` (distinct molecules
          found within the scanned prefix, NOT the whole file),
          ``sampled_mols``, ``sampled_n``.
    """
    if data_file is None and opt_dir is None and themol_file is None:
        raise ValueError("at least one of (data_file + opt_dir) or themol_file is required")

    combined: list[InstanceRecord] = []
    stats: dict = {"qca": None, "themol": None}

    if data_file is not None or opt_dir is not None:
        if data_file is None or opt_dir is None:
            raise ValueError("QCA source requires BOTH data_file and opt_dir")
        from mlip_optimizer.analysis.bundle import load_openff_bundle

        bundle = load_openff_bundle(
            Path(data_file),
            potential_name=ff,
            optimized_root=Path(opt_dir),
            use_cache=use_cache,
            label_forcefield_name=ff,
        )
        records = bundle["records"]
        qm_results = bundle["qm_results"]
        ff_param_lookups = bundle["ff_param_lookups"]
        if ff_param_lookups is None:
            raise RuntimeError("ff_param_lookups is None — was label_forcefield_name set?")

        # Matching the small QCA set against param_id is the cheap, unavoidable
        # pass — it IS the point of QCA (it's where the MM error signal lives).
        # Every match is kept; QCA is never sampled/capped.
        qca_instances = extract_param_instances(
            records, qm_results, ff_param_lookups,
            param_id, param_type,
            potential_names=potential_names or [],
        )
        combined.extend(qca_instances)
        stats["qca"] = {
            "total_records": len(records),
            "matched_n": len(qca_instances),
            "matched_mols": len({r.mol_idx for r in qca_instances}),
        }

    if themol_file is not None:
        themol_path = Path(themol_file)

        # Sample from a bounded PREFIX of the file, not the whole thing — see
        # sample_themol_prefix(). Some THEMol exports are 40M+ rows (a1: 46.9M
        # rows / 2.5M molecules); scanning every row just to enumerate the
        # full molecule population before sampling max_molecules of them is
        # unnecessary and, for the largest files, memory-risky. Non-parquet
        # (JSON/JSONL) THEMol files are small enough in practice to just load
        # whole and sample after.
        if themol_path.suffix == ".parquet":
            themol_instances, themol_stats = sample_themol_prefix(
                themol_path, max_molecules=max_molecules, seed=seed,
                reconstruct_rdmol=reconstruct_rdmol,
            )
        else:
            pre = load_instances(themol_path, reconstruct_rdmol=False)
            matched_mols = sorted({r.mol_idx for r in pre})
            del pre
            rng = random.Random(seed)
            sampled_mols = set(rng.sample(matched_mols, min(max_molecules, len(matched_mols))))
            themol_instances = load_instances(
                themol_path, reconstruct_rdmol=reconstruct_rdmol, mol_idx_filter=sampled_mols,
            )
            themol_stats = {
                "total_rows_in_file": len(matched_mols),
                "prefix_rows_scanned": len(matched_mols),
                "prefix_matched_mols": len(matched_mols),
                "sampled_mols": len(sampled_mols),
                "sampled_n": len(themol_instances),
            }

        combined.extend(themol_instances)
        stats["themol"] = themol_stats

    return combined, stats
