#!/usr/bin/env python
"""Compare pre-computed optimized geometries against QM reference.

Gathers optimized SDF files produced by ``optimize_single_model.py``
and compares them against the QM reference data, generating CSV
summaries and a PDF report.

Phase 2 of a two-phase workflow:

1. Read QM reference data (parquet or SDF) -- same files used in Phase 1
2. Discover optimized SDF files in the output directory
3. Compare each model vs QM (RMSD, bond/angle/torsion differences)
4. Write CSV (detail + summary) and PDF report

Usage
-----
::

    python examples/07_single_model_benchmark/compare_models.py <config.json>

JSON configuration
------------------
::

    {
        "data_files": [
            "path/to/nitrogen.parquet",
            "path/to/sulfur.parquet"
        ],
        "optimized_directory": "../outputs/optimized",
        "output_directory": "../outputs/comparison",
        "forcefield_name": "openff-2.3.0",
        "max_molecules": 100000,
        "max_conformers_per_molecule": 10,
        "bond_threshold": 0.1,
        "angle_threshold": 5.0,
        "torsion_threshold": 40.0
    }

Fields:

- **data_files** *(required)*: List of parquet or SDF files -- same ones
  used during Phase 1. A single string ``"data_file"`` is also accepted.
- **optimized_directory** *(required)*: Directory containing the
  dataset subdirectories with ``optimized_*.sdf`` files produced by
  Phase 1 (one subdirectory per input data file).
- **output_directory** *(required)*: Where to write CSV and PDF reports.
  A subdirectory is created per dataset.
- **forcefield_name** *(required)*: OpenFF ForceField name (e.g.
  ``"openff-2.3.0"``) used to annotate per-parameter IDs and SMIRKS in
  the differential tables. Must be installed and loadable.
- **max_molecules** / **max_conformers_per_molecule** *(optional)*:
  Must match the values used during Phase 1 so molecule ordering is
  identical.
- **bond_threshold** / **angle_threshold** / **torsion_threshold**
  *(optional)*: Minimum difference thresholds for reporting tables.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from mlip_optimizer import compute_overall_statistics, evaluate_against_qm
from mlip_optimizer.data import load_records
from mlip_optimizer.io import read_optimized_sdf, write_qm_comparison_csv
from mlip_optimizer.visualization.fg_report import build_report as fg_build_report
from mlip_optimizer.visualization.mol_report import build_report as mol_build_report

# Allow importing the shared helpers from the same directory.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _shared import load_json_config, resolve_path  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)


# ---------------------------------------------------------------------------
# Helpers to build fg_report data structures from qm_comparison_results
# ---------------------------------------------------------------------------


def _build_stats_dfs(
    overall_stats: dict,
    potential_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (stats_df, worst_df) from OverallErrorStatistics."""
    stats_rows = []
    worst_rows = []
    for pot in potential_names:
        s = overall_stats.get(pot)
        if s is None:
            stats_rows.append({"Potential": pot})
            worst_rows.append({"Potential": pot})
            continue
        stats_rows.append({
            "Potential":      pot,
            "N_conformers":   s.n_conformers_total,
            "RMSD_mean":      round(s.rmsd_mean, 3),
            "RMSD_max":       round(s.rmsd_max, 3),
            "Bond_mean_Å":    round(s.bond_mean, 2),
            "Bond_max_Å":     round(s.bond_max, 2),
            "Angle_mean_°":   round(s.angle_mean, 2),
            "Angle_max_°":    round(s.angle_max, 2),
            "Torsion_mean_°": round(s.torsion_mean, 2),
            "Torsion_max_°":  round(s.torsion_max, 2),
        })
        worst_rows.append({
            "Potential":    pot,
            "RMSD_max_ID":  s.rmsd_max_id,
            "Bond_max_ID":  s.bond_max_id,
            "Angle_max_ID": s.angle_max_id,
            "Tors_max_ID":  s.torsion_max_id,
        })
    return pd.DataFrame(stats_rows), pd.DataFrame(worst_rows)


def _build_threshold_df(
    qm_results: list,
    potential_names: list[str],
    bond_thresh: float,
    angle_thresh: float,
    torsion_thresh: float,
) -> pd.DataFrame:
    """Count molecules where any conformer has any single parameter exceeding the threshold."""
    rows = []
    for pot in potential_names:
        n_bond = n_angle = n_tors = n_any = 0
        for qm_comp in qm_results:
            if qm_comp is None:
                continue
            mol_bond = mol_angle = mol_tors = False
            for m in qm_comp.per_potential.get(pot, []):
                if m.opt_failed:
                    continue
                if m.max_bond_diff > bond_thresh:
                    mol_bond = True
                if m.max_angle_diff > angle_thresh:
                    mol_angle = True
                if m.max_torsion_diff > torsion_thresh:
                    mol_tors = True
            if mol_bond:
                n_bond += 1
            if mol_angle:
                n_angle += 1
            if mol_tors:
                n_tors += 1
            if mol_bond or mol_angle or mol_tors:
                n_any += 1
        rows.append({
            "Potential":          pot,
            "N_mol_any_crossing": n_any,
            "N_mol_bond":         n_bond,
            "N_mol_angle":        n_angle,
            "N_mol_torsion":      n_tors,
        })
    return pd.DataFrame(rows)


def _build_distribution_data(
    qm_results: list,
    potential_names: list[str],
) -> dict[str, dict[str, list[float]]]:
    """Collect per-conformer mean errors AND pooled actual/QM-ref values per potential."""
    data: dict[str, Any] = {
        "bond":    {p: [] for p in potential_names},
        "angle":   {p: [] for p in potential_names},
        "torsion": {p: [] for p in potential_names},
        # Actual parameter values pooled across all conformers/bonds/angles/torsions
        "bond_actual":    {p: [] for p in potential_names},
        "angle_actual":   {p: [] for p in potential_names},
        "torsion_actual": {p: [] for p in potential_names},
        # QM reference values pooled (same pools as actuals, for side-by-side comparison)
        "bond_qm":    [],
        "angle_qm":   [],
        "torsion_qm": [],
    }
    for qm_comp in qm_results:
        if qm_comp is None:
            continue
        # QM reference value pools
        for vals in qm_comp.bond_ref_values.values():
            data["bond_qm"].extend(v for v in vals if not np.isnan(v))
        for vals in qm_comp.angle_ref_values.values():
            data["angle_qm"].extend(v for v in vals if not np.isnan(v))
        for vals in qm_comp.torsion_ref_values.values():
            data["torsion_qm"].extend(v for v in vals if not np.isnan(v))
        for pot in potential_names:
            for m in qm_comp.per_potential.get(pot, []):
                if m.opt_failed:
                    continue
                for key, attr in [("bond", "mean_bond_diff"),
                                   ("angle", "mean_angle_diff"),
                                   ("torsion", "mean_torsion_diff")]:
                    val = getattr(m, attr, None)
                    if val is not None and not np.isnan(val):
                        data[key][pot].append(val)
                # Actual value pools
                data["bond_actual"][pot].extend(
                    v for v in m.bond_values.values() if not np.isnan(v)
                )
                data["angle_actual"][pot].extend(
                    v for v in m.angle_values.values() if not np.isnan(v)
                )
                data["torsion_actual"][pot].extend(
                    v for v in m.torsion_values.values() if not np.isnan(v)
                )
    return data


def _build_groups(
    qm_results: list,
    records: list,
    potential_names: list[str],
    fg_df,
    dataset_name: str,
) -> list[dict]:
    """Build fg_report group dicts from SMARTS matching."""
    from rdkit import Chem

    rdmols = []
    for rec in records:
        try:
            rdmols.append(rec.molecule.to_rdkit())
        except Exception:
            rdmols.append(None)

    groups = []
    for _, row in fg_df.iterrows():
        fg_name    = str(row.get("Functional Group", "") or "")
        smarts     = str(row.get("SMARTS", "") or "")
        hybrid     = str(row.get("Hybridization", "") or "")
        geometry   = str(row.get("Geometry", "") or "")

        if not smarts or smarts.lower() == "nan":
            continue
        try:
            query = Chem.MolFromSmarts(smarts)
        except Exception:
            query = None
        if query is None:
            continue

        matching = [
            i for i, rdmol in enumerate(rdmols)
            if rdmol is not None and i < len(qm_results)
            and qm_results[i] is not None
            and rdmol.HasSubstructMatch(query)
        ]
        if not matching:
            continue

        per_pot: dict[str, dict[str, list[float]]] = {
            pot: {"bond": [], "angle": [], "torsion": []}
            for pot in potential_names
        }
        n_conf = 0
        for mol_idx in matching:
            qm_comp = qm_results[mol_idx]
            for pot in potential_names:
                for m in qm_comp.per_potential.get(pot, []):
                    if m.opt_failed:
                        continue
                    per_pot[pot]["bond"].append(m.mean_bond_diff)
                    per_pot[pot]["angle"].append(m.mean_angle_diff)
                    per_pot[pot]["torsion"].append(m.mean_torsion_diff)
                    n_conf = max(n_conf, len(per_pot[pot]["bond"]))

        groups.append({
            "group_name": fg_name,
            "smarts":     smarts,
            "metadata": {
                "n_molecules":   len(matching),
                "n_conformers":  n_conf,
                "hybridization": hybrid,
                "geometry":      geometry,
                "dataset":       dataset_name,
            },
            "per_potential_data": per_pot,
        })
    return groups


def _build_param_data(
    qm_results: list,
    records: list,
    potential_names: list[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Aggregate raw errors/actuals for FF parameters that exceeded deviation thresholds.

    Reads the already-annotated diff tables (param_id + smirks appended by evaluate_against_qm
    when an OpenFF FF was detected) to identify which atom-key → param_id mappings had severe
    deviations, then pools raw per-conformer values from QMComparisonMetrics for those keys only.
    Falls back to element-type keys (C-C, C-N-C, …) when FF annotation is absent.

    Returns
    -------
    dict with keys 'bond', 'angle', 'torsion', each mapping:
        param_key → {'qm_ref': [...], 'errors': {pot: [...]}, 'actuals': {pot: [...]},
                     'smirks': str, 'elem_key': str}
    """
    param_data: dict[str, dict] = {"bond": {}, "angle": {}, "torsion": {}}
    n_base = 2 + len(potential_names)   # cols before optional (param_id, smirks) suffix

    def _elem_key_for(rdmol, ptype, key):
        try:
            atoms = [rdmol.GetAtomWithIdx(i).GetSymbol() for i in key]
            if ptype == "bond":
                return "-".join(sorted(atoms))
            if ptype == "angle":
                ends = sorted([atoms[0], atoms[2]])
                return f"{ends[0]}-{atoms[1]}-{ends[1]}"
            a, b, c, d = atoms
            if (b, c) > (c, b):
                a, b, c, d = d, c, b, a
            return f"{a}-{b}-{c}-{d}"
        except Exception:
            return None

    def _entry(ptype, pkey, smirks="", elem_key=""):
        bucket = param_data[ptype].setdefault(pkey, {
            "qm_ref": [],
            "errors":  {p: [] for p in potential_names},
            "actuals": {p: [] for p in potential_names},
            "smirks":   smirks,
            "elem_key": elem_key,
        })
        if smirks and not bucket["smirks"]:
            bucket["smirks"] = smirks
        if elem_key and not bucket["elem_key"]:
            bucket["elem_key"] = elem_key
        return bucket

    for rec, qm_comp in zip(records, qm_results):
        if qm_comp is None:
            continue
        try:
            rdmol = rec.molecule.to_rdkit()
        except Exception:
            rdmol = None

        # Step 1: Build atom_key → (param_key, smirks, elem_key) from annotated diff tables.
        # Only keys that exceeded the threshold appear in these tables (threshold-filtered).
        # FF-annotated rows have len > n_base; otherwise fall back to element-type key.
        ptype_lut: dict[str, dict[tuple, tuple[str, str, str]]] = {
            "bond": {}, "angle": {}, "torsion": {}
        }
        for ptype, table_attr in [
            ("bond",    "bond_diff_table"),
            ("angle",   "angle_diff_table"),
            ("torsion", "torsion_diff_table"),
        ]:
            for row in getattr(qm_comp, table_attr, []):
                if not (isinstance(row, (list, tuple)) and row and isinstance(row[0], tuple)):
                    continue
                key = row[0]
                ek = _elem_key_for(rdmol, ptype, key) if rdmol is not None else ""
                if len(row) > n_base:
                    pid    = str(row[n_base]).strip()     if row[n_base]     else ""
                    smirks = str(row[n_base + 1]).strip() if len(row) > n_base + 1 and row[n_base + 1] else ""
                else:
                    pid, smirks = "", ""
                pkey = pid if pid else (ek or None)
                if pkey:
                    ptype_lut[ptype][key]          = (pkey, smirks, ek or "")
                    ptype_lut[ptype][key[::-1]]    = (pkey, smirks, ek or "")

        if not any(ptype_lut[t] for t in ptype_lut):
            continue

        # Step 2: Pool raw per-conformer QM reference values for threshold-flagged keys.
        for ptype, ref_attr in [
            ("bond",    "bond_ref_values"),
            ("angle",   "angle_ref_values"),
            ("torsion", "torsion_ref_values"),
        ]:
            for key, vals in getattr(qm_comp, ref_attr, {}).items():
                lut = ptype_lut[ptype].get(key)
                if lut:
                    _entry(ptype, lut[0], lut[1], lut[2])["qm_ref"].extend(
                        v for v in vals if not np.isnan(v)
                    )

        # Step 3: Pool raw per-conformer errors and actual values.
        for pot in potential_names:
            for m in qm_comp.per_potential.get(pot, []):
                if m.opt_failed:
                    continue
                for ptype, diff_attr, val_attr in [
                    ("bond",    "bond_diffs",    "bond_values"),
                    ("angle",   "angle_diffs",   "angle_values"),
                    ("torsion", "torsion_diffs", "torsion_values"),
                ]:
                    for key, err in getattr(m, diff_attr, {}).items():
                        lut = ptype_lut[ptype].get(key)
                        if lut and not np.isnan(err):
                            _entry(ptype, lut[0], lut[1], lut[2])["errors"][pot].append(err)
                    for key, val in getattr(m, val_attr, {}).items():
                        lut = ptype_lut[ptype].get(key)
                        if lut and not np.isnan(val):
                            _entry(ptype, lut[0], lut[1], lut[2])["actuals"][pot].append(val)

    return param_data


# ---------------------------------------------------------------------------
# Helper for parallel evaluation (must be top-level for pickling)
# ---------------------------------------------------------------------------


def _evaluate_one_molecule(
    mol_idx: int,
    rec_molecule,
    opt_mols: dict,
    bond_thresh: float,
    angle_thresh: float,
    torsion_thresh: float,
    inchi_key: str,
    smiles: str,
    molecule_name: str,
    record_ids: list[int],
    forcefield_name: str = "",
):
    """Evaluate a single molecule against QM reference (worker function)."""
    return mol_idx, evaluate_against_qm(
        rec_molecule,
        opt_mols,
        bond_threshold=bond_thresh,
        angle_threshold=angle_thresh,
        torsion_threshold=torsion_thresh,
        inchi_key=inchi_key,
        smiles=smiles,
        molecule_name=molecule_name,
        record_ids=record_ids,
        forcefield_name=forcefield_name or None,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(config_path: str | Path) -> None:
    """Load optimized SDFs, compare vs QM, and write reports."""
    config = load_json_config(
        config_path,
        required_keys=(
            "optimized_directory",
            "output_directory",
        ),
    )

    # --- 1. Resolve data files ---
    raw_files = config.get("data_files", config.get("data_file"))
    if raw_files is None:
        raise ValueError("Config must contain 'data_files' (list) or 'data_file' (string)")
    if isinstance(raw_files, str):
        raw_files = [raw_files]

    opt_base = resolve_path(config["optimized_directory"], config_path)
    output_base = resolve_path(config["output_directory"], config_path)

    bond_thresh = config.get("bond_threshold", 0.1)
    angle_thresh = config.get("angle_threshold", 5.0)
    torsion_thresh = config.get("torsion_threshold", 40.0)

    # Load functional group SMARTS CSV files if provided.
    # These are dataset-agnostic (element taxonomy) so loaded once here.
    fg_df = None
    fg_files_raw = config.get("functional_groups_files", [])
    if isinstance(fg_files_raw, str):
        fg_files_raw = [fg_files_raw]
    if fg_files_raw:
        import pandas as pd
        _dfs = []
        for _fg_raw in fg_files_raw:
            _fg_path = resolve_path(_fg_raw, config_path)
            _df = pd.read_csv(str(_fg_path), delimiter=",")
            # Strip trailing whitespace from all string columns
            for _col in _df.select_dtypes(include="object").columns:
                _df[_col] = _df[_col].str.rstrip().fillna("")
            _dfs.append(_df)
        if _dfs:
            fg_df = pd.concat(_dfs, ignore_index=True)
            logger.info("  Loaded %d functional group SMARTS entries", len(fg_df))

    # --- 2. Process each dataset independently ---
    all_qm_results: list = []   # accumulate across datasets for combined report
    all_records:    list = []

    for file_idx, raw in enumerate(raw_files):
        data_file = resolve_path(raw, config_path)
        dataset_name = data_file.stem
        logger.info(
            "=== Dataset %d/%d: %s ===",
            file_idx + 1,
            len(raw_files),
            dataset_name,
        )

        records = load_records(
            data_file,
            max_molecules=config.get("max_molecules"),
            max_conformers_per_molecule=config.get("max_conformers_per_molecule"),
        )
        logger.info(
            "  QM reference: %d molecules (%d conformers)",
            len(records),
            sum(len(r.record_ids) for r in records),
        )

        if not records:
            logger.warning("  No molecules -- skipping.")
            continue

        # Discover optimized SDF files in the dataset subdirectory
        dataset_opt_dir = opt_base / dataset_name
        if not dataset_opt_dir.is_dir():
            logger.warning(
                "  Optimized directory not found: %s -- skipping.",
                dataset_opt_dir,
            )
            continue

        sdf_files = sorted(dataset_opt_dir.glob("optimized_*.sdf"))
        if not sdf_files:
            logger.warning(
                "  No optimized_*.sdf files in %s -- skipping.",
                dataset_opt_dir,
            )
            continue

        logger.info("  Found %d optimized SDF files:", len(sdf_files))
        for f in sdf_files:
            logger.info("    %s", f.name)

        # Read back all optimized results for this dataset
        optimized_results: dict[str, list] = {}
        potential_names: list[str] = []

        for sdf_file in sdf_files:
            model_name, molecules = read_optimized_sdf(sdf_file, records)

            if model_name in optimized_results:
                logger.info(
                    "  %s: duplicate for %s -- using latest file",
                    sdf_file.name,
                    model_name,
                )
                optimized_results[model_name] = molecules
                continue

            optimized_results[model_name] = molecules
            potential_names.append(model_name)
            n_ok = sum(1 for m in molecules if m is not None)
            n_fail = sum(1 for m in molecules if m is None)
            logger.info(
                "  Loaded %s: %d molecules (%d optimized, %d failed)",
                model_name, len(molecules), n_ok, n_fail,
            )

        if not potential_names:
            logger.warning("  No valid optimized results for %s.", dataset_name)
            continue

        # Set up per-dataset output directory
        output_dir = output_base / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Compare each model vs QM -- parallel evaluation
        n_workers = os.cpu_count() or 1
        logger.info("  Evaluating with %d workers ...", n_workers)

        # Determine the reference OpenFF ForceField for parameter labeling.
        # Priority: explicit config > detected from potential names.
        # Must succeed; no silent fallbacks.
        from mlip_optimizer.comparison import _detect_forcefield_name
        ff_name = (
            config.get("forcefield_name")
            or _detect_forcefield_name([(p, "") for p in potential_names])
        )
        if not ff_name:
            raise ValueError(
                "OpenFF ForceField not found. Provide 'forcefield_name' in config "
                "or add an OpenFF forcefield (e.g. 'openff-2.3.0') to potential names."
            )
        logger.info("  FF parameter annotation: %s", ff_name)

        # Build argument tuples for each molecule
        futures_args = []
        for mol_idx, rec in enumerate(records):
            opt_mols = {
                pot_name: optimized_results[pot_name][mol_idx]
                for pot_name in potential_names
            }
            futures_args.append((
                mol_idx,
                rec.molecule,
                opt_mols,
                bond_thresh,
                angle_thresh,
                torsion_thresh,
                rec.inchi_key,
                rec.smiles,
                rec.inchi_key or rec.smiles,
                rec.record_ids,
                ff_name,
            ))

        qm_comparison_results: list = [None] * len(records)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_evaluate_one_molecule, *args): args[0]
                for args in futures_args
            }
            for future in futures:
                mol_idx, qm_comp = future.result()
                qm_comparison_results[mol_idx] = qm_comp
                logger.info(
                    "  [%d/%d] %s  done",
                    mol_idx + 1,
                    len(records),
                    records[mol_idx].smiles,
                )

        # Write per-molecule PDF report (sequential -- matplotlib is not
        # process-safe for shared PdfPages)
        pdf_path = output_dir / "benchmark_report.pdf"
        mol_build_report(
            output_path=str(pdf_path),
            title=f"QM Benchmark Report: {dataset_name}",
            records=records,
            qm_results=qm_comparison_results,
            potential_names=potential_names,
        )
        logger.info("  PDF report: %s", pdf_path)

        # --- Overall error statistics PDF (reportlab-based) ---
        overall_stats = compute_overall_statistics(
            qm_comparison_results, potential_names,
        )
        stats_df, worst_df = _build_stats_dfs(overall_stats, potential_names)
        threshold_df = _build_threshold_df(
            qm_comparison_results, potential_names,
            bond_thresh, angle_thresh, torsion_thresh,
        )
        distribution_data = _build_distribution_data(
            qm_comparison_results, potential_names,
        )
        param_data = _build_param_data(
            qm_comparison_results, records, potential_names,
        )
        groups = []
        if fg_df is not None:
            groups = _build_groups(
                qm_comparison_results, records, potential_names,
                fg_df, dataset_name,
            )

        stats_pdf_path = output_dir / "error_statistics.pdf"
        fg_build_report(
            output_path=str(stats_pdf_path),
            title="QM Benchmark — Error Statistics",
            potential_names=potential_names,
            n_molecules=len(records),
            stats_df=stats_df,
            worst_df=worst_df,
            threshold_df=threshold_df,
            distribution_data=distribution_data,
            groups=groups,
            dataset_name=dataset_name,
            param_data=param_data,
        )
        logger.info("  Statistics PDF: %s", stats_pdf_path)

        # Write CSV
        detail_csv, summary_csv = write_qm_comparison_csv(
            qm_comparison_results,
            records,
            potential_names,
            output_dir,
        )
        logger.info("  Detail CSV:  %s", detail_csv)
        logger.info("  Summary CSV: %s", summary_csv)

        all_qm_results.extend(r for r in qm_comparison_results if r is not None)
        all_records.extend(records)

    # --- 3. Combined report across all datasets ---
    if len(raw_files) > 1 and all_qm_results:
        logger.info("=== Generating combined report ===")
        combined_dir = output_base / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)

        overall_stats = compute_overall_statistics(all_qm_results, potential_names)
        stats_df, worst_df = _build_stats_dfs(overall_stats, potential_names)
        threshold_df = _build_threshold_df(
            all_qm_results, potential_names,
            bond_thresh, angle_thresh, torsion_thresh,
        )
        distribution_data = _build_distribution_data(all_qm_results, potential_names)
        param_data = _build_param_data(all_qm_results, all_records, potential_names)

        combined_pdf = combined_dir / "error_statistics.pdf"
        fg_build_report(
            output_path=str(combined_pdf),
            title="QM Benchmark — Combined Error Statistics",
            potential_names=potential_names,
            n_molecules=len(all_records),
            stats_df=stats_df,
            worst_df=worst_df,
            threshold_df=threshold_df,
            distribution_data=distribution_data,
            groups=[],
            dataset_name="All datasets combined",
            param_data=param_data,
        )
        logger.info("  Combined PDF: %s", combined_pdf)

    logger.info("Comparison complete.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compare_models.py <config.json>")
        print()
        print("Example:")
        print(
            "  python examples/07_single_model_benchmark/"
            "compare_models.py \\"
        )
        print(
            "      examples/07_single_model_benchmark/"
            "inputs/compare_config.json"
        )
        sys.exit(1)

    main(sys.argv[1])
