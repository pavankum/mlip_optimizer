"""Geometry comparison between optimization results.

Provides both pairwise optimizer-vs-optimizer comparison
(:func:`evaluate_model_pairs`) and multi-potential comparison against
a QM reference (:func:`evaluate_against_qm`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from openff.toolkit import Molecule

from mlip_optimizer.geometry import (
    compute_geometry_diffs,
    compute_rmsd,
    get_conformer_geometry,
)


@dataclass
class ComparisonResult:
    """Results of pairwise geometry comparison.

    Each field is a list of table rows suitable for display with
    ``tabulate`` or inclusion in reports.

    Attributes
    ----------
    bond_diffs : list[list]
        Rows of ``[bond_key, model1_val, model2_val, mean_diff]``
        for bonds exceeding the threshold.
    angle_diffs : list[list]
        Rows of ``[angle_key, model1_val, model2_val, mean_diff]``
        for angles exceeding the threshold.
    torsion_diffs : list[list]
        Rows of ``[torsion_key, model1_val, model2_val, mean_diff]``
        for torsions exceeding the threshold.
    """

    bond_diffs: list[list] = field(default_factory=list)
    angle_diffs: list[list] = field(default_factory=list)
    torsion_diffs: list[list] = field(default_factory=list)


def evaluate_model_pairs(
    optimized_results: dict[str, Molecule],
    reference_molecule: Molecule,
    model_pairs: list[tuple[str, str]],
    *,
    bond_threshold: float = 0.1,
    angle_threshold: float = 5.0,
    torsion_threshold: float = 40.0,
) -> ComparisonResult:
    """Evaluate pairwise geometry differences across all conformers.

    For each model pair and each conformer, computes the absolute
    difference in bond lengths, angles, and torsions.  Results are
    aggregated (mean +/- std) across conformers and filtered by the
    given thresholds.

    Parameters
    ----------
    optimized_results : dict[str, Molecule]
        Map from optimizer name to optimized molecule.  All molecules
        must have the same number of conformers.
    reference_molecule : Molecule
        The original (pre-optimization) molecule, used to determine
        the number of conformers to iterate over.
    model_pairs : list[tuple[str, str]]
        Pairs of optimizer names to compare,
        e.g. ``[("openff-2.3.0.offxml", "aceff-2.0")]``.
    bond_threshold : float, optional
        Minimum mean bond length difference in Angstroms to include
        in the report.  Default is ``0.1``.
    angle_threshold : float, optional
        Minimum mean angle difference in degrees to include.
        Default is ``5.0``.
    torsion_threshold : float, optional
        Minimum mean absolute torsion difference in degrees to include.
        Default is ``40.0``.

    Returns
    -------
    ComparisonResult
        Filtered comparison data with bond, angle, and torsion
        differences exceeding the thresholds.

    Examples
    --------
    >>> results = {"sage": sage_mol, "aceff": aceff_mol}
    >>> comp = evaluate_model_pairs(results, orig_mol, [("sage", "aceff")])
    >>> len(comp.bond_diffs)  # number of bonds with large differences
    3
    """
    bond_diffs_list: list[tuple] = []
    angle_diffs_list: list[tuple] = []
    torsion_diffs_list: list[tuple] = []

    n_conformers = len(reference_molecule.conformers)

    for conf_idx in range(n_conformers):
        # Compute geometry for each model at this conformer
        geom_data = {}
        for model_name in optimized_results:
            geom_data[model_name] = get_conformer_geometry(
                optimized_results[model_name], conf_idx
            )

        for model1, model2 in model_pairs:
            geom1 = geom_data[model1]
            geom2 = geom_data[model2]

            # Bond differences
            for bond_key in geom1.bond_lengths:
                val1 = geom1.bond_lengths[bond_key]
                val2 = geom2.bond_lengths[bond_key]
                diff = abs(val1 - val2)
                bond_diffs_list.append((bond_key, val1, val2, diff))

            # Angle differences
            for angle_key in geom1.bond_angles:
                val1 = geom1.bond_angles[angle_key]
                val2 = geom2.bond_angles[angle_key]
                diff = abs(val1 - val2)
                angle_diffs_list.append((angle_key, val1, val2, diff))

            # Torsion differences
            for torsion_key in geom1.torsion_angles:
                val1 = geom1.torsion_angles[torsion_key]
                val2 = geom2.torsion_angles[torsion_key]
                diff = abs(val1 - val2)
                torsion_diffs_list.append((torsion_key, val1, val2, diff))

    # --- Aggregate bonds ---
    bond_table = _aggregate_diffs(bond_diffs_list, bond_threshold)

    # --- Aggregate angles ---
    angle_table = _aggregate_diffs(angle_diffs_list, angle_threshold)

    # --- Aggregate torsions (normalize to [-180, 180]) ---
    torsion_table = _aggregate_diffs(
        torsion_diffs_list, torsion_threshold, normalize_torsion=True
    )

    return ComparisonResult(
        bond_diffs=bond_table,
        angle_diffs=angle_table,
        torsion_diffs=torsion_table,
    )


def _aggregate_diffs(
    diffs_list: list[tuple],
    threshold: float,
    normalize_torsion: bool = False,
) -> list[list]:
    """Aggregate per-conformer differences and filter by threshold.

    Parameters
    ----------
    diffs_list : list[tuple]
        Each element is ``(key, val1, val2, diff)``.
    threshold : float
        Minimum mean difference to report.
    normalize_torsion : bool
        If True, normalize differences to ``[-180, 180]`` before
        aggregation (appropriate for torsion angles).

    Returns
    -------
    list[list]
        Rows of ``[key, "mean1 +/- std1", "mean2 +/- std2",
        "mean_diff +/- std_diff"]`` for entries exceeding the threshold.
    """
    summary: dict[tuple, dict[str, list[float]]] = {}

    for key, val1, val2, diff in diffs_list:
        if normalize_torsion:
            diff = (diff + 180) % 360 - 180
        if key not in summary:
            summary[key] = {"diffs": [], "model1_vals": [], "model2_vals": []}
        summary[key]["diffs"].append(diff)
        summary[key]["model1_vals"].append(val1)
        summary[key]["model2_vals"].append(val2)

    table: list[list] = []
    for key, data in summary.items():
        mean_diff = float(np.mean(data["diffs"]))
        std_diff = float(np.std(data["diffs"]))
        mean_val1 = float(np.mean(data["model1_vals"]))
        std_val1 = float(np.std(data["model1_vals"]))
        mean_val2 = float(np.mean(data["model2_vals"]))
        std_val2 = float(np.std(data["model2_vals"]))

        check_val = abs(mean_diff) if normalize_torsion else mean_diff
        if check_val > threshold:
            table.append([
                key,
                f"{mean_val1:.2f} \u00b1 {std_val1:.2f}",
                f"{mean_val2:.2f} \u00b1 {std_val2:.2f}",
                f"{mean_diff:.2f} \u00b1 {std_diff:.2f}",
            ])

    return table


# ---------------------------------------------------------------------------
# QM-reference comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QMComparisonMetrics:
    """Per-conformer comparison metrics of one potential vs QM reference.

    Attributes
    ----------
    rmsd : float
        Heavy-atom RMSD vs QM (Angstroms).
    max_bond_diff : float
        Maximum absolute bond length difference (Angstroms).
    mean_bond_diff : float
        Mean absolute bond length difference (Angstroms).
    max_angle_diff : float
        Maximum absolute angle difference (degrees).
    mean_angle_diff : float
        Mean absolute angle difference (degrees).
    max_torsion_diff : float
        Maximum absolute torsion difference (degrees).
    mean_torsion_diff : float
        Mean absolute torsion difference (degrees).
    bond_diffs : dict[tuple, float]
        Per-bond absolute differences.
    angle_diffs : dict[tuple, float]
        Per-angle absolute differences.
    torsion_diffs : dict[tuple, float]
        Per-torsion absolute differences.
    """

    rmsd: float
    max_bond_diff: float
    mean_bond_diff: float
    max_angle_diff: float
    mean_angle_diff: float
    max_torsion_diff: float
    mean_torsion_diff: float
    bond_diffs: dict[tuple, float]
    angle_diffs: dict[tuple, float]
    torsion_diffs: dict[tuple, float]


@dataclass
class QMComparisonResult:
    """Comparison of multiple potentials against QM reference for one molecule.

    Attributes
    ----------
    inchi_key : str
        InChI key identifying the molecule.
    smiles : str
        Canonical SMILES.
    n_conformers : int
        Number of conformers compared.
    per_potential : dict[str, list[QMComparisonMetrics]]
        Map from potential name to a list of per-conformer metrics.
    bond_diff_table : list[list]
        Aggregated bond difference rows for tabular reporting.
    angle_diff_table : list[list]
        Aggregated angle difference rows.
    torsion_diff_table : list[list]
        Aggregated torsion difference rows.
    molecule_name : str
        Human-readable molecule name or label.
    record_ids : list[int]
        QCArchive record IDs, one per conformer.
    """

    inchi_key: str = ""
    smiles: str = ""
    n_conformers: int = 0
    per_potential: dict[str, list[QMComparisonMetrics]] = field(
        default_factory=dict
    )
    bond_diff_table: list[list] = field(default_factory=list)
    angle_diff_table: list[list] = field(default_factory=list)
    torsion_diff_table: list[list] = field(default_factory=list)
    molecule_name: str = ""
    record_ids: list[int] = field(default_factory=list)


def evaluate_against_qm(
    qm_molecule: Molecule,
    optimized_molecules: dict[str, Molecule],
    *,
    heavy_atoms_only: bool = True,
    bond_threshold: float = 0.1,
    angle_threshold: float = 5.0,
    torsion_threshold: float = 40.0,
    inchi_key: str = "",
    smiles: str = "",
    molecule_name: str = "",
    record_ids: list[int] | None = None,
) -> QMComparisonResult:
    """Compare optimized geometries from multiple potentials against QM.

    For each conformer and each potential, computes RMSD and per-parameter
    geometry differences against the QM reference coordinates in
    *qm_molecule*.  Results are aggregated (mean +/- std) across
    conformers and filtered by thresholds for inclusion in the
    difference tables.

    Parameters
    ----------
    qm_molecule : Molecule
        Molecule with QM-optimized conformer geometries (the reference).
    optimized_molecules : dict[str, Molecule]
        Map from potential name to optimized molecule.  Each must have
        the same number of conformers as *qm_molecule*.
    heavy_atoms_only : bool, optional
        Use only heavy atoms for RMSD.  Default is ``True``.
    bond_threshold : float, optional
        Minimum mean bond length difference (Angstroms) to include in
        the report tables.  Default is ``0.1``.
    angle_threshold : float, optional
        Minimum mean angle difference (degrees).  Default is ``5.0``.
    torsion_threshold : float, optional
        Minimum mean absolute torsion difference (degrees).
        Default is ``40.0``.
    inchi_key : str, optional
        InChI key for the molecule (stored in result metadata).
    smiles : str, optional
        Canonical SMILES (stored in result metadata).
    molecule_name : str, optional
        Human-readable molecule name or label.
    record_ids : list[int] or None, optional
        QCArchive record IDs, one per conformer.

    Returns
    -------
    QMComparisonResult
    """
    n_conformers = len(qm_molecule.conformers)
    potential_names = list(optimized_molecules.keys())

    # --- Per-conformer metrics ---
    per_potential: dict[str, list[QMComparisonMetrics]] = {
        name: [] for name in potential_names
    }

    # Accumulate per-key diffs for aggregation:
    # {param_key: {pot_name: [diff_values_per_conf]}}
    bond_accum: dict[tuple, dict[str, list[float]]] = {}
    angle_accum: dict[tuple, dict[str, list[float]]] = {}
    torsion_accum: dict[tuple, dict[str, list[float]]] = {}
    # QM reference values for the table:
    # {param_key: [ref_values_per_conf]}
    bond_ref_accum: dict[tuple, list[float]] = {}
    angle_ref_accum: dict[tuple, list[float]] = {}
    torsion_ref_accum: dict[tuple, list[float]] = {}

    for conf_idx in range(n_conformers):
        geom_qm = get_conformer_geometry(qm_molecule, conf_idx)

        # Store QM reference values
        for key, val in geom_qm.bond_lengths.items():
            bond_ref_accum.setdefault(key, []).append(val)
        for key, val in geom_qm.bond_angles.items():
            angle_ref_accum.setdefault(key, []).append(val)
        for key, val in geom_qm.torsion_angles.items():
            torsion_ref_accum.setdefault(key, []).append(val)

        for pot_name in potential_names:
            opt_mol = optimized_molecules[pot_name]

            rmsd = compute_rmsd(
                qm_molecule, conf_idx, opt_mol, conf_idx,
                heavy_atoms_only=heavy_atoms_only,
            )

            geom_opt = get_conformer_geometry(opt_mol, conf_idx)
            b_diffs, a_diffs, t_diffs = compute_geometry_diffs(
                geom_opt, geom_qm
            )

            b_vals = list(b_diffs.values()) if b_diffs else [0.0]
            a_vals = list(a_diffs.values()) if a_diffs else [0.0]
            t_vals = list(t_diffs.values()) if t_diffs else [0.0]

            metrics = QMComparisonMetrics(
                rmsd=rmsd,
                max_bond_diff=float(max(b_vals)),
                mean_bond_diff=float(np.mean(b_vals)),
                max_angle_diff=float(max(a_vals)),
                mean_angle_diff=float(np.mean(a_vals)),
                max_torsion_diff=float(max(t_vals)),
                mean_torsion_diff=float(np.mean(t_vals)),
                bond_diffs=b_diffs,
                angle_diffs=a_diffs,
                torsion_diffs=t_diffs,
            )
            per_potential[pot_name].append(metrics)

            # Accumulate diffs for aggregation
            for key, diff in b_diffs.items():
                bond_accum.setdefault(key, {pot: [] for pot in potential_names})
                bond_accum[key][pot_name].append(diff)
            for key, diff in a_diffs.items():
                angle_accum.setdefault(key, {pot: [] for pot in potential_names})
                angle_accum[key][pot_name].append(diff)
            for key, diff in t_diffs.items():
                torsion_accum.setdefault(key, {pot: [] for pot in potential_names})
                torsion_accum[key][pot_name].append(diff)

    # --- Build aggregated tables ---
    bond_table = _aggregate_qm_diffs(
        bond_accum, bond_ref_accum, potential_names, bond_threshold,
    )
    angle_table = _aggregate_qm_diffs(
        angle_accum, angle_ref_accum, potential_names, angle_threshold,
    )
    torsion_table = _aggregate_qm_diffs(
        torsion_accum, torsion_ref_accum, potential_names, torsion_threshold,
    )

    return QMComparisonResult(
        inchi_key=inchi_key,
        smiles=smiles,
        n_conformers=n_conformers,
        per_potential=per_potential,
        bond_diff_table=bond_table,
        angle_diff_table=angle_table,
        torsion_diff_table=torsion_table,
        molecule_name=molecule_name,
        record_ids=record_ids if record_ids is not None else [],
    )


@dataclass(frozen=True)
class OverallErrorStatistics:
    """Aggregated error statistics across all molecules for each potential.

    Attributes
    ----------
    potential_name : str
        Name of the potential model.
    n_molecules : int
        Number of molecules analysed.
    n_conformers_total : int
        Total number of conformers across all molecules.
    rmsd_mean : float
    rmsd_std : float
    rmsd_median : float
    rmsd_min : float
    rmsd_max : float
    rmsd_max_id : str
        Molecule identifier (record ID or molecule name) with the largest RMSD.
    bond_mean : float
    bond_std : float
    bond_median : float
    bond_min : float
    bond_max : float
    bond_max_id : str
    angle_mean : float
    angle_std : float
    angle_median : float
    angle_min : float
    angle_max : float
    angle_max_id : str
    torsion_mean : float
    torsion_std : float
    torsion_median : float
    torsion_min : float
    torsion_max : float
    torsion_max_id : str
    """

    potential_name: str
    n_molecules: int
    n_conformers_total: int
    rmsd_mean: float
    rmsd_std: float
    rmsd_median: float
    rmsd_min: float
    rmsd_max: float
    rmsd_max_id: str
    bond_mean: float
    bond_std: float
    bond_median: float
    bond_min: float
    bond_max: float
    bond_max_id: str
    angle_mean: float
    angle_std: float
    angle_median: float
    angle_min: float
    angle_max: float
    angle_max_id: str
    torsion_mean: float
    torsion_std: float
    torsion_median: float
    torsion_min: float
    torsion_max: float
    torsion_max_id: str


def compute_overall_statistics(
    qm_results: list[QMComparisonResult],
    potential_names: list[str],
) -> dict[str, OverallErrorStatistics]:
    """Compute overall error statistics per potential across all molecules.

    For each potential, aggregates per-conformer RMSD and per-conformer
    max bond/angle/torsion differences across *all* molecules, then
    computes summary statistics (mean, std, median, min, max) and
    identifies the molecule with the worst (max) error for each metric.

    Parameters
    ----------
    qm_results : list[QMComparisonResult]
        One per molecule, from :func:`evaluate_against_qm`.
    potential_names : list[str]
        Ordered list of potential names.

    Returns
    -------
    dict[str, OverallErrorStatistics]
        Map from potential name to its aggregated statistics.
    """
    stats: dict[str, OverallErrorStatistics] = {}

    for pot_name in potential_names:
        all_rmsds: list[float] = []
        all_max_bond: list[float] = []
        all_max_angle: list[float] = []
        all_max_torsion: list[float] = []
        # Track which molecule each value belongs to (index into qm_results)
        mol_ids_rmsd: list[str] = []
        mol_ids_bond: list[str] = []
        mol_ids_angle: list[str] = []
        mol_ids_torsion: list[str] = []

        total_conformers = 0

        for qm_comp in qm_results:
            mol_name = (
                qm_comp.molecule_name
                or qm_comp.inchi_key
                or qm_comp.smiles
            )
            metrics_list = qm_comp.per_potential.get(pot_name, [])
            for conf_idx, m in enumerate(metrics_list):
                total_conformers += 1
                # Use record_id if available, otherwise molecule name
                if conf_idx < len(qm_comp.record_ids):
                    mol_id = str(qm_comp.record_ids[conf_idx])
                else:
                    mol_id = mol_name
                all_rmsds.append(m.rmsd)
                mol_ids_rmsd.append(mol_id)
                all_max_bond.append(m.max_bond_diff)
                mol_ids_bond.append(mol_id)
                all_max_angle.append(m.max_angle_diff)
                mol_ids_angle.append(mol_id)
                all_max_torsion.append(m.max_torsion_diff)
                mol_ids_torsion.append(mol_id)

        if not all_rmsds:
            continue

        arr_rmsd = np.array(all_rmsds)
        arr_bond = np.array(all_max_bond)
        arr_angle = np.array(all_max_angle)
        arr_torsion = np.array(all_max_torsion)

        stats[pot_name] = OverallErrorStatistics(
            potential_name=pot_name,
            n_molecules=len(qm_results),
            n_conformers_total=total_conformers,
            rmsd_mean=float(np.mean(arr_rmsd)),
            rmsd_std=float(np.std(arr_rmsd)),
            rmsd_median=float(np.median(arr_rmsd)),
            rmsd_min=float(np.min(arr_rmsd)),
            rmsd_max=float(np.max(arr_rmsd)),
            rmsd_max_id=mol_ids_rmsd[int(np.argmax(arr_rmsd))],
            bond_mean=float(np.mean(arr_bond)),
            bond_std=float(np.std(arr_bond)),
            bond_median=float(np.median(arr_bond)),
            bond_min=float(np.min(arr_bond)),
            bond_max=float(np.max(arr_bond)),
            bond_max_id=mol_ids_bond[int(np.argmax(arr_bond))],
            angle_mean=float(np.mean(arr_angle)),
            angle_std=float(np.std(arr_angle)),
            angle_median=float(np.median(arr_angle)),
            angle_min=float(np.min(arr_angle)),
            angle_max=float(np.max(arr_angle)),
            angle_max_id=mol_ids_angle[int(np.argmax(arr_angle))],
            torsion_mean=float(np.mean(arr_torsion)),
            torsion_std=float(np.std(arr_torsion)),
            torsion_median=float(np.median(arr_torsion)),
            torsion_min=float(np.min(arr_torsion)),
            torsion_max=float(np.max(arr_torsion)),
            torsion_max_id=mol_ids_torsion[int(np.argmax(arr_torsion))],
        )

    return stats


def _aggregate_qm_diffs(
    accum: dict[tuple, dict[str, list[float]]],
    ref_accum: dict[tuple, list[float]],
    potential_names: list[str],
    threshold: float,
) -> list[list]:
    """Aggregate per-conformer diffs into tabular rows.

    Rows are: ``[param_key, QM_ref_mean+/-std, pot1_diff_mean+/-std, ...]``
    A row is included if **any** potential's mean diff exceeds the threshold.

    Parameters
    ----------
    accum : dict
        ``{param_key: {pot_name: [diff_values_per_conf]}}``
    ref_accum : dict
        ``{param_key: [ref_values_per_conf]}``
    potential_names : list[str]
        Ordered list of potential names.
    threshold : float
        Minimum mean diff for any potential to include the row.

    Returns
    -------
    list[list]
    """
    table: list[list] = []

    for key in accum:
        # Check if any potential exceeds threshold
        any_exceeds = False
        pot_stats: list[str] = []
        for pot_name in potential_names:
            diffs = accum[key].get(pot_name, [])
            if diffs:
                mean_d = float(np.mean(diffs))
                std_d = float(np.std(diffs))
                if abs(mean_d) > threshold:
                    any_exceeds = True
                pot_stats.append(f"{mean_d:.3f} \u00b1 {std_d:.3f}")
            else:
                pot_stats.append("N/A")

        if not any_exceeds:
            continue

        ref_vals = ref_accum.get(key, [])
        if ref_vals:
            ref_mean = float(np.mean(ref_vals))
            ref_std = float(np.std(ref_vals))
            ref_str = f"{ref_mean:.3f} \u00b1 {ref_std:.3f}"
        else:
            ref_str = "N/A"

        row: list = [key, ref_str] + pot_stats
        table.append(row)

    return table
