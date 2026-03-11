"""Molecule I/O utilities for SDF export and CSV reporting.

Provides helpers to write optimized molecules (with metadata) to SDF files,
preserving model name, conformer index, and arbitrary user-defined
properties.  Also provides batch SDF and CSV writers for QM-reference
comparison workflows.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
from openff.toolkit import Molecule
from openff.units import unit
from rdkit import Chem

from mlip_optimizer.data.grouping import MoleculeRecord

logger = logging.getLogger(__name__)


def molecule_to_sdf(
    molecule: Molecule,
    path: str | Path,
    *,
    model_name: str = "",
    extra_properties: dict[str, str] | None = None,
) -> None:
    """Write all conformers of a molecule to an SDF file.

    Each conformer is written as a separate record with SD properties
    for the model name, conformer index, canonical SMILES, and any
    additional user-supplied properties.

    Parameters
    ----------
    molecule : openff.toolkit.Molecule
        Molecule with one or more optimized conformers.
    path : str or Path
        Output SDF file path.  Parent directories are created if needed.
    model_name : str, optional
        Name of the optimizer that produced these conformers.
    extra_properties : dict[str, str] or None, optional
        Additional SD properties to attach to each record.

    Examples
    --------
    >>> from openff.toolkit import Molecule
    >>> mol = Molecule.from_smiles("CCO")
    >>> mol.generate_conformers(n_conformers=1)
    >>> molecule_to_sdf(mol, "ethanol.sdf", model_name="openff-2.3.0")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rdmol = molecule.to_rdkit()
    writer = Chem.SDWriter(str(path))
    writer.SetKekulize(False)

    props = extra_properties or {}
    canonical_smiles = molecule.to_smiles()

    for conf_idx in range(len(molecule.conformers)):
        rdmol.SetProp("MODEL_NAME", model_name)
        rdmol.SetProp("CONFORMER_ID", str(conf_idx))
        rdmol.SetProp("SMILES_CANONICAL", canonical_smiles)
        for key, value in props.items():
            rdmol.SetProp(key, value)
        rdmol.SetProp("_Name", f"{model_name}_conf{conf_idx}")
        writer.write(rdmol, confId=conf_idx)

    writer.close()


def molecules_to_sdf(
    molecules: dict[str, Molecule],
    output_dir: str | Path,
    *,
    prefix: str = "",
    extra_properties: dict[str, str] | None = None,
) -> dict[str, Path]:
    """Write multiple optimized molecules to separate SDF files.

    Creates one SDF file per optimizer, named using the optimizer
    name (sanitized for filesystem safety).

    Parameters
    ----------
    molecules : dict[str, Molecule]
        Map from optimizer name to optimized molecule.
    output_dir : str or Path
        Directory for output files (created if it does not exist).
    prefix : str, optional
        Filename prefix, e.g. ``"smarts_0_mol_1_"``.
    extra_properties : dict[str, str] or None, optional
        Additional SD properties to attach to each record.

    Returns
    -------
    dict[str, Path]
        Map from optimizer name to the output file path.

    Examples
    --------
    >>> paths = molecules_to_sdf(
    ...     {"sage": sage_mol, "aceff": aceff_mol},
    ...     "./output/sdf",
    ...     prefix="mol_42_",
    ... )
    >>> paths["sage"]
    PosixPath('output/sdf/mol_42_sage.sdf')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for model_name, mol in molecules.items():
        safe_name = (
            model_name.replace("/", "_")
            .replace(".", "_")
            .replace(":", "_")[:30]
        )
        path = output_dir / f"{prefix}{safe_name}.sdf"
        molecule_to_sdf(
            mol, path, model_name=model_name, extra_properties=extra_properties
        )
        paths[model_name] = path

    return paths


# ---------------------------------------------------------------------------
# Batch outputs for QM-reference comparison
# ---------------------------------------------------------------------------


def write_qm_comparison_csv(
    results: list,
    record_data: list,
    potential_names: list[str],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write per-conformer detail and per-molecule summary CSVs.

    Parameters
    ----------
    results : list[QMComparisonResult]
        One per molecule, from :func:`evaluate_against_qm`.
    record_data : list[MoleculeRecord]
        Original molecule records (for record IDs and QM energies).
        Must be parallel to *results*.
    potential_names : list[str]
        Ordered list of potential names (determines column order).
    output_dir : str or Path
        Output directory (created if needed).

    Returns
    -------
    tuple[Path, Path]
        ``(detail_csv_path, summary_csv_path)``
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_path = output_dir / "qm_comparison_detail.csv"
    summary_path = output_dir / "qm_comparison_summary.csv"

    # --- Detail CSV ---
    detail_header = [
        "inchi_key", "smiles", "record_id", "conformer_idx", "potential",
        "rmsd", "max_bond_diff", "mean_bond_diff",
        "max_angle_diff", "mean_angle_diff",
        "max_torsion_diff", "mean_torsion_diff", "qm_energy_au",
    ]

    with open(detail_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(detail_header)

        for qm_comp, mol_rec in zip(results, record_data):
            for pot_name in potential_names:
                metrics_list = qm_comp.per_potential.get(pot_name, [])
                for conf_idx, metrics in enumerate(metrics_list):
                    record_id = (
                        mol_rec.record_ids[conf_idx]
                        if conf_idx < len(mol_rec.record_ids)
                        else ""
                    )
                    energy = (
                        mol_rec.energies[conf_idx]
                        if conf_idx < len(mol_rec.energies)
                        else ""
                    )
                    writer.writerow([
                        qm_comp.inchi_key,
                        qm_comp.smiles,
                        record_id,
                        conf_idx,
                        pot_name,
                        f"{metrics.rmsd:.6f}",
                        f"{metrics.max_bond_diff:.6f}",
                        f"{metrics.mean_bond_diff:.6f}",
                        f"{metrics.max_angle_diff:.4f}",
                        f"{metrics.mean_angle_diff:.4f}",
                        f"{metrics.max_torsion_diff:.4f}",
                        f"{metrics.mean_torsion_diff:.4f}",
                        energy,
                    ])

    logger.info("Wrote detail CSV: %s", detail_path)

    # --- Summary CSV ---
    summary_header = ["inchi_key", "smiles", "n_conformers"]
    for pot in potential_names:
        summary_header.extend([
            f"{pot}_rmsd_mean",
            f"{pot}_rmsd_std",
            f"{pot}_mean_bond_diff",
            f"{pot}_mean_angle_diff",
            f"{pot}_mean_torsion_diff",
        ])

    with open(summary_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(summary_header)

        for qm_comp in results:
            row: list = [
                qm_comp.inchi_key,
                qm_comp.smiles,
                qm_comp.n_conformers,
            ]
            for pot in potential_names:
                metrics_list = qm_comp.per_potential.get(pot, [])
                if metrics_list:
                    rmsds = [m.rmsd for m in metrics_list]
                    bond_diffs = [m.mean_bond_diff for m in metrics_list]
                    angle_diffs = [m.mean_angle_diff for m in metrics_list]
                    torsion_diffs = [m.mean_torsion_diff for m in metrics_list]
                    row.extend([
                        f"{float(np.mean(rmsds)):.6f}",
                        f"{float(np.std(rmsds)):.6f}",
                        f"{float(np.mean(bond_diffs)):.6f}",
                        f"{float(np.mean(angle_diffs)):.4f}",
                        f"{float(np.mean(torsion_diffs)):.4f}",
                    ])
                else:
                    row.extend(["", "", "", "", ""])
            writer.writerow(row)

    logger.info("Wrote summary CSV: %s", summary_path)
    return detail_path, summary_path


def write_batch_sdf(
    molecule_records: list,
    optimized_results: dict[str, list[Molecule]],
    output_dir: str | Path,
    file_suffix: str = "",
) -> dict[str, Path]:
    """Write one SDF per potential, containing all molecules in input order.

    Each SDF record includes metadata: potential name, SMILES, InChI key,
    QM energy, conformer index, record ID, and molecule index.

    Parameters
    ----------
    molecule_records : list[MoleculeRecord]
        Original molecule data (for metadata).
    optimized_results : dict[str, list[Molecule]]
        Map from potential name to list of optimized molecules
        (parallel to *molecule_records*).
    output_dir : str or Path
        Output directory (created if needed).
    file_suffix : str, optional
        Extra string appended to the filename before ``.sdf``
        (e.g. ``"_20260311T143022"``).

    Returns
    -------
    dict[str, Path]
        Map from potential name to the output SDF file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    for pot_name, opt_mols in optimized_results.items():
        safe_name = (
            pot_name.replace("/", "_")
            .replace(".", "_")
            .replace(":", "_")[:30]
        )
        sdf_path = output_dir / f"optimized_{safe_name}{file_suffix}.sdf"
        writer = Chem.SDWriter(str(sdf_path))
        writer.SetKekulize(False)

        written = 0
        for mol_idx, (mol_rec, opt_mol) in enumerate(
            zip(molecule_records, opt_mols)
        ):
            try:
                rdmol = opt_mol.to_rdkit()
            except Exception:
                logger.warning(
                    "Failed to convert molecule %d (%s) for %s",
                    mol_idx, mol_rec.inchi_key, pot_name,
                )
                continue

            n_confs = len(opt_mol.conformers)
            for conf_idx in range(n_confs):
                rdmol.SetProp("MODEL_NAME", pot_name)
                rdmol.SetProp("SMILES", mol_rec.smiles)
                rdmol.SetProp("INCHI_KEY", mol_rec.inchi_key)
                rdmol.SetProp("CMILES", mol_rec.cmiles)
                rdmol.SetProp("MOLECULE_IDX", str(mol_idx))
                rdmol.SetProp("CONFORMER_IDX", str(conf_idx))
                if conf_idx < len(mol_rec.energies):
                    rdmol.SetProp(
                        "QM_ENERGY_AU", str(mol_rec.energies[conf_idx])
                    )
                if conf_idx < len(mol_rec.record_ids):
                    rdmol.SetProp(
                        "RECORD_ID", str(mol_rec.record_ids[conf_idx])
                    )
                rdmol.SetProp(
                    "_Name",
                    f"{pot_name}_mol{mol_idx}_conf{conf_idx}",
                )
                writer.write(rdmol, confId=conf_idx)
                written += 1

        writer.close()
        paths[pot_name] = sdf_path
        logger.info("Wrote SDF: %s  (%d records)", sdf_path, written)

    return paths


def read_optimized_sdf(
    path: str | Path,
    qm_records: list[MoleculeRecord],
) -> tuple[str, list[Molecule]]:
    """Read an optimized SDF and reconstruct molecules matching QM order.

    Coordinates from the SDF are mapped onto copies of the QM reference
    molecules, guaranteeing that atom indices match for comparison.
    This is the inverse of :func:`write_batch_sdf`.

    Parameters
    ----------
    path : str or Path
        Path to an ``optimized_*.sdf`` file produced by
        :func:`write_batch_sdf`.
    qm_records : list[MoleculeRecord]
        QM reference records (same list used during optimization).

    Returns
    -------
    tuple[str, list[Molecule]]
        ``(model_name, molecules)`` where *molecules* is parallel to
        *qm_records* with optimized conformer coordinates.
    """
    path = Path(path)
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)

    # Group SDF entries by molecule index
    groups: dict[int, list[tuple[int, np.ndarray]]] = {}
    model_name: str | None = None

    for rdmol in supplier:
        if rdmol is None:
            continue
        props = rdmol.GetPropsAsDict()
        mol_idx = int(props.get("MOLECULE_IDX", 0))
        conf_idx = int(props.get("CONFORMER_IDX", 0))
        if model_name is None:
            model_name = str(props.get("MODEL_NAME", path.stem))
        coords = rdmol.GetConformer().GetPositions()
        groups.setdefault(mol_idx, []).append((conf_idx, coords))

    if model_name is None:
        model_name = path.stem.replace("optimized_", "").replace("_", ".")

    # Reconstruct multi-conformer molecules using QM topology
    molecules: list[Molecule] = []
    for mol_idx in range(len(qm_records)):
        if mol_idx not in groups:
            logger.warning(
                "No optimized data for molecule %d in %s", mol_idx, path.name
            )
            molecules.append(qm_records[mol_idx].molecule)
            continue

        entries = sorted(groups[mol_idx], key=lambda x: x[0])
        opt_mol = Molecule(qm_records[mol_idx].molecule)
        opt_mol.clear_conformers()
        for _, coords in entries:
            opt_mol.add_conformer(coords * unit.angstrom)
        molecules.append(opt_mol)

    return model_name, molecules
