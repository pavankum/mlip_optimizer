"""Group QCArchive parquet rows into multi-conformer molecules.

Parquet tables downloaded from QCArchive contain one row per optimization
record.  Multiple rows may share the same ``inchi_key`` -- these correspond
to different conformers of the same molecule.  This module reconstructs
OpenFF :class:`~openff.toolkit.Molecule` objects with multiple conformers
from such tables.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
from openff.toolkit import Molecule
from openff.units import unit
from rdkit import Chem

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MoleculeRecord:
    """A molecule with one or more QM-optimized conformer geometries.

    Attributes
    ----------
    inchi_key : str
        InChI key identifying this molecule.
    smiles : str
        Canonical SMILES (without atom maps).
    cmiles : str
        Canonical mapped SMILES used to reconstruct atom ordering.
    molecule : Molecule
        OpenFF Molecule with one conformer per QCA record.
    record_ids : list[int]
        QCArchive record IDs, one per conformer (same order as conformers).
    energies : list[float]
        QM energies in atomic units, one per conformer.
    dataset_name : str
        Name of the source QCArchive dataset.
    """

    inchi_key: str
    smiles: str
    cmiles: str
    molecule: Molecule
    record_ids: list[int]
    energies: list[float]
    dataset_name: str


def group_by_molecule(
    table: pa.Table,
    *,
    max_molecules: int | None = None,
    max_conformers_per_molecule: int | None = None,
) -> list[MoleculeRecord]:
    """Group parquet rows by ``inchi_key`` and build multi-conformer molecules.

    Each unique ``inchi_key`` becomes one :class:`MoleculeRecord` whose
    :pyattr:`molecule` attribute carries one conformer per row in the group.

    Parameters
    ----------
    table : pa.Table
        Parquet table with columns ``id``, ``inchi_key``, ``cmiles``,
        ``smiles``, ``dataset_name``, ``energy``, ``geometry``.
    max_molecules : int or None, optional
        Limit the number of unique molecules returned.
    max_conformers_per_molecule : int or None, optional
        Limit the number of conformers kept per molecule.

    Returns
    -------
    list[MoleculeRecord]
        One record per unique molecule, ordered by first appearance in
        the table.
    """
    df = table.to_pandas()

    # Preserve first-appearance order via an ordered list of unique keys
    seen: set[str] = set()
    ordered_keys: list[str] = []
    for key in df["inchi_key"]:
        if key not in seen:
            seen.add(key)
            ordered_keys.append(key)

    if max_molecules is not None:
        ordered_keys = ordered_keys[:max_molecules]

    records: list[MoleculeRecord] = []

    for inchi_key in ordered_keys:
        group = df[df["inchi_key"] == inchi_key]

        cmiles = group.iloc[0]["cmiles"]
        smiles = group.iloc[0]["smiles"]
        dataset_name = group.iloc[0]["dataset_name"]

        try:
            mol = Molecule.from_mapped_smiles(
                cmiles, allow_undefined_stereo=True
            )
        except Exception as exc:
            logger.warning(
                "Skipping molecule %s: cannot parse CMILES: %s", inchi_key, exc
            )
            continue

        record_ids: list[int] = []
        energies: list[float] = []
        conformers_added = 0

        for _, row in group.iterrows():
            geom = row["geometry"]

            # Skip rows without valid geometry
            if geom is None:
                continue
            if not hasattr(geom, "__len__") or len(geom) == 0:
                continue

            if (
                max_conformers_per_molecule is not None
                and conformers_added >= max_conformers_per_molecule
            ):
                break

            try:
                coords = np.array(geom, dtype=float).reshape(-1, 3)
                mol.add_conformer(coords * unit.angstrom)
                record_ids.append(int(row["id"]))
                energies.append(float(row["energy"]))
                conformers_added += 1
            except Exception as exc:
                logger.warning(
                    "Skipping conformer for %s (record %s): %s",
                    inchi_key,
                    row["id"],
                    exc,
                )

        if conformers_added == 0:
            logger.warning(
                "No valid conformers for molecule %s; skipping.", inchi_key
            )
            continue

        records.append(
            MoleculeRecord(
                inchi_key=inchi_key,
                smiles=smiles,
                cmiles=cmiles,
                molecule=mol,
                record_ids=record_ids,
                energies=energies,
                dataset_name=dataset_name,
            )
        )

    logger.info(
        "Grouped %d rows into %d molecules (%d total conformers)",
        len(df),
        len(records),
        sum(len(r.record_ids) for r in records),
    )
    return records


def group_sdf_by_molecule(
    path: str | Path,
    *,
    max_molecules: int | None = None,
    max_conformers_per_molecule: int | None = None,
) -> list[MoleculeRecord]:
    """Group SDF entries by InChI key into multi-conformer MoleculeRecords.

    Analogous to :func:`group_by_molecule` but reads from an SDF file
    instead of a pyarrow Table.  Molecule identity is determined by the
    ``INCHI_KEY`` SD property, falling back to an RDKit-computed InChI key
    and finally canonical SMILES.

    Parameters
    ----------
    path : str or Path
        Path to an ``.sdf`` file.
    max_molecules : int or None, optional
        Limit the number of unique molecules returned.
    max_conformers_per_molecule : int or None, optional
        Limit the number of conformers kept per molecule.

    Returns
    -------
    list[MoleculeRecord]
        One record per unique molecule, ordered by first appearance.
    """
    path = Path(path)
    logger.info("Reading SDF: %s", path)
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)

    groups: dict[str, dict] = {}
    order: list[str] = []

    for rdmol in supplier:
        if rdmol is None:
            continue
        props = rdmol.GetPropsAsDict()

        # --- Identify the molecule ---
        inchi_key = str(props.get("INCHI_KEY", ""))
        if not inchi_key:
            try:
                inchi_key = Chem.inchi.InchiToInchiKey(
                    Chem.inchi.MolToInchi(rdmol)
                )
            except Exception:
                inchi_key = Chem.MolToSmiles(rdmol)

        # --- First encounter: create a new group ---
        if inchi_key not in groups:
            if max_molecules is not None and len(groups) >= max_molecules:
                continue
            groups[inchi_key] = {
                "smiles": str(props.get("SMILES", Chem.MolToSmiles(rdmol))),
                "cmiles": str(props.get("CMILES", "")),
                "energies": [],
                "record_ids": [],
                "rdmols": [],
            }
            order.append(inchi_key)

        group = groups[inchi_key]
        if (
            max_conformers_per_molecule is not None
            and len(group["rdmols"]) >= max_conformers_per_molecule
        ):
            continue

        group["rdmols"].append(rdmol)

        # --- Extract optional numeric metadata ---
        energy_str = str(
            props.get(
                "QM_ENERGY_AU",
                props.get("ENERGY_AU", props.get("energy", "")),
            )
        )
        try:
            group["energies"].append(float(energy_str))
        except (ValueError, TypeError):
            group["energies"].append(0.0)

        record_str = str(props.get("RECORD_ID", props.get("id", "")))
        try:
            group["record_ids"].append(int(float(record_str)))
        except (ValueError, TypeError):
            group["record_ids"].append(0)

    # --- Build MoleculeRecords ---
    records: list[MoleculeRecord] = []
    for inchi_key in order:
        group = groups[inchi_key]
        try:
            off_mol = Molecule.from_rdkit(
                group["rdmols"][0], allow_undefined_stereo=True
            )
        except Exception as exc:
            logger.warning("Skipping %s: %s", inchi_key, exc)
            continue

        off_mol.clear_conformers()
        for rdm in group["rdmols"]:
            coords = rdm.GetConformer().GetPositions()
            off_mol.add_conformer(coords * unit.angstrom)

        cmiles = group["cmiles"]
        if not cmiles:
            try:
                cmiles = off_mol.to_smiles(mapped=True)
            except Exception:
                cmiles = group["smiles"]

        records.append(
            MoleculeRecord(
                inchi_key=inchi_key,
                smiles=group["smiles"],
                cmiles=cmiles,
                molecule=off_mol,
                record_ids=group["record_ids"],
                energies=group["energies"],
                dataset_name="sdf_input",
            )
        )

    logger.info(
        "Grouped SDF into %d molecules (%d conformers)",
        len(records),
        sum(len(r.record_ids) for r in records),
    )
    return records
