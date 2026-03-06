"""Readers for downloaded QCArchive data (parquet tables and SDF files).

Provides utilities to:

* Read individual ``.parquet`` files back to pyarrow Tables or pandas
  DataFrames.
* Read ``.sdf`` files back to lists of :class:`openff.toolkit.Molecule`
  or RDKit ``Mol`` objects, preserving SD-property metadata.
* Discover and list datasets that have been downloaded into a data
  directory tree (as created by :func:`~mlip_optimizer.data.download.download_datasets`).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from openff.toolkit import Molecule
from rdkit import Chem

from mlip_optimizer.data.grouping import (
    MoleculeRecord,
    group_by_molecule,
    group_sdf_by_molecule,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parquet readers
# ---------------------------------------------------------------------------


def read_parquet(path: str | Path) -> pa.Table:
    """Read a single parquet file to a pyarrow Table.

    Parameters
    ----------
    path : str or Path
        Path to a ``.parquet`` file.

    Returns
    -------
    pa.Table
    """
    return pq.read_table(str(path))


def read_parquet_as_pandas(path: str | Path):
    """Read a single parquet file to a pandas DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to a ``.parquet`` file.

    Returns
    -------
    pandas.DataFrame
    """
    return read_parquet(path).to_pandas()


def read_dataset_parquets(directory: str | Path) -> pa.Table:
    """Read and concatenate all parquet files in *directory*.

    Parameters
    ----------
    directory : str or Path
        A directory (possibly with subdirectories) containing
        ``.parquet`` files.

    Returns
    -------
    pa.Table
        A single table concatenated from every parquet file found.

    Raises
    ------
    FileNotFoundError
        If *directory* does not exist or contains no parquet files.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Not a directory: {directory}")

    parquet_files = sorted(directory.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found under {directory}")

    tables = [pq.read_table(str(f)) for f in parquet_files]
    return pa.concat_tables(tables, promote_options="default")


# ---------------------------------------------------------------------------
# SDF readers
# ---------------------------------------------------------------------------


def read_sdf(
    path: str | Path,
    *,
    as_openff: bool = True,
) -> list[Molecule] | list[Chem.rdchem.Mol]:
    """Read molecules from an SDF file, preserving SD-property metadata.

    Parameters
    ----------
    path : str or Path
        Path to an ``.sdf`` file.
    as_openff : bool
        If ``True`` (default), return OpenFF ``Molecule`` objects.
        If ``False``, return RDKit ``Mol`` objects with SD properties
        intact.

    Returns
    -------
    list[Molecule] or list[rdkit.Chem.rdchem.Mol]
        One entry per SDF record.
    """
    path = Path(path)
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)

    molecules: list = []
    for rdmol in supplier:
        if rdmol is None:
            continue
        if as_openff:
            try:
                off_mol = Molecule.from_rdkit(rdmol, allow_undefined_stereo=True)
                # Carry SD properties over as molecule metadata
                for prop_name in rdmol.GetPropsAsDict():
                    off_mol.properties[prop_name] = rdmol.GetProp(prop_name)
                molecules.append(off_mol)
            except Exception as exc:
                logger.warning("Skipping molecule: %s", exc)
        else:
            molecules.append(rdmol)

    logger.info("Read %d molecules from %s", len(molecules), path)
    return molecules


def read_sdf_metadata(path: str | Path) -> list[dict[str, str]]:
    """Read only the SD-property metadata from an SDF file.

    This is lighter weight than :func:`read_sdf` when you only need the
    metadata (SMILES, InChI keys, energies, etc.) without constructing
    full molecule objects.

    Parameters
    ----------
    path : str or Path
        Path to an ``.sdf`` file.

    Returns
    -------
    list[dict[str, str]]
        One dict per SDF record, mapping property names to string values.
    """
    path = Path(path)
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)

    metadata: list[dict[str, str]] = []
    for rdmol in supplier:
        if rdmol is None:
            continue
        props = rdmol.GetPropsAsDict()
        # Ensure all values are strings for consistency
        metadata.append({k: str(v) for k, v in props.items()})

    return metadata


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------


def list_datasets(
    data_directory: str | Path = "data",
) -> dict[str, list[Path]]:
    """Discover downloaded datasets organised by category.

    Scans the directory tree created by
    :func:`~mlip_optimizer.data.download.download_datasets` and returns
    a mapping from category (``"optimization"`` / ``"torsiondrive"``) to
    a sorted list of dataset sub-directory paths.

    Parameters
    ----------
    data_directory : str or Path
        Root data directory.  Default is ``"data"``.

    Returns
    -------
    dict[str, list[Path]]
        ``{"optimization": [...], "torsiondrive": [...]}``.
    """
    data_directory = Path(data_directory)
    result: dict[str, list[Path]] = {}

    for category in ("optimization", "torsiondrive"):
        cat_dir = data_directory / category
        if not cat_dir.is_dir():
            result[category] = []
            continue
        # Each child directory that contains at least one parquet is a dataset
        dirs = sorted(
            d for d in cat_dir.iterdir()
            if d.is_dir() and list(d.glob("*.parquet"))
        )
        result[category] = dirs

    return result


def load_dataset(
    dataset_path: str | Path,
) -> tuple[pa.Table, list[Molecule] | None]:
    """Load a single downloaded dataset (parquet + optional SDF).

    Parameters
    ----------
    dataset_path : str or Path
        Path to a dataset directory containing a ``.parquet`` file and
        optionally a ``.sdf`` file.

    Returns
    -------
    tuple[pa.Table, list[Molecule] or None]
        The pyarrow table and, if an SDF file exists, the list of
        OpenFF molecules.  ``None`` if no SDF was found.
    """
    dataset_path = Path(dataset_path)

    parquet_files = sorted(dataset_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files in {dataset_path}")
    table = pq.read_table(str(parquet_files[0]))

    sdf_files = sorted(dataset_path.glob("*.sdf"))
    molecules = read_sdf(sdf_files[0]) if sdf_files else None

    return table, molecules


# ---------------------------------------------------------------------------
# Unified record loader
# ---------------------------------------------------------------------------


def load_records(
    data_file: str | Path,
    *,
    max_molecules: int | None = None,
    max_conformers_per_molecule: int | None = None,
) -> list[MoleculeRecord]:
    """Load QM reference data into :class:`MoleculeRecord` objects.

    Dispatches on file extension:

    * ``.parquet`` -- uses :func:`read_parquet` + :func:`group_by_molecule`
    * ``.sdf`` -- uses :func:`group_sdf_by_molecule`

    Parameters
    ----------
    data_file : str or Path
        Path to a ``.parquet`` or ``.sdf`` input file.
    max_molecules : int or None, optional
        Limit the number of unique molecules returned.
    max_conformers_per_molecule : int or None, optional
        Limit the number of conformers kept per molecule.

    Returns
    -------
    list[MoleculeRecord]
        One record per unique molecule.
    """
    data_file = Path(data_file)
    suffix = data_file.suffix.lower()

    if suffix == ".parquet":
        logger.info("Reading parquet: %s", data_file)
        table = read_parquet(data_file)
        return group_by_molecule(
            table,
            max_molecules=max_molecules,
            max_conformers_per_molecule=max_conformers_per_molecule,
        )

    if suffix == ".sdf":
        return group_sdf_by_molecule(
            data_file,
            max_molecules=max_molecules,
            max_conformers_per_molecule=max_conformers_per_molecule,
        )

    raise ValueError(
        f"Unsupported input file type: '{suffix}'. Use .parquet or .sdf."
    )
