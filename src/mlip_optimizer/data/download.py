"""Download QM datasets from QCArchive as parquet tables and SDF files.

Connects to the MolSSI QCArchive server and downloads optimization and
torsion-drive datasets.  Each dataset is saved as a pyarrow parquet file
and, optionally, as an SDF file containing all unique molecules with
metadata.  Output is organised into a central data directory with
subdirectories named ``<dataset_slug>_<timestamp>``.

The module can be used programmatically or as a CLI via
``python -m mlip_optimizer.data.download``.

Dataset selection is driven by a JSON configuration file (or an
equivalent Python dict) with the following schema::

    {
        "optimization": ["Dataset Name 1", "Dataset Name 2"],
        "torsiondrive": ["Dataset Name 3"],
        "ignore_iodine": ["Dataset Name 1"]
    }

All three keys are optional.  See ``examples/sage_rc2_datasets.json``
for a complete example containing every dataset from the OpenFF Sage
RC2 fitting pipeline.

Heavily inspired by the OpenFF Sage fitting pipeline download scripts.
"""

from __future__ import annotations

import datetime
import functools
import json
import logging
import multiprocessing
import pathlib
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from openff.toolkit import Molecule
from openff.units import unit
from rdkit import Chem

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

QCFRACTAL_URL = "https://api.qcarchive.molssi.org:443/"


# ---------------------------------------------------------------------------
# JSON config helpers
# ---------------------------------------------------------------------------

DatasetConfig = dict[str, list[str]]
"""Type alias for the dataset configuration dictionary.

Expected keys (all optional):

- ``"optimization"`` -- list of optimization dataset names
- ``"torsiondrive"`` -- list of torsion-drive dataset names
- ``"ignore_iodine"`` -- list of dataset names to filter iodine from
"""


def load_dataset_config(path: str | pathlib.Path) -> DatasetConfig:
    """Load a dataset configuration from a JSON file.

    Parameters
    ----------
    path : str or Path
        Path to a JSON file with the schema described in :data:`DatasetConfig`.

    Returns
    -------
    DatasetConfig

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the JSON does not contain a top-level object.
    """
    path = pathlib.Path(path)
    with open(path) as fh:
        config = json.load(fh)
    if not isinstance(config, dict):
        raise ValueError(
            f"Expected a JSON object at top level, got {type(config).__name__}"
        )
    return config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slugify(name: str) -> str:
    """Turn a dataset name into a filesystem-safe slug (no spaces)."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name)
    return slug.strip("_")


def _timestamp() -> str:
    """Compact UTC timestamp for directory naming."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")


@functools.cache
def _sanitize_smiles(smiles: str) -> str:
    """Remove atom-map numbers and return canonical SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def _get_cmiles(entry) -> str:
    """Extract canonical mapped SMILES from a QCArchive entry."""
    key = "canonical_isomeric_explicit_hydrogen_mapped_smiles"
    try:
        return entry.attributes[key]
    except KeyError:
        return getattr(entry.initial_molecule.identifiers, key)


@functools.cache
def _cmiles_to_inchi(cmiles: str) -> str:
    """Convert mapped SMILES to an InChI key."""
    return Molecule.from_mapped_smiles(
        cmiles, allow_undefined_stereo=True
    ).to_inchikey(fixed_hydrogens=True)


_BOHR_TO_ANGSTROM = 0.529177210903


def _extract_geometry(record) -> list[float] | None:
    """Return the final optimized geometry as a flat list in Angstrom.

    Works for optimization records (``record.final_molecule``) and
    returns *None* on any failure so the download can continue.
    """
    try:
        final_mol = record.final_molecule
        geom_bohr = np.array(final_mol.geometry).reshape(-1, 3)
        geom_ang = geom_bohr * _BOHR_TO_ANGSTROM
        return geom_ang.flatten().tolist()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-record processing (run in worker pool)
# ---------------------------------------------------------------------------


def _process_optimization_record(item):
    """Process one optimization record into a dict row (or *None*)."""
    from openff.qcsubmit.results.filters import (
        ConnectivityFilter,
        UnperceivableStereoFilter,
    )

    record, molecule, cmiles, dataset_name = item

    try:
        if not record.status.upper() == "COMPLETE":
            return None
        if not ConnectivityFilter()._filter_function(None, None, molecule):  # type: ignore
            return None
        if not UnperceivableStereoFilter()._filter_function(None, None, molecule):  # type: ignore
            return None
    except Exception:
        return None

    try:
        smiles = _sanitize_smiles(cmiles)
    except ValueError:
        return None
    try:
        inchi_key = _cmiles_to_inchi(cmiles)
    except Exception:
        return None

    return {
        "id": record.id,
        "inchi_key": inchi_key,
        "cmiles": cmiles,
        "smiles": smiles,
        "dataset_name": dataset_name,
        "energy": record.energies[-1],
        "geometry": _extract_geometry(record),
    }


def _process_torsiondrive_record(item):
    """Process one torsion-drive record into a dict row (or *None*)."""
    from openff.qcsubmit.results.filters import (
        ConnectivityFilter,
        HydrogenBondFilter,
        UnperceivableStereoFilter,
    )

    record, molecule, cmiles, dataset_name = item

    try:
        if not record.status.upper() == "COMPLETE":
            return None
        if not ConnectivityFilter()._filter_function(None, None, molecule):  # type: ignore
            return None
        if not UnperceivableStereoFilter()._filter_function(None, None, molecule):  # type: ignore
            return None
        if not HydrogenBondFilter()._filter_function(None, None, molecule):  # type: ignore
            return None
    except Exception:
        return None

    try:
        smiles = _sanitize_smiles(cmiles)
    except ValueError:
        return None
    try:
        inchi_key = _cmiles_to_inchi(cmiles)
    except Exception:
        return None

    dihedrals = record.specification.keywords.dihedrals
    dihedral: list[int] = []
    for dih in dihedrals:
        dihedral.extend(dih)

    return {
        "id": record.id,
        "inchi_key": inchi_key,
        "cmiles": cmiles,
        "smiles": smiles,
        "dataset_name": dataset_name,
        "dihedral": list(dihedral),
        "n_dihedrals": len(dihedrals),
        "geometry": None,  # torsion drives have per-angle geometries
    }


# ---------------------------------------------------------------------------
# Dataset downloaders
# ---------------------------------------------------------------------------


def download_optimization(
    client,
    dataset_name: str,
    *,
    n_processes: int = 4,
) -> pa.Table:
    """Download an optimization dataset from QCArchive.

    Parameters
    ----------
    client : qcportal.PortalClient
        An authenticated QCArchive portal client.
    dataset_name : str
        Exact name of the optimization dataset on QCArchive.
    n_processes : int, optional
        Worker pool size.  Default is ``4``.

    Returns
    -------
    pa.Table
        Columns: ``id``, ``inchi_key``, ``cmiles``, ``smiles``,
        ``dataset_name``, ``energy``.
    """
    import qcportal as ptl
    from openff.qcsubmit.results import OptimizationResultCollection
    from openff.qcsubmit.utils.utils import portal_client_manager
    from tqdm import tqdm

    try:
        collection = OptimizationResultCollection.from_server(
            client=client, datasets=[dataset_name], spec_name="default"
        )
    except KeyError:
        collection = OptimizationResultCollection.from_server(
            client=client, datasets=[dataset_name], spec_name="spec_1"
        )

    ids_to_cmiles: dict[int, str] = {
        entry.record_id: entry.cmiles
        for entries in collection.entries.values()
        for entry in entries
    }

    # Only to_records() needs the portal_client_manager context.
    # The pool must run OUTSIDE so workers don't inherit the context.
    with portal_client_manager(lambda x: ptl.PortalClient(x, cache_dir=".")):
        records_and_molecules = list(collection.to_records())

    items = [
        (record, molecule, ids_to_cmiles[record.id], dataset_name)
        for record, molecule in records_and_molecules
    ]
    logger.info("Processing %d records for '%s'", len(items), dataset_name)

    with multiprocessing.Pool(n_processes) as pool:
        results = list(tqdm(
            pool.imap(_process_optimization_record, items),
            total=len(items),
            desc=f"Processing {dataset_name}",
        ))

    data_entries = [r for r in results if r is not None]
    logger.info(
        "Kept %d / %d records for '%s'",
        len(data_entries), len(items), dataset_name,
    )
    return pa.Table.from_pylist(data_entries)


def download_torsiondrive(
    client,
    dataset_name: str,
    *,
    n_processes: int = 4,
) -> pa.Table:
    """Download a torsion-drive dataset from QCArchive.

    Parameters
    ----------
    client : qcportal.PortalClient
        An authenticated QCArchive portal client.
    dataset_name : str
        Exact name of the torsion-drive dataset on QCArchive.
    n_processes : int, optional
        Worker pool size.  Default is ``4``.

    Returns
    -------
    pa.Table
        Columns: ``id``, ``inchi_key``, ``cmiles``, ``smiles``,
        ``dataset_name``, ``dihedral``, ``n_dihedrals``.
    """
    import qcportal as ptl
    from openff.qcsubmit.results import TorsionDriveResultCollection
    from openff.qcsubmit.utils.utils import portal_client_manager
    from tqdm import tqdm

    try:
        collection = TorsionDriveResultCollection.from_server(
            client=client, datasets=[dataset_name], spec_name="default"
        )
    except KeyError:
        collection = TorsionDriveResultCollection.from_server(
            client=client, datasets=[dataset_name], spec_name="spec_1"
        )

    ids_to_cmiles: dict[int, str] = {
        entry.record_id: entry.cmiles
        for entries in collection.entries.values()
        for entry in entries
    }

    # Only to_records() needs the portal_client_manager context.
    # The pool must run OUTSIDE so workers don't inherit the context.
    with portal_client_manager(lambda x: ptl.PortalClient(x, cache_dir=".")):
        records_and_molecules = list(collection.to_records())

    items = [
        (record, molecule, ids_to_cmiles[record.id], dataset_name)
        for record, molecule in records_and_molecules
    ]
    logger.info("Processing %d records for '%s'", len(items), dataset_name)

    with multiprocessing.Pool(n_processes) as pool:
        results = list(tqdm(
            pool.imap(_process_torsiondrive_record, items),
            total=len(items),
            desc=f"Processing {dataset_name}",
        ))

    data_entries = [r for r in results if r is not None]
    logger.info(
        "Kept %d / %d records for '%s'",
        len(data_entries), len(items), dataset_name,
    )
    return pa.Table.from_pylist(data_entries)


# ---------------------------------------------------------------------------
# SDF export from a downloaded parquet table
# ---------------------------------------------------------------------------


def table_to_sdf(table: pa.Table, path: str | pathlib.Path) -> None:
    """Write molecules from a pyarrow table to an SDF file.

    When a ``geometry`` column is present (optimization datasets), the
    QCArchive-optimized 3D coordinates are used directly.  The molecule
    is constructed from the CMILES via ``Molecule.from_mapped_smiles``
    which guarantees atom ordering matches the QCA geometry (atom *i*
    corresponds to map number *i + 1*).  Records without geometry are
    skipped.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = table.to_pandas()
    has_geometry_col = "geometry" in df.columns
    writer = Chem.SDWriter(str(path))
    writer.SetKekulize(False)

    written = 0
    for _, row in df.iterrows():
        smiles = row["smiles"]

        geom = row["geometry"] if has_geometry_col else None
        if geom is not None and (not hasattr(geom, "__len__") or len(geom) == 0):
            geom = None
        if geom is None:
            continue

        try:
            off_mol = Molecule.from_mapped_smiles(
                row["cmiles"], allow_undefined_stereo=True
            )
            coords = np.array(geom).reshape(-1, 3) * unit.angstrom
            off_mol.add_conformer(coords)
            mol = off_mol.to_rdkit()
        except Exception:
            logger.debug("Failed to build mol for record %s", row["id"])
            continue

        mol.SetProp("_Name", str(row["id"]))
        mol.SetProp("RECORD_ID", str(row["id"]))
        mol.SetProp("SMILES", smiles)
        mol.SetProp("INCHI_KEY", str(row["inchi_key"]))
        mol.SetProp("CMILES", str(row["cmiles"]))
        mol.SetProp("DATASET_NAME", str(row["dataset_name"]))
        if "energy" in row.index:
            mol.SetProp("ENERGY_AU", str(row["energy"]))
        if "n_dihedrals" in row.index:
            mol.SetProp("N_DIHEDRALS", str(row["n_dihedrals"]))

        writer.write(mol)
        written += 1

    writer.close()
    logger.info("Wrote SDF: %s  (%d records)", path, written)


# ---------------------------------------------------------------------------
# Dataset status checking
# ---------------------------------------------------------------------------

_DATASET_TYPE_MAP: dict[str, str] = {
    "optimization": "Optimization",
    "torsiondrive": "TorsionDrive",
}
"""Map config-file keys (lowercase) to QCArchive dataset type strings."""


def get_dataset_status(
    dataset_name: str,
    dataset_type: str = "optimization",
    *,
    client=None,
) -> dict:
    """Query the record-level status of a single QCArchive dataset.

    Parameters
    ----------
    dataset_name : str
        Exact name of the dataset on QCArchive.
    dataset_type : str
        Config-style type key: ``"optimization"`` or ``"torsiondrive"``.
        Case-insensitive.
    client : qcportal.PortalClient or None, optional
        An existing portal client.  If *None*, a new client is created
        using :data:`QCFRACTAL_URL`.

    Returns
    -------
    dict
        Status dict as returned by ``dataset.status()`` (maps status
        labels to counts, e.g. ``{"COMPLETE": 120, "ERROR": 3}``).

    Raises
    ------
    KeyError
        If *dataset_type* is not ``"optimization"`` or ``"torsiondrive"``.
    """
    import qcportal as ptl

    dtype = dataset_type.lower()
    if dtype not in _DATASET_TYPE_MAP:
        raise KeyError(
            f"Unknown dataset type {dataset_type!r}; "
            f"expected one of {list(_DATASET_TYPE_MAP)}"
        )

    if client is None:
        client = ptl.PortalClient(address=QCFRACTAL_URL, cache_dir=".")

    qca_type = _DATASET_TYPE_MAP[dtype]
    ds = client.get_dataset(qca_type, dataset_name)
    return ds.status()


def check_dataset_status(
    config: str | pathlib.Path | DatasetConfig,
    *,
    client=None,
) -> dict[str, dict]:
    """Check completion status of all datasets in a configuration.

    Connects to QCArchive and queries ``dataset.status()`` for every
    optimization and torsion-drive dataset listed in *config*.  The
    ``"ignore_iodine"`` key, if present, is silently skipped.

    Parameters
    ----------
    config : str, Path, or DatasetConfig
        A path to a JSON configuration file, or a dict with the same
        schema as used by :func:`download_datasets`.
    client : qcportal.PortalClient or None, optional
        An existing portal client.  If *None*, a new client is created
        using :data:`QCFRACTAL_URL`.

    Returns
    -------
    dict[str, dict]
        Mapping of ``"<type>/<dataset_name>"`` to the status dict
        returned by the QCArchive server.  For example::

            {
                "optimization/OpenFF Optimization Set 1": {
                    "COMPLETE": 592,
                    "ERROR": 3,
                },
                "torsiondrive/OpenFF Group1 Torsions": {
                    "COMPLETE": 120,
                },
            }

        If a dataset could not be queried, its value is a dict with
        a single ``"error"`` key containing the error message.
    """
    import qcportal as ptl

    if isinstance(config, (str, pathlib.Path)):
        cfg = load_dataset_config(config)
    else:
        cfg = config

    if client is None:
        client = ptl.PortalClient(address=QCFRACTAL_URL, cache_dir=".")

    results: dict[str, dict] = {}

    for config_key, qca_type in _DATASET_TYPE_MAP.items():
        dataset_names: list[str] = cfg.get(config_key, [])
        for dsname in dataset_names:
            key = f"{config_key}/{dsname}"
            logger.info("Checking status: [%s] %s", qca_type, dsname)
            try:
                ds = client.get_dataset(qca_type, dsname)
                results[key] = ds.status()
            except Exception as exc:
                logger.warning(
                    "Failed to get status for '%s' (%s): %s",
                    dsname,
                    qca_type,
                    exc,
                )
                results[key] = {"error": str(exc)}

    return results


# ---------------------------------------------------------------------------
# Top-level download orchestrator
# ---------------------------------------------------------------------------


def download_datasets(
    config: str | pathlib.Path | DatasetConfig,
    *,
    output_directory: str | pathlib.Path = "data",
    write_sdf: bool = True,
    n_processes: int = 4,
) -> pathlib.Path:
    """Download datasets from QCArchive, saving parquet tables and SDF files.

    Dataset selection is driven entirely by *config* -- either a path to
    a JSON file or an equivalent Python dict.  See
    ``examples/sage_rc2_datasets.json`` for a complete example with the
    full OpenFF Sage RC2 dataset list; copy and trim it to your needs.

    JSON schema::

        {
            "optimization": ["Dataset Name 1", ...],
            "torsiondrive": ["Dataset Name 3", ...],
            "ignore_iodine": ["Dataset Name 1", ...]
        }

    All three keys are optional.  Omit a key (or set it to an empty
    list) to skip that category.

    Output directory layout::

        data/
          optimization/
            OpenFF_Optimization_Set_1_20260303T142500/
              OpenFF_Optimization_Set_1.parquet
              OpenFF_Optimization_Set_1.sdf
          torsiondrive/
            OpenFF_Group1_Torsions_20260303T142510/
              OpenFF_Group1_Torsions.parquet
              OpenFF_Group1_Torsions.sdf

    Parameters
    ----------
    config : str, Path, or dict
        A path to a JSON configuration file, or a dict with the same
        schema.  See :func:`load_dataset_config`.
    output_directory : str or Path
        Root data directory.  Default is ``"data"``.
    write_sdf : bool
        Also write an SDF file per dataset.  Default is ``True``.
    n_processes : int
        Worker pool size per dataset download.  Default is ``4``.

    Returns
    -------
    pathlib.Path
        The root *output_directory* path.

    Examples
    --------
    Download from a JSON file:

    >>> download_datasets("my_datasets.json", output_directory="data")

    Download from an inline dict:

    >>> download_datasets({
    ...     "optimization": ["OpenFF Optimization Set 1"],
    ... })
    """
    import qcportal as ptl
    from tqdm import tqdm

    # Resolve config
    if isinstance(config, (str, pathlib.Path)):
        cfg = load_dataset_config(config)
    else:
        cfg = config

    optimization_datasets: list[str] = cfg.get("optimization", [])
    torsiondrive_datasets: list[str] = cfg.get("torsiondrive", [])
    ignore_iodine: list[str] = cfg.get("ignore_iodine", [])

    output_directory = pathlib.Path(output_directory)
    client = ptl.PortalClient(address=QCFRACTAL_URL, cache_dir=".")

    # --- Optimizations ---
    if optimization_datasets:
        opt_root = output_directory / "optimization"
        for dsname in tqdm(optimization_datasets, desc="Downloading Optimizations"):
            slug = _slugify(dsname)
            ts = _timestamp()
            ds_dir = opt_root / f"{slug}_{ts}"
            ds_dir.mkdir(parents=True, exist_ok=True)

            table = download_optimization(client, dsname, n_processes=n_processes)

            if dsname in ignore_iodine:
                df = table.to_pandas()
                mask = np.array(["I" in smi for smi in df["smiles"].values])
                table = pa.Table.from_pandas(pd.DataFrame(df[~mask]))

            pq.write_table(table, ds_dir / f"{slug}.parquet")
            logger.info("Saved parquet: %s", ds_dir / f"{slug}.parquet")

            if write_sdf:
                table_to_sdf(table, ds_dir / f"{slug}.sdf")

    # --- Torsion drives ---
    if torsiondrive_datasets:
        td_root = output_directory / "torsiondrive"
        for dsname in tqdm(torsiondrive_datasets, desc="Downloading TorsionDrives"):
            slug = _slugify(dsname)
            ts = _timestamp()
            ds_dir = td_root / f"{slug}_{ts}"
            ds_dir.mkdir(parents=True, exist_ok=True)

            table = download_torsiondrive(client, dsname, n_processes=n_processes)

            if dsname in ignore_iodine:
                df = table.to_pandas()
                mask = np.array(["I" in smi for smi in df["smiles"].values])
                table = pa.Table.from_pandas(pd.DataFrame(df[~mask]))

            pq.write_table(table, ds_dir / f"{slug}.parquet")
            logger.info("Saved parquet: %s", ds_dir / f"{slug}.parquet")

            if write_sdf:
                table_to_sdf(table, ds_dir / f"{slug}.sdf")

    return output_directory


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _cli():
    """Command-line interface for downloading QCArchive datasets."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Download QM datasets from QCArchive as parquet + SDF.  "
            "See examples/sage_rc2_datasets.json for an example config."
        ),
    )
    parser.add_argument(
        "config",
        help="Path to a JSON config file listing datasets to download.",
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        type=str,
        default="data",
        help="Root directory for downloaded data (default: data).",
    )
    parser.add_argument(
        "--no-sdf",
        action="store_true",
        help="Skip writing SDF files.",
    )
    parser.add_argument(
        "-n",
        "--n-processes",
        type=int,
        default=4,
        help="Number of worker processes per dataset (default: 4).",
    )

    args = parser.parse_args()

    download_datasets(
        config=args.config,
        output_directory=args.output_directory,
        write_sdf=not args.no_sdf,
        n_processes=args.n_processes,
    )


if __name__ == "__main__":
    _cli()
