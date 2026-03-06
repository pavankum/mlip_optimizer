"""Data downloading, reading, and grouping utilities for QCArchive datasets.

Downloading
-----------
- :func:`download_datasets` -- batch-download datasets from QCArchive,
  driven by a JSON config file (or equivalent dict).
- :func:`download_optimization` -- download a single optimization dataset.
- :func:`download_torsiondrive` -- download a single torsion-drive dataset.
- :func:`table_to_sdf` -- write unique molecules from a pyarrow table to SDF.

Status Checking
---------------
- :func:`check_dataset_status` -- check completion status of datasets in a config.
- :func:`get_dataset_status` -- check completion status of a single dataset.

Configuration
-------------
- :func:`load_dataset_config` -- load a dataset config from a JSON file.
- See ``examples/sage_rc2_datasets.json`` for a complete example config.

Reading
-------
- :func:`read_parquet` -- read a parquet file to a pyarrow Table.
- :func:`read_parquet_as_pandas` -- read a parquet file to a pandas DataFrame.
- :func:`read_dataset_parquets` -- concatenate all parquets in a directory.
- :func:`read_sdf` -- read an SDF file to OpenFF Molecules (or RDKit Mols).
- :func:`read_sdf_metadata` -- read only SD-property metadata from an SDF.
- :func:`list_datasets` -- discover downloaded datasets by category.
- :func:`load_dataset` -- convenience loader for a single dataset directory.
- :func:`load_records` -- unified loader: parquet or SDF to MoleculeRecords.

Grouping
--------
- :class:`MoleculeRecord` -- a molecule with multiple QM-optimized conformers.
- :func:`group_by_molecule` -- group parquet rows by molecule identity.
- :func:`group_sdf_by_molecule` -- group SDF entries by InChI key.
"""

from mlip_optimizer.data.download import (
    check_dataset_status,
    download_datasets,
    download_optimization,
    download_torsiondrive,
    get_dataset_status,
    load_dataset_config,
    table_to_sdf,
)
from mlip_optimizer.data.grouping import MoleculeRecord, group_by_molecule, group_sdf_by_molecule
from mlip_optimizer.data.readers import (
    list_datasets,
    load_dataset,
    load_records,
    read_dataset_parquets,
    read_parquet,
    read_parquet_as_pandas,
    read_sdf,
    read_sdf_metadata,
)

__all__ = [
    # Download
    "check_dataset_status",
    "download_datasets",
    "download_optimization",
    "download_torsiondrive",
    "get_dataset_status",
    "table_to_sdf",
    # Config
    "load_dataset_config",
    # Read
    "read_parquet",
    "read_parquet_as_pandas",
    "read_dataset_parquets",
    "read_sdf",
    "read_sdf_metadata",
    "list_datasets",
    "load_dataset",
    "load_records",
    # Grouping
    "MoleculeRecord",
    "group_by_molecule",
    "group_sdf_by_molecule",
]
