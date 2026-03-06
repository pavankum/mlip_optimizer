#!/usr/bin/env python
"""Download NSP optimization datasets from QCArchive.

Demonstrates the mlip_optimizer data download and read-back workflow:
1. Read dataset names from nsp_datasets.json
2. Download completed records for the default specification
3. Save as parquet tables and SDF files with metadata
4. Read back the downloaded data and inspect it

Requirements
------------
Install the qcarchive extras first::

    pip install -e ".[qcarchive]"

Usage
-----
As a script::

    python examples/01_download_datasets/download_datasets.py
"""

import json
from pathlib import Path

from mlip_optimizer.data import (
    download_datasets,
    list_datasets,
    load_dataset,
    read_parquet_as_pandas,
)

# ============================================================
# Configuration
# ============================================================
DATASETS_JSON = Path(__file__).parent / "inputs" / "nsp_datasets.json"
OUTPUT_DIR = Path(__file__).parent / "outputs"


def main():
    # --- 1. Download the datasets ---
    # Read dataset names from the companion JSON config file.
    with open(DATASETS_JSON) as f:
        dataset_config = json.load(f)

    print(f"Datasets config: {DATASETS_JSON}")
    print(f"Output directory: {OUTPUT_DIR}")
    for ds_type, ds_names in dataset_config.items():
        print(f"  {ds_type}: {ds_names}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    download_datasets(
        dataset_config,
        output_directory=OUTPUT_DIR,
    )

    print("\nDownload complete.")

    # --- 2. Discover what was downloaded ---
    datasets = list_datasets(OUTPUT_DIR)
    print(f"\nDownloaded optimization datasets: {len(datasets['optimization'])}")
    for ds_path in datasets["optimization"]:
        print(f"  {ds_path.name}")

    # --- 3. Load and inspect the dataset ---
    ds_path = datasets["optimization"][-1]  # most recent download
    table, molecules = load_dataset(ds_path)

    print(f"\n--- Dataset: {ds_path.name} ---")
    print(f"Records (parquet rows): {table.num_rows}")
    print(f"Columns: {table.column_names}")

    if molecules is not None:
        print(f"Unique molecules (SDF): {len(molecules)}")

    # --- 4. Inspect the parquet table as a DataFrame ---
    df = table.to_pandas()
    print(f"\nFirst 5 records:")
    print(df[["id", "smiles", "energy", "dataset_name"]].head().to_string(index=False))

    print(f"\nEnergy range: {df['energy'].min():.6f} to {df['energy'].max():.6f} a.u.")

    # --- 5. Show SDF metadata for the first molecule ---
    if molecules is not None and len(molecules) > 0:
        mol = molecules[0]
        print(f"\nFirst molecule SMILES: {mol.properties.get('_Name', 'N/A')}")
        print(f"  InChI Key:   {mol.properties.get('INCHI_KEY', 'N/A')}")
        print(f"  Energy (au): {mol.properties.get('ENERGY_AU', 'N/A')}")
        print(f"  Dataset:     {mol.properties.get('DATASET_NAME', 'N/A')}")

    # --- 6. Also show how to read the parquet directly ---
    parquet_files = sorted(ds_path.glob("*.parquet"))
    if parquet_files:
        df2 = read_parquet_as_pandas(parquet_files[0])
        print(f"\nDirect parquet read: {len(df2)} rows from {parquet_files[0].name}")


if __name__ == "__main__":
    main()
