import marimo

__generated_with = "0.20.2"
app = marimo.App()


@app.cell
def _():
    #!/usr/bin/env python
    """Download the NSP Nitrogen optimization dataset from QCArchive.

    Demonstrates the mlip_optimizer data download and read-back workflow:
    1. Download completed records for the default specification from
       "OpenFF NSP Optimization Set 1 Nitrogen v4.0"
    2. Save as a parquet table and SDF file with metadata
    3. Read back the downloaded data and inspect it

    Requirements
    ------------
    Install the qcarchive extras first::

        pip install -e ".[qcarchive]"

    Usage
    -----
    As a script::

        python examples/download_nsp_nitrogen.py

    Or via the CLI with the companion JSON config::

        python -m mlip_optimizer.data.download examples/nsp_nitrogen_dataset.json
    """

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
    DATASET_NAME = "OpenFF NSP Optimization Set 1 Phosphorus v4.0"
    OUTPUT_DIR = Path(__file__).parent / "outputs"


    def main():
        # --- 1. Download the dataset ---
        # Pass the dataset selection as a plain dict.  Only the
        # "optimization" key is needed -- no torsion drives, no iodine
        # filtering.
        print(f"Downloading: {DATASET_NAME}")
        print(f"Output directory: {OUTPUT_DIR}")

        download_datasets(
            {"optimization": [DATASET_NAME]},
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

    def _main_():
        main()

    _main_()
    return


if __name__ == "__main__":
    app.run()
