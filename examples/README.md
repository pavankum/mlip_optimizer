# Examples

Each example lives in its own numbered directory with `inputs/` and `outputs/` subdirectories where applicable.

## Prerequisites

```bash
pip install -e ".[qcarchive]"
```

## Examples

### 01_download_datasets

Download QCArchive optimization datasets (parquet + SDF) using a JSON config that lists dataset names. Ships with two configs: `nsp_datasets.json` (3 NSP element sets) and `sage_rc2_datasets.json` (full Sage RC2 training corpus).

```bash
python examples/01_download_datasets/download_datasets.py
```

### 02_download_notebook

Marimo notebook version of the dataset download workflow. Provides the same download-and-inspect logic in an interactive notebook UI.

```bash
marimo run examples/02_download_notebook/download_notebook.py
```

### 03_check_dataset_status

Query QCArchive for completion status of datasets listed in a config file. Prints per-dataset record counts by status (complete, error, etc.).

```bash
python examples/03_check_dataset_status/check_dataset_status.py
```

### 04_single_molecule

End-to-end demo: load a single molecule from SMILES, generate conformers, optimize with SAGE and aceff-2.0, compare bond/angle/torsion differences, and produce a PDF report + SDF files.

```bash
python examples/04_single_molecule/single_molecule.py
```

### 05_batch_smarts

Batch optimization over SMARTS-grouped molecules. Reads a SMARTS-to-SMILES dictionary, optimizes each group with multiple potentials, and writes per-group PDF reports and SDF files. Place `chembl35_smarts_dict.json` in `inputs/` before running.

```bash
python examples/05_batch_smarts/batch_smarts.py
```

### 06_run_benchmark

Config-driven benchmarking pipeline. Supports two workflows based on input file type: **parquet** (compare optimized geometries against QM reference, write CSV + PDF + SDF) or **SDF** (pairwise potential comparison, write PDF + SDF).

```bash
# Parquet / QM-reference workflow
python examples/06_run_benchmark/run_benchmark.py examples/06_run_benchmark/inputs/benchmark_config.json

# SDF / pairwise workflow
python examples/06_run_benchmark/run_benchmark.py examples/06_run_benchmark/inputs/benchmark_config_sdf.json
```

### 07_single_model_benchmark

Two-phase workflow for HPC / batch environments. **Phase 1** optimizes molecules with a single potential and saves results to SDF. **Phase 2** gathers all optimized SDF files and compares them against the QM reference, producing CSV summaries and a PDF report. Input data can be parquet or SDF.

```bash
# Phase 1 -- run each model independently (can be separate jobs)
python examples/07_single_model_benchmark/optimize_single_model.py \
    examples/07_single_model_benchmark/inputs/optimize_aceff.json

python examples/07_single_model_benchmark/optimize_single_model.py \
    examples/07_single_model_benchmark/inputs/optimize_sage.json

# Phase 2 -- compare all accumulated results against QM
python examples/07_single_model_benchmark/compare_models.py \
    examples/07_single_model_benchmark/inputs/compare_config.json
```

### 08_torsion_scan

Constrained torsion (dihedral) scan using yammbs. Rotates a selected dihedral through an angle grid, minimizes at each point with one or more methods, and produces a CSV of energies plus a PDF plot. Requires the `torsion` extra (`pip install mlip-optimizer[torsion]`).

```bash
python examples/08_torsion_scan/torsion_scan.py \
    examples/08_torsion_scan/inputs/torsion_scan_config.json
```
