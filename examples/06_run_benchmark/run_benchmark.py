#!/usr/bin/env python
"""Run a full benchmark: load data, optimize, compare vs QM, write CSV + PDF + SDF.

This script ties together the complete mlip_optimizer pipeline driven
entirely by a JSON configuration file:

1. Load input data from a **parquet** file (QCArchive QM reference) or an
   **SDF** file (pre-built molecules without QM reference).
2. Reconstruct multi-conformer molecules (parquet) or load them directly (SDF).
3. Optimize every molecule with each requested potential.
4. Compare optimized geometries against QM reference (parquet) or pairwise (SDF).
5. Write output:
   - CSV files  (detail + summary)  -- only for parquet/QM workflows
   - PDF report (molecule images + difference tables)
   - SDF files  (one per potential with metadata)

Usage
-----
::

    python examples/06_run_benchmark/run_benchmark.py examples/06_run_benchmark/inputs/benchmark_config.json

JSON configuration schema
-------------------------
::

    {
        "data_file": "path/to/input.parquet",   # or "path/to/input.sdf"
        "output_directory": "./benchmark_output",
        "potentials": [
            {"type": "openff",    "forcefield": "openff-2.3.0.offxml"},
            {"type": "openmm_ml", "potential_name": "aceff-2.0"}
        ],
        "max_molecules": 5,
        "max_conformers_per_molecule": 3,
        "bond_threshold": 0.1,
        "angle_threshold": 5.0,
        "torsion_threshold": 40.0
    }

Fields:

- **data_file** *(required)*: Path to a ``.parquet`` or ``.sdf`` input file.
  Parquet files are expected to follow the QCArchive schema produced by
  ``mlip_optimizer.data.download`` (columns: id, inchi_key, cmiles,
  smiles, dataset_name, energy, geometry).
- **output_directory** *(required)*: Where to write CSV, PDF, and SDF output.
- **potentials** *(required)*: List of optimizer specifications.  Each entry
  must have a ``"type"`` key (``"openff"`` or ``"openmm_ml"``) plus the
  type-specific parameter (``"forcefield"`` or ``"potential_name"``).
- **max_molecules** *(optional)*: Limit the number of molecules processed.
- **max_conformers_per_molecule** *(optional)*: Limit conformers per molecule.
- **bond_threshold** *(optional)*: Min bond diff (Angstrom) for tables.  Default 0.1.
- **angle_threshold** *(optional)*: Min angle diff (degrees) for tables.  Default 5.0.
- **torsion_threshold** *(optional)*: Min torsion diff (degrees) for tables.  Default 40.0.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages
from openff.toolkit import Molecule

from mlip_optimizer import (
    OpenFFOptimizer,
    OpenMMMLOptimizer,
    evaluate_against_qm,
    evaluate_model_pairs,
)
from mlip_optimizer.data import (
    MoleculeRecord,
    group_by_molecule,
    read_parquet,
    read_sdf,
)
from mlip_optimizer.io import (
    molecules_to_sdf,
    write_batch_sdf,
    write_qm_comparison_csv,
)
from mlip_optimizer.visualization import (
    create_comparison_report,
    create_qm_comparison_report,
    create_title_page,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_benchmark_config(path: str | Path) -> dict:
    """Read and validate a benchmark JSON configuration file.

    Parameters
    ----------
    path : str or Path
        Path to the JSON config.

    Returns
    -------
    dict
        The parsed configuration.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If required keys are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as fh:
        config = json.load(fh)

    for key in ("data_file", "output_directory", "potentials"):
        if key not in config:
            raise ValueError(f"Missing required config key: '{key}'")

    if not config["potentials"]:
        raise ValueError("'potentials' list must not be empty")

    return config


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------


def build_optimizers(potential_specs: list[dict]) -> dict[str, object]:
    """Instantiate optimizers from JSON potential specifications.

    Parameters
    ----------
    potential_specs : list[dict]
        Each dict must have ``"type"`` (``"openff"`` or ``"openmm_ml"``)
        plus the relevant parameter.

    Returns
    -------
    dict[str, optimizer]
        Map from a display name to the optimizer instance.
    """
    optimizers: dict = {}

    for spec in potential_specs:
        pot_type = spec["type"]

        if pot_type == "openff":
            ff = spec["forcefield"]
            opt = OpenFFOptimizer(forcefield=ff)
            name = ff.replace(".offxml", "")
            optimizers[name] = opt

        elif pot_type == "openmm_ml":
            pot_name = spec["potential_name"]
            opt = OpenMMMLOptimizer(potential_name=pot_name)
            optimizers[pot_name] = opt

        else:
            raise ValueError(f"Unknown potential type: '{pot_type}'")

    return optimizers


# ---------------------------------------------------------------------------
# Parquet workflow (QM reference comparison)
# ---------------------------------------------------------------------------


def run_parquet_benchmark(
    config: dict,
    optimizers: dict,
) -> None:
    """Run the QM-reference benchmark from a parquet input file.

    Steps:
    1. Read parquet -> group into MoleculeRecords
    2. Optimize each molecule with every potential
    3. Compare vs QM reference
    4. Write CSV (detail + summary), PDF report, and batch SDF
    """
    data_file = Path(config["data_file"])
    output_dir = Path(config["output_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    max_mols = config.get("max_molecules")
    max_confs = config.get("max_conformers_per_molecule")
    bond_thresh = config.get("bond_threshold", 0.1)
    angle_thresh = config.get("angle_threshold", 5.0)
    torsion_thresh = config.get("torsion_threshold", 40.0)

    potential_names = list(optimizers.keys())

    # --- 1. Load data ---
    logger.info("Reading parquet: %s", data_file)
    table = read_parquet(data_file)
    records = group_by_molecule(
        table,
        max_molecules=max_mols,
        max_conformers_per_molecule=max_confs,
    )
    logger.info(
        "Loaded %d molecules (%d total conformers)",
        len(records),
        sum(len(r.record_ids) for r in records),
    )

    if not records:
        logger.warning("No molecules to process. Exiting.")
        return

    # --- 2. Optimize ---
    # optimized_results[pot_name][mol_idx] = Molecule with optimized conformers
    optimized_results: dict[str, list[Molecule | None]] = {
        name: [] for name in potential_names
    }
    qm_comparison_results = []

    pdf_path = output_dir / "benchmark_report.pdf"

    with PdfPages(str(pdf_path)) as pdf:
        title_parts = ["QM Benchmark Report", ""]
        title_parts.append("Potentials: " + ", ".join(potential_names))
        title_parts.append(f"Molecules: {len(records)}")
        create_title_page(pdf, "\n".join(title_parts))

        for mol_idx, rec in enumerate(records):
            logger.info(
                "[%d/%d] %s  (%d conformers)",
                mol_idx + 1, len(records), rec.smiles, len(rec.record_ids),
            )

            opt_mols: dict[str, Molecule | None] = {}
            for pot_name, optimizer in optimizers.items():
                try:
                    opt_mols[pot_name] = optimizer.optimize(rec.molecule)
                except Exception as exc:
                    logger.warning(
                        "  %s failed for %s: %s", pot_name, rec.inchi_key, exc
                    )
                    opt_mols[pot_name] = None  # mark as Opt. fail

            for pot_name in potential_names:
                optimized_results[pot_name].append(opt_mols[pot_name])

            # --- 3. Compare vs QM ---
            qm_comp = evaluate_against_qm(
                rec.molecule,
                opt_mols,
                bond_threshold=bond_thresh,
                angle_threshold=angle_thresh,
                torsion_threshold=torsion_thresh,
                inchi_key=rec.inchi_key,
                smiles=rec.smiles,
                molecule_name=rec.inchi_key or rec.smiles,
                record_ids=rec.record_ids,
            )
            qm_comparison_results.append(qm_comp)

            # Add page to PDF
            create_qm_comparison_report(
                rec.molecule,
                rec.smiles,
                qm_comp,
                potential_names,
                pdf,
                molecule_label=f"mol_{mol_idx} (QCA: {', '.join(str(r) for r in rec.record_ids)})",
            )

    logger.info("PDF report: %s", pdf_path)

    # --- 4. Write CSV ---
    detail_csv, summary_csv = write_qm_comparison_csv(
        qm_comparison_results,
        records,
        potential_names,
        output_dir,
    )
    logger.info("Detail CSV:  %s", detail_csv)
    logger.info("Summary CSV: %s", summary_csv)

    # --- 5. Write batch SDF ---
    sdf_dir = output_dir / "sdf"
    sdf_paths = write_batch_sdf(records, optimized_results, sdf_dir)
    for pot_name, sdf_path in sdf_paths.items():
        logger.info("SDF (%s): %s", pot_name, sdf_path)


# ---------------------------------------------------------------------------
# SDF workflow (pairwise comparison, no QM reference)
# ---------------------------------------------------------------------------


def run_sdf_benchmark(
    config: dict,
    optimizers: dict,
) -> None:
    """Run a pairwise benchmark from an SDF input file.

    When no QM reference is available (i.e. input is SDF, not parquet),
    we compare optimizers against each other rather than against QM.

    Steps:
    1. Read SDF -> list of Molecules
    2. Optimize each molecule with every potential
    3. Compare potentials pairwise
    4. Write PDF report and per-molecule SDF files
    """
    data_file = Path(config["data_file"])
    output_dir = Path(config["output_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    max_mols = config.get("max_molecules")
    bond_thresh = config.get("bond_threshold", 0.1)
    angle_thresh = config.get("angle_threshold", 5.0)
    torsion_thresh = config.get("torsion_threshold", 40.0)

    potential_names = list(optimizers.keys())

    # Build all unique pairs for comparison
    model_pairs = []
    for i, n1 in enumerate(potential_names):
        for n2 in potential_names[i + 1 :]:
            model_pairs.append((n1, n2))

    if not model_pairs:
        logger.warning("Need at least 2 potentials for pairwise comparison.")
        return

    # --- 1. Load SDF ---
    logger.info("Reading SDF: %s", data_file)
    molecules = read_sdf(data_file, as_openff=True)
    if max_mols is not None:
        molecules = molecules[:max_mols]
    logger.info("Loaded %d molecules from SDF", len(molecules))

    if not molecules:
        logger.warning("No molecules to process. Exiting.")
        return

    # --- 2. Optimize and compare ---
    pdf_path = output_dir / "benchmark_report.pdf"
    sdf_dir = output_dir / "sdf"

    with PdfPages(str(pdf_path)) as pdf:
        pair_strs = [f"{a} vs {b}" for a, b in model_pairs]
        title = (
            "Pairwise Benchmark Report\n\n"
            + "Potentials: " + ", ".join(potential_names) + "\n"
            + f"Molecules: {len(molecules)}"
        )
        create_title_page(pdf, title)

        for mol_idx, mol in enumerate(molecules):
            smiles = mol.to_smiles()
            logger.info("[%d/%d] %s", mol_idx + 1, len(molecules), smiles)

            # Generate conformers if the molecule has none
            if not mol.conformers or len(mol.conformers) == 0:
                try:
                    mol.generate_conformers(n_conformers=1)
                except Exception as exc:
                    logger.warning("  Cannot generate conformers: %s", exc)
                    continue

            opt_mols: dict[str, Molecule | None] = {}
            for pot_name, optimizer in optimizers.items():
                try:
                    opt_mols[pot_name] = optimizer.optimize(mol)
                except Exception as exc:
                    logger.warning("  %s failed: %s", pot_name, exc)
                    opt_mols[pot_name] = None

            # Compare pairwise
            comparison = evaluate_model_pairs(
                opt_mols,
                mol,
                model_pairs,
                bond_threshold=bond_thresh,
                angle_threshold=angle_thresh,
                torsion_threshold=torsion_thresh,
            )

            # PDF page
            create_comparison_report(
                mol,
                smiles,
                comparison,
                model_pairs,
                pdf,
                molecule_label=f"mol_{mol_idx}",
            )

            # Per-molecule SDF
            molecules_to_sdf(
                opt_mols,
                sdf_dir,
                prefix=f"mol_{mol_idx}_",
            )

    logger.info("PDF report: %s", pdf_path)
    logger.info("SDF files:  %s/", sdf_dir)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(config_path: str | Path) -> None:
    """Run the benchmark driven by a JSON config file.

    Automatically detects whether the input is parquet (QM-reference
    workflow) or SDF (pairwise workflow) based on the file extension.

    Parameters
    ----------
    config_path : str or Path
        Path to the JSON configuration file.
    """
    config = load_benchmark_config(config_path)

    data_file = Path(config["data_file"])
    if not data_file.exists():
        # Try resolving relative to the config file's directory
        config_dir = Path(config_path).resolve().parent
        data_file = config_dir / config["data_file"]
        if not data_file.exists():
            raise FileNotFoundError(
                f"Data file not found: {config['data_file']}\n"
                f"Also tried: {data_file}"
            )
        config["data_file"] = str(data_file)

    suffix = data_file.suffix.lower()
    logger.info("Input file: %s (type: %s)", data_file, suffix)

    optimizers = build_optimizers(config["potentials"])
    logger.info(
        "Potentials: %s", ", ".join(optimizers.keys())
    )

    if suffix == ".parquet":
        run_parquet_benchmark(config, optimizers)
    elif suffix == ".sdf":
        run_sdf_benchmark(config, optimizers)
    else:
        raise ValueError(
            f"Unsupported input file type: '{suffix}'. "
            "Use a .parquet or .sdf file."
        )

    logger.info("Benchmark complete. Output: %s", config["output_directory"])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_benchmark.py <config.json>")
        print()
        print("Example:")
        print("  python examples/06_run_benchmark/run_benchmark.py examples/06_run_benchmark/inputs/benchmark_config.json")
        sys.exit(1)

    main(sys.argv[1])
