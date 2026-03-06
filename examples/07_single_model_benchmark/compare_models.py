#!/usr/bin/env python
"""Compare pre-computed optimized geometries against QM reference.

Gathers optimized SDF files produced by ``optimize_single_model.py``
and compares them against the QM reference data, generating CSV
summaries and a PDF report.

Phase 2 of a two-phase workflow:

1. Read QM reference data (parquet or SDF) -- same file used in Phase 1
2. Discover optimized SDF files in the output directory
3. Compare each model vs QM (RMSD, bond/angle/torsion differences)
4. Write CSV (detail + summary) and PDF report

Usage
-----
::

    python examples/07_single_model_benchmark/compare_models.py <config.json>

JSON configuration
------------------
::

    {
        "data_file": "path/to/input.parquet",
        "optimized_directory": "../outputs/optimized",
        "output_directory": "../outputs/comparison",
        "max_molecules": 2,
        "max_conformers_per_molecule": 3,
        "bond_threshold": 0.1,
        "angle_threshold": 5.0,
        "torsion_threshold": 40.0
    }

Fields:

- **data_file** *(required)*: Same parquet or SDF used during Phase 1.
- **optimized_directory** *(required)*: Directory containing the
  ``optimized_*.sdf`` files produced by Phase 1.
- **output_directory** *(required)*: Where to write CSV and PDF reports.
- **max_molecules** / **max_conformers_per_molecule** *(optional)*:
  Must match the values used during Phase 1 so molecule ordering is
  identical.
- **bond_threshold** / **angle_threshold** / **torsion_threshold**
  *(optional)*: Minimum difference thresholds for reporting tables.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages

from mlip_optimizer import evaluate_against_qm
from mlip_optimizer.data import load_records
from mlip_optimizer.io import read_optimized_sdf, write_qm_comparison_csv
from mlip_optimizer.visualization import (
    create_qm_comparison_report,
    create_title_page,
)

# Allow importing the shared helpers from the same directory.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _shared import load_json_config, resolve_path  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(config_path: str | Path) -> None:
    """Load optimized SDFs, compare vs QM, and write reports."""
    config = load_json_config(
        config_path,
        required_keys=(
            "data_file",
            "optimized_directory",
            "output_directory",
        ),
    )

    # --- 1. Load QM reference ---
    data_file = resolve_path(config["data_file"], config_path)
    records = load_records(
        data_file,
        max_molecules=config.get("max_molecules"),
        max_conformers_per_molecule=config.get("max_conformers_per_molecule"),
    )
    logger.info(
        "QM reference: %d molecules (%d conformers)",
        len(records),
        sum(len(r.record_ids) for r in records),
    )

    if not records:
        logger.warning("No molecules in QM reference.")
        return

    # --- 2. Discover optimized SDF files ---
    opt_dir = resolve_path(config["optimized_directory"], config_path)
    sdf_files = sorted(opt_dir.glob("optimized_*.sdf"))

    if not sdf_files:
        logger.error("No optimized_*.sdf files found in %s", opt_dir)
        sys.exit(1)

    logger.info("Found %d optimized SDF files:", len(sdf_files))
    for f in sdf_files:
        logger.info("  %s", f.name)

    # --- 3. Read back all optimized results ---
    optimized_results: dict[str, list] = {}
    potential_names: list[str] = []

    for sdf_file in sdf_files:
        model_name, molecules = read_optimized_sdf(sdf_file, records)

        if len(molecules) != len(records):
            logger.warning(
                "%s has %d molecules but QM reference has %d -- skipping",
                sdf_file.name,
                len(molecules),
                len(records),
            )
            continue

        optimized_results[model_name] = molecules
        potential_names.append(model_name)
        logger.info("Loaded %s: %d molecules", model_name, len(molecules))

    if not potential_names:
        logger.error("No valid optimized results to compare.")
        sys.exit(1)

    # --- 4. Set up output ---
    output_dir = resolve_path(config["output_directory"], config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    bond_thresh = config.get("bond_threshold", 0.1)
    angle_thresh = config.get("angle_threshold", 5.0)
    torsion_thresh = config.get("torsion_threshold", 40.0)

    # --- 5. Compare each model vs QM ---
    qm_comparison_results = []
    pdf_path = output_dir / "benchmark_report.pdf"

    with PdfPages(str(pdf_path)) as pdf:
        title_parts = [
            "QM Benchmark Report",
            "",
            "Potentials: " + ", ".join(potential_names),
            f"Molecules: {len(records)}",
        ]
        create_title_page(pdf, "\n".join(title_parts))

        for mol_idx, rec in enumerate(records):
            logger.info(
                "[%d/%d] %s  (%d conformers)",
                mol_idx + 1,
                len(records),
                rec.smiles,
                len(rec.record_ids),
            )

            opt_mols = {
                pot_name: optimized_results[pot_name][mol_idx]
                for pot_name in potential_names
            }

            qm_comp = evaluate_against_qm(
                rec.molecule,
                opt_mols,
                bond_threshold=bond_thresh,
                angle_threshold=angle_thresh,
                torsion_threshold=torsion_thresh,
                inchi_key=rec.inchi_key,
                smiles=rec.smiles,
            )
            qm_comparison_results.append(qm_comp)

            create_qm_comparison_report(
                rec.molecule,
                rec.smiles,
                qm_comp,
                potential_names,
                pdf,
                molecule_label=f"mol_{mol_idx} ({rec.inchi_key})",
            )

    logger.info("PDF report: %s", pdf_path)

    # --- 6. Write CSV ---
    detail_csv, summary_csv = write_qm_comparison_csv(
        qm_comparison_results,
        records,
        potential_names,
        output_dir,
    )
    logger.info("Detail CSV:  %s", detail_csv)
    logger.info("Summary CSV: %s", summary_csv)

    logger.info("Comparison complete. Output: %s", output_dir)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compare_models.py <config.json>")
        print()
        print("Example:")
        print(
            "  python examples/07_single_model_benchmark/"
            "compare_models.py \\"
        )
        print(
            "      examples/07_single_model_benchmark/"
            "inputs/compare_config.json"
        )
        sys.exit(1)

    main(sys.argv[1])
