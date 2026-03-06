#!/usr/bin/env python
"""Optimize molecules with a single potential and save results to SDF.

Designed for workflows where different models run independently (e.g.,
separate HPC jobs) and results are compared afterward.

Phase 1 of a two-phase workflow:

1. Read QM reference data (parquet or SDF)
2. Optimize all molecules with one potential
3. Save optimized geometries as an ordered SDF file

Use ``compare_models.py`` (Phase 2) to load the optimized SDF files and
compare them against the QM reference.

Usage
-----
::

    python examples/07_single_model_benchmark/optimize_single_model.py <config.json>

JSON configuration
------------------
::

    {
        "data_file": "path/to/input.parquet",
        "output_directory": "../outputs/optimized",
        "potential": {"type": "openmm_ml", "potential_name": "aceff-2.0"},
        "max_molecules": 2,
        "max_conformers_per_molecule": 3
    }

Fields:

- **data_file** *(required)*: Path to a ``.parquet`` or ``.sdf`` input
  file containing QM reference geometries.
- **output_directory** *(required)*: Directory for the output SDF file.
- **potential** *(required)*: A single optimizer specification with
  ``"type"`` (``"openff"`` or ``"openmm_ml"``) plus the type-specific
  parameter (``"forcefield"`` or ``"potential_name"``).
- **max_molecules** *(optional)*: Limit the number of molecules processed.
- **max_conformers_per_molecule** *(optional)*: Limit conformers per molecule.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from openff.toolkit import Molecule

from mlip_optimizer import OpenFFOptimizer, OpenMMMLOptimizer
from mlip_optimizer.data import load_records
from mlip_optimizer.io import write_batch_sdf

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
# Optimizer factory
# ---------------------------------------------------------------------------


def build_optimizer(spec: dict) -> tuple[str, object]:
    """Return ``(display_name, optimizer)`` from a potential spec."""
    pot_type = spec["type"]
    if pot_type == "openff":
        ff = spec["forcefield"]
        return ff.replace(".offxml", ""), OpenFFOptimizer(forcefield=ff)
    if pot_type == "openmm_ml":
        name = spec["potential_name"]
        return name, OpenMMMLOptimizer(potential_name=name)
    raise ValueError(f"Unknown potential type: '{pot_type}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(config_path: str | Path) -> None:
    """Run single-model optimization driven by a JSON config."""
    config = load_json_config(
        config_path,
        required_keys=("data_file", "output_directory", "potential"),
    )

    # Build optimizer
    pot_name, optimizer = build_optimizer(config["potential"])
    logger.info("Potential: %s", pot_name)

    # Load QM reference data
    data_file = resolve_path(config["data_file"], config_path)
    records = load_records(
        data_file,
        max_molecules=config.get("max_molecules"),
        max_conformers_per_molecule=config.get("max_conformers_per_molecule"),
    )
    logger.info(
        "Loaded %d molecules (%d total conformers)",
        len(records),
        sum(len(r.record_ids) for r in records),
    )

    if not records:
        logger.warning("No molecules to process.")
        return

    # Optimize each molecule, preserving order
    optimized: list[Molecule] = []
    for mol_idx, rec in enumerate(records):
        logger.info(
            "[%d/%d] %s  (%d conformers)",
            mol_idx + 1,
            len(records),
            rec.smiles,
            len(rec.record_ids),
        )
        try:
            opt_mol = optimizer.optimize(rec.molecule)
        except Exception as exc:
            logger.warning("  Failed: %s -- keeping unoptimized geometry", exc)
            opt_mol = rec.molecule
        optimized.append(opt_mol)

    # Write output SDF (one file named after the model)
    output_dir = resolve_path(config["output_directory"], config_path)
    sdf_paths = write_batch_sdf(records, {pot_name: optimized}, output_dir)
    for name, path in sdf_paths.items():
        logger.info("Output SDF: %s", path)

    logger.info("Optimization complete.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python optimize_single_model.py <config.json>")
        print()
        print("Example:")
        print(
            "  python examples/07_single_model_benchmark/"
            "optimize_single_model.py \\"
        )
        print(
            "      examples/07_single_model_benchmark/"
            "inputs/optimize_aceff.json"
        )
        sys.exit(1)

    main(sys.argv[1])
