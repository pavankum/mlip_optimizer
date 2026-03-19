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
        "data_files": [
            "path/to/nitrogen.parquet",
            "path/to/sulfur.parquet"
        ],
        "output_directory": "../outputs/optimized",
        "potential": {"type": "openmm_ml", "potential_name": "aceff-2.0"},
        "max_molecules": 100000,
        "max_conformers_per_molecule": 10
    }

Fields:

- **data_files** *(required)*: List of paths to ``.parquet`` or ``.sdf``
  input files containing QM reference geometries. A single string
  ``"data_file"`` is also accepted for backward compatibility.
- **output_directory** *(required)*: Directory for the output SDF files.
  A subdirectory is created per input data file (named after the file stem).
- **potential** *(required)*: A single optimizer specification with
  ``"type"`` (``"openff"`` or ``"openmm_ml"``) plus the type-specific
  parameter (``"forcefield"`` or ``"potential_name"``).
  For ``"openmm_ml"`` potentials that support custom checkpoints
  (MACE variants, AIMNet2, ANI models), an optional ``"model_path"``
  field can point to a checkpoint file (relative to config or absolute).
- **max_molecules** *(optional)*: Limit the number of molecules processed
  (applied per file).
- **max_conformers_per_molecule** *(optional)*: Limit conformers per
  molecule (applied per file).
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
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


def build_optimizer(spec: dict, config_path: str | Path | None = None) -> tuple[str, object]:
    """Return ``(display_name, optimizer)`` from a potential spec."""
    pot_type = spec["type"]
    if pot_type == "openff":
        ff = spec["forcefield"]
        return ff.replace(".offxml", ""), OpenFFOptimizer(forcefield=ff)
    if pot_type == "openmm_ml":
        name = spec["potential_name"]
        model_path = spec.get("model_path")
        if model_path is not None and config_path is not None:
            model_path = resolve_path(model_path, config_path)
        return name, OpenMMMLOptimizer(
            potential_name=name, model_path=model_path,
        )
    raise ValueError(f"Unknown potential type: '{pot_type}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(config_path: str | Path) -> None:
    """Run single-model optimization driven by a JSON config."""
    config = load_json_config(
        config_path,
        required_keys=("output_directory", "potential"),
    )

    # Build optimizer
    pot_name, optimizer = build_optimizer(config["potential"], config_path)
    logger.info("Potential: %s", pot_name)

    # Normalize data_files / data_file to a list
    raw_files = config.get("data_files", config.get("data_file"))
    if raw_files is None:
        raise ValueError("Config must contain 'data_files' (list) or 'data_file' (string)")
    if isinstance(raw_files, str):
        raw_files = [raw_files]

    output_dir = resolve_path(config["output_directory"], config_path)
    timestamp = datetime.now().strftime("_%Y%m%dT%H%M%S")

    # Process each data file independently
    for file_idx, raw in enumerate(raw_files):
        data_file = resolve_path(raw, config_path)
        dataset_name = data_file.stem
        logger.info(
            "=== Dataset %d/%d: %s ===",
            file_idx + 1,
            len(raw_files),
            dataset_name,
        )

        records = load_records(
            data_file,
            max_molecules=config.get("max_molecules"),
            max_conformers_per_molecule=config.get("max_conformers_per_molecule"),
        )
        logger.info(
            "  %d molecules (%d conformers)",
            len(records),
            sum(len(r.record_ids) for r in records),
        )

        if not records:
            logger.warning("  No molecules -- skipping.")
            continue

        # Optimize each molecule, preserving order.  Failed molecules are
        # stored as None so that MOLECULE_IDX stays aligned with records.
        optimized: list[Molecule | None] = []
        n_failed = 0
        for mol_idx, rec in enumerate(records):
            logger.info(
                "  [%d/%d] %s  (%d conformers)",
                mol_idx + 1,
                len(records),
                rec.smiles,
                len(rec.record_ids),
            )
            try:
                opt_mol = optimizer.optimize(rec.molecule) # type: ignore
            except Exception as exc:
                logger.error("    Failed: %s -- skipping molecule", exc)
                optimized.append(None)
                n_failed += 1
                continue
            optimized.append(opt_mol)

        if n_failed:
            logger.warning(
                "  %d/%d molecules failed for %s",
                n_failed, len(records), dataset_name,
            )

        if all(m is None for m in optimized):
            logger.warning("  All molecules failed for %s -- skipping SDF output.", dataset_name)
            continue

        # Write output SDF into a dataset-specific subdirectory
        dataset_dir = output_dir / dataset_name
        sdf_paths = write_batch_sdf(
            records, {pot_name: optimized}, dataset_dir,
            file_suffix=timestamp,
        )
        for name, path in sdf_paths.items():
            logger.info("  Output SDF: %s", path)

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
