#!/usr/bin/env python
"""Run a constrained torsion scan and plot the energy profile.

Reads a JSON config specifying the molecule (SMILES), force field,
dihedral to scan, angle grid, and minimization methods.  Produces a
CSV of energies and a PDF plot comparing the methods.

Requires the ``torsion`` extra::

    pip install mlip-optimizer[torsion]

Usage
-----
::

    python examples/08_torsion_scan/torsion_scan.py <config.json>

JSON configuration
------------------
::

    {
        "smiles": "NC=O",
        "force_field": "openff-2.1.0",
        "dihedral": [3, 0, 1, 5],
        "angle_start": -180,
        "angle_stop": 180,
        "angle_step": 15,
        "methods": [
            {"method": "openmm_torsion_restrained", "restraint_k": 1.0},
            {"method": "openmm_torsion_atoms_frozen", "restraint_k": 1.0}
        ],
        "output_directory": "../outputs"
    }

Fields:

- **smiles** *(required)*: SMILES string for the molecule.
- **force_field** *(required)*: OpenFF force-field name
  (e.g. ``"openff-2.1.0"``).
- **dihedral** *(required)*: List of four atom indices
  ``[i, j, k, l]`` defining the torsion to scan.
- **angle_start** / **angle_stop** / **angle_step** *(optional)*:
  Define the angle grid in degrees.  Defaults: ``-180``, ``180``, ``15``.
- **methods** *(required)*: List of ``{"method": ..., "restraint_k": ...}``
  dicts defining the minimization methods to compare.
- **output_directory** *(required)*: Where to write CSV and PDF output.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from openff.toolkit import Molecule

from mlip_optimizer.geometry import compute_dihedral
from mlip_optimizer.torsion import TorsionScanResult, run_torsion_scan

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)


# ---------------------------------------------------------------------------
# Config helpers (self-contained -- no _shared dependency)
# ---------------------------------------------------------------------------


def _load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as fh:
        return json.load(fh)


def _resolve(raw: str, config_path: str | Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return (Path(config_path).resolve().parent / p).resolve()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(config_path: str | Path) -> None:
    config = _load_config(config_path)

    smiles = config["smiles"]
    force_field = config["force_field"]
    methods = config["methods"]
    output_dir = _resolve(config["output_directory"], config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build molecule and conformer
    mol = Molecule.from_smiles(smiles)
    mol.generate_conformers(n_conformers=1)

    # Select dihedral
    dihedral = tuple(config["dihedral"])
    if len(dihedral) != 4:
        raise ValueError(
            f"'dihedral' must have exactly 4 atom indices, got {len(dihedral)}"
        )

    mapped_smiles = mol.to_smiles(mapped=True)
    coordinates = mol.conformers[0].m_as("angstrom")

    # Angle grid
    angle_start = config.get("angle_start", -180)
    angle_stop = config.get("angle_stop", 180)
    angle_step = config.get("angle_step", 15)
    angle_grid = np.arange(angle_start, angle_stop, angle_step, dtype=float)

    logger.info("Molecule: %s", mapped_smiles)
    logger.info("Dihedral indices: %s", dihedral)
    logger.info("Angle grid: %d points (%d to %d by %d)",
                len(angle_grid), angle_start, angle_stop, angle_step)

    # Run scans
    scan_results: dict[str, TorsionScanResult] = {}
    for spec in methods:
        method = spec["method"]
        restraint_k = spec.get("restraint_k", 1.0)
        logger.info("Running %s (restraint_k=%.1f) ...", method, restraint_k)

        result = run_torsion_scan(
            mapped_smiles=mapped_smiles,
            dihedral_indices=dihedral,
            coordinates=coordinates,
            mol=mol,
            angle_grid=angle_grid,
            force_field=force_field,
            method=method,
            restraint_k=restraint_k,
        )
        label = f"{method}_k{restraint_k}"
        scan_results[label] = result

        ok = np.isfinite(result.energies)
        logger.info(
            "  done -- %d/%d converged, energy range: %.2f .. %.2f kcal/mol",
            ok.sum(), len(result.energies),
            np.nanmin(result.energies), np.nanmax(result.energies),
        )

    # Write CSV
    csv_path = output_dir / "torsion_scan.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        header = ["angle_deg"] + [
            f"{label}_energy_kcal_mol" for label in scan_results
        ] + [
            f"{label}_actual_dihedral_deg" for label in scan_results
        ]
        writer.writerow(header)

        for i, angle in enumerate(angle_grid):
            row: list[object] = [float(angle)]
            for result in scan_results.values():
                row.append(float(result.energies[i]))
            for result in scan_results.values():
                coords = result.coordinates[i]
                if coords is not None:
                    actual = compute_dihedral(coords, dihedral)
                    row.append(f"{actual:.2f}")
                else:
                    row.append("")
            writer.writerow(row)

    logger.info("CSV: %s", csv_path)

    # Plot PDF
    pdf_path = output_dir / "torsion_scan.pdf"
    with PdfPages(str(pdf_path)) as pdf:
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, result in scan_results.items():
            e = result.energies.copy()
            ok = np.isfinite(e)
            if not ok.any():
                continue
            e[ok] -= e[ok].min()
            ax.plot(angle_grid[ok], e[ok], "o-", markersize=4, label=label)

        ax.set_xlabel("Dihedral angle (deg)")
        ax.set_ylabel("Relative energy (kcal/mol)")
        ax.set_title(f"Torsion scan: {smiles}  (dihedral {dihedral})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    logger.info("PDF: %s", pdf_path)
    logger.info("Torsion scan complete.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python torsion_scan.py <config.json>")
        print()
        print("Example:")
        print(
            "  python examples/08_torsion_scan/torsion_scan.py \\"
        )
        print(
            "      examples/08_torsion_scan/inputs/torsion_scan_config.json"
        )
        sys.exit(1)

    main(sys.argv[1])
