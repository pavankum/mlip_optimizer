#!/usr/bin/env python
"""Run a constrained 2-D torsion scan and plot the energy surface.

Reads a JSON config specifying the molecule (SMILES), force field,
two dihedrals to scan, angle grid, and minimization methods.  Produces a
CSV of energies and a PDF contour plot comparing the methods.

Requires the ``torsion`` extra::

    pip install mlip-optimizer[torsion]

Usage
-----
::

    python examples/09_torsion_scan_2d/torsion_scan_2d.py <config.json>

JSON configuration
------------------
::

    {
        "smiles": "CCCC",
        "force_field": "openff-2.3.0",
        "dihedral_index_1": 0,
        "dihedral_index_2": 1,
        "angle_start": -180,
        "angle_stop": 180,
        "angle_step": 24,
        "methods": [
            {"method": "openmm_torsion_restrained", "restraint_k": 1.0}
        ],
        "output_directory": "../outputs"
    }

Fields:

- **smiles** *(required)*: SMILES string for the molecule.
- **force_field** *(required)*: OpenFF force-field name.
- **dihedral_index_1** / **dihedral_index_2** *(optional)*: Zero-based
  indices into the molecule's ``propers`` list.  Defaults ``0`` and ``1``.
- **angle_start** / **angle_stop** / **angle_step** *(optional)*:
  Define the angle grid (same for both dihedrals).
  Defaults: ``-180``, ``180``, ``24``.
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

from mlip_optimizer.torsion import TorsionScanResult, run_torsion_scan

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)


# ---------------------------------------------------------------------------
# Config helpers
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

    # Select dihedrals
    propers = list(mol.propers)
    dih_idx_1 = config.get("dihedral_index_1", 0)
    dih_idx_2 = config.get("dihedral_index_2", 1)
    for label, idx in [("dihedral_index_1", dih_idx_1), ("dihedral_index_2", dih_idx_2)]:
        if idx >= len(propers):
            raise ValueError(
                f"{label} {idx} out of range "
                f"(molecule has {len(propers)} proper torsions)"
            )
    dihedral_1 = tuple(a.molecule_atom_index for a in propers[dih_idx_1])
    dihedral_2 = tuple(a.molecule_atom_index for a in propers[dih_idx_2])

    mapped_smiles = mol.to_smiles(mapped=True)
    coordinates = mol.conformers[0].m_as("angstrom")

    # Angle grid (shared for both dihedrals)
    angle_start = config.get("angle_start", -180)
    angle_stop = config.get("angle_stop", 180)
    angle_step = config.get("angle_step", 24)
    angle_grid = np.arange(angle_start, angle_stop, angle_step, dtype=float)

    logger.info("Molecule: %s", mapped_smiles)
    logger.info("Dihedral 1 indices: %s", dihedral_1)
    logger.info("Dihedral 2 indices: %s", dihedral_2)
    logger.info(
        "Angle grid: %d x %d = %d points (%d to %d by %d)",
        len(angle_grid), len(angle_grid), len(angle_grid) ** 2,
        angle_start, angle_stop, angle_step,
    )

    # Run 2-D scans
    scan_results: dict[str, TorsionScanResult] = {}
    for spec in methods:
        method = spec["method"]
        restraint_k = spec.get("restraint_k", 1.0)
        logger.info("Running %s (restraint_k=%.1f) ...", method, restraint_k)

        result = run_torsion_scan(
            mapped_smiles=mapped_smiles,
            dihedral_indices=[dihedral_1, dihedral_2],
            coordinates=coordinates,
            mol=mol,
            angle_grid=[angle_grid, angle_grid],
            force_field=force_field,
            method=method,
            restraint_k=restraint_k,
        )
        label = f"{method}_k{restraint_k}"
        scan_results[label] = result

        ok = np.isfinite(result.energies)
        logger.info(
            "  done -- %d/%d converged, energy range: %.2f .. %.2f kcal/mol",
            ok.sum(), result.energies.size,
            np.nanmin(result.energies), np.nanmax(result.energies),
        )

    # Write CSV
    csv_path = output_dir / "torsion_scan_2d.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        header = ["angle_1_deg", "angle_2_deg"] + [
            f"{label}_energy_kcal_mol" for label in scan_results
        ]
        writer.writerow(header)

        for i, a1 in enumerate(angle_grid):
            for j, a2 in enumerate(angle_grid):
                row: list[object] = [float(a1), float(a2)]
                for result in scan_results.values():
                    row.append(float(result.energies[i, j]))
                writer.writerow(row)

    logger.info("CSV: %s", csv_path)

    # Plot PDF -- one contour page per method
    pdf_path = output_dir / "torsion_scan_2d.pdf"
    with PdfPages(str(pdf_path)) as pdf:
        for label, result in scan_results.items():
            e = result.energies.copy()
            ok = np.isfinite(e)
            if not ok.any():
                continue
            e[ok] -= np.nanmin(e)

            fig, ax = plt.subplots(figsize=(7, 6))
            A1, A2 = np.meshgrid(angle_grid, angle_grid, indexing="ij")
            cf = ax.contourf(A1, A2, e, levels=20, cmap="viridis")
            fig.colorbar(cf, ax=ax, label="Relative energy (kcal/mol)")
            ax.contour(A1, A2, e, levels=20, colors="k", linewidths=0.3, alpha=0.4)
            ax.set_xlabel(f"Dihedral 1 (deg)  {dihedral_1}")
            ax.set_ylabel(f"Dihedral 2 (deg)  {dihedral_2}")
            ax.set_title(f"2-D Torsion scan: {smiles}\n{label}")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    logger.info("PDF: %s", pdf_path)
    logger.info("2-D torsion scan complete.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python torsion_scan_2d.py <config.json>")
        print()
        print("Example:")
        print(
            "  python examples/09_torsion_scan_2d/torsion_scan_2d.py \\"
        )
        print(
            "      examples/09_torsion_scan_2d/inputs/"
            "torsion_scan_2d_config.json"
        )
        sys.exit(1)

    main(sys.argv[1])
