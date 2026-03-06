#!/usr/bin/env python
"""Optimize a single molecule with multiple methods, compare, and report.

Demonstrates the full mlip_optimizer workflow:
1. Load a molecule from SMILES via OpenFF toolkit
2. Generate conformers
3. Optimize with OpenFF (classical) and OpenMM-ML (ML potential)
4. Compare geometries
5. Generate a PDF report with molecule visualization and difference tables
6. Export to SDF
"""

from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages
from openff.toolkit import Molecule

from mlip_optimizer import (
    OpenFFOptimizer,
    OpenMMMLOptimizer,
    evaluate_model_pairs,
    get_conformer_geometry,
)
from mlip_optimizer.io import molecule_to_sdf
from mlip_optimizer.visualization import create_comparison_report, create_title_page

# ============================================================
# Configuration
# ============================================================
OUTPUT_DIR = Path(__file__).parent / "outputs"
PDF_FILENAME = "comparison_report.pdf"


def main():
    # --- 1. Load molecule from SMILES ---
    smiles = "CC#CC(=O)N1CC(=CC2(C1)CC2)c3c(cc(c4c3c(c([nH]4)C)C#N)C(=O)N)F"
    mol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    print(f"Loaded molecule: {smiles}")
    print(f"  Atoms: {mol.n_atoms}, Bonds: {mol.n_bonds}")

    # --- 2. Generate conformers ---
    mol.generate_conformers(n_conformers=2)
    print(f"  Generated {len(mol.conformers)} conformers") # type: ignore

    # --- 3. Set up optimizers ---
    sage = OpenFFOptimizer(forcefield="openff-2.3.0.offxml")
    aceff = OpenMMMLOptimizer(potential_name="aceff-2.0")

    optimizers = {"sage": sage, "aceff-2.0": aceff}

    # --- 4. Optimize with each method ---
    results = {}
    for name, opt in optimizers.items():
        print(f"\nOptimizing with {opt.name}...")
        results[name] = opt.optimize(mol)
        print(f"  Done. {len(results[name].conformers)} conformers optimized.")

    # --- 5. Compare geometries ---
    model_pairs = [("sage", "aceff-2.0")]
    comparison = evaluate_model_pairs(results, mol, model_pairs)

    print("\n--- Comparison Summary ---")
    print(f"Bond differences > 0.1 A:     {len(comparison.bond_diffs)}")
    print(f"Angle differences > 5 deg:    {len(comparison.angle_diffs)}")
    print(f"Torsion differences > 40 deg: {len(comparison.torsion_diffs)}")

    # --- 6. Inspect a single conformer geometry ---
    geom = get_conformer_geometry(results["aceff-2.0"], conf_idx=0)
    print(f"\naceff-2.0 conformer 0: {len(geom.bond_lengths)} bonds, "
          f"{len(geom.bond_angles)} angles, {len(geom.torsion_angles)} torsions")

    # --- 7. Generate PDF report ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / PDF_FILENAME

    with PdfPages(str(pdf_path)) as pdf:
        create_title_page(
            pdf,
            "Single Molecule Optimization Comparison\n"
            "SAGE (OpenFF 2.3.0) vs aceff-2.0",
        )

        create_comparison_report(
            mol,
            smiles,
            comparison,
            model_pairs,
            pdf,
            molecule_label="single_molecule",
        )

    print(f"\nPDF report written to: {pdf_path}")

    # --- 8. Export to SDF ---
    sdf_dir = OUTPUT_DIR / "sdf"
    sdf_dir.mkdir(parents=True, exist_ok=True)

    for name, opt_mol in results.items():
        path = sdf_dir / f"optimized_{name.replace('.', '_')}.sdf"
        molecule_to_sdf(opt_mol, str(path), model_name=name)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
