#!/usr/bin/env python
"""Batch optimization over SMARTS-grouped molecules with PDF reporting.

Replicates the workflow from the original scripts, using the refactored
mlip_optimizer package:
1. Load a SMARTS dictionary (JSON mapping SMARTS -> list of SMILES)
2. For each SMARTS group, optimize molecules with multiple methods
3. Compare geometry differences across methods
4. Export SDF files and generate a PDF report
"""

import json
from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages
from openff.toolkit import Molecule
from tqdm import tqdm

from mlip_optimizer import (
    OpenFFOptimizer,
    OpenMMMLOptimizer,
    evaluate_model_pairs,
)
from mlip_optimizer.io import molecules_to_sdf
from mlip_optimizer.visualization import create_comparison_report, create_title_page

# ============================================================
# Configuration -- adjust these for your use case
# ============================================================
SMARTS_FILE = Path(__file__).parent / "inputs" / "chembl35_smarts_dict.json"
OUTPUT_DIR = Path(__file__).parent / "outputs"
NUM_MOLS_PER_SMARTS = 10  # max molecules per SMARTS group
NUM_CONFORMERS = 5  # conformers per molecule


def main():
    # --- Set up optimizers ---
    sage = OpenFFOptimizer(forcefield="openff-2.3.0.offxml")
    aceff = OpenMMMLOptimizer(potential_name="aceff-2.0")

    optimizers = {"sage": sage, "aceff-2.0": aceff}
    model_pairs = [("sage", "aceff-2.0")]

    # --- Load SMARTS dictionary ---
    with open(SMARTS_FILE) as f:
        smarts_dict: dict[str, list[str]] = json.load(f)

    print(f"Loaded {len(smarts_dict)} SMARTS patterns")
    for key, val in smarts_dict.items():
        print(f"  {key}: {len(val)} molecules")

    # --- Create output directories ---
    pdf_dir = OUTPUT_DIR / "reports"
    sdf_dir = OUTPUT_DIR / "optimized_molecules"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # --- Process each SMARTS group ---
    for smarts_idx, (smarts, smiles_list) in tqdm(
        enumerate(smarts_dict.items()), total=len(smarts_dict)
    ):
        pdf_path = pdf_dir / f"optimization_results_{smarts_idx}.pdf"

        with PdfPages(str(pdf_path)) as pdf:
            create_title_page(
                pdf, f"SMARTS Pattern {smarts_idx}:\n{smarts}"
            )

            for mol_idx, smi in enumerate(smiles_list[:NUM_MOLS_PER_SMARTS]):
                mol = Molecule.from_smiles(smi, allow_undefined_stereo=True)
                try:
                    mol.generate_conformers(n_conformers=NUM_CONFORMERS)
                except Exception as e:
                    print(f"  Skipping {smi}: {e}")
                    continue

                # Optimize with each method
                results = {}
                for name, opt in optimizers.items():
                    results[name] = opt.optimize(mol)

                # Export SDF files
                molecules_to_sdf(
                    results,
                    sdf_dir,
                    prefix=f"smarts_{smarts_idx}_mol_{mol_idx}_",
                    extra_properties={
                        "SMARTS": smarts,
                        "SMARTS_ID": f"smarts_{smarts_idx}",
                        "MOLECULE_ID": f"mol_{mol_idx}",
                        "INPUT_SMILES": smi,
                    },
                )

                # Compare geometries
                comparison = evaluate_model_pairs(results, mol, model_pairs)

                # Add comparison page to PDF
                label = f"smarts_{smarts_idx}_mol_{mol_idx}"
                create_comparison_report(
                    mol,
                    smi,
                    comparison,
                    model_pairs,
                    pdf,
                    molecule_label=label,
                )

        print(f"Wrote report: {pdf_path}")

    print(f"\nAll reports saved to: {pdf_dir}")
    print(f"All SDF files saved to: {sdf_dir}")


if __name__ == "__main__":
    main()
