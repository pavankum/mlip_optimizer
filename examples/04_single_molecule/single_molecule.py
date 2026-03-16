#!/usr/bin/env python
"""Optimize a single molecule with multiple methods, compare, and report.

Demonstrates the full mlip_optimizer workflow:
1. Load a molecule from SMILES via OpenFF toolkit
2. Generate conformers
3. Optimize with OpenFF (classical) and several OpenMM-ML (ML potential) models
4. Compare geometries
5. Generate a PDF report with molecule visualization and difference tables
6. Export to SDF
7. Print a timing report for each optimizer
"""

import json
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import torch

from matplotlib.backends.backend_pdf import PdfPages
from openff.toolkit import Molecule

from mlip_optimizer import (
    GeometryOptimizer,
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
POTENTIALS_JSON = Path(__file__).parent / "inputs" / "potentials.json"


@dataclass
class _OptimizerInfo:
    """Optimizer instance together with its metadata."""
    optimizer: GeometryOptimizer
    pot_type: str  # e.g. "openmm_ml" or "openff"
    device: str    # e.g. "cuda" or "cpu"


def _resolve_path(raw: str, relative_to: Path) -> Path:
    """Resolve *raw* relative to *relative_to*'s parent directory."""
    p = Path(raw)
    return p if p.is_absolute() else (relative_to.parent / p).resolve()


def _build_optimizers(config_path: Path) -> dict[str, _OptimizerInfo]:
    """Build optimizer instances from a JSON config file."""
    with open(config_path) as fh:
        config = json.load(fh)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    infos: dict[str, _OptimizerInfo] = {}
    for spec in config["potentials"]:
        pot_type = spec["type"]
        if pot_type == "openff":
            ff = spec["forcefield"]
            name = ff.replace(".offxml", "")
            infos[name] = _OptimizerInfo(
                optimizer=OpenFFOptimizer(forcefield=ff),
                pot_type=pot_type,
                device="cpu",  # OpenFF/OpenMM classical always CPU
            )
        elif pot_type == "openmm_ml":
            name = spec["potential_name"]
            model_path = spec.get("model_path")
            if model_path is not None:
                model_path = str(_resolve_path(model_path, config_path))
            infos[name] = _OptimizerInfo(
                optimizer=OpenMMMLOptimizer(
                    potential_name=name, model_path=model_path,
                ),
                pot_type=pot_type,
                device=device,
            )
        else:
            raise ValueError(f"Unknown potential type: '{pot_type}'")
    return infos


def main():
    # --- 1. Load molecule from SMILES ---
    smiles = "CC#CC(=O)N1CC(=CC2(C1)CC2)c3c(cc(c4c3c(c([nH]4)C)C#N)C(=O)N)F"
    mol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    print(f"Loaded molecule: {smiles}")
    print(f"  Atoms: {mol.n_atoms}, Bonds: {mol.n_bonds}")

    # --- 2. Generate conformers ---
    mol.generate_conformers(n_conformers=10)
    print(f"  Generated {len(mol.conformers)} conformers") # type: ignore

    # --- 3. Set up optimizers from JSON config ---
    optimizer_infos = _build_optimizers(POTENTIALS_JSON)

    # --- 4. Optimize with each method and record timings ---
    results = {}
    timings: dict[str, float] = {}
    for name, info in optimizer_infos.items():
        print(f"\nOptimizing with {info.optimizer.name} [{info.pot_type}, {info.device}]...")
        t0 = time.perf_counter()
        results[name] = info.optimizer.optimize(mol)
        elapsed = time.perf_counter() - t0
        timings[name] = elapsed
        print(f"  Done. {len(results[name].conformers)} conformers optimized in {elapsed:.2f}s.")

    # --- 5. Compare geometries (all pairs) ---
    model_names = list(results.keys())
    model_pairs = list(combinations(model_names, 2))
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

    title_models = ", ".join(model_names)
    with PdfPages(str(pdf_path)) as pdf:
        create_title_page(
            pdf,
            f"Single Molecule Optimization Comparison\n{title_models}",
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

    # --- 9. Timing report ---
    print("\n" + "=" * 80)
    print("  Optimization Timing Report")
    print("=" * 80)
    print(f"  {'Model':<35} {'Type':<12} {'Device':<8} {'Time (s)':>10}")
    print("  " + "-" * 76)
    for name, elapsed in sorted(timings.items(), key=lambda x: x[1]):
        info = optimizer_infos[name]
        print(f"  {name:<35} {info.pot_type:<12} {info.device:<8} {elapsed:>10.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
