# mlip_optimizer

Molecular geometry optimization using ML interatomic potentials, built on
[OpenFF Toolkit](https://docs.openforcefield.org/projects/toolkit/) and
[OpenMM-ML](https://github.com/openmm/openmm-ml).

## Installation

### Using Pixi (recommended)

[Pixi](https://pixi.sh) manages both conda and PyPI dependencies from a
single `pyproject.toml` and avoids the conda/pip version conflicts that
commonly affect the PyTorch + OpenMM stack.

```bash
# Install pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# CPU-only (laptop / login node)
pixi install

# GPU with CUDA 13 (cluster nodes)
pixi install -e cuda
```

After installation, activate the environment with:

```bash
# CPU
pixi shell

# GPU
pixi shell -e cuda
```

Then run any example:

```bash
# Single molecule optimization
python examples/04_single_molecule/single_molecule.py

# 2-D torsion scan
python examples/09_torsion_scan_2d/torsion_scan_2d.py \
    examples/09_torsion_scan_2d/inputs/torsion_scan_2d_config.json

# Or use pixi run without entering the shell
pixi run python examples/04_single_molecule/single_molecule.py
pixi run -e cuda python examples/04_single_molecule/single_molecule.py
```

### Using pip

```bash
# Core (OpenFF + OpenMM-ML)
pip install -e .

# With visualization support
pip install -e ".[viz]"

# With ASE-based optimizers (ORB, EGRET)
pip install -e ".[orb,egret]"

# Everything
pip install -e ".[all]"
```

## Quick Start

```python
from openff.toolkit import Molecule
from mlip_optimizer import OpenFFOptimizer, OpenMMMLOptimizer

# Load a molecule from SMILES
mol = Molecule.from_smiles("c1ccccc1")
mol.generate_conformers(n_conformers=3)

# Optimize with a classical force field
sage = OpenFFOptimizer(forcefield="openff-2.3.0.offxml")
sage_result = sage.optimize(mol)

# Optimize with an ML potential
aceff = OpenMMMLOptimizer(potential_name="aceff-2.0")
aceff_result = aceff.optimize(mol)
```

## Available Optimizers

### OpenMM-based (always available)

| Class | Description |
|-------|-------------|
| `OpenFFOptimizer` | Classical OpenFF force fields (Sage, etc.) via OpenMM |
| `OpenMMMLOptimizer` | Any ML potential supported by OpenMM-ML |

**Supported OpenMM-ML potentials:**

NEED TO VERIFY SOME OF THESE ON CHARGE SUPPORT - Yes and No list seems fine, others lack information

| Category | Potential(s) | Charged Molecules | Notes |
|----------|-------------|-------------------|-------|
| `anipotential` | `ani1ccx`, `ani2x` | ❌ No | Neutral only (ANI training sets) |
| `aimnet2potential` | `aimnet2` | ✅ Yes | Trained on neutral + charged molecules |
| `deepmdpotential` | `deepmd` | ⚠️ Depends | Depends on model weights; framework supports charges |
| `fennixpotential` | `fennix` | ⚠️ Depends | Base model; charge support depends on weights |
| | `fennix-bio1-small` | ⚠️ Partial | Bio-focused; limited charge support |
| | `fennix-bio1-small-finetune-ions` | ✅ Yes | Ions fine-tune variant |
| | `fennix-bio1-medium` | ⚠️ Partial | Bio-focused; limited charge support |
| | `fennix-bio1-medium-finetune-ions` | ✅ Yes | Ions fine-tune variant |
| `macepotential` | `aceff-1.0`, `aceff-1.1`, `aceff-2.0` | ✅ Yes | AceMD/HTMD force field; trained on SPICE dataset including charged species |
| | `mace` | ⚠️ Depends | Depends on model weights; framework supports charges |
| | `mace-off23-small/medium/large` | ❌ No | Neutral only (SPICE subset) |
| | `mace-off24-medium` | ✅ Yes | Trained on expanded SPICE 2 including charged species |
| | `mace-mpa-0-medium` | ❌ No | Materials Project Alexandria; materials-focused, inorganic |
| | `mace-omat-0-small/medium` | ❌ No | Materials-focused (Open MatSci); not suited for molecules |
| | `mace-omol-0-extra-large` | ✅ Yes | Trained on large organic molecule set including ions |
| `nequippotential` | `nequip` | ⚠️ Depends | Depends on model weights; framework supports charges |
| `torchmdnetpotential` | `torchmdnet` | ⚠️ Depends | Depends on model weights; some pretrained models include charges |

To verify available potentials in your environment:
```python
from openmmml import MLPotential

print(MLPotential._implFactories.keys())
```

### ASE-based (optional dependencies)

| Class | Description | Install extra |
|-------|-------------|--------------|
| `ORBOptimizer` | ORB v3 conservative force field | `pip install -e ".[orb]"` |
| `EGRETOptimizer` | MACE-architecture models | `pip install -e ".[egret]"` |

## Geometry Analysis

```python
from mlip_optimizer import get_conformer_geometry

geom = get_conformer_geometry(optimized_mol, conf_idx=0)
print(geom.bond_lengths)    # dict of (i, j) -> Angstroms
print(geom.bond_angles)     # dict of (i, j, k) -> degrees
print(geom.torsion_angles)  # dict of (i, j, k, l) -> degrees
```

## Comparing Optimizers

```python
from mlip_optimizer import evaluate_model_pairs

results = {"sage": sage_result, "aceff-2.0": aceff_result}
comparison = evaluate_model_pairs(
    results, mol, model_pairs=[("sage", "aceff-2.0")]
)

print(f"Large bond diffs: {len(comparison.bond_diffs)}")
print(f"Large angle diffs: {len(comparison.angle_diffs)}")
print(f"Large torsion diffs: {len(comparison.torsion_diffs)}")
```

## SDF Export

```python
from mlip_optimizer.io import molecule_to_sdf, molecules_to_sdf

# Single molecule
molecule_to_sdf(aceff_result, "optimized.sdf", model_name="aceff-2.0")

# Multiple models at once
paths = molecules_to_sdf(results, "./output/sdf", prefix="benzene_")
```

## PDF Reports

```python
from matplotlib.backends.backend_pdf import PdfPages
from mlip_optimizer.visualization import create_title_page, create_comparison_report

with PdfPages("report.pdf") as pdf:
    create_title_page(pdf, "My Optimization Report")
    create_comparison_report(
        mol, "c1ccccc1", comparison,
        model_pairs=[("sage", "aceff-2.0")],
        pdf_pages=pdf,
        molecule_label="benzene",
    )
```

## Custom ASE Optimizer

Subclass `ASEOptimizer` to add any ASE-compatible calculator:

```python
from mlip_optimizer.optimizers.ase_base import ASEOptimizer

class MyOptimizer(ASEOptimizer):
    def __init__(self, model_path, **kwargs):
        super().__init__(**kwargs)
        self._model_path = model_path

    @property
    def name(self):
        return "my-model"

    def _create_calculator(self):
        from my_package import MyCalculator
        return MyCalculator(self._model_path)
```

Or implement the `GeometryOptimizer` protocol directly (no inheritance
needed -- just provide a `name` property and `optimize(molecule)` method).

## Examples

See the [`examples/`](examples/) directory — each example has its own
numbered subdirectory with inputs and outputs. Start with
`examples/04_single_molecule/` for a minimal end-to-end demo or
`examples/06_run_benchmark/` for the full config-driven pipeline.
