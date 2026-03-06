"""mlip_optimizer: Molecular geometry optimization using ML interatomic potentials.

This package provides a unified API for optimizing molecular conformer
geometries using classical force fields (OpenFF), machine learning
potentials via OpenMM-ML, and pluggable ASE-based calculators.

Quick start
-----------
>>> from openff.toolkit import Molecule
>>> from mlip_optimizer import OpenMMMLOptimizer
>>>
>>> mol = Molecule.from_smiles("CCO")
>>> mol.generate_conformers(n_conformers=1)
>>> opt = OpenMMMLOptimizer(potential_name="ani2x")
>>> result = opt.optimize(mol)

Available OpenMM-ML potentials
------------------------------
aceff-1.0, aceff-1.1, aceff-2.0, aimnet2, ani1ccx, ani2x, deepmd,
mace, mace-mpa-0-medium, mace-off23-large, mace-off23-medium,
mace-off23-small, mace-off24-medium, mace-omat-0-medium,
mace-omat-0-small, nequip, torchmdnet

QCArchive data downloading
--------------------------
Install with ``pip install mlip-optimizer[qcarchive]`` then use::

    from mlip_optimizer.data import download_datasets, read_parquet, read_sdf
"""

from mlip_optimizer.comparison import (
    ComparisonResult,
    QMComparisonMetrics,
    QMComparisonResult,
    evaluate_against_qm,
    evaluate_model_pairs,
)
from mlip_optimizer.geometry import (
    ConformerGeometry,
    compute_geometry_diffs,
    compute_rmsd,
    get_conformer_geometry,
)
from mlip_optimizer.optimizers._base import GeometryOptimizer
from mlip_optimizer.optimizers.openff import OpenFFOptimizer
from mlip_optimizer.optimizers.openmm_ml import OpenMMMLOptimizer

# ASE-based optimizers are optional imports; they require extra dependencies.
# Users should install with: pip install mlip-optimizer[orb] or [egret]
try:
    from mlip_optimizer.optimizers.ase_base import ASEOptimizer
except ImportError:
    pass

try:
    from mlip_optimizer.optimizers.orb import ORBOptimizer
except ImportError:
    pass

try:
    from mlip_optimizer.optimizers.egret import EGRETOptimizer
except ImportError:
    pass

__all__ = [
    "ConformerGeometry",
    "get_conformer_geometry",
    "compute_rmsd",
    "compute_geometry_diffs",
    "GeometryOptimizer",
    "OpenFFOptimizer",
    "OpenMMMLOptimizer",
    "ASEOptimizer",
    "ORBOptimizer",
    "EGRETOptimizer",
    "ComparisonResult",
    "evaluate_model_pairs",
    "QMComparisonMetrics",
    "QMComparisonResult",
    "evaluate_against_qm",
]
