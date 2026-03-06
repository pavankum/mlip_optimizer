"""Geometry optimizer implementations.

Core optimizers:
    - :class:`OpenFFOptimizer` -- Classical OpenFF force fields via OpenMM
    - :class:`OpenMMMLOptimizer` -- ML potentials via OpenMM-ML

ASE-based optimizers (optional dependencies):
    - :class:`ASEOptimizer` -- Abstract base for ASE calculator backends
    - :class:`ORBOptimizer` -- ORB v3 conservative force field
    - :class:`EGRETOptimizer` -- MACE-architecture models (EGRET, etc.)
"""

from mlip_optimizer.optimizers._base import GeometryOptimizer
from mlip_optimizer.optimizers.openff import OpenFFOptimizer
from mlip_optimizer.optimizers.openmm_ml import OpenMMMLOptimizer

__all__ = [
    "GeometryOptimizer",
    "OpenFFOptimizer",
    "OpenMMMLOptimizer",
]

# Conditionally export ASE-based optimizers
try:
    from mlip_optimizer.optimizers.ase_base import ASEOptimizer  # noqa: F401

    __all__.append("ASEOptimizer")
except ImportError:
    pass

try:
    from mlip_optimizer.optimizers.orb import ORBOptimizer  # noqa: F401

    __all__.append("ORBOptimizer")
except ImportError:
    pass

try:
    from mlip_optimizer.optimizers.egret import EGRETOptimizer  # noqa: F401

    __all__.append("EGRETOptimizer")
except ImportError:
    pass
