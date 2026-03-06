"""Abstract interface for geometry optimizers.

All geometry optimizers -- whether classical force fields, ML potentials via
OpenMM-ML, or ASE-based calculators -- satisfy the :class:`GeometryOptimizer`
protocol.  Any object with a ``name`` property and an ``optimize`` method
matching the signature below is a valid optimizer; no inheritance is required.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from openff.toolkit import Molecule


@runtime_checkable
class GeometryOptimizer(Protocol):
    """Structural-typing protocol for geometry optimizers.

    Implementations must:

    * Accept a :class:`~openff.toolkit.Molecule` with one or more conformers.
    * Return a **new** Molecule whose conformers are the optimized geometries.
    * **Not** mutate the input molecule.

    Examples
    --------
    >>> isinstance(my_optimizer, GeometryOptimizer)
    True
    >>> result = my_optimizer.optimize(molecule)
    """

    @property
    def name(self) -> str:
        """Human-readable name for this optimizer (used in reports)."""
        ...

    def optimize(self, molecule: Molecule) -> Molecule:
        """Optimize all conformers of *molecule*.

        Parameters
        ----------
        molecule : openff.toolkit.Molecule
            Input molecule with at least one conformer.

        Returns
        -------
        openff.toolkit.Molecule
            A new molecule with optimized conformer geometries.
        """
        ...
