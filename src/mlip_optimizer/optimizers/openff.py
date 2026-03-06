"""OpenFF classical force field optimizer via OpenMM.

Uses the `openff-toolkit` :class:`ForceField` to parameterize a molecule
and OpenMM's energy minimization to optimize conformer geometries.
"""

from __future__ import annotations

from openff.toolkit import ForceField, Molecule
from openff.units import unit
from openmm import VerletIntegrator
from openmm import unit as omm_unit
from openmm.app import Simulation


class OpenFFOptimizer:
    """Geometry optimizer using an OpenFF classical force field.

    Parameters
    ----------
    forcefield : str
        Name of the OpenFF force field file, e.g. ``"openff-2.3.0.offxml"``.
    tolerance : float or None, optional
        Convergence tolerance in kJ/mol/nm for OpenMM's energy minimizer.
        ``None`` uses OpenMM's default (10 kJ/mol/nm).
    max_iterations : int, optional
        Maximum number of minimization iterations.  ``0`` (default) means
        run until convergence.

    Examples
    --------
    >>> from openff.toolkit import Molecule
    >>> mol = Molecule.from_smiles("CCO")
    >>> mol.generate_conformers(n_conformers=1)
    >>> opt = OpenFFOptimizer(forcefield="openff-2.3.0.offxml")
    >>> result = opt.optimize(mol)
    >>> len(result.conformers)
    1
    """

    def __init__(
        self,
        forcefield: str = "openff-2.3.0.offxml",
        tolerance: float | None = None,
        max_iterations: int = 0,
    ) -> None:
        self._forcefield_name = forcefield
        self._ff = ForceField(forcefield)
        self._tolerance = tolerance
        self._max_iterations = max_iterations

    @property
    def name(self) -> str:
        """Name of the force field file."""
        return self._forcefield_name

    def optimize(self, molecule: Molecule) -> Molecule:
        """Optimize all conformers using the OpenFF force field.

        Parameters
        ----------
        molecule : openff.toolkit.Molecule
            Input molecule with at least one conformer.

        Returns
        -------
        openff.toolkit.Molecule
            New molecule with optimized conformer geometries.
        """
        result = Molecule(molecule)
        topology = result.to_topology()
        system = self._ff.create_openmm_system(topology)

        original_conformers = list(result.conformers)
        result.clear_conformers()

        for conformer in original_conformers:
            integrator = VerletIntegrator(0.001)
            simulation = Simulation(topology.to_openmm(), system, integrator)
            simulation.context.setPositions(conformer.to_openmm())

            minimize_kwargs: dict = {"maxIterations": self._max_iterations}
            if self._tolerance is not None:
                minimize_kwargs["tolerance"] = (
                    self._tolerance
                    * omm_unit.kilojoule_per_mole
                    / omm_unit.nanometer
                )

            simulation.minimizeEnergy(**minimize_kwargs)

            optimized_coords = (
                simulation.context.getState(getPositions=True)
                .getPositions(asNumpy=True)
            )
            result.add_conformer(optimized_coords * unit.nanometer)

        return result
