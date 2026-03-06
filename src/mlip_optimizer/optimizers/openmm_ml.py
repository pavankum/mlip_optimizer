"""OpenMM-ML machine learning potential optimizer.

Wraps the ``openmmml.MLPotential`` API to optimize molecular geometries
using any of the supported ML interatomic potentials:

    aceff-1.0, aceff-1.1, aceff-2.0, aimnet2, ani1ccx, ani2x, deepmd,
    mace, mace-mpa-0-medium, mace-off23-large, mace-off23-medium,
    mace-off23-small, mace-off24-medium, mace-omat-0-medium,
    mace-omat-0-small, nequip, torchmdnet
"""

from __future__ import annotations

import openmm
from openff.toolkit import Molecule
from openff.units import unit
from openmm import unit as omm_unit
from openmm.app import Simulation
from openmmml import MLPotential


class OpenMMMLOptimizer:
    """Geometry optimizer using an OpenMM-ML machine learning potential.

    Parameters
    ----------
    potential_name : str
        Name of the ML potential recognized by ``openmmml.MLPotential``,
        e.g. ``"aceff-2.0"``, ``"ani2x"``, ``"mace-off23-medium"``.
    tolerance : float, optional
        Convergence tolerance in kJ/mol/nm.  Default is ``10.0``.
    max_iterations : int, optional
        Maximum minimization iterations.  ``0`` (default) means run until
        convergence.

    Examples
    --------
    >>> from openff.toolkit import Molecule
    >>> mol = Molecule.from_smiles("CCO")
    >>> mol.generate_conformers(n_conformers=1)
    >>> opt = OpenMMMLOptimizer(potential_name="ani2x")
    >>> result = opt.optimize(mol)
    >>> len(result.conformers)
    1
    """

    def __init__(
        self,
        potential_name: str = "aceff-2.0",
        tolerance: float = 10.0,
        max_iterations: int = 0,
    ) -> None:
        self._potential_name = potential_name
        self._potential = MLPotential(potential_name)
        self._tolerance = tolerance
        self._max_iterations = max_iterations

    @property
    def name(self) -> str:
        """Name of the ML potential."""
        return self._potential_name

    def optimize(self, molecule: Molecule) -> Molecule:
        """Optimize all conformers using the ML potential.

        Creates an OpenMM ``System`` from the ML potential, runs energy
        minimization for each conformer, and returns a new molecule with
        the optimized coordinates.

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
        off_topology = result.to_topology()
        system = self._potential.createSystem(off_topology.to_openmm())

        original_conformers = list(result.conformers)
        result.clear_conformers()

        for conformer in original_conformers:
            positions = conformer.m_as(unit.nanometer)

            integrator = openmm.LangevinIntegrator(
                300 * omm_unit.kelvin,
                1.0 / omm_unit.picoseconds,
                1.0 * omm_unit.femtosecond,
            )
            simulation = Simulation(off_topology.to_openmm(), system, integrator)
            simulation.context.setPositions(positions)

            simulation.minimizeEnergy(
                tolerance=(
                    self._tolerance
                    * omm_unit.kilojoule_per_mole
                    / omm_unit.nanometer
                ),
                maxIterations=self._max_iterations,
            )

            optimized_coords = (
                simulation.context.getState(getPositions=True)
                .getPositions(asNumpy=True)
            )
            result.add_conformer(optimized_coords * unit.nanometer)

        return result
