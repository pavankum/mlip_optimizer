"""OpenMM-ML machine learning potential optimizer.

Wraps the ``openmmml.MLPotential`` API to optimize molecular geometries
using any of the supported ML interatomic potentials:

    aceff-1.0, aceff-1.1, aceff-2.0, aimnet2, ani1ccx, ani2x, deepmd,
    mace, mace-mpa-0-medium, mace-off23-large, mace-off23-medium,
    mace-off23-small, mace-off24-medium, mace-omat-0-medium,
    mace-omat-0-small, nequip, torchmdnet
"""

from __future__ import annotations

from pathlib import Path

import openmm
from openff.toolkit import Molecule
from openff.units import unit
from openmm import unit as omm_unit
from openmm.app import Simulation
from openmmml import MLPotential

# Mapping from potential name to the ``createSystem`` keyword used to pass a
# custom model/checkpoint path.  Only potentials that support loading from
# a user-supplied file are listed here.
_MODEL_PATH_KWARG: dict[str, str] = {
    # MACE family  →  modelPath
    "mace": "modelPath",
    "mace-mpa-0-medium": "modelPath",
    "mace-off23-large": "modelPath",
    "mace-off23-medium": "modelPath",
    "mace-off23-small": "modelPath",
    "mace-off24-medium": "modelPath",
    "mace-omat-0-medium": "modelPath",
    "mace-omat-0-small": "modelPath",
    # AIMNet / ANI  →  modelPath
    "aimnet2": "modelPath",
    "ani1ccx": "modelPath",
    "ani2x": "modelPath",
    # AcePotential (AceFF)  →  ckpt_path
    "aceff-1.0": "ckpt_path",
    "aceff-1.1": "ckpt_path",
    "aceff-2.0": "ckpt_path",
    # FeNNIx  →  modelPath
    "fennix-bio1-medium": "modelPath",
    "fennix-bio1-medium-finetune-ions": "modelPath",
    "fennix-bio1-small": "modelPath",
    "fennix-bio1-small-finetune-ions": "modelPath",
}


class OpenMMMLOptimizer:
    """Geometry optimizer using an OpenMM-ML machine learning potential.

    Parameters
    ----------
    potential_name : str
        Name of the ML potential recognized by ``openmmml.MLPotential``,
        e.g. ``"aceff-2.0"``, ``"ani2x"``, ``"mace-off23-medium"``.
    model_path : str or Path or None, optional
        Path to a custom model checkpoint file.  Only used for potentials
        listed in :data:`_MODEL_PATH_KWARG` (MACE variants, AIMNet2,
        ANI models, AceFF, FeNNIx).  When ``None`` (default) the
        built-in model shipped with the potential package is used.
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

    Using a custom checkpoint:

    >>> opt = OpenMMMLOptimizer(
    ...     potential_name="mace-off23-medium",
    ...     model_path="models/my_finetuned_mace.model",
    ... )
    """

    def __init__(
        self,
        potential_name: str = "aceff-2.0",
        model_path: str | Path | None = None,
        tolerance: float = 10.0,
        max_iterations: int = 0,
    ) -> None:
        self._potential_name = potential_name
        self._model_path = str(model_path) if model_path is not None else None
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

        create_kwargs: dict = {}
        if self._model_path is not None:
            kwarg_name = _MODEL_PATH_KWARG.get(self._potential_name)
            if kwarg_name is not None:
                create_kwargs[kwarg_name] = self._model_path

        system = self._potential.createSystem(
            off_topology.to_openmm(), **create_kwargs
        )

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
