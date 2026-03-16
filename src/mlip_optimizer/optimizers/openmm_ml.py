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
import torch
from openff.toolkit import Molecule
from openff.units import unit
from openmm import unit as omm_unit
from openmm.app import Simulation
from openmmml import MLPotential

# Mapping from potential name to the generic ``MLPotential`` name that
# accepts a ``modelPath`` keyword for loading a user-supplied checkpoint.
# When ``model_path`` is provided, the potential name is replaced with this
# generic name so that ``MLPotential`` loads the local file instead of
# downloading a built-in model.
_GENERIC_NAME_FOR_LOCAL_MODEL: dict[str, str] = {
    # MACE family  →  generic name 'mace'
    "mace": "mace",
    "mace-mpa-0-medium": "mace",
    "mace-off23-large": "mace",
    "mace-off23-medium": "mace",
    "mace-off23-small": "mace",
    "mace-off24-medium": "mace",
    "mace-omat-0-medium": "mace",
    "mace-omat-0-small": "mace",
    "mace-omol-0-extra-large": "mace",
    # AceFF / TorchMDNet  →  generic name 'torchmdnet'
    "aceff-1.0": "torchmdnet",
    "aceff-1.1": "torchmdnet",
    "aceff-2.0": "torchmdnet",
    "torchmdnet": "torchmdnet",
    # FeNNIx  →  generic name 'fennix'
    "fennix": "fennix",
    "fennix-bio1-medium": "fennix",
    "fennix-bio1-medium-finetune-ions": "fennix",
    "fennix-bio1-small": "fennix",
    "fennix-bio1-small-finetune-ions": "fennix",
}


class OpenMMMLOptimizer:
    """Geometry optimizer using an OpenMM-ML machine learning potential.

    Parameters
    ----------
    potential_name : str
        Name of the ML potential recognized by ``openmmml.MLPotential``,
        e.g. ``"aceff-2.0"``, ``"ani2x"``, ``"mace-off23-medium"``.
    model_path : str or Path or None, optional
        Path to a custom model checkpoint file.  When set, the potential
        name is remapped to the generic loader name (``'mace'``,
        ``'torchmdnet'``, or ``'fennix'``) so that ``MLPotential`` loads
        the local file via its ``modelPath`` argument instead of
        downloading a built-in model.  See
        :data:`_GENERIC_NAME_FOR_LOCAL_MODEL` for supported potentials.
        When ``None`` (default) the built-in model is used.
    device : str or None, optional
        Torch device for the ML model: ``"cpu"``, ``"cuda"``, etc.
        When ``None`` (default), automatically selects ``"cuda"`` if
        available, otherwise ``"cpu"``.
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
        device: str | None = None,
        tolerance: float = 10.0,
        max_iterations: int = 0,
    ) -> None:
        self._potential_name = potential_name
        self._model_path = str(model_path) if model_path is not None else None
        self._device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self._tolerance = tolerance
        self._max_iterations = max_iterations

        # When a local model file is provided, swap to the generic loader
        # name so MLPotential routes through the local-file code path.
        init_kwargs: dict = {}
        if self._model_path is not None:
            generic = _GENERIC_NAME_FOR_LOCAL_MODEL.get(potential_name)
            if generic is not None:
                potential_name = generic
                init_kwargs["modelPath"] = self._model_path
        self._potential = MLPotential(potential_name, **init_kwargs)

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

        system = self._potential.createSystem(
            off_topology.to_openmm(),
            device=torch.device(self._device),
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
