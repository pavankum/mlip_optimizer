"""OpenMM-ML machine learning potential optimizer.

Wraps the ``openmmml.MLPotential`` API to optimize molecular geometries
using any of the supported ML interatomic potentials:

    aceff-1.0, aceff-1.1, aceff-2.0, aimnet2, ani1ccx, ani2x, deepmd,
    mace, mace-mpa-0-medium, mace-off23-large, mace-off23-medium,
    mace-off23-small, mace-off24-medium, mace-omat-0-medium,
    mace-omat-0-small, nequip, torchmdnet
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import openmm
import torch
from openff.toolkit import Molecule
from openff.units import unit
from openmm import unit as omm_unit
from openmm.app import Simulation
from openmmml import MLPotential
from scipy.optimize import minimize as scipy_minimize

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

        # Resolve the generic loader name and kwargs once so that
        # _create_potential() can recreate the MLPotential cheaply.
        self._ml_potential_name = potential_name
        self._ml_potential_kwargs: dict = {}
        if self._model_path is not None:
            generic = _GENERIC_NAME_FOR_LOCAL_MODEL.get(potential_name)
            if generic is not None:
                self._ml_potential_name = generic
                self._ml_potential_kwargs["modelPath"] = self._model_path

    @staticmethod
    def _system_uses_python_force(system: openmm.System) -> bool:
        """Return True if *system* contains an ``openmm.PythonForce``."""
        for i in range(system.getNumForces()):
            if type(system.getForce(i)).__name__ == "PythonForce":
                return True
        return False

    @staticmethod
    def _scipy_minimize(
        context: openmm.Context,
        tolerance: float,
        max_iterations: int,
    ) -> None:
        """Minimize energy using scipy L-BFGS-B via the OpenMM Context.

        ``LocalEnergyMinimizer`` (C++) cannot propagate Python exceptions
        raised inside a ``PythonForce`` callback — it emits an empty
        ``OpenMMException`` and corrupts the context.  This method
        avoids the C++ minimizer entirely: it reads energy/forces with
        ``getState()`` and writes positions with ``setPositions()``,
        letting scipy drive L-BFGS-B in Python space.
        """
        n_atoms = context.getSystem().getNumParticles()
        n_dof = n_atoms * 3

        # Units: positions in nm, energy in kJ/mol, forces in kJ/mol/nm.
        def _objective(x: np.ndarray):
            positions = x.reshape(n_atoms, 3)
            context.setPositions(positions)
            state = context.getState(getEnergy=True, getForces=True)
            energy = state.getPotentialEnergy().value_in_unit(
                omm_unit.kilojoule_per_mole,
            )
            forces = state.getForces(asNumpy=True).value_in_unit(
                omm_unit.kilojoule_per_mole / omm_unit.nanometer,
            )
            grad = -forces.flatten()  # gradient = -force
            return energy, grad

        state0 = context.getState(getPositions=True)
        x0 = state0.getPositions(asNumpy=True).value_in_unit(
            omm_unit.nanometer,
        ).flatten()

        # tolerance is in kJ/mol/nm (force units); scipy's ftol/gtol
        # are dimensionless but gtol maps to the max gradient component.
        opts: dict = {"maxiter": max_iterations if max_iterations > 0 else 15000}
        opts["gtol"] = tolerance

        result = scipy_minimize(
            _objective, x0, method="L-BFGS-B", jac=True, options=opts,
        )

        # Write final positions back into the context.
        context.setPositions(result.x.reshape(n_atoms, 3))

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

        potential = MLPotential(self._ml_potential_name, **self._ml_potential_kwargs)
        system = potential.createSystem(off_topology.to_openmm())

        original_conformers = list(result.conformers)
        result.clear_conformers()

        # MACE TorchScript models conflict with the OpenMM CUDA
        # platform during iterative minimization (repeated TorchScript
        # calls), causing "invalid resource handle" errors.  Use
        # OpenCL for MACE — the TorchForce still evaluates the model
        # on GPU via PyTorch.  Other TorchScript models (AceFF/
        # TorchMDNet) work fine on CUDA and OpenCL is ~2x slower for
        # them.
        if self._potential_name.startswith("mace") and self._device.startswith("cuda"):
            platform = openmm.Platform.getPlatformByName("OpenCL")
        elif self._device.startswith("cuda"):
            platform = openmm.Platform.getPlatformByName("CUDA")
        else:
            platform = openmm.Platform.getPlatformByName("CPU")

        # Create a single Simulation (and Context) and reuse it for all
        # conformers.  Creating a new Context per conformer leaks GPU
        # resources and causes "invalid resource handle" errors.
        integrator = openmm.LangevinIntegrator(
            300 * omm_unit.kelvin,
            1.0 / omm_unit.picoseconds,
            1.0 * omm_unit.femtosecond,
        )
        simulation = Simulation(
            off_topology.to_openmm(), system, integrator, platform,
        )

        for conf_idx, conformer in enumerate(original_conformers):
            positions = conformer.m_as(unit.nanometer)
            simulation.context.setPositions(positions)

            try:
                simulation.minimizeEnergy(
                    tolerance=(
                        self._tolerance
                        * omm_unit.kilojoule_per_mole
                        / omm_unit.nanometer
                    ),
                    maxIterations=self._max_iterations,
                )
            except Exception as e:
                diag = (
                    f"  Conformer index: {conf_idx}\n"
                    f"  Potential:        {self._potential_name}\n"
                    f"  Platform:         {platform.getName()}\n"
                )
                try:
                    state = simulation.context.getState(
                        getPositions=True, getEnergy=True, getForces=True,
                    )
                    pos = state.getPositions(asNumpy=True)
                    energy = state.getPotentialEnergy()
                    forces = state.getForces(asNumpy=True)
                    has_nan_pos = bool(np.any(np.isnan(pos)))
                    max_force = float(np.max(np.linalg.norm(forces, axis=1)))
                    diag += (
                        f"\n  Potential energy: {energy}"
                        f"\n  NaN in positions: {has_nan_pos}"
                        f"\n  Max force norm:   {max_force}"
                    )
                except Exception:
                    diag += "\n  (context state unavailable — likely CUDA error)"
                raise type(e)(f"{e}\n{diag}") from e

            optimized_coords = (
                simulation.context.getState(getPositions=True)
                .getPositions(asNumpy=True)
            )
            result.add_conformer(optimized_coords * unit.nanometer)

        # Explicitly release GPU resources so that subsequent optimizers
        # (possibly using a different TorchScript model) do not collide
        # with stale CUDA handles.  Synchronize first to ensure all
        # pending CUDA kernels complete before tearing down the context.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        del simulation.context
        del simulation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result
