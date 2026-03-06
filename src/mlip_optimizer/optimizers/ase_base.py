"""Base class for ASE-calculator-based geometry optimizers.

Provides the shared logic for converting between OpenFF Molecules and ASE
Atoms, running BFGS optimization, and managing the PyTorch default dtype
that some ML models modify as a side effect.

Subclasses need only implement :meth:`_create_calculator` and the
:attr:`name` property.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize import BFGS
from openff.toolkit import Molecule
from openff.units import unit
from openff.units.elements import SYMBOLS


class ASEOptimizer(ABC):
    """Abstract base for optimizers backed by an ASE calculator + BFGS.

    Parameters
    ----------
    fmax : float, optional
        Maximum force convergence threshold in eV/Angstrom.
        Default is ``0.05``.
    device : str, optional
        Compute device: ``"cpu"`` or ``"cuda"``.  Default is ``"cpu"``.
    rattle : float, optional
        Random perturbation magnitude (Angstroms) applied to atom positions
        before optimization.  Helps escape shallow local minima near the
        starting geometry.  Set to ``0.0`` to disable.  Default is ``0.1``.
    logfile : str or None, optional
        Path for BFGS optimizer log output.  ``None`` (default) suppresses
        log output.

    Examples
    --------
    Subclass and implement ``_create_calculator`` and ``name``::

        class MyOptimizer(ASEOptimizer):
            @property
            def name(self) -> str:
                return "my-model"

            def _create_calculator(self) -> Calculator:
                from my_package import MyCalc
                return MyCalc(device=self._device)
    """

    def __init__(
        self,
        fmax: float = 0.05,
        device: str = "cpu",
        rattle: float = 0.1,
        logfile: str | None = None,
    ) -> None:
        self._fmax = fmax
        self._device = device
        self._rattle = rattle
        self._logfile = logfile

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this optimizer."""
        ...

    @abstractmethod
    def _create_calculator(self) -> Calculator:
        """Return an ASE Calculator instance for energy/force evaluation.

        Called once before iterating over conformers.  Implementations may
        cache or re-create the calculator as needed.
        """
        ...

    @staticmethod
    def _offmol_to_ase_atoms(molecule: Molecule, conf_idx: int) -> Atoms:
        """Convert one conformer of an OpenFF Molecule to an ASE Atoms object.

        Parameters
        ----------
        molecule : openff.toolkit.Molecule
            Source molecule.
        conf_idx : int
            Index of the conformer whose coordinates to use.

        Returns
        -------
        ase.Atoms
            ASE Atoms with positions (Angstroms), element symbols, and
            formal charges.
        """
        positions = molecule.conformers[conf_idx].m  # Angstroms as numpy
        symbols = [SYMBOLS[atom.atomic_number] for atom in molecule.atoms]
        formal_charges = [
            atom.formal_charge.m_as(unit.elementary_charge)
            for atom in molecule.atoms
        ]
        return Atoms(symbols=symbols, positions=positions, charges=formal_charges)

    def optimize(self, molecule: Molecule) -> Molecule:
        """Optimize all conformers using the ASE calculator and BFGS.

        Saves and restores the PyTorch default dtype to avoid side effects
        from model loading.

        Parameters
        ----------
        molecule : openff.toolkit.Molecule
            Input molecule with at least one conformer.

        Returns
        -------
        openff.toolkit.Molecule
            New molecule with optimized conformer geometries.
        """
        original_dtype = torch.get_default_dtype()
        try:
            calc = self._create_calculator()
            result = Molecule(molecule)
            original_conformers = list(result.conformers)
            result.clear_conformers()

            for conf_idx in range(len(original_conformers)):
                atoms = self._offmol_to_ase_atoms(molecule, conf_idx)
                atoms.calc = calc

                if self._rattle > 0:
                    atoms.rattle(self._rattle)

                dyn = BFGS(atoms, logfile=self._logfile)
                dyn.run(fmax=self._fmax)

                optimized_coords = atoms.get_positions() * unit.angstrom
                result.add_conformer(optimized_coords)

            return result
        finally:
            torch.set_default_dtype(original_dtype)
