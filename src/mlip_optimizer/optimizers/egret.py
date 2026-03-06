"""EGRET / MACE-architecture optimizer via ASE.

Uses the ``mace`` package to load a MACE-architecture model (such as EGRET)
and optimize molecular geometries through ASE's BFGS.

Requires the ``egret`` optional dependency group::

    pip install mlip-optimizer[egret]
"""

from __future__ import annotations

from pathlib import Path

from ase.calculators.calculator import Calculator

from mlip_optimizer.optimizers.ase_base import ASEOptimizer


class EGRETOptimizer(ASEOptimizer):
    """Geometry optimizer using a MACE-architecture model (e.g. EGRET).

    Parameters
    ----------
    model_path : str
        Path to the compiled MACE/EGRET model file, or a ``mace_off``
        preset name (e.g. ``"medium"``).
    precision : str, optional
        Numeric precision: ``"float32"`` or ``"float64"``.
        Default is ``"float64"``.
    fmax : float, optional
        Maximum force convergence threshold in eV/Angstrom.
        Default is ``0.05``.
    device : str, optional
        Compute device: ``"cpu"`` or ``"cuda"``.  Default is ``"cpu"``.
    rattle : float, optional
        Pre-optimization perturbation magnitude in Angstroms.
        Default is ``0.1``.

    Examples
    --------
    >>> from openff.toolkit import Molecule
    >>> mol = Molecule.from_smiles("CCO")
    >>> mol.generate_conformers(n_conformers=1)
    >>> opt = EGRETOptimizer(model_path="/path/to/EGRET_1.model")
    >>> result = opt.optimize(mol)
    """

    def __init__(
        self,
        model_path: str,
        precision: str = "float64",
        fmax: float = 0.05,
        device: str = "cpu",
        rattle: float = 0.1,
    ) -> None:
        super().__init__(fmax=fmax, device=device, rattle=rattle)
        self._model_path = model_path
        self._precision = precision

    @property
    def name(self) -> str:
        """Optimizer name derived from model filename."""
        return f"egret:{Path(self._model_path).stem}"

    def _create_calculator(self) -> Calculator:
        """Load the MACE model and return an ASE calculator."""
        from mace.calculators import mace_off

        return mace_off(
            model=self._model_path,
            device=self._device,
            default_dtype=self._precision,
        )
