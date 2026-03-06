"""ORB (Orbital Materials) force field optimizer via ASE.

Uses the ``orb_models`` package to load a pretrained ORB v3 conservative
force field and optimize molecular geometries through ASE's BFGS.

Requires the ``orb`` optional dependency group::

    pip install mlip-optimizer[orb]
"""

from __future__ import annotations

from ase.calculators.calculator import Calculator

from mlip_optimizer.optimizers.ase_base import ASEOptimizer


class ORBOptimizer(ASEOptimizer):
    """Geometry optimizer using the ORB v3 conservative force field.

    Parameters
    ----------
    precision : str, optional
        Numeric precision: ``"float32-highest"`` or ``"float64"``.
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
    >>> opt = ORBOptimizer(precision="float64", device="cpu")
    >>> result = opt.optimize(mol)
    """

    def __init__(
        self,
        precision: str = "float64",
        fmax: float = 0.05,
        device: str = "cpu",
        rattle: float = 0.1,
    ) -> None:
        super().__init__(fmax=fmax, device=device, rattle=rattle)
        self._precision = precision

    @property
    def name(self) -> str:
        """Optimizer name."""
        return "orb-v3-conservative"

    def _create_calculator(self) -> Calculator:
        """Load the ORB v3 conservative model and return an ORBCalculator."""
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        orbff = pretrained.orb_v3_conservative_inf_omat(
            device=self._device,
            precision=self._precision,
        )
        return ORBCalculator(orbff, device=self._device)
