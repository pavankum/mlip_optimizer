"""Torsion (dihedral) scan using yammbs constrained minimization.

Provides :func:`run_torsion_scan` which drives a constrained torsion
scan over an angle grid.  Coordinates are pre-rotated to the target
angle before each minimization so the optimizer starts near the
desired dihedral.

Requirements
------------
Install the torsion extras::

    pip install mlip-optimizer[torsion]

This pulls in `yammbs <https://github.com/openforcefield/yammbs>`_.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from openff.toolkit import Molecule

from mlip_optimizer.geometry import set_dihedral

logger = logging.getLogger(__name__)


@dataclass
class TorsionScanResult:
    """Result of a torsion scan at a series of angles.

    Attributes
    ----------
    angles : np.ndarray
        The angle grid in degrees.
    energies : np.ndarray
        Energies at each grid point (kcal/mol).  ``nan`` for failed
        minimizations.
    coordinates : list[np.ndarray | None]
        Minimized coordinates at each grid point, or ``None`` on
        failure.
    method : str
        The minimization method used.
    force_field : str
        The force-field / potential used.
    """

    angles: np.ndarray
    energies: np.ndarray
    coordinates: list[np.ndarray | None]
    method: str
    force_field: str


def run_torsion_scan(
    mapped_smiles: str,
    dihedral_indices: tuple[int, int, int, int],
    coordinates: np.ndarray,
    mol: Molecule,
    angle_grid: np.ndarray,
    force_field: str,
    method: str = "openmm_torsion_restrained",
    restraint_k: float = 1.0,
) -> TorsionScanResult:
    """Run a constrained torsion scan at each angle in *angle_grid*.

    At each grid point the input geometry is pre-rotated to the
    target dihedral (via :func:`~mlip_optimizer.geometry.set_dihedral`)
    and then minimized with the requested *method* using yammbs.

    Parameters
    ----------
    mapped_smiles : str
        Mapped SMILES string for the molecule.
    dihedral_indices : tuple of four ints
        Atom indices ``(i, j, k, l)`` defining the torsion to scan.
    coordinates : np.ndarray, shape (N, 3)
        Starting coordinates in Angstroms.
    mol : openff.toolkit.Molecule
        Molecule used for fragment detection when rotating the dihedral.
    angle_grid : np.ndarray
        1-D array of dihedral angles in degrees (e.g. ``np.arange(-180, 180, 15)``).
    force_field : str
        OpenFF force-field name (e.g. ``"openff-2.1.0"``).
    method : str, optional
        Minimization method passed to yammbs, one of
        ``"openmm_torsion_restrained"`` or ``"openmm_torsion_atoms_frozen"``.
    restraint_k : float, optional
        Restraint force constant (kcal/mol/rad^2).  Default ``1.0``.

    Returns
    -------
    TorsionScanResult
        Dataclass with ``angles``, ``energies``, ``coordinates``,
        ``method``, and ``force_field`` fields.
    """
    from yammbs.torsion._minimize import (
        ConstrainedMinimizationInput,
        _run_minimization_constrained,
    )

    energies: list[float] = []
    coords_out: list[np.ndarray | None] = []

    for angle in angle_grid:
        rotated = set_dihedral(coordinates, dihedral_indices, float(angle), mol)

        inp = ConstrainedMinimizationInput(
            torsion_id=1,
            mapped_smiles=mapped_smiles,
            dihedral_indices=dihedral_indices,
            force_field=force_field,
            coordinates=rotated,
            grid_id=float(angle),
            method=method,
            restraint_k=float(restraint_k),
        )
        result = _run_minimization_constrained(inp)
        if result is None:
            logger.warning("Minimization failed at angle %.1f°", angle)
            energies.append(float("nan"))
            coords_out.append(None)
        else:
            energies.append(result.energy)
            coords_out.append(result.coordinates)

    return TorsionScanResult(
        angles=np.asarray(angle_grid, dtype=float),
        energies=np.array(energies),
        coordinates=coords_out,
        method=method,
        force_field=force_field,
    )
