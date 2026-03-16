"""Torsion (dihedral) scan using yammbs constrained minimization.

Provides :func:`run_torsion_scan` which drives a constrained torsion
scan over one or more dihedral angle grids.  Coordinates are pre-rotated
to the target angles before each minimization so the optimizer starts
near the desired geometry.

For a single dihedral the result is a 1-D energy profile.  Pass
multiple dihedrals and angle grids to get an N-D scan (e.g. a 2-D
Ramachandran-style surface).

Requirements
------------
Install the torsion extras::

    pip install mlip-optimizer[torsion]

This pulls in `yammbs <https://github.com/openforcefield/yammbs>`_.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass

import numpy as np
from openff.toolkit import Molecule

from mlip_optimizer.geometry import set_dihedral

logger = logging.getLogger(__name__)


@dataclass
class TorsionScanResult:
    """Result of a (possibly multi-dimensional) torsion scan.

    Attributes
    ----------
    angles : list[np.ndarray]
        One angle grid per scanned dihedral (each 1-D, in degrees).
        For a single-dihedral scan this is a one-element list.
    energies : np.ndarray
        Energies at each grid point (kcal/mol).  Shape matches the
        Cartesian product of the angle grids, i.e.
        ``(len(angles[0]), len(angles[1]), ...)``.
        ``nan`` for failed minimizations.
    coordinates : np.ndarray
        Object array of the same shape as *energies*.  Each element is
        either an ``np.ndarray`` of minimized coordinates or ``None``
        on failure.
    method : str
        The minimization method used.
    force_field : str
        The force-field / potential used.
    """

    angles: list[np.ndarray]
    energies: np.ndarray
    coordinates: np.ndarray  # object array; entries are ndarray | None
    method: str
    force_field: str

    @property
    def ndim(self) -> int:
        """Number of scanned dihedrals."""
        return len(self.angles)


def run_torsion_scan(
    mapped_smiles: str,
    dihedral_indices: (
        tuple[int, int, int, int]
        | list[tuple[int, int, int, int]]
    ),
    coordinates: np.ndarray,
    mol: Molecule,
    angle_grid: np.ndarray | list[np.ndarray],
    force_field: str,
    method: str = "openmm_torsion_restrained",
    restraint_k: float = 1.0,
) -> TorsionScanResult:
    """Run a constrained torsion scan over one or more dihedrals.

    For a **1-D scan**, pass a single dihedral tuple and a single angle
    grid::

        run_torsion_scan(..., dihedral_indices=(0,1,2,3),
                         angle_grid=np.arange(-180, 180, 15), ...)

    For an **N-D scan**, pass lists of dihedrals and angle grids::

        run_torsion_scan(
            ...,
            dihedral_indices=[(0,1,2,3), (1,2,3,4)],
            angle_grid=[grid_a, grid_b],
            ...
        )

    At each grid point all dihedrals are pre-rotated to the target
    angles (via :func:`~mlip_optimizer.geometry.set_dihedral`) and then
    minimized with the requested *method* using yammbs.

    Parameters
    ----------
    mapped_smiles : str
        Mapped SMILES string for the molecule.
    dihedral_indices : tuple or list of tuples
        Atom indices ``(i, j, k, l)`` defining each torsion to scan.
        A single tuple is treated as a 1-D scan.
    coordinates : np.ndarray, shape (N, 3)
        Starting coordinates in Angstroms.
    mol : openff.toolkit.Molecule
        Molecule used for fragment detection when rotating the dihedral.
    angle_grid : np.ndarray or list of np.ndarray
        Angle grid(s) in degrees.  A single array is treated as a 1-D
        scan.  For N-D scans supply one array per dihedral.
    force_field : str
        OpenFF force-field name (e.g. ``"openff-2.1.0"``).
    method : str, optional
        Minimization method passed to yammbs, e.g.
        ``"openmm_torsion_restrained"`` or ``"openmm_torsion_atoms_frozen"``.
    restraint_k : float, optional
        Restraint force constant (kcal/mol/rad^2).  Default ``1.0``.

    Returns
    -------
    TorsionScanResult
        Dataclass with ``angles`` (list of grids), ``energies``
        (N-D array), ``coordinates`` (N-D object array), ``method``,
        and ``force_field``.
    """
    from yammbs.torsion._minimize import (
        ConstrainedMinimizationInput,
        _run_minimization_constrained,
    )

    # --- normalise inputs to lists ---
    if isinstance(dihedral_indices, tuple) and isinstance(dihedral_indices[0], int):
        dihedral_indices = [dihedral_indices]  # type: ignore[list-item]
    dihedral_list: list[tuple[int, int, int, int]] = list(dihedral_indices)  # type: ignore[arg-type]

    if isinstance(angle_grid, np.ndarray):
        angle_grids = [angle_grid]
    else:
        angle_grids = list(angle_grid)

    if len(dihedral_list) != len(angle_grids):
        raise ValueError(
            f"Number of dihedrals ({len(dihedral_list)}) must match "
            f"number of angle grids ({len(angle_grids)})"
        )

    # --- prepare output arrays ---
    grid_shape = tuple(len(g) for g in angle_grids)
    energies = np.full(grid_shape, np.nan)
    coords_out = np.empty(grid_shape, dtype=object)

    # --- iterate over the full Cartesian product ---
    grid_iters = [enumerate(g) for g in angle_grids]
    total = int(np.prod(grid_shape))
    done = 0

    for combo in itertools.product(*grid_iters):
        done += 1
        indices = tuple(c[0] for c in combo)
        target_angles = tuple(float(c[1]) for c in combo)

        angle_str = ", ".join(f"{a:.1f}°" for a in target_angles)
        logger.info("  [%d/%d] %s", done, total, angle_str)

        # Pre-rotate to each target dihedral sequentially
        rotated = coordinates
        for dih, ang in zip(dihedral_list, target_angles):
            rotated = set_dihedral(rotated, dih, ang, mol)

        inp = ConstrainedMinimizationInput(
            torsion_id=1,
            mapped_smiles=mapped_smiles,
            dihedral_indices=dihedral_list[0],
            force_field=force_field,
            coordinates=rotated,
            grid_id=target_angles[0],
            method=method,
            restraint_k=float(restraint_k),
        )
        result = _run_minimization_constrained(inp)
        if result is None:
            logger.warning("Minimization failed at (%s)", angle_str)
            coords_out[indices] = None
        else:
            energies[indices] = result.energy
            coords_out[indices] = result.coordinates

    return TorsionScanResult(
        angles=[np.asarray(g, dtype=float) for g in angle_grids],
        energies=energies,
        coordinates=coords_out,
        method=method,
        force_field=force_field,
    )
