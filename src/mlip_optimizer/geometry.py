"""Molecular geometry analysis and manipulation.

Computes bond lengths, bond angles, and proper torsion (dihedral) angles
from the 3D coordinates of a molecule's conformers.  Also provides
utilities for computing and setting dihedral angles on raw coordinate
arrays, useful for torsion-scan workflows.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from openff.toolkit import Molecule
from openff.units import unit

from mlip_optimizer._types import BondAngles, BondLengths, TorsionAngles


@dataclass(frozen=True)
class ConformerGeometry:
    """Geometry data for a single conformer.

    Attributes
    ----------
    bond_lengths : dict[tuple[int, int], float]
        Bond lengths in Angstroms, keyed by ``(atom_idx1, atom_idx2)``.
    bond_angles : dict[tuple[int, int, int], float]
        Bond angles in degrees, keyed by ``(atom_i, center_j, atom_k)``
        where ``j`` is the central atom.
    torsion_angles : dict[tuple[int, int, int, int], float]
        Proper torsion (dihedral) angles in degrees, keyed by
        ``(i, j, k, l)`` atom indices.
    """

    bond_lengths: BondLengths
    bond_angles: BondAngles
    torsion_angles: TorsionAngles


def get_conformer_geometry(off_mol: Molecule, conf_idx: int) -> ConformerGeometry:
    """Compute bond lengths, angles, and torsions for one conformer.

    Parameters
    ----------
    off_mol : openff.toolkit.Molecule
        Molecule with at least ``conf_idx + 1`` conformers.
    conf_idx : int
        Zero-based index of the conformer to analyze.

    Returns
    -------
    ConformerGeometry
        A frozen dataclass containing ``bond_lengths`` (Angstroms),
        ``bond_angles`` (degrees), and ``torsion_angles`` (degrees).

    Raises
    ------
    IndexError
        If ``conf_idx`` is out of range for the molecule's conformers.

    Examples
    --------
    >>> from openff.toolkit import Molecule
    >>> mol = Molecule.from_smiles("CCO")
    >>> mol.generate_conformers(n_conformers=1)
    >>> geom = get_conformer_geometry(mol, 0)
    >>> len(geom.bond_lengths) == mol.n_bonds
    True
    """
    coords = off_mol.conformers[conf_idx].m_as(unit.angstrom)

    # Bond lengths
    bond_lengths: BondLengths = {}
    for bond in off_mol.bonds:
        idx1, idx2 = bond.atom1_index, bond.atom2_index
        dist = float(np.linalg.norm(coords[idx1] - coords[idx2]))
        bond_lengths[(idx1, idx2)] = dist

    # Bond angles
    bond_angles: BondAngles = {}
    for angle in off_mol.angles:
        i, j, k = [atom.molecule_atom_index for atom in angle]
        v1 = coords[i] - coords[j]
        v2 = coords[k] - coords[j]
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle_deg = float(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))))
        bond_angles[(i, j, k)] = angle_deg

    # Proper torsion (dihedral) angles
    torsion_angles: TorsionAngles = {}
    for torsion in off_mol.propers:
        i, j, k, m = [atom.molecule_atom_index for atom in torsion]
        b1 = coords[j] - coords[i]
        b2 = coords[k] - coords[j]
        b3 = coords[m] - coords[k]

        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)

        # Guard against degenerate (linear) geometries
        if n1_norm < 1e-10 or n2_norm < 1e-10:
            torsion_angles[(i, j, k, m)] = 0.0
            continue

        n1 = n1 / n1_norm
        n2 = n2 / n2_norm

        p1 = np.cross(n1, b2 / np.linalg.norm(b2))
        x = np.dot(n1, n2)
        y = np.dot(p1, n2)
        torsion_deg = float(np.degrees(np.arctan2(y, x)))
        torsion_angles[(i, j, k, m)] = torsion_deg

    return ConformerGeometry(
        bond_lengths=bond_lengths,
        bond_angles=bond_angles,
        torsion_angles=torsion_angles,
    )


def compute_rmsd(
    mol_a: Molecule,
    conf_idx_a: int,
    mol_b: Molecule,
    conf_idx_b: int,
    *,
    heavy_atoms_only: bool = True,
) -> float:
    """Compute RMSD between two conformer coordinate sets.

    No alignment or superposition is applied -- coordinates are compared
    directly.  This is appropriate when both molecules share the same
    atom ordering (e.g. from CMILES mapping in QCArchive data).

    Parameters
    ----------
    mol_a : Molecule
        First molecule.
    conf_idx_a : int
        Conformer index in *mol_a*.
    mol_b : Molecule
        Second molecule (must have the same number of atoms).
    conf_idx_b : int
        Conformer index in *mol_b*.
    heavy_atoms_only : bool, optional
        If ``True`` (default), exclude hydrogen atoms from the RMSD
        calculation.

    Returns
    -------
    float
        RMSD in Angstroms.

    Raises
    ------
    ValueError
        If the two molecules have different atom counts.
    """
    coords_a = mol_a.conformers[conf_idx_a].m_as(unit.angstrom)
    coords_b = mol_b.conformers[conf_idx_b].m_as(unit.angstrom)

    if coords_a.shape != coords_b.shape:
        raise ValueError(
            f"Atom count mismatch: {coords_a.shape[0]} vs {coords_b.shape[0]}"
        )

    if heavy_atoms_only:
        heavy_mask = np.array(
            [atom.atomic_number != 1 for atom in mol_a.atoms]
        )
        coords_a = coords_a[heavy_mask]
        coords_b = coords_b[heavy_mask]

    diff = coords_a - coords_b
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))


def compute_geometry_diffs(
    geom_opt: ConformerGeometry,
    geom_ref: ConformerGeometry,
) -> tuple[BondLengths, BondAngles, TorsionAngles]:
    """Compute per-parameter absolute differences between two geometries.

    Parameters
    ----------
    geom_opt : ConformerGeometry
        Optimized (test) geometry.
    geom_ref : ConformerGeometry
        Reference (QM) geometry.

    Returns
    -------
    tuple[dict, dict, dict]
        ``(bond_diffs, angle_diffs, torsion_diffs)`` -- each mapping the
        parameter key to the absolute difference.  Torsion differences
        are normalized to ``[-180, 180]`` before taking the absolute
        value.
    """
    bond_diffs: BondLengths = {}
    for key in geom_ref.bond_lengths:
        if key in geom_opt.bond_lengths:
            bond_diffs[key] = abs(
                geom_opt.bond_lengths[key] - geom_ref.bond_lengths[key]
            )

    angle_diffs: BondAngles = {}
    for key in geom_ref.bond_angles:
        if key in geom_opt.bond_angles:
            angle_diffs[key] = abs(
                geom_opt.bond_angles[key] - geom_ref.bond_angles[key]
            )

    torsion_diffs: TorsionAngles = {}
    for key in geom_ref.torsion_angles:
        if key in geom_opt.torsion_angles:
            raw = geom_opt.torsion_angles[key] - geom_ref.torsion_angles[key]
            normalized = (raw + 180) % 360 - 180
            torsion_diffs[key] = abs(normalized)

    return bond_diffs, angle_diffs, torsion_diffs


# ---------------------------------------------------------------------------
# Dihedral manipulation utilities
# ---------------------------------------------------------------------------


def compute_dihedral(
    positions: np.ndarray,
    indices: tuple[int, int, int, int],
) -> float:
    """Compute the dihedral angle for four atom indices.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 3)
        Cartesian coordinates in Angstroms.
    indices : tuple of four ints
        Atom indices ``(i, j, k, l)`` defining the dihedral.

    Returns
    -------
    float
        Dihedral angle in degrees, in the range ``(-180, 180]``.
    """
    i, j, k, l = indices
    b1 = positions[j] - positions[i]
    b2 = positions[k] - positions[j]
    b3 = positions[l] - positions[k]
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    return float(np.degrees(np.arctan2(np.dot(m1, n2), np.dot(n1, n2))))


def _rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues' rotation matrix around *axis* by *angle_rad*."""
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * K @ K


def set_dihedral(
    positions: np.ndarray,
    indices: tuple[int, int, int, int],
    target_deg: float,
    mol: Molecule,
) -> np.ndarray:
    """Return a copy of *positions* with the dihedral set to *target_deg*.

    Atoms on the far side of the central bond (the *k*-side) are rotated
    to achieve the target angle.  The molecular connectivity in *mol* is
    used to determine which atoms to rotate via BFS from atom *k*
    (excluding atom *j*).

    Parameters
    ----------
    positions : np.ndarray, shape (N, 3)
        Cartesian coordinates in Angstroms.
    indices : tuple of four ints
        Atom indices ``(i, j, k, l)`` defining the dihedral.
    target_deg : float
        Desired dihedral angle in degrees.
    mol : openff.toolkit.Molecule
        Molecule whose bond connectivity is used to identify the
        rotating fragment.

    Returns
    -------
    np.ndarray, shape (N, 3)
        New coordinates with the dihedral set to *target_deg*.
    """
    current = compute_dihedral(positions, indices)
    delta = np.radians(target_deg - current)

    i, j, k, l = indices
    R = _rotation_matrix(positions[k] - positions[j], delta)

    # BFS from k, excluding j, to find atoms on the k-side of the bond
    bonds: dict[int, set[int]] = {a.molecule_atom_index: set() for a in mol.atoms}
    for bond in mol.bonds:
        bonds[bond.atom1_index].add(bond.atom2_index)
        bonds[bond.atom2_index].add(bond.atom1_index)

    visited: set[int] = set()
    queue: deque[int] = deque([k])
    visited.add(k)
    while queue:
        node = queue.popleft()
        for nb in bonds[node]:
            if nb not in visited and nb != j:
                visited.add(nb)
                queue.append(nb)

    new_pos = positions.copy()
    origin = positions[j]
    for idx in visited:
        new_pos[idx] = origin + R @ (positions[idx] - origin)
    return new_pos
