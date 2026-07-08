"""Four-axis local-chemistry featurizer for valence parameter instances.

Uses the RDKit molecule carried in each ``InstanceRecord`` directly — no
SMILES re-parsing, no canonical-index guessing.  Because ``rdmol`` is
derived from ``Molecule.from_mapped_smiles(cmiles).to_rdkit()``, the atom
at index *j* in ``rdmol`` is the same physical atom as index *j* in the
QM geometry and in the OpenFF FF labeling output.

The four PRIMARY axes, in priority order (matching the methodology; these are
graph depth d=0 per ``reference/besmarts_concepts.md`` — only the primary
mapped atoms :1/:2/:3 themselves, never their neighbors):

1. **Bond order** on each side of the central atom (single/aromatic/double).
2. **Flanking-atom elements** (atomic symbols of atoms 1 and 3).
3. **Ring membership + ring size** of the central atom.
4. **Formal charge** of the central atom.

``depth > 0`` (d=1: one-bond neighbors of the primary atoms; d=2: two-bond)
adds a fifth, "attachment context" axis — what the flank atoms are themselves
bonded to, and what the center's non-angle substituents are — via
:func:`_bfs_shell_elements`. This is Axis 5 / the "second-shell sweep" in
``reference/methodology.md``, previously done by hand; ``depth`` makes it a
real, swept parameter instead of prose.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import rdchem


# ---------------------------------------------------------------------------
# Bond-order label helpers
# ---------------------------------------------------------------------------

def _bo_char(bond: rdchem.Bond) -> str:
    bt = bond.GetBondTypeAsDouble()
    if bt == 1.0:
        return "s"   # single
    if bt == 1.5:
        return "a"   # aromatic
    if bt == 2.0:
        return "d"   # double
    if bt == 3.0:
        return "t"   # triple
    return f"bo{bt}"


def _bo_smarts_char(bond: rdchem.Bond) -> str:
    """SMARTS bond symbol for explicit bond-order discrimination."""
    bt = bond.GetBondTypeAsDouble()
    if bt == 1.0:
        return "-"
    if bt == 1.5:
        return ":"
    if bt == 2.0:
        return "="
    if bt == 3.0:
        return "#"
    return "~"


# ---------------------------------------------------------------------------
# FeatureSet
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureSet:
    """Local-chemistry fingerprint for a single parameter instance.

    All fields are derived from the RDKit molecule using native atom indices
    (no SMILES re-parsing).

    Attributes
    ----------
    bond_orders : tuple[float, ...]
        Sorted bond-order values on each side of the central atom
        (1.0 = single, 1.5 = aromatic, 2.0 = double).
        For a bond parameter this has two entries (the atom's other bonds
        are not included).
    bo_chars : tuple[str, ...]
        Human-readable bond-order labels (``'s'``, ``'a'``, ``'d'``, ``'t'``),
        sorted to match *bond_orders*.
    flank_elements : tuple[str, ...]
        Sorted atomic symbols of the flanking atoms (atoms :1 and :3 for
        angles; atoms :1 and :2 for bonds).
    center_element : str
        Atomic symbol of the central atom (:2 for angles, relevant side
        for bonds).
    center_ring_sizes : tuple[int, ...]
        SSSR ring sizes the central atom belongs to, sorted ascending.
        Empty tuple if the atom is acyclic.
    center_in_ring : bool
        Shortcut: ``bool(center_ring_sizes)``.
    center_aromatic : bool
        Whether the central atom is flagged aromatic by RDKit.
    center_formal_charge : int
        Formal charge on the central atom.
    feature_key : str
        Compact string summary in priority-axis order, usable as a dict key
        and as a human-readable cluster label.
        Format: ``bo_label|elem_label|ring_label|fc_label``
        Example: ``'s+d|C/N|r6|fc0'``. When ``depth > 0`` a ``|d{depth}:{sig}``
        suffix is appended (see *shell_elements*/*center_substituent_elements*),
        so d=0 grouping is unaffected by depth-aware callers.
    depth : int
        Graph depth this FeatureSet was computed at (0 = primary atoms only).
    shell_elements : tuple[str, ...]
        Only populated when ``depth > 0``. Sorted, deduped element symbols
        found by BFS from each flank atom out to *depth* bonds, excluding the
        primary triple's own atoms — "what are the flank atoms bonded to"
        (methodology.md Axis 5, BESMARTS d=1/d=2).
    center_substituent_elements : tuple[str, ...]
        Only populated when ``depth > 0``. Same BFS, started from the
        center atom's neighbors that are NOT part of the angle/bond itself —
        "the center's non-angle substituents" (methodology.md Axis 5).
    """

    bond_orders: tuple
    bo_chars: tuple
    flank_elements: tuple
    center_element: str
    center_ring_sizes: tuple
    center_in_ring: bool
    center_aromatic: bool
    center_formal_charge: int
    feature_key: str

    # SMARTS-builder helpers (populated by featurize_instance)
    bo_smarts: tuple = ()         # SMARTS bond chars, same order as bond_orders (unsorted)
    center_valence: int = 0       # X-count (total connections)
    flank_in_ring: tuple = ()     # bool per flank atom

    # Depth-aware attachment-context axis (populated only when depth > 0)
    depth: int = 0
    shell_elements: tuple = ()
    center_substituent_elements: tuple = ()


def _make_feature_key(
    bo_chars_sorted: tuple,
    flank_elements_sorted: tuple,
    center_ring_sizes: tuple,
    center_formal_charge: int,
) -> str:
    bo_part = "+".join(bo_chars_sorted)
    elem_part = "/".join(flank_elements_sorted)
    if center_ring_sizes:
        min_r = min(center_ring_sizes)
        ring_part = f"r{min_r}_strained" if min_r <= 4 else f"r{min_r}"
    else:
        ring_part = "acyclic"
    fc_part = f"fc{center_formal_charge:+d}"
    return f"{bo_part}|{elem_part}|{ring_part}|{fc_part}"


def _bfs_shell_elements(
    rdmol: Chem.Mol,
    start_idxs: tuple,
    exclude_idxs: set,
    depth: int,
) -> tuple[str, ...]:
    """BFS out to *depth* bonds from each atom in *start_idxs*, collecting the
    sorted/deduped element symbols of every atom REACHED (not the start atoms
    themselves). *exclude_idxs* (the primary :1/:2/:3 atoms) are never
    traversed into or reported, so this only sees genuinely "outside" atoms —
    the BESMARTS d=1/d=2 attachment-context axis (methodology.md Axis 5).

    Two call shapes cover both Axis-5 questions:
    - ``start_idxs=flank_idxs`` → "what are the flank atoms bonded to?"
    - ``start_idxs=(center_idx,)`` → "what are the center's non-angle
      substituents?" (``exclude_idxs`` already excludes the flank atoms, so
      depth=1 from the center lands exactly on its non-angle neighbors).
    """
    if depth <= 0:
        return ()
    seen_elements: set = set()
    visited = set(exclude_idxs) | set(start_idxs)
    frontier = list(start_idxs)
    for _ in range(depth):
        next_frontier = []
        for idx in frontier:
            for nbr in rdmol.GetAtomWithIdx(idx).GetNeighbors():
                nidx = nbr.GetIdx()
                if nidx in visited:
                    continue
                visited.add(nidx)
                seen_elements.add(nbr.GetSymbol())
                next_frontier.append(nidx)
        frontier = next_frontier
        if not frontier:
            break
    return tuple(sorted(seen_elements))


def featurize_instance(
    rdmol: Chem.Mol,
    atom_key: tuple,
    param_type: str = "angle",
    depth: int = 0,
) -> FeatureSet | None:
    """Compute the local-chemistry feature fingerprint for one parameter instance.

    Parameters
    ----------
    rdmol : rdkit.Chem.Mol
        RDKit molecule with the atom ordering from ``to_rdkit()``.
        Must not require any re-canonicalization.
    atom_key : tuple[int, ...]
        Atom indices in *rdmol*: ``(i, j)`` for bonds or ``(i, j, k)`` for
        angles or ``(i, j, k, m)`` for torsions.
    param_type : str
        ``'angle'``, ``'bond'``, or ``'torsion'``.
    depth : int
        Graph depth (BESMARTS d). ``0`` (default) computes only the four
        primary axes — identical output to before this parameter existed.
        ``> 0`` additionally computes ``shell_elements`` /
        ``center_substituent_elements`` (Axis 5) and appends a depth-tagged
        suffix to ``feature_key``, so d=0 and d>0 groupings never collide.

    Returns
    -------
    FeatureSet or None
        ``None`` if the atom key references atoms not present in *rdmol*
        or if the bonds cannot be found.
    """
    if rdmol is None:
        return None

    n_atoms = rdmol.GetNumAtoms()
    if any(idx >= n_atoms for idx in atom_key):
        return None

    ri = rdmol.GetRingInfo()

    if param_type == "angle":
        if len(atom_key) != 3:
            return None
        idx_i, idx_j, idx_k = atom_key
        center_idx = idx_j
        flank_idxs = (idx_i, idx_k)

        bond_ij = rdmol.GetBondBetweenAtoms(idx_i, idx_j)
        bond_jk = rdmol.GetBondBetweenAtoms(idx_j, idx_k)
        if bond_ij is None or bond_jk is None:
            return None

        bo_i = bond_ij.GetBondTypeAsDouble()
        bo_k = bond_jk.GetBondTypeAsDouble()
        bo_smarts_i = _bo_smarts_char(bond_ij)
        bo_smarts_k = _bo_smarts_char(bond_jk)

        # Sort both flanks together for order-independent key
        if bo_i <= bo_k:
            bond_orders_raw = (bo_i, bo_k)
            bo_chars_raw = (_bo_char(bond_ij), _bo_char(bond_jk))
            bo_smarts_raw = (bo_smarts_i, bo_smarts_k)
        else:
            bond_orders_raw = (bo_k, bo_i)
            bo_chars_raw = (_bo_char(bond_jk), _bo_char(bond_ij))
            bo_smarts_raw = (bo_smarts_k, bo_smarts_i)

    elif param_type == "bond":
        if len(atom_key) != 2:
            return None
        idx_i, idx_j = atom_key
        center_idx = idx_j   # treat second atom as center by convention
        flank_idxs = (idx_i,)

        bond_ij = rdmol.GetBondBetweenAtoms(idx_i, idx_j)
        if bond_ij is None:
            return None

        bo_v = bond_ij.GetBondTypeAsDouble()
        bond_orders_raw = (bo_v,)
        bo_chars_raw = (_bo_char(bond_ij),)
        bo_smarts_raw = (_bo_smarts_char(bond_ij),)

    elif param_type == "torsion":
        if len(atom_key) != 4:
            return None
        idx_i, idx_j, idx_k, idx_m = atom_key
        center_idx = idx_j   # use second atom as "center" for ring/charge
        flank_idxs = (idx_i, idx_m)

        bond_ij = rdmol.GetBondBetweenAtoms(idx_i, idx_j)
        bond_jk = rdmol.GetBondBetweenAtoms(idx_j, idx_k)
        bond_km = rdmol.GetBondBetweenAtoms(idx_k, idx_m)
        if bond_ij is None or bond_jk is None or bond_km is None:
            return None

        bo_v = bond_jk.GetBondTypeAsDouble()
        bond_orders_raw = (bo_v,)
        bo_chars_raw = (_bo_char(bond_jk),)
        bo_smarts_raw = (_bo_smarts_char(bond_jk),)

    else:
        raise ValueError(f"param_type must be 'angle', 'bond', or 'torsion'; got {param_type!r}")

    center_atom = rdmol.GetAtomWithIdx(center_idx)
    center_element = center_atom.GetSymbol()
    center_aromatic = center_atom.GetIsAromatic()
    center_formal_charge = center_atom.GetFormalCharge()
    center_valence = center_atom.GetTotalDegree()

    ring_sizes = tuple(sorted(len(r) for r in ri.AtomRings() if center_idx in r))
    center_in_ring = bool(ring_sizes)

    flank_elements = tuple(
        sorted(rdmol.GetAtomWithIdx(fi).GetSymbol() for fi in flank_idxs)
    )
    flank_in_ring = tuple(rdmol.GetAtomWithIdx(fi).IsInRing() for fi in flank_idxs)

    feature_key = _make_feature_key(
        bo_chars_raw, flank_elements, ring_sizes, center_formal_charge
    )

    shell_elements: tuple = ()
    center_substituent_elements: tuple = ()
    if depth > 0:
        primary_idxs = set(atom_key)
        shell_elements = _bfs_shell_elements(rdmol, flank_idxs, primary_idxs, depth)
        center_substituent_elements = _bfs_shell_elements(rdmol, (center_idx,), primary_idxs, depth)
        shell_sig = "+".join(shell_elements) or "none"
        sub_sig = "+".join(center_substituent_elements) or "none"
        feature_key = f"{feature_key}|d{depth}:{shell_sig}/{sub_sig}"

    return FeatureSet(
        bond_orders=bond_orders_raw,
        bo_chars=bo_chars_raw,
        flank_elements=flank_elements,
        center_element=center_element,
        center_ring_sizes=ring_sizes,
        center_in_ring=center_in_ring,
        center_aromatic=center_aromatic,
        center_formal_charge=center_formal_charge,
        feature_key=feature_key,
        bo_smarts=bo_smarts_raw,
        center_valence=center_valence,
        flank_in_ring=flank_in_ring,
        depth=depth,
        shell_elements=shell_elements,
        center_substituent_elements=center_substituent_elements,
    )


def group_by_features(
    instances,
    param_type: str = "angle",
    depth: int = 0,
) -> dict[str, list]:
    """Group InstanceRecords by their feature key (four primary axes, plus
    the Axis-5 attachment-context signature when ``depth > 0``).

    Returns a dict mapping ``feature_key → list[InstanceRecord]``.
    Instances whose rdmol is None or whose atom_key references missing
    atoms are silently dropped.
    """
    groups: dict[str, list] = {}
    for rec in instances:
        fs = featurize_instance(rec.rdmol, rec.atom_key, param_type, depth=depth)
        if fs is None:
            continue
        groups.setdefault(fs.feature_key, []).append(rec)
    return groups
