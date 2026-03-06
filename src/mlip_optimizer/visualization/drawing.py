"""Molecule drawing utilities using RDKit.

Renders 2D depictions of molecules as SVG strings, with optional
highlighting of atoms and bonds, atom/bond annotations, and control
over hydrogen display.
"""

from __future__ import annotations

import unicodedata

from openff.toolkit import Molecule
from rdkit import Chem
from rdkit.Chem import Draw

from mlip_optimizer._types import BondIndices, Color


def asciify(s: str) -> str:
    """Normalize a Unicode string to ASCII, dropping non-ASCII characters.

    Parameters
    ----------
    s : str
        Input string (may contain Unicode).

    Returns
    -------
    str
        ASCII-safe string.
    """
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def draw_molecule(
    molecule: Molecule | Chem.rdchem.Mol,
    *,
    width: int = -1,
    height: int = 300,
    highlight_atoms: list[int] | dict[int, Color] | None = None,
    highlight_bonds: list[BondIndices] | dict[BondIndices, Color] | None = None,
    atom_notes: dict[int, str] | None = None,
    bond_notes: dict[BondIndices, str] | None = None,
    deemphasize_atoms: list[int] | None = None,
    explicit_hydrogens: bool | None = None,
    color_by_element: bool | None = None,
    legend: str = "",
) -> str:
    """Draw a molecule as an SVG string.

    Parameters
    ----------
    molecule : Molecule or rdkit.Chem.rdchem.Mol
        The molecule to draw (OpenFF or RDKit).
    width : int, optional
        Image width in pixels.  ``-1`` (default) auto-sizes.
    height : int, optional
        Image height in pixels.  Default is ``300``.
    highlight_atoms : list[int] or dict[int, Color] or None
        Atom indices to highlight, or a map from indices to RGB color
        triplets (floats in ``[0, 1]``).
    highlight_bonds : list[BondIndices] or dict[BondIndices, Color] or None
        Bond atom-index pairs to highlight, or a map to RGB colors.
    atom_notes : dict[int, str] or None
        Labels to display near specific atoms.
    bond_notes : dict[BondIndices, str] or None
        Labels to display near specific bonds.
    deemphasize_atoms : list[int] or None
        Atom indices to render in light grey.
    explicit_hydrogens : bool or None
        ``True`` forces all H explicit; ``False`` hides uncharged monovalent
        H; ``None`` defers to the molecule.
    color_by_element : bool or None
        ``True`` colors heteroatoms by element; ``False`` uses monochrome.
        ``None`` (default) uses color unless highlights are provided.
    legend : str, optional
        Text legend below the drawing.

    Returns
    -------
    str
        SVG XML string.

    Raises
    ------
    KeyError
        If a highlighted atom or bond is missing from the drawn molecule.

    Examples
    --------
    >>> from IPython.display import SVG
    >>> SVG(draw_molecule(Molecule.from_smiles("CCO")))
    """
    # Convert to RDKit molecule
    try:
        rdmol = Chem.rdchem.Mol(molecule.to_rdkit())
    except AttributeError:
        rdmol = Chem.rdchem.Mol(molecule)

    # Process color_by_element
    if color_by_element is None:
        color_by_element = highlight_atoms is None and highlight_bonds is None

    if color_by_element:
        def set_atom_palette(draw_options):
            return draw_options.useDefaultAtomPalette()
    else:
        def set_atom_palette(draw_options):
            return draw_options.useBWAtomPalette()

    # Process explicit_hydrogens and build atom index map
    if explicit_hydrogens is None:
        idx_map = {i: i for i in range(rdmol.GetNumAtoms())}
    elif explicit_hydrogens:
        idx_map = {i: i for i in range(rdmol.GetNumAtoms())}
        rdmol = Chem.AddHs(rdmol, explicitOnly=True)
    else:
        idx_map = {
            old: new
            for new, old in enumerate(
                a.GetIdx()
                for a in rdmol.GetAtoms()
                if a.GetAtomicNum() != 1 and a.GetMass() != 1
            )
        }
        rdmol = Chem.RemoveHs(rdmol, updateExplicitCount=True)

    # Process highlight_atoms
    highlight_atom_colors: dict[int, tuple] | None
    if highlight_atoms is None:
        highlight_atoms = []
        highlight_atom_colors = None
    elif isinstance(highlight_atoms, dict):
        highlight_atom_colors = {
            idx_map[i]: tuple(c)
            for i, c in highlight_atoms.items()
            if i in idx_map
        }
        highlight_atoms = list(highlight_atoms.keys())
    else:
        highlight_atoms = [idx_map[i] for i in highlight_atoms if i in idx_map]
        highlight_atom_colors = None

    # Process highlight_bonds
    highlight_bond_indices: list[int]
    highlight_bond_colors: dict[int, tuple] | None
    if highlight_bonds is None:
        highlight_bond_indices = []
        highlight_bond_colors = None
    elif isinstance(highlight_bonds, dict):
        highlight_bond_colors = {
            rdmol.GetBondBetweenAtoms(idx_map[i_a], idx_map[i_b]).GetIdx(): tuple(v)
            for (i_a, i_b), v in highlight_bonds.items()
            if i_a in idx_map and i_b in idx_map
        }
        highlight_bond_indices = list(highlight_bond_colors.keys())
    else:
        highlight_bond_indices = [
            rdmol.GetBondBetweenAtoms(idx_map[i_a], idx_map[i_b]).GetIdx()
            for i_a, i_b in highlight_bonds
            if i_a in idx_map and i_b in idx_map
        ]
        highlight_bond_colors = None

    # Place bond notes
    if bond_notes is not None:
        for (i_a, i_b), note in bond_notes.items():
            if i_a not in idx_map or i_b not in idx_map:
                continue
            rdbond = rdmol.GetBondBetweenAtoms(idx_map[i_a], idx_map[i_b])
            rdbond.SetProp("bondNote", asciify(str(note)))

    # Place atom notes
    if atom_notes is not None:
        for i, note in atom_notes.items():
            if i not in idx_map:
                continue
            rdatom = rdmol.GetAtomWithIdx(idx_map[i])
            rdatom.SetProp("atomNote", asciify(str(note)))

    # Kekulize for consistent depiction
    Chem.rdmolops.Kekulize(rdmol, clearAromaticFlags=True)

    # Compute 2D coordinates
    Chem.rdDepictor.Compute2DCoords(rdmol)
    Chem.rdDepictor.StraightenDepiction(rdmol)

    # Draw
    drawer = Draw.MolDraw2DSVG(width, height)
    draw_options = drawer.drawOptions()
    set_atom_palette(draw_options)
    draw_options.setBondNoteColour((0.7, 0.7, 0.7))

    if deemphasize_atoms:
        draw_options.setHighlightColour((255 / 255, 176 / 255, 103 / 255))
        draw_options.continuousHighlight = False
        draw_options.circleAtoms = False
        highlight_atoms += deemphasize_atoms
        highlight_atom_colors = (
            {} if highlight_atom_colors is None else highlight_atom_colors
        )
        highlight_atom_colors.update(
            {i: (0.8, 0.8, 0.8) for i in deemphasize_atoms}
        )
    else:
        draw_options.setHighlightColour((255 / 255, 202 / 255, 154 / 255))

    drawer.DrawMolecule(
        rdmol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_atom_colors,
        highlightBonds=highlight_bond_indices,
        highlightBondColors=highlight_bond_colors,
        legend=legend,
    )

    drawer.FinishDrawing()
    return drawer.GetDrawingText()
