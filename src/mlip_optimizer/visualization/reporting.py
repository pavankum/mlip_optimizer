"""PDF report generation for geometry optimization comparisons.

Creates multi-page PDF reports containing molecule images alongside
tabulated bond, angle, and torsion differences between optimizers.
Supports both pairwise comparisons and multi-potential vs QM reference.
"""

from __future__ import annotations

import io

import numpy as np
import matplotlib.pyplot as plt
from cairosvg import svg2png
from matplotlib.backends.backend_pdf import PdfPages
from openff.toolkit import Molecule
from PIL import Image
from tabulate import tabulate

from mlip_optimizer.comparison import ComparisonResult, QMComparisonResult
from mlip_optimizer.visualization.drawing import draw_molecule


def create_title_page(
    pdf_pages: PdfPages,
    title: str,
    *,
    figsize: tuple[float, float] = (11, 8),
    dpi: int = 300,
) -> None:
    """Add a title page to an open PDF.

    Parameters
    ----------
    pdf_pages : PdfPages
        Open PdfPages object.
    title : str
        Title text (may contain newlines).
    figsize : tuple[float, float], optional
        Figure size in inches.  Default is ``(11, 8)``.
    dpi : int, optional
        Resolution.  Default is ``300``.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        title,
        fontsize=14,
        ha="center",
        va="center",
        wrap=True,
    )
    pdf_pages.savefig(fig, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def create_comparison_report(
    molecule: Molecule,
    smiles: str,
    comparison: ComparisonResult,
    model_pairs: list[tuple[str, str]],
    pdf_pages: PdfPages,
    *,
    molecule_label: str = "",
    num_conformers: int | None = None,
    image_width: int = 1200,
    image_height: int = 1200,
    dpi: int = 300,
) -> None:
    """Add a comparison page (molecule image + difference tables) to a PDF.

    The left side of the page shows a 2D depiction of the molecule with
    atom indices annotated.  The right side shows tabulated bond, angle,
    and torsion differences exceeding the configured thresholds.

    Parameters
    ----------
    molecule : Molecule
        The molecule to draw (atom indices are annotated automatically).
    smiles : str
        SMILES string displayed in the page title.
    comparison : ComparisonResult
        Comparison data from :func:`evaluate_model_pairs`.
    model_pairs : list[tuple[str, str]]
        The optimizer name pairs that were compared (used for column
        headers in the tables).
    pdf_pages : PdfPages
        Open PdfPages object to write into.
    molecule_label : str, optional
        Label for the molecule (shown in the page header).
    num_conformers : int or None, optional
        Number of conformers to display in the header.  If ``None``,
        uses ``len(molecule.conformers)``.
    image_width : int, optional
        Width of the molecule SVG in pixels.  Default is ``1200``.
    image_height : int, optional
        Height of the molecule SVG in pixels.  Default is ``1200``.
    dpi : int, optional
        Resolution for the page.  Default is ``300``.
    """
    if num_conformers is None:
        num_conformers = len(molecule.conformers)

    # Render molecule image
    svg_data = draw_molecule(
        molecule,
        atom_notes={i: str(i) for i in range(molecule.n_atoms)},
        width=image_width,
        height=image_height,
    )
    png_data = svg2png(bytestring=svg_data.encode("utf-8"), dpi=dpi)
    img = Image.open(io.BytesIO(png_data))

    # Get model names from the first pair
    model1_name, model2_name = model_pairs[0]

    # Build the figure
    fig = plt.figure(figsize=(17, 11), dpi=dpi)

    # Molecule image on the left
    ax_img = fig.add_axes([0.02, 0.05, 0.35, 0.9])
    ax_img.axis("off")
    ax_img.imshow(img)
    ax_img.set_title(
        f"{molecule_label}\nSMILES: {smiles}",
        fontsize=11,
        wrap=True,
        pad=10,
    )

    # Tables on the right
    ax_tables = fig.add_axes([0.40, 0.05, 0.58, 0.9])
    ax_tables.axis("off")

    tables_text: list[str] = []
    tables_text.append(
        f"Molecule: {molecule_label}, num_conformers: {num_conformers}\n"
    )

    if comparison.bond_diffs:
        headers = [
            "Bond",
            f"{model1_name}\n(\u00c5)",
            f"{model2_name}\n(\u00c5)",
            "Difference (\u00c5)",
        ]
        tables_text.append(
            "BOND DIFFERENCES OF > threshold\n" + "=" * 100
        )
        tables_text.append(
            tabulate(comparison.bond_diffs, headers=headers, tablefmt="simple")
        )
        tables_text.append("\n")

    if comparison.angle_diffs:
        headers = [
            "Angle",
            f"{model1_name}\n(\u00b0)",
            f"{model2_name}\n(\u00b0)",
            "Difference (\u00b0)",
        ]
        tables_text.append(
            "ANGLE DIFFERENCES OF > threshold\n" + "=" * 100
        )
        tables_text.append(
            tabulate(comparison.angle_diffs, headers=headers, tablefmt="simple")
        )
        tables_text.append("\n")

    if comparison.torsion_diffs:
        headers = [
            "Torsion",
            f"{model1_name}\n(\u00b0)",
            f"{model2_name}\n(\u00b0)",
            "Difference (\u00b0)",
        ]
        tables_text.append(
            "TORSION DIFFERENCES OF > threshold\n" + "=" * 100
        )
        tables_text.append(
            tabulate(
                comparison.torsion_diffs, headers=headers, tablefmt="simple"
            )
        )
        tables_text.append("\n")

    if len(tables_text) == 1:
        tables_text.append(
            "No significant differences found within configured thresholds."
        )

    full_text = "\n".join(tables_text)
    ax_tables.text(
        0.0,
        1.0,
        full_text,
        fontsize=9,
        family="monospace",
        verticalalignment="top",
        wrap=True,
    )

    pdf_pages.savefig(fig, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    plt.close("all")


def create_qm_comparison_report(
    molecule: Molecule,
    smiles: str,
    qm_comparison: QMComparisonResult,
    potential_names: list[str],
    pdf_pages: PdfPages,
    *,
    molecule_label: str = "",
    num_conformers: int | None = None,
    image_width: int = 1200,
    image_height: int = 1200,
    dpi: int = 300,
) -> None:
    """Add a QM-comparison page to a PDF report.

    The left side shows a 2D molecule depiction with atom indices.
    The right side shows an RMSD summary and tabulated bond, angle,
    and torsion differences with one column per potential, all
    compared against the QM reference geometry.

    Parameters
    ----------
    molecule : Molecule
        The molecule to draw.
    smiles : str
        SMILES string displayed in the page title.
    qm_comparison : QMComparisonResult
        Comparison data from :func:`evaluate_against_qm`.
    potential_names : list[str]
        Ordered list of potential names (column order in tables).
    pdf_pages : PdfPages
        Open PdfPages object to write into.
    molecule_label : str, optional
        Label for the molecule (shown in the page header).
    num_conformers : int or None, optional
        Number of conformers to display in the header.
    image_width : int, optional
        Width of the molecule SVG in pixels.  Default is ``1200``.
    image_height : int, optional
        Height of the molecule SVG in pixels.  Default is ``1200``.
    dpi : int, optional
        Resolution for the page.  Default is ``300``.
    """
    if num_conformers is None:
        num_conformers = qm_comparison.n_conformers

    # Render molecule image
    svg_data = draw_molecule(
        molecule,
        atom_notes={i: str(i) for i in range(molecule.n_atoms)},
        width=image_width,
        height=image_height,
    )
    png_data = svg2png(bytestring=svg_data.encode("utf-8"), dpi=dpi)
    img = Image.open(io.BytesIO(png_data))

    # Build the figure
    fig = plt.figure(figsize=(17, 11), dpi=dpi)

    # Molecule image on the left
    ax_img = fig.add_axes([0.02, 0.05, 0.35, 0.9])
    ax_img.axis("off")
    ax_img.imshow(img)
    ax_img.set_title(
        f"{molecule_label}\nSMILES: {smiles}",
        fontsize=11,
        wrap=True,
        pad=10,
    )

    # Tables on the right
    ax_tables = fig.add_axes([0.40, 0.05, 0.58, 0.9])
    ax_tables.axis("off")

    tables_text: list[str] = []
    tables_text.append(
        f"Molecule: {molecule_label}, num_conformers: {num_conformers}\n"
    )

    # --- RMSD summary ---
    rmsd_lines = ["RMSD vs QM (Angstrom)  [mean +/- std across conformers]"]
    rmsd_lines.append("-" * 60)
    for pot_name in potential_names:
        metrics_list = qm_comparison.per_potential.get(pot_name, [])
        if metrics_list:
            rmsds = [m.rmsd for m in metrics_list]
            mean_r = float(np.mean(rmsds))
            std_r = float(np.std(rmsds))
            rmsd_lines.append(f"  {pot_name:30s}  {mean_r:.4f} +/- {std_r:.4f}")
        else:
            rmsd_lines.append(f"  {pot_name:30s}  N/A")
    tables_text.append("\n".join(rmsd_lines))
    tables_text.append("\n")

    # --- Build diff table headers ---
    diff_headers = ["Key", "QM Ref"]
    for pot in potential_names:
        diff_headers.append(f"{pot}\n(diff)")

    # --- Bond differences ---
    if qm_comparison.bond_diff_table:
        tables_text.append(
            "BOND DIFF vs QM (Angstrom) > threshold\n" + "=" * 80
        )
        tables_text.append(
            tabulate(
                qm_comparison.bond_diff_table,
                headers=diff_headers,
                tablefmt="simple",
            )
        )
        tables_text.append("\n")

    # --- Angle differences ---
    if qm_comparison.angle_diff_table:
        tables_text.append(
            "ANGLE DIFF vs QM (degrees) > threshold\n" + "=" * 80
        )
        tables_text.append(
            tabulate(
                qm_comparison.angle_diff_table,
                headers=diff_headers,
                tablefmt="simple",
            )
        )
        tables_text.append("\n")

    # --- Torsion differences ---
    if qm_comparison.torsion_diff_table:
        tables_text.append(
            "TORSION DIFF vs QM (degrees) > threshold\n" + "=" * 80
        )
        tables_text.append(
            tabulate(
                qm_comparison.torsion_diff_table,
                headers=diff_headers,
                tablefmt="simple",
            )
        )
        tables_text.append("\n")

    if len(tables_text) <= 3:
        tables_text.append(
            "No significant differences found within configured thresholds."
        )

    # Scale font size if many potentials
    font_size = max(6, 9 - len(potential_names))

    full_text = "\n".join(tables_text)
    ax_tables.text(
        0.0,
        1.0,
        full_text,
        fontsize=font_size,
        family="monospace",
        verticalalignment="top",
        wrap=True,
    )

    pdf_pages.savefig(fig, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    plt.close("all")
