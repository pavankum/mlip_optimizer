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

from mlip_optimizer.comparison import (
    ComparisonResult,
    OverallErrorStatistics,
    QMComparisonResult,
)
from mlip_optimizer.visualization.drawing import draw_molecule

plt.rcParams.update({"font.size": 18})


def _param_failure_note(table: list[list], label: str) -> str | None:
    """Return a footnote string for any rows where FF parametrization failed.

    A row is considered a failure when the second-to-last column (param_id)
    is an empty string, meaning the atom-index key was not found in the FF
    lookup dict.  Returns ``None`` when all rows have a valid param_id or
    when the table has no param columns.
    """
    if not table:
        return None
    # param columns are only present when len(row) is at least 3 and the
    # last two entries were appended by _annotate_table_with_ff_params.
    # We detect them by checking that row[-2] is a str (param_id).
    failed_keys = [
        row[0] for row in table
        if len(row) >= 3 and isinstance(row[-2], str) and row[-2] == ""
    ]
    if not failed_keys:
        return None
    key_strs = ", ".join(str(k) for k in failed_keys)
    return f"  [!] {label} parametrization failed (no FF match) for atom indices: {key_strs}"


def _escape_mpl_text(text: str) -> str:
    """Escape characters in *text* that clash with matplotlib's mathtext parser.

    ``$`` starts/ends math mode; ``\\`` can trigger escape sequences.
    Both are replaced with safe equivalents so SMIRKS patterns render
    correctly as plain text.
    """
    # Backslash first to avoid double-escaping the replacement for $
    text = text.replace("\\", "\\\\")
    text = text.replace("$", r"\$")
    return text


def _parse_mean_diff(s: str) -> float:
    """Parse the mean value from a formatted ``'mean +/- std'`` string."""
    try:
        return abs(float(s.split("+/-")[0].strip()))
    except (ValueError, IndexError, AttributeError):
        return 0.0


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
        fontsize=18,
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

    # Detect whether FF param columns were appended (6 cols vs 4)
    _has_params = (
        (comparison.bond_diffs and len(comparison.bond_diffs[0]) == 6)
        or (comparison.angle_diffs and len(comparison.angle_diffs[0]) == 6)
        or (comparison.torsion_diffs and len(comparison.torsion_diffs[0]) == 6)
    )
    _param_headers = ["Param ID", "SMIRKS"] if _has_params else []

    if comparison.bond_diffs:
        headers = [
            "Bond",
            f"{model1_name}\n(\u00c5)",
            f"{model2_name}\n(\u00c5)",
            "Difference (\u00c5)",
        ] + _param_headers
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
        ] + _param_headers
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
        ] + _param_headers
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

    full_text = _escape_mpl_text("\n".join(tables_text))
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

    # Detect whether FF param columns were appended by _annotate_table_with_ff_params.
    # Rows have len == 2 + n_potentials when no FF, 2 + n_potentials + 2 when FF present.
    _n_base_cols = 2 + len(potential_names)
    _has_params = (
        (qm_comparison.bond_diff_table and len(qm_comparison.bond_diff_table[0]) == _n_base_cols + 2)
        or (qm_comparison.angle_diff_table and len(qm_comparison.angle_diff_table[0]) == _n_base_cols + 2)
        or (qm_comparison.torsion_diff_table and len(qm_comparison.torsion_diff_table[0]) == _n_base_cols + 2)
    )
    if _has_params:
        diff_headers += ["Param ID", "SMIRKS"]

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
        _note = _param_failure_note(qm_comparison.bond_diff_table, "Bond")
        if _note:
            tables_text.append(_note)
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
        _note = _param_failure_note(qm_comparison.angle_diff_table, "Angle")
        if _note:
            tables_text.append(_note)
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
        _note = _param_failure_note(qm_comparison.torsion_diff_table, "Torsion")
        if _note:
            tables_text.append(_note)
        tables_text.append("\n")

    if len(tables_text) <= 3:
        tables_text.append(
            "No significant differences found within configured thresholds."
        )

    # Scale font size if many potentials
    font_size = max(6, 9 - len(potential_names))

    full_text = _escape_mpl_text("\n".join(tables_text))
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


def create_statistics_report(
    stats: dict[str, OverallErrorStatistics],
    potential_names: list[str],
    pdf_pages: PdfPages,
    *,
    dataset_name: str = "",
    dpi: int = 300,
    qm_results: list | None = None,
) -> None:
    """Add overall error statistics pages to a PDF report.

    Creates one page per metric (RMSD, bond, angle, torsion) with a
    summary table comparing all potentials, including max-error molecule
    identifiers.

    Parameters
    ----------
    stats : dict[str, OverallErrorStatistics]
        Map from potential name to its aggregated statistics, from
        :func:`compute_overall_statistics`.
    potential_names : list[str]
        Ordered list of potential names (row order in tables).
    pdf_pages : PdfPages
        Open PdfPages object to write into.
    dataset_name : str, optional
        Dataset label for page titles.
    dpi : int, optional
        Resolution.  Default is ``300``.
    """
    if not stats:
        return

    _PARAM_PAGE_INFO = [
        ("bond_diff_table",    "bond_diffs",    "Bond",    0.1),
        ("angle_diff_table",   "angle_diffs",   "Angle",   5.0),
        ("torsion_diff_table", "torsion_diffs", "Torsion", 40.0),
    ]
    _bond_thresh, _angle_thresh, _torsion_thresh = (
        _PARAM_PAGE_INFO[0][3], _PARAM_PAGE_INFO[1][3], _PARAM_PAGE_INFO[2][3],
    )

    # --- Page 1: Summary overview table (+ threshold-subset table) ---
    _add_summary_overview_page(
        stats, potential_names, pdf_pages, dataset_name, dpi,
        qm_results=qm_results,
        bond_thresh=_bond_thresh,
        angle_thresh=_angle_thresh,
        torsion_thresh=_torsion_thresh,
    )

    # --- Page 2+: Per-metric detail tables ---
    metrics_info = [
        ("RMSD (Angstrom)", "rmsd"),
        ("Mean Bond Diff per conformer (Angstrom)", "bond"),
        ("Mean Angle Diff per conformer (degrees)", "angle"),
        ("Mean Torsion Diff per conformer (degrees)", "torsion"),
    ]
    for title_label, prefix in metrics_info:
        _add_metric_detail_page(
            stats, potential_names, pdf_pages, title_label, prefix,
            dataset_name, dpi,
        )

    # --- Per-parameter error table + bar chart pages ---
    if qm_results:
        for attr, metric_attr, label, threshold in _PARAM_PAGE_INFO:
            _add_param_error_table_and_chart_page(
                qm_results, potential_names, pdf_pages, attr, metric_attr,
                label, threshold, dataset_name, dpi,
            )

    # --- Param histogram, distribution, and violin pages ---
    if qm_results:
        for attr, metric_attr, label, threshold in _PARAM_PAGE_INFO:
            _add_param_histogram_page(
                qm_results, potential_names, pdf_pages, attr, label,
                dataset_name, dpi, threshold=threshold,
            )
            _add_overall_distribution_and_violin_page(
                qm_results, potential_names, pdf_pages, attr, metric_attr,
                label, dataset_name, dpi,
            )
            _add_param_error_distribution_page(
                qm_results, potential_names, pdf_pages, attr, metric_attr,
                label, dataset_name, dpi, threshold=threshold,
            )


def _add_summary_overview_page(
    stats: dict[str, OverallErrorStatistics],
    potential_names: list[str],
    pdf_pages: PdfPages,
    dataset_name: str,
    dpi: int,
    *,
    qm_results: list | None = None,
    bond_thresh: float = 0.1,
    angle_thresh: float = 5.0,
    torsion_thresh: float = 40.0,
) -> None:
    """Add the overview page with one row per potential and key statistics."""
    fig, ax = plt.subplots(figsize=(17, 11), dpi=dpi)
    ax.axis("off")

    headers = [
        "Potential",
        "N conf",
        "RMSD\nmean\u00b1std",
        "RMSD\nmax",
        "Mean bond diff\n(\u00c5) mean\u00b1std",
        "Mean bond diff\n(\u00c5) max",
        "Mean angle diff\n(\u00b0) mean\u00b1std",
        "Mean angle diff\n(\u00b0) max",
        "Mean torsion diff\n(\u00b0) mean\u00b1std",
        "Mean torsion diff\n(\u00b0) max",
    ]

    rows: list[list[str]] = []
    for pot in potential_names:
        s = stats.get(pot)
        if s is None:
            rows.append([pot] + ["N/A"] * 9)
            continue
        rows.append([
            pot,
            str(s.n_conformers_total),
            f"{s.rmsd_mean:.4f}\u00b1{s.rmsd_std:.4f}",
            f"{s.rmsd_max:.4f}",
            f"{s.bond_mean:.4f}\u00b1{s.bond_std:.4f}",
            f"{s.bond_max:.4f}",
            f"{s.angle_mean:.2f}\u00b1{s.angle_std:.2f}",
            f"{s.angle_max:.2f}",
            f"{s.torsion_mean:.2f}\u00b1{s.torsion_std:.2f}",
            f"{s.torsion_max:.2f}",
        ])

    text_parts: list[str] = []
    text_parts.append(f"Overall Error Statistics: {dataset_name}")
    text_parts.append("=" * 120)
    text_parts.append("")
    text_parts.append(
        tabulate(rows, headers=headers, tablefmt="simple", stralign="right")
    )

    # Max-error molecule identifiers table
    text_parts.append("")
    text_parts.append("")
    text_parts.append("Worst-Case Molecule Identifiers (max error)")
    text_parts.append("-" * 120)
    id_headers = [
        "Potential", "RMSD max ID", "Bond max ID", "Angle max ID", "Torsion max ID",
    ]
    id_rows: list[list[str]] = []
    for pot in potential_names:
        s = stats.get(pot)
        if s is None:
            id_rows.append([pot] + ["N/A"] * 4)
            continue
        id_rows.append([
            pot,
            s.rmsd_max_id,
            s.bond_max_id,
            s.angle_max_id,
            s.torsion_max_id,
        ])
    text_parts.append(
        tabulate(id_rows, headers=id_headers, tablefmt="simple", stralign="left")
    )

    # ---- Threshold-crossing subset table ----
    if qm_results:
        text_parts.append("")
        text_parts.append("")
        text_parts.append(
            f"Threshold-Crossing Subset Statistics  "
            f"(bond > {bond_thresh}\u00c5  /  angle > {angle_thresh}\u00b0  /  torsion > {torsion_thresh}\u00b0)"
        )
        text_parts.append(
            "  (N conf = conformers exceeding that metric's threshold; "
            "RMSD column uses union of all three thresholds)"
        )
        text_parts.append("=" * 120)

        th_headers = [
            "Potential",
            "N conf\n(any thresh)",
            "RMSD\nmean\u00b1std",
            "RMSD\nmax",
            "Mean bond diff\n(\u00c5) mean\u00b1std",
            "Bond diff\nmax",
            "N bond\nconfs",
            "Mean angle diff\n(\u00b0) mean\u00b1std",
            "Angle diff\nmax",
            "N angle\nconfs",
            "Mean torsion diff\n(\u00b0) mean\u00b1std",
            "Torsion diff\nmax",
            "N torsion\nconfs",
        ]
        th_rows: list[list[str]] = []
        for pot in potential_names:
            r_vals: list[float] = []
            b_vals: list[float] = []
            a_vals: list[float] = []
            t_vals: list[float] = []
            for qm_comp in qm_results:
                for m in qm_comp.per_potential.get(pot, []):
                    if m.opt_failed:
                        continue
                    b_cross = m.mean_bond_diff > bond_thresh
                    a_cross = m.mean_angle_diff > angle_thresh
                    t_cross = m.mean_torsion_diff > torsion_thresh
                    if b_cross or a_cross or t_cross:
                        r_vals.append(m.rmsd)
                    if b_cross:
                        b_vals.append(m.mean_bond_diff)
                    if a_cross:
                        a_vals.append(m.mean_angle_diff)
                    if t_cross:
                        t_vals.append(m.mean_torsion_diff)

            def _s(vals: list[float], fmt: str) -> tuple[str, str]:
                if not vals:
                    return "N/A", "N/A"
                return (
                    f"{float(np.mean(vals)):{fmt}}\u00b1{float(np.std(vals)):{fmt}}",
                    f"{float(np.max(vals)):{fmt}}",
                )

            r_ms, r_mx = _s(r_vals, ".4f")
            b_ms, b_mx = _s(b_vals, ".4f")
            a_ms, a_mx = _s(a_vals, ".2f")
            t_ms, t_mx = _s(t_vals, ".2f")
            th_rows.append([
                pot,
                str(len(r_vals)),
                r_ms, r_mx,
                b_ms, b_mx, str(len(b_vals)),
                a_ms, a_mx, str(len(a_vals)),
                t_ms, t_mx, str(len(t_vals)),
            ])

        text_parts.append(
            tabulate(th_rows, headers=th_headers, tablefmt="simple", stralign="right")
        )

    font_size = max(5, 8 - len(potential_names))
    ax.text(
        0.02, 0.98, "\n".join(text_parts),
        fontsize=font_size, family="monospace",
        verticalalignment="top", transform=ax.transAxes,
    )

    pdf_pages.savefig(fig, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def _add_metric_detail_page(
    stats: dict[str, OverallErrorStatistics],
    potential_names: list[str],
    pdf_pages: PdfPages,
    title_label: str,
    prefix: str,
    dataset_name: str,
    dpi: int,
) -> None:
    """Add a detail page for a single metric (rmsd/bond/angle/torsion)."""
    fig, ax = plt.subplots(figsize=(17, 11), dpi=dpi)
    ax.axis("off")

    headers = [
        "Potential", "Mean", "Std", "Median", "Min", "Max", "Max Error Molecule",
    ]

    fmt = ".4f" if prefix in ("rmsd", "bond") else ".2f"

    rows: list[list[str]] = []
    for pot in potential_names:
        s = stats.get(pot)
        if s is None:
            rows.append([pot] + ["N/A"] * 6)
            continue
        rows.append([
            pot,
            f"{getattr(s, f'{prefix}_mean'):{fmt}}",
            f"{getattr(s, f'{prefix}_std'):{fmt}}",
            f"{getattr(s, f'{prefix}_median'):{fmt}}",
            f"{getattr(s, f'{prefix}_min'):{fmt}}",
            f"{getattr(s, f'{prefix}_max'):{fmt}}",
            getattr(s, f"{prefix}_max_id"),
        ])

    text_parts: list[str] = []
    text_parts.append(f"{title_label} — {dataset_name}")
    text_parts.append("=" * 120)
    text_parts.append(f"Per-conformer statistics across all molecules")
    text_parts.append("")
    text_parts.append(
        tabulate(rows, headers=headers, tablefmt="simple", stralign="right")
    )

    font_size = max(5, 9 - len(potential_names))
    ax.text(
        0.02, 0.98, "\n".join(text_parts),
        fontsize=font_size, family="monospace",
        verticalalignment="top", transform=ax.transAxes,
    )

    pdf_pages.savefig(fig, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def _add_param_histogram_page(
    qm_results: list,
    potential_names: list[str],
    pdf_pages: PdfPages,
    table_attr: str,
    label: str,
    dataset_name: str,
    dpi: int,
    *,
    threshold: float = 0.0,
) -> None:
    """Add a bar-chart page showing how often each FF param ID crossed the threshold.

    For each potential, counts how many times each ``param_id`` appears in
    the per-molecule diff table (col index 4 of annotated rows).  Rows from
    ``_aggregate_qm_diffs`` have the structure:
    ``[atom_key, qm_ref, pot1_diff, pot2_diff, ..., param_id, smirks]``
    where ``param_id`` is at index ``2 + len(potential_names)`` when FF
    annotation is present.

    If no rows carry a param_id column the page is skipped.
    """
    from collections import Counter

    n_pots = len(potential_names)
    # param_id column index in annotated QM diff-table rows:
    # [atom_key, qm_ref, pot1, pot2, ..., param_id, smirks]
    param_col = 2 + n_pots

    # Count occurrences per potential
    counters: dict[str, Counter] = {p: Counter() for p in potential_names}
    has_params = False

    for qm_comp in qm_results:
        rows: list = getattr(qm_comp, table_attr, [])
        for row in rows:
            if len(row) <= param_col:
                continue  # not annotated
            pid = row[param_col]
            if not pid:
                continue
            has_params = True
            # Only attribute to potentials whose own diff crossed the threshold
            for pot_idx, pot_name in enumerate(potential_names):
                diff_str = row[2 + pot_idx] if len(row) > 2 + pot_idx else "N/A"
                if diff_str and diff_str != "N/A" and _parse_mean_diff(diff_str) > threshold:
                    counters[pot_name][pid] += 1

    if not has_params:
        return

    # Collect all param IDs that appear across any potential
    all_pids = sorted(
        {pid for c in counters.values() for pid in c},
        key=lambda p: -max(c[p] for c in counters.values()),
    )
    if not all_pids:
        return

    # Build figure: one subplot per potential, stacked vertically
    n_rows_pots = len(potential_names)
    fig, axes = plt.subplots(
        n_rows_pots, 1,
        figsize=(max(12, len(all_pids) * 0.6 + 2), 4 * n_rows_pots),
        dpi=dpi,
        squeeze=False,
    )
    fig.suptitle(
        f"{label} Parameter Threshold-Crossing Count — {dataset_name}",
    )

    x = np.arange(len(all_pids))
    for row_idx, pot_name in enumerate(potential_names):
        ax = axes[row_idx][0]
        counts = [counters[pot_name].get(pid, 0) for pid in all_pids]
        bars = ax.bar(x, counts, color="steelblue", edgecolor="white")
        ax.set_title(pot_name)
        ax.set_xticks(x)
        ax.set_xticklabels(all_pids, rotation=45, ha="right")
        ax.set_ylabel("Count")
        ax.set_xlabel(f"{label} Param ID")
        # Annotate bars with count value
        for bar, cnt in zip(bars, counts):
            if cnt > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    str(cnt),
                    ha="center", va="bottom",
                )

    plt.tight_layout(rect=(0, 0.02, 1, 0.95))
    pdf_pages.savefig(fig, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def _add_param_error_distribution_page(
    qm_results: list,
    potential_names: list[str],
    pdf_pages: PdfPages,
    table_attr: str,
    metric_attr: str,
    label: str,
    dataset_name: str,
    dpi: int,
    *,
    threshold: float = 0.0,
) -> None:
    """Add pages of per-parameter-ID error distributions across potentials.

    For each parameter ID that appears in threshold-crossing diff table rows,
    plots two side-by-side panels:

    - Left (A): all per-conformer absolute error values for that parameter,
      regardless of whether they crossed the threshold.
    - Right (B): only the individual data points where ``|error| > threshold``.

    Up to two parameters are shown per page (one row per param, two cols per
    param).  Dashed vertical lines mark each potential's mean; a red dotted
    line marks the threshold when > 0.
    """
    from collections import defaultdict

    n_pots = len(potential_names)
    param_col = 2 + n_pots

    # First pass: build global atom_key -> param_id mapping from all molecules
    global_key_to_pid: dict[tuple, str] = {}
    for qm_comp in qm_results:
        rows: list = getattr(qm_comp, table_attr, [])
        for row in rows:
            if len(row) > param_col:
                pid = row[param_col]
                if pid:
                    atom_key = row[0]
                    if atom_key not in global_key_to_pid:
                        global_key_to_pid[atom_key] = pid

    if not global_key_to_pid:
        return

    # Second pass: collect ALL per-conformer absolute errors and
    # THRESHOLD-CROSSING-ONLY values, both keyed by (pid, potential).
    pid_errors_all: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    pid_errors_thresh: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for qm_comp in qm_results:
        for pot_name in potential_names:
            metrics_list = qm_comp.per_potential.get(pot_name, [])
            for m in metrics_list:
                if m.opt_failed:
                    continue
                diffs: dict = getattr(m, metric_attr, {})
                for atom_key, val in diffs.items():
                    pid = global_key_to_pid.get(atom_key)
                    if pid is None:
                        continue
                    abs_val = abs(val)
                    pid_errors_all[pid][pot_name].append(abs_val)
                    if abs_val > threshold:
                        pid_errors_thresh[pid][pot_name].append(abs_val)

    if not pid_errors_all:
        return

    # Sort pids by total sample count descending (most-seen params first)
    all_pids = sorted(
        pid_errors_all.keys(),
        key=lambda p: -sum(len(v) for v in pid_errors_all[p].values()),
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    pot_colors = {p: colors[i % len(colors)] for i, p in enumerate(potential_names)}
    unit = "\u00c5" if label == "Bond" else "\u00b0"

    def _plot_panel(ax, pid: str, errors_by_pot: dict, title: str) -> None:
        all_vals = [v for vals in errors_by_pot.values() for v in vals]
        if not all_vals:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(title)
            return
        vmin, vmax = min(all_vals), max(all_vals)
        bins = np.linspace(vmin, vmax, 20).tolist() if vmax > vmin else 10
        for pot_name in potential_names:
            vals = errors_by_pot.get(pot_name, [])
            if not vals:
                continue
            ax.hist(vals, bins=bins, alpha=0.5, label=pot_name,
                    color=pot_colors[pot_name], edgecolor="none")
            ax.axvline(float(np.mean(vals)), color=pot_colors[pot_name],
                       linestyle="--", linewidth=1.0)
        if threshold > 0:
            ax.axvline(threshold, color="red", linestyle=":", linewidth=1.2,
                       alpha=0.8, label=f"threshold ({threshold})")
        ax.set_title(title)
        ax.set_xlabel(f"|error| ({unit})")
        ax.set_ylabel("Count")
        ax.legend(loc="upper right")
        ax.tick_params()

    params_per_page = 2
    for page_start in range(0, len(all_pids), params_per_page):
        page_pids = all_pids[page_start : page_start + params_per_page]
        nrows = len(page_pids)

        fig, axes = plt.subplots(
            nrows, 2,
            figsize=(16, 4 * nrows),
            dpi=dpi,
            squeeze=False,
        )
        fig.suptitle(
            f"{label} Error Distributions per Parameter \u2014 {dataset_name}",
        )

        for row_i, pid in enumerate(page_pids):
            _plot_panel(
                axes[row_i][0],
                pid,
                pid_errors_all[pid],
                f"{label} param {pid} \u2014 all conformer data",
            )
            _plot_panel(
                axes[row_i][1],
                pid,
                pid_errors_thresh[pid],
                f"{label} param {pid} \u2014 threshold-crossing only (|err| > {threshold})",
            )

        plt.tight_layout(rect=(0, 0.0, 1, 0.95))
        pdf_pages.savefig(fig, bbox_inches="tight", dpi=dpi)
        plt.close(fig)


def _add_overall_distribution_and_violin_page(
    qm_results: list,
    potential_names: list[str],
    pdf_pages: PdfPages,
    table_attr: str,
    metric_attr: str,
    label: str,
    dataset_name: str,
    dpi: int,
) -> None:
    """Add a single page with an overlapping histogram of all errors (left) and
    a violin plot per potential (right), both restricted to atom keys that
    appear in any threshold-crossing diff-table row.
    """
    from collections import defaultdict

    # Collect atom keys that appeared in threshold-crossing rows
    threshold_keys: set[tuple] = set()
    for qm_comp in qm_results:
        rows: list = getattr(qm_comp, table_attr, [])
        for row in rows:
            threshold_keys.add(row[0])

    if not threshold_keys:
        return

    pot_errors: dict[str, list[float]] = defaultdict(list)
    for qm_comp in qm_results:
        for pot_name in potential_names:
            metrics_list = qm_comp.per_potential.get(pot_name, [])
            for metrics in metrics_list:
                if metrics.opt_failed:
                    continue
                diffs: dict = getattr(metrics, metric_attr, {})
                for atom_key, val in diffs.items():
                    if atom_key in threshold_keys:
                        pot_errors[pot_name].append(abs(val))

    plot_pots = [p for p in potential_names if len(pot_errors.get(p, [])) >= 2]
    if not plot_pots:
        return

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    pot_colors = {p: colors[i % len(colors)] for i, p in enumerate(potential_names)}
    unit = "\u00c5" if label == "Bond" else "\u00b0"

    fig, (ax_hist, ax_violin) = plt.subplots(
        1, 2, figsize=(16, 6), dpi=dpi
    )
    fig.suptitle(
        f"{label} Overall Error Distributions (threshold-crossing params) \u2014 {dataset_name}",
    )

    # --- Left: overlapping histograms ---
    all_vals = [v for p in plot_pots for v in pot_errors[p]]
    vmin, vmax = min(all_vals), max(all_vals)
    bins: int | np.ndarray = np.linspace(vmin, vmax, 30) if vmax > vmin else 10
    for pot_name in plot_pots:
        ax_hist.hist(
            pot_errors[pot_name],
            bins=bins,
            alpha=0.5,
            label=pot_name,
            color=pot_colors[pot_name],
            edgecolor="none",
        )
        ax_hist.axvline(
            float(np.mean(pot_errors[pot_name])),
            color=pot_colors[pot_name],
            linestyle="--",
            linewidth=1.2,
        )
    ax_hist.set_xlabel(f"|{label} error| ({unit})")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Overlapping distributions")
    ax_hist.legend()

    # --- Right: violin plot ---
    positions = list(range(1, len(plot_pots) + 1))
    data = [pot_errors[p] for p in plot_pots]
    parts = ax_violin.violinplot(data, positions=positions, showmedians=True, showextrema=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(pot_colors[plot_pots[i]])
        pc.set_alpha(0.6)
    for part_name in ("cbars", "cmins", "cmaxes", "cmedians"):
        if part_name in parts:
            parts[part_name].set_color("black")
            parts[part_name].set_linewidth(1.0)
    ax_violin.set_xticks(positions)
    ax_violin.set_xticklabels(plot_pots, rotation=30, ha="right")
    ax_violin.set_ylabel(f"|{label} error| ({unit})")
    ax_violin.set_xlabel("Potential")
    ax_violin.set_title("Violin plot")
    ax_violin.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=(0, 0.0, 1, 0.93))
    pdf_pages.savefig(fig, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def _add_param_error_table_and_chart_page(
    qm_results: list,
    potential_names: list[str],
    pdf_pages: PdfPages,
    table_attr: str,
    metric_attr: str,
    label: str,
    threshold: float,
    dataset_name: str,
    dpi: int,
) -> None:
    """Add per-parameter-ID error table + horizontal bar chart pages for one metric.

    For each FF parameter ID found in threshold-crossing diff-table rows, collects
    the raw signed per-conformer per-atom-key diff values, splits into:

    - **Overall**: all conformer diffs mapped to that param_id
    - **Threshold-crossing subset**: only values where ``|diff| > threshold``

    Outputs two page groups per potential:

    1. Text table (param_id rows × overall+thresh columns).
    2. Horizontal bar chart — left panel: overall mean ± std per param_id;
       right panel: threshold-crossing subset mean ± std per param_id.
       Red dashed line at zero; error bars show ± std.
    """
    from collections import defaultdict

    n_pots = len(potential_names)
    param_col = 2 + n_pots
    smirks_col = param_col + 1

    # ---- Build atom_key → param_id / smirks from threshold-crossing diff rows ----
    global_key_to_pid: dict[tuple, str] = {}
    global_key_to_smirks: dict[tuple, str] = {}
    for qm_comp in qm_results:
        for row in getattr(qm_comp, table_attr, []):
            if len(row) > param_col and row[param_col] and row[0] not in global_key_to_pid:
                global_key_to_pid[row[0]] = row[param_col]
                global_key_to_smirks[row[0]] = row[smirks_col] if len(row) > smirks_col else ""

    if not global_key_to_pid:
        return

    # ---- Collect raw signed diff values keyed by (pid, potential) ----
    pid_all: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    pid_thresh: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for qm_comp in qm_results:
        for pot_name in potential_names:
            for m in qm_comp.per_potential.get(pot_name, []):
                if m.opt_failed:
                    continue
                for atom_key, val in getattr(m, metric_attr, {}).items():
                    pid = global_key_to_pid.get(atom_key)
                    if pid is None:
                        continue
                    pid_all[pid][pot_name].append(val)
                    if abs(val) > threshold:
                        pid_thresh[pid][pot_name].append(val)

    if not pid_all:
        return

    pid_to_smirks: dict[str, str] = {
        pid: global_key_to_smirks.get(k, "")
        for k, pid in global_key_to_pid.items()
    }
    all_pids = sorted(pid_all.keys())
    unit = "\u00c5" if label == "Bond" else "\u00b0"
    val_fmt = ".4f" if label == "Bond" else ".3f"

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    pot_colors = {p: colors[i % len(colors)] for i, p in enumerate(potential_names)}

    # ---- TABLE PAGE: one per potential ----
    for pot_name in potential_names:
        headers = [
            "Param ID", "SMIRKS (truncated)",
            f"Overall mean\n({unit})", "Overall\nstd",
            f"Overall min\n({unit})", f"Overall max\n({unit})", "Overall\nN",
            f"Thresh mean\n({unit})", "Thresh\nstd",
            f"Thresh min\n({unit})", f"Thresh max\n({unit})", "Thresh\nN",
        ]
        rows_table = []
        for pid in all_pids:
            o_vals = pid_all[pid].get(pot_name, [])
            t_vals = pid_thresh[pid].get(pot_name, [])
            smirks_s = _escape_mpl_text(pid_to_smirks.get(pid, "")[:50])
            if o_vals:
                row: list = [
                    pid, smirks_s,
                    f"{float(np.mean(o_vals)):{val_fmt}}",
                    f"{float(np.std(o_vals)):{val_fmt}}",
                    f"{float(np.min(o_vals)):{val_fmt}}",
                    f"{float(np.max(o_vals)):{val_fmt}}",
                    str(len(o_vals)),
                ]
            else:
                row = [pid, smirks_s, "N/A", "N/A", "N/A", "N/A", "0"]
            if t_vals:
                row += [
                    f"{float(np.mean(t_vals)):{val_fmt}}",
                    f"{float(np.std(t_vals)):{val_fmt}}",
                    f"{float(np.min(t_vals)):{val_fmt}}",
                    f"{float(np.max(t_vals)):{val_fmt}}",
                    str(len(t_vals)),
                ]
            else:
                row += ["\u2014", "\u2014", "\u2014", "\u2014", "0"]
            rows_table.append(row)

        fig_h = max(8.0, len(all_pids) * 0.35 + 3.0)
        fig, ax = plt.subplots(figsize=(17, fig_h), dpi=dpi)
        ax.axis("off")
        title_text = (
            f"{label} Per-Parameter Error Statistics \u2014 "
            f"{_escape_mpl_text(pot_name)} \u2014 {_escape_mpl_text(dataset_name)}\n"
            f"Overall: all conformer diffs  |  "
            f"Thresh: |diff| > {threshold}\u202f{unit}"
        )
        full_text = title_text + "\n" + "=" * 140 + "\n"
        full_text += tabulate(rows_table, headers=headers, tablefmt="simple")
        font_size = max(5, 7 - len(all_pids) // 25)
        ax.text(0.01, 0.99, full_text, fontsize=font_size, family="monospace",
                verticalalignment="top", transform=ax.transAxes)
        pdf_pages.savefig(fig, bbox_inches="tight", dpi=dpi)
        plt.close(fig)

    # ---- BAR CHART PAGE: one per potential ----
    for pot_name in potential_names:
        color = pot_colors[pot_name]
        n_pids = len(all_pids)
        fig_h = max(8.0, n_pids * 0.45 + 2.5)

        fig, axes = plt.subplots(
            1, 2, figsize=(16, fig_h), dpi=dpi,
            gridspec_kw={"wspace": 0.45},
        )
        fig.suptitle(
            f"{label} Per-Parameter Error Distribution \u2014 "
            f"{_escape_mpl_text(pot_name)} \u2014 {_escape_mpl_text(dataset_name)}\n"
            f"Left: all conformers  |  "
            f"Right: |diff| > {threshold}\u202f{unit} (threshold-crossing subset)",
            y=0.998,
        )

        y_pos = np.arange(n_pids)
        for ax_panel, use_thresh in zip(axes, [False, True]):
            means_list: list[float] = []
            stds_list: list[float] = []
            for pid in all_pids:
                vals = (pid_thresh if use_thresh else pid_all)[pid].get(pot_name, [])
                if vals:
                    means_list.append(float(np.mean(vals)))
                    stds_list.append(float(np.std(vals)))
                else:
                    means_list.append(0.0)
                    stds_list.append(0.0)

            means_arr = np.array(means_list)
            stds_arr = np.array(stds_list)

            ax_panel.barh(
                y_pos, means_arr,
                color=color if not use_thresh else "none",
                alpha=0.82 if not use_thresh else 1.0,
                hatch="" if not use_thresh else "///",
                edgecolor=color, linewidth=0.8,
            )
            ax_panel.errorbar(
                means_arr, y_pos, xerr=stds_arr,
                fmt="none", ecolor="black",
                elinewidth=0.7, capsize=2, capthick=0.7,
            )
            ax_panel.axvline(0, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
            ax_panel.set_yticks(y_pos)
            ax_panel.set_yticklabels(all_pids)
            ax_panel.set_xlabel(f"Mean diff ({unit})")
            subset_str = "All conformers" if not use_thresh else f"|diff| > {threshold}\u202f{unit}"
            ax_panel.set_title(f"{subset_str}  (mean \u00b1 std)")
            ax_panel.grid(axis="x", alpha=0.3)
            ax_panel.tick_params(axis="x")

        plt.tight_layout(rect=(0, 0, 1, 0.93))
        pdf_pages.savefig(fig, bbox_inches="tight", dpi=dpi)
        plt.close(fig)


def create_smarts_error_report(
    qm_results: list,
    records: list,
    potential_names: list[str],
    pdf_pages: PdfPages,
    functional_groups_df,
    *,
    dataset_name: str = "",
    dpi: int = 300,
) -> None:
    """Add per-functional-group bond/angle/torsion deviation pages to a PDF.

    For each SMARTS pattern in *functional_groups_df*, finds matching
    molecules in *records* / *qm_results* and plots per-conformer
    mean bond, angle, and torsion deviation distributions as three
    side-by-side violin subplots (one violin per potential per panel).

    A section title page is prepended before the first matching group.
    Groups with no matching molecules or insufficient data are silently
    skipped.

    Parameters
    ----------
    qm_results : list[QMComparisonResult]
        Per-molecule comparison results, parallel to *records*.
    records : list[MoleculeRecord]
        QM reference records (each must have a ``.molecule`` OpenFF Molecule).
    potential_names : list[str]
        Ordered list of potential model names.
    pdf_pages : PdfPages
        Open PdfPages object to append pages into.
    functional_groups_df : pd.DataFrame
        DataFrame with at least ``"Functional Group"`` and ``"SMARTS"``
        columns.  Optional: ``"Element"``, ``"Hybridization"``, ``"Geometry"``.
    dataset_name : str, optional
        Dataset label shown in page titles.
    dpi : int, optional
        Resolution.  Default ``300``.
    """
    from collections import defaultdict
    from rdkit import Chem

    required = {"Functional Group", "SMARTS"}
    if not required.issubset(functional_groups_df.columns):
        return

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    pot_colors = {p: colors[i % len(colors)] for i, p in enumerate(potential_names)}

    # Pre-convert all QM molecules to RDKit once.
    rdmols: list = []
    for rec in records:
        try:
            rdmols.append(rec.molecule.to_rdkit())
        except Exception:
            rdmols.append(None)

    any_page_added = False

    # Metrics to plot: (attr on QMComparisonMetrics, axis label, unit)
    _SMARTS_METRICS = [
        ("mean_bond_diff",    "Mean Bond Deviation",    "\u00c5"),
        ("mean_angle_diff",   "Mean Angle Deviation",   "\u00b0"),
        ("mean_torsion_diff", "Mean Torsion Deviation", "\u00b0"),
    ]

    for _, row in functional_groups_df.iterrows():
        fg_name = str(row.get("Functional Group", "Unknown") or "Unknown")
        smarts = str(row.get("SMARTS", "") or "")
        element = str(row.get("Element", "") or "")
        hybridization = str(row.get("Hybridization", "") or "")
        geometry = str(row.get("Geometry", "") or "")

        if not smarts or smarts.lower() == "nan":
            continue

        try:
            query = Chem.MolFromSmarts(smarts)
        except Exception:
            query = None
        if query is None:
            continue

        # Substructure-match every molecule
        matching_indices: list[int] = []
        for mol_idx, rdmol in enumerate(rdmols):
            if rdmol is None or mol_idx >= len(qm_results):
                continue
            try:
                if rdmol.HasSubstructMatch(query):
                    matching_indices.append(mol_idx)
            except Exception:
                continue

        if not matching_indices:
            continue

        # Collect per-conformer bond/angle/torsion mean diffs for each potential
        pot_data: dict[str, dict[str, list[float]]] = {
            attr: defaultdict(list) for attr, _, _ in _SMARTS_METRICS
        }
        for mol_idx in matching_indices:
            if mol_idx >= len(qm_results) or qm_results[mol_idx] is None:
                continue
            qm_comp = qm_results[mol_idx]
            for pot_name in potential_names:
                for m in qm_comp.per_potential.get(pot_name, []):
                    if m.opt_failed:
                        continue
                    for attr, _, _ in _SMARTS_METRICS:
                        val = getattr(m, attr, None)
                        if val is not None and not np.isnan(val):
                            pot_data[attr][pot_name].append(val)

        # Keep potentials that have at least 2 data points in any metric
        plot_pots = [
            p for p in potential_names
            if any(len(pot_data[attr].get(p, [])) >= 2 for attr, _, _ in _SMARTS_METRICS)
        ]
        if not plot_pots:
            continue

        # Section title page on first successful SMARTS group
        if not any_page_added:
            create_title_page(
                pdf_pages,
                (
                    "Functional Group SMARTS Error Analysis\n\n"
                    f"{dataset_name}\n\n"
                    "Bond / Angle / Torsion Deviations \u2014 per SMARTS functional group"
                ),
                dpi=dpi,
            )
            any_page_added = True

        safe_fg = _escape_mpl_text(fg_name)
        safe_smarts = _escape_mpl_text(smarts)
        safe_element = _escape_mpl_text(element)
        safe_hybrid = _escape_mpl_text(hybridization)
        safe_geom = _escape_mpl_text(geometry)
        safe_dataset = _escape_mpl_text(dataset_name)

        n_mol_matched = len(matching_indices)
        # Conformer count from the first available metric / potential
        n_conf_total = 0
        for attr, _, _ in _SMARTS_METRICS:
            for p in plot_pots:
                v = pot_data[attr].get(p, [])
                if v:
                    n_conf_total = len(v)
                    break
            if n_conf_total:
                break

        # --- Build figure: 1 row × 3 violin subplots ---
        fig, axes = plt.subplots(
            1, 3, figsize=(22, 9.5), dpi=dpi,
            gridspec_kw={"wspace": 0.45},
        )

        title_main = f"{safe_element}  \u2014  {safe_fg}"
        title_smarts = f"SMARTS:  {safe_smarts}"
        subtitle = (
            f"{n_mol_matched} molecule(s) matched  "
            f"|  {n_conf_total} conformer(s) [first potential]  "
            f"|  hybridization: {safe_hybrid}  |  geometry: {safe_geom}  "
            f"|  dataset: {safe_dataset}"
        )
        fig.suptitle(f"{title_main}\n{title_smarts}", y=0.98)
        fig.text(
            0.5, 0.915, subtitle,
            ha="center", style="italic", color="#444444",
        )

        legend_handles: list = []
        legend_labels: list[str] = []

        for ax, (attr, metric_label, unit) in zip(axes, _SMARTS_METRICS):
            # Only include potentials with enough data for this metric
            metric_pots = [p for p in plot_pots if len(pot_data[attr].get(p, [])) >= 2]
            if not metric_pots:
                ax.set_visible(False)
                continue

            positions = list(range(1, len(metric_pots) + 1))
            data = [pot_data[attr][p] for p in metric_pots]

            parts = ax.violinplot(
                data, positions=positions, showmedians=True, showextrema=True,
            )
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(pot_colors[metric_pots[i]])
                pc.set_alpha(0.65)
                # Collect legend proxies from the first subplot only
                if ax is axes[0]:
                    import matplotlib.patches as mpatches
                    legend_handles.append(
                        mpatches.Patch(
                            facecolor=pot_colors[metric_pots[i]], alpha=0.65,
                            label=_escape_mpl_text(metric_pots[i]),
                        )
                    )
                    legend_labels.append(_escape_mpl_text(metric_pots[i]))
            for part_name in ("cbars", "cmins", "cmaxes", "cmedians"):
                if part_name in parts:
                    parts[part_name].set_color("black")
                    parts[part_name].set_linewidth(1.0)

            ax.set_xticks(positions)
            ax.set_xticklabels(
                [_escape_mpl_text(p) for p in metric_pots],
                rotation=35, ha="right",
            )
            ax.set_ylabel(f"{metric_label} ({unit})")
            ax.set_xlabel("Potential")
            ax.set_title(metric_label)
            ax.grid(axis="y", alpha=0.3)

        # Legend below all panels, outside the axes
        if legend_handles:
            fig.legend(
                legend_handles, legend_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.01),
                ncol=min(len(legend_handles), 4),
                frameon=True,
            )

        # ~14 % headroom at top for title block, ~12 % at bottom for legend
        plt.tight_layout(rect=(0.0, 0.12, 1.0, 0.86))
        pdf_pages.savefig(fig, bbox_inches="tight", dpi=dpi)
        plt.close(fig)
        plt.close("all")