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

    # --- Page 1: Summary overview table ---
    _add_summary_overview_page(stats, potential_names, pdf_pages, dataset_name, dpi)

    # --- Page 2+: Per-metric detail tables ---
    metrics_info = [
        ("RMSD (Angstrom)", "rmsd"),
        ("Max Bond Diff (Angstrom)", "bond"),
        ("Max Angle Diff (degrees)", "angle"),
        ("Max Torsion Diff (degrees)", "torsion"),
    ]
    for title_label, prefix in metrics_info:
        _add_metric_detail_page(
            stats, potential_names, pdf_pages, title_label, prefix,
            dataset_name, dpi,
        )

    # --- Param histogram, distribution, and violin pages ---
    if qm_results:
        _PARAM_PAGE_INFO = [
            ("bond_diff_table",    "bond_diffs",    "Bond",    0.1),
            ("angle_diff_table",   "angle_diffs",   "Angle",   5.0),
            ("torsion_diff_table", "torsion_diffs", "Torsion", 40.0),
        ]
        for attr, metric_attr, label, threshold in _PARAM_PAGE_INFO:
            _add_param_histogram_page(
                qm_results, potential_names, pdf_pages, attr, label,
                dataset_name, dpi, threshold=threshold,
            )
            _add_param_error_distribution_page(
                qm_results, potential_names, pdf_pages, attr, metric_attr,
                label, dataset_name, dpi, threshold=threshold,
            )
            _add_error_violin_page(
                qm_results, potential_names, pdf_pages, attr, metric_attr,
                label, dataset_name, dpi,
            )


def _add_summary_overview_page(
    stats: dict[str, OverallErrorStatistics],
    potential_names: list[str],
    pdf_pages: PdfPages,
    dataset_name: str,
    dpi: int,
) -> None:
    """Add the overview page with one row per potential and key statistics."""
    fig, ax = plt.subplots(figsize=(17, 11), dpi=dpi)
    ax.axis("off")

    headers = [
        "Potential",
        "N conf",
        "RMSD\nmean\u00b1std",
        "RMSD\nmax",
        "Bond\nmean\u00b1std",
        "Bond\nmax",
        "Angle\nmean\u00b1std",
        "Angle\nmax",
        "Torsion\nmean\u00b1std",
        "Torsion\nmax",
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
        fontsize=12,
    )

    x = np.arange(len(all_pids))
    for row_idx, pot_name in enumerate(potential_names):
        ax = axes[row_idx][0]
        counts = [counters[pot_name].get(pid, 0) for pid in all_pids]
        bars = ax.bar(x, counts, color="steelblue", edgecolor="white")
        ax.set_title(pot_name, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(all_pids, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Count", fontsize=8)
        ax.set_xlabel(f"{label} Param ID", fontsize=8)
        # Annotate bars with count value
        for bar, cnt in zip(bars, counts):
            if cnt > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    str(cnt),
                    ha="center", va="bottom", fontsize=6,
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
    plots overlapping normalised histograms of absolute error values (collected
    across all conformers and molecules) for each potential, with dashed
    vertical lines at each potential's mean.  Up to four parameters are shown
    per page in a 2x2 grid.  A red dotted vertical line marks the threshold.
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

    # Second pass: collect per-conformer absolute error values per (pid, potential)
    pid_errors: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for qm_comp in qm_results:
        for pot_name in potential_names:
            metrics_list = qm_comp.per_potential.get(pot_name, [])
            for metrics in metrics_list:
                if metrics.opt_failed:
                    continue
                diffs: dict = getattr(metrics, metric_attr, {})
                for atom_key, val in diffs.items():
                    pid = global_key_to_pid.get(atom_key)
                    if pid:
                        pid_errors[pid][pot_name].append(abs(val))

    if not pid_errors:
        return

    # Sort pids by total sample count descending (most-seen params first)
    all_pids = sorted(
        pid_errors.keys(),
        key=lambda p: -sum(len(v) for v in pid_errors[p].values()),
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    pot_colors = {p: colors[i % len(colors)] for i, p in enumerate(potential_names)}
    unit = "\u00c5" if label == "Bond" else "\u00b0"

    params_per_page = 4
    for page_start in range(0, len(all_pids), params_per_page):
        page_pids = all_pids[page_start : page_start + params_per_page]
        n_on_page = len(page_pids)
        ncols = min(2, n_on_page)
        nrows = (n_on_page + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(8 * ncols, 4 * nrows),
            dpi=dpi,
            squeeze=False,
        )
        fig.suptitle(
            f"{label} Error Distributions per Parameter \u2014 {dataset_name}",
            fontsize=11,
        )

        for idx, pid in enumerate(page_pids):
            row_i, col_i = divmod(idx, ncols)
            ax = axes[row_i][col_i]

            all_vals = [v for vals in pid_errors[pid].values() for v in vals]
            if not all_vals:
                ax.axis("off")
                continue

            vmin, vmax = min(all_vals), max(all_vals)
            bins: int | np.ndarray = (
                np.linspace(vmin, vmax, 20) if vmax > vmin else 10
            )
            plotted_any = False
            for pot_name in potential_names:
                vals = pid_errors[pid].get(pot_name, [])
                if not vals:
                    continue
                ax.hist(
                    vals,
                    bins=bins,
                    alpha=0.5,
                    label=pot_name,
                    color=pot_colors[pot_name],
                    edgecolor="none",
                    density=True,
                )
                ax.axvline(
                    float(np.mean(vals)),
                    color=pot_colors[pot_name],
                    linestyle="--",
                    linewidth=1.0,
                )
                plotted_any = True

            if threshold > 0:
                ax.axvline(
                    threshold,
                    color="red",
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.8,
                    label=f"threshold ({threshold})",
                )

            ax.set_title(f"{label} param {pid}", fontsize=9)
            ax.set_xlabel(f"|error| ({unit})", fontsize=8)
            ax.set_ylabel("Density", fontsize=8)
            if plotted_any:
                ax.legend(fontsize=6, loc="upper right")
            ax.tick_params(labelsize=7)

        for idx in range(n_on_page, nrows * ncols):
            row_i, col_i = divmod(idx, ncols)
            axes[row_i][col_i].axis("off")

        plt.tight_layout(rect=(0, 0.0, 1, 0.95))
        pdf_pages.savefig(fig, bbox_inches="tight", dpi=dpi)
        plt.close(fig)


def _add_error_violin_page(
    qm_results: list,
    potential_names: list[str],
    pdf_pages: PdfPages,
    table_attr: str,
    metric_attr: str,
    label: str,
    dataset_name: str,
    dpi: int,
) -> None:
    """Add a violin plot page showing per-potential error distributions.

    Collects absolute error values for all parameter keys that appear in
    threshold-crossing rows across all molecules and conformers, then draws
    one violin per potential.  Potentials with fewer than two data points
    are skipped.
    """
    from collections import defaultdict

    # Collect atom keys that crossed threshold in any molecule
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

    plot_data = [
        (p, pot_errors[p]) for p in potential_names if len(pot_errors.get(p, [])) >= 2
    ]
    if not plot_data:
        return

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    unit = "\u00c5" if label == "Bond" else "\u00b0"

    fig, ax = plt.subplots(
        figsize=(max(8, len(plot_data) * 2 + 2), 6), dpi=dpi
    )

    positions = list(range(1, len(plot_data) + 1))
    data = [vals for _, vals in plot_data]
    pot_labels = [name for name, _ in plot_data]

    parts = ax.violinplot(data, positions=positions, showmedians=True, showextrema=True)

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.6)
    for part_name in ("cbars", "cmins", "cmaxes", "cmedians"):
        if part_name in parts:
            parts[part_name].set_color("black")
            parts[part_name].set_linewidth(1.0)

    ax.set_xticks(positions)
    ax.set_xticklabels(pot_labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(f"|{label} error| ({unit})", fontsize=9)
    ax.set_xlabel("Potential", fontsize=9)
    ax.set_title(
        f"{label} Error Distribution (threshold-crossing params) \u2014 {dataset_name}",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    pdf_pages.savefig(fig, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
