"""Functional-group benchmark PDF report.

Mixes reportlab (tables, assembly) with matplotlib (figures embedded as PNG).
Each per-group page has identical dimensions, fixed margins, and no text overlap.

Library split
-------------
reportlab  : title page, all table pages, PDF document assembly
matplotlib : distribution pages, per-group violin pages  →  rendered to BytesIO
             PNG and embedded via reportlab.platypus.Image

Dimension guarantee
-------------------
Every per-group matplotlib figure uses the module constants FIG_WIDTH /
FIG_HEIGHT / DPI plus a fixed fig.subplots_adjust() call.  bbox_inches is
deliberately omitted (defaults to None / rcParams) so matplotlib cannot
resize the canvas between loop iterations.

Text-overlap prevention
-----------------------
suptitle y=0.97   subplot titles pad=8   SUBPLOTS_ADJUST bottom=0.18
footer fig.text(0.5, 0.04)   legend bbox_to_anchor=(1.02, 0.5) inside the
18 % right margin reserved by right=0.82
"""

from __future__ import annotations

import io
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ---------------------------------------------------------------------------
# Module-level rendering constants  — define ONCE, used everywhere
# ---------------------------------------------------------------------------

PAGE_SIZE = letter
_PAGE_W, _PAGE_H = letter          # 612 pt × 792 pt

FIG_WIDTH:  float = 9.5            # inches — every per-group figure
FIG_HEIGHT: float = 4.2            # inches — per-group / single-row pages
_DIST_FIG_HEIGHT: float = 6.0     # inches — 2×2 distribution pages
DPI:        int   = 200

TITLE_FS: int = 11                 # suptitle / subplot titles
AXIS_FS:  int = 9                  # axis labels
TICK_FS:  int = 8                  # tick labels
META_FS:  int = 7                  # metadata footer

# Fixed subplot margins — applied on every per-group page, no tight_layout
SUBPLOTS_ADJUST: dict[str, float] = dict(
    left=0.07, right=0.82, top=0.84, bottom=0.22, wspace=0.30,
)

# Populated by the caller; _FALLBACK used when a potential has no entry
POTENTIAL_COLORS: dict[str, str] = {}
_FALLBACK_PALETTE = plt.rcParams["axes.prop_cycle"].by_key()["color"]

X_TICK_ROTATION: int = 30          # degrees; ha='right' below

# reportlab table layout
_TABLE_FONT_SIZE: int   = 9
_TABLE_MARGIN:    float = 0.5 * inch
_MAX_TABLE_W:     float = _PAGE_W - 2 * _TABLE_MARGIN
_HDR_BG   = colors.HexColor("#D3D3D3")
_ALT_BG   = colors.HexColor("#F5F5F5")
_GRID_CLR = colors.HexColor("#AAAAAA")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pot_color(name: str, idx: int) -> str:
    return POTENTIAL_COLORS.get(name, _FALLBACK_PALETTE[idx % len(_FALLBACK_PALETTE)])


def _escape_mpl(text: str) -> str:
    return text.replace("\\", "\\\\").replace("$", r"\$")


def _sanitize_for_matplotlib(text: str) -> str:
    """Escape $ to prevent matplotlib mathtext interpretation."""
    return text.replace("$", r"\$")


def _wrap_title(text: str, max_width: int = 80) -> str:
    """Wrap long titles to multiple lines intelligently.

    Breaks at SMIRKS boundary (after ]) or at max_width, whichever comes first.
    """
    if len(text) <= max_width:
        return text

    # Try to break after the first SMIRKS pattern (ends with ])
    for i in range(max_width, len(text)):
        if text[i] == ']' and i + 2 < len(text) and text[i + 1:i + 3] == '  ':
            return text[:i + 1] + "\n" + text[i + 3:]

    # Fallback: break at max_width at a space
    for i in range(max_width, max_width // 2, -1):
        if text[i] == ' ':
            return text[:i] + "\n" + text[i + 1:]

    return text


def _shorten_pot_name(name: str, max_len: int = 20) -> str:
    """Abbreviate common long patterns in model names for plot display."""
    s = name
    s = s.replace("finetuneIons", "ftI").replace("finetune-ions", "ftI")
    s = s.replace("finetune", "ft")
    s = s.replace("-medium", "-M").replace("-small", "-S").replace("-large", "-L")
    if len(s) > max_len:
        s = s[:max_len - 1] + "\u2026"
    return s


# ---------------------------------------------------------------------------
# reportlab table builder
# ---------------------------------------------------------------------------

def _make_rl_table(
    headers: list[str],
    rows: list[list[Any]],
    *,
    numeric_cols: set[int] | None = None,
    decimal_places: dict[int, int] | None = None,
    title: str = "",
) -> list:
    """Build reportlab Table flowables from headers + rows.

    If the table is too wide for the page, it is split into column-groups
    (key column 0 repeated on each group).  Font size is NEVER shrunk.
    Numeric columns are right-aligned and formatted using decimal_places or .4g.
    Alternating row shading is applied when the table has more than 6 rows.

    Parameters
    ----------
    decimal_places : dict, optional
        Maps column index → number of decimal places to use (e.g. {2: 2, 3: 2}).
        Columns not in this dict use .4g format (4 significant figures).
    """
    if numeric_cols is None:
        numeric_cols = set()
    if decimal_places is None:
        decimal_places = {}

    _hdr_style = ParagraphStyle(
        "rpt_th",
        fontName="Helvetica-Bold",
        fontSize=_TABLE_FONT_SIZE,
        alignment=TA_CENTER,
        leading=_TABLE_FONT_SIZE + 2,
    )
    _cell_l = ParagraphStyle(
        "rpt_td",
        fontName="Helvetica",
        fontSize=_TABLE_FONT_SIZE,
        alignment=TA_LEFT,
        leading=_TABLE_FONT_SIZE + 2,
    )
    _cell_r = ParagraphStyle(
        "rpt_td_r",
        fontName="Helvetica",
        fontSize=_TABLE_FONT_SIZE,
        alignment=TA_RIGHT,
        leading=_TABLE_FONT_SIZE + 2,
    )

    def _fmt(val: Any, ci: int) -> Paragraph:
        if isinstance(val, float):
            if ci in decimal_places:
                txt = f"{val:.{decimal_places[ci]}f}"
            else:
                txt = f"{val:.4g}"
        elif isinstance(val, (int, np.integer)):
            txt = str(int(val))
        else:
            txt = str(val)
        return Paragraph(txt, _cell_r if ci in numeric_cols else _cell_l)

    def _build_subtable(col_indices: list[int]) -> Table:
        hdr_row = [Paragraph(headers[ci], _hdr_style) for ci in col_indices]
        body = [
            [_fmt(row[ci], ci) for ci in col_indices]
            for row in rows
        ]
        data = [hdr_row] + body
        col_w = _MAX_TABLE_W / len(col_indices)

        ts_cmds = [
            ("BACKGROUND",  (0, 0), (-1, 0),  _HDR_BG),
            ("GRID",        (0, 0), (-1, -1),  0.25, _GRID_CLR),
            ("VALIGN",      (0, 0), (-1, -1),  "MIDDLE"),
            ("TOPPADDING",  (0, 0), (-1, -1),  3),
            ("BOTTOMPADDING",(0, 0),(-1, -1),  3),
        ]
        if len(rows) > 6:
            for ri in range(2, len(rows) + 1, 2):   # even body rows (1-indexed)
                ts_cmds.append(("BACKGROUND", (0, ri), (-1, ri), _ALT_BG))

        t = Table(data, colWidths=[col_w] * len(col_indices), repeatRows=1)
        t.setStyle(TableStyle(ts_cmds))
        return t

    # Estimate total width using character counts at ~0.55 pt per char
    char_w = _TABLE_FONT_SIZE * 0.55
    max_chars = [
        max(len(str(headers[c])), *(len(str(row[c])) for row in rows))
        for c in range(len(headers))
    ]
    estimated_w = sum(char_w * mc for mc in max_chars)

    flowables: list = []
    if title:
        sty = getSampleStyleSheet()
        flowables.append(Paragraph(f"<b>{title}</b>", sty["Heading3"]))
        flowables.append(Spacer(1, 4))

    if not rows:
        flowables.append(Paragraph("(no data)", getSampleStyleSheet()["Normal"]))
        return flowables

    if estimated_w <= _MAX_TABLE_W:
        flowables.append(_build_subtable(list(range(len(headers)))))
    else:
        # Split: always keep col 0 (key); partition remaining cols into groups
        key = 0
        char_w_key = char_w * max_chars[key]
        budget = _MAX_TABLE_W - char_w_key
        group: list[int] = []
        group_w = 0.0
        groups: list[list[int]] = []
        for ci in range(1, len(headers)):
            cw = char_w * max_chars[ci]
            if group and group_w + cw > budget:
                groups.append(group)
                group = []
                group_w = 0.0
            group.append(ci)
            group_w += cw
        if group:
            groups.append(group)

        for g in groups:
            col_indices = [key] + g
            flowables.append(_build_subtable(col_indices))
            flowables.append(Spacer(1, 8))

    return flowables


# ---------------------------------------------------------------------------
# matplotlib → reportlab image helper
# ---------------------------------------------------------------------------

def _param_error_stats_table(
    err_dict: dict[str, list[float]],
    potential_names: list[str],
    unit: str,
) -> list:
    """Compact error-statistics table for a single FF parameter.

    Columns: Potential | N | Mean | Std | Median | P95 | Max  (all errors in *unit*)
    Returns a list of flowables (Spacer + Table).
    Formatting: bonds use 2 decimals, angles/torsions use 2 decimals.
    """
    headers = [
        "Potential",
        "N",
        f"Mean ({unit})",
        f"Std ({unit})",
        f"Median ({unit})",
        f"P95 ({unit})",
        f"Max ({unit})",
    ]
    rows = []
    for pot in potential_names:
        vals = err_dict.get(pot, [])
        if not vals:
            continue
        a = np.array(vals)
        rows.append([
            _shorten_pot_name(pot),
            len(a),
            float(np.mean(a)),
            float(np.std(a)),
            float(np.median(a)),
            float(np.percentile(a, 95)),
            float(np.max(a)),
        ])
    if not rows:
        return []
    # Determine decimal places: 2 for all distance/angle metrics (Å, °)
    decimals = {2: 2, 3: 2, 4: 2, 5: 2, 6: 2}
    return [Spacer(1, 6)] + _make_rl_table(
        headers, rows, numeric_cols={1, 2, 3, 4, 5, 6}, decimal_places=decimals
    )


def _fig_to_rl_image(fig: plt.Figure, *, exact: bool = False) -> RLImage:
    """Save a matplotlib figure to a reportlab Image flowable.

    Parameters
    ----------
    exact : bool
        If True, save at exact canvas size (no bbox cropping) to guarantee
        identical output across loop iterations.  Use for per-group pages.
        If False, use bbox_inches='tight' (fine for one-off pages).
    """
    buf = io.BytesIO()
    kwargs: dict[str, Any] = {"format": "png", "dpi": DPI}
    if not exact:
        kwargs["bbox_inches"] = "tight"
    fig.savefig(buf, **kwargs)
    buf.seek(0)
    # Derive aspect ratio from the figure's actual canvas size
    fig_w, fig_h = fig.get_size_inches()
    plt.close(fig)

    rl_w = _PAGE_W - inch
    rl_h = rl_w * (fig_h / fig_w)
    return RLImage(buf, width=rl_w, height=rl_h)


# ---------------------------------------------------------------------------
# Public API — page renderers
# ---------------------------------------------------------------------------

def render_title_page(
    title: str,
    potentials: list[str],
    n_molecules: int,
    dataset_name: str = "",
) -> list:
    """Return a list of reportlab flowables for the title page."""
    _title_sty = ParagraphStyle(
        "rpt_title",
        fontName="Helvetica-Bold",
        fontSize=22,
        alignment=TA_CENTER,
        spaceAfter=14,
    )
    _sub_sty = ParagraphStyle(
        "rpt_sub",
        fontName="Helvetica",
        fontSize=12,
        alignment=TA_CENTER,
        spaceAfter=6,
    )
    _bullet_sty = ParagraphStyle(
        "rpt_bullet",
        fontName="Helvetica",
        fontSize=10,
        alignment=TA_LEFT,
        leftIndent=72,
        spaceAfter=3,
    )

    flowables: list = [
        Spacer(1, 1.5 * inch),
        Paragraph(title, _title_sty),
    ]
    if dataset_name:
        flowables.append(Paragraph(f"Dataset: {dataset_name}", _sub_sty))
    flowables += [
        Paragraph(f"Total molecules: {n_molecules:,}", _sub_sty),
        Spacer(1, 0.3 * inch),
        HRFlowable(width="80%", thickness=0.5, color=colors.gray),
        Spacer(1, 0.2 * inch),
        Paragraph("<b>Potentials compared:</b>", _sub_sty),
    ]
    for pot in potentials:
        flowables.append(Paragraph(f"\u2022  {pot}", _bullet_sty))
    flowables.append(PageBreak())
    return flowables


def render_table_page(title: str, df: pd.DataFrame) -> list:
    """Return reportlab flowables for a full-page-width table.

    Numeric columns are formatted with appropriate decimal places:
    - Bonds: 2 decimals (Å)
    - Angles/Torsions: 2 decimals (°)
    - RMSD: 3 decimals (Å)
    Wide tables are wrapped into column groups — font size is never shrunk.
    Rows > 6 get alternating shading.
    """
    numeric_cols = {
        i for i, dt in enumerate(df.dtypes)
        if pd.api.types.is_numeric_dtype(dt)
    }
    # Determine decimal places from column names
    decimal_places = {}
    for i, col in enumerate(df.columns):
        if i in numeric_cols:
            if "Bond" in col:
                decimal_places[i] = 2
            elif "Angle" in col or "Torsion" in col:
                decimal_places[i] = 2
            elif "RMSD" in col:
                decimal_places[i] = 3
    headers = list(df.columns)
    rows = [list(r) for r in df.itertuples(index=False, name=None)]
    flowables = _make_rl_table(
        headers, rows, numeric_cols=numeric_cols, decimal_places=decimal_places, title=title
    )
    flowables.append(PageBreak())
    return flowables


def render_distribution_page(
    per_potential_errors: dict[str, list[float]],
    potential_names: list[str],
    label: str,
    unit: str,
    dataset_name: str = "",
    per_potential_actuals: dict[str, list[float]] | None = None,
    qm_ref_values: list[float] | None = None,
    per_potential_actuals_all: dict[str, list[float]] | None = None,
    qm_ref_all: list[float] | None = None,
) -> RLImage | None:
    """2×2 grid: actual-value distributions (left) and error distributions (right).

    Row 0: overlapping histograms.
    Row 1: violin plots.

    When *qm_ref_all* / *per_potential_actuals_all* are provided the actual-value
    panels show both the full-dataset distribution (step/outline) and the
    high-error subset (filled), so you can see how the problematic subset relates
    to the overall population.
    Legend is placed outside the axes to the right.
    """
    label = _sanitize_for_matplotlib(label)
    plot_pots = [p for p in potential_names if per_potential_errors.get(p)]
    if not plot_pots:
        return None

    has_actual = bool(
        per_potential_actuals and any(per_potential_actuals.get(p) for p in plot_pots)
    )
    has_all = bool(qm_ref_all) or bool(
        per_potential_actuals_all and any(per_potential_actuals_all.get(p) for p in plot_pots)
    )

    pot_colors = {p: _pot_color(p, i) for i, p in enumerate(potential_names)}
    short_names = {p: _shorten_pot_name(p) for p in potential_names}

    # Width scales with the number of violins when (all + high-error) pairs are shown
    n_violin_actual = 0
    if has_actual:
        n_violin_actual = (1 if (qm_ref_values or qm_ref_all) else 0) + len(plot_pots)
        if has_all:
            n_violin_actual *= 2
    fig_w = max(FIG_WIDTH, n_violin_actual * 1.05 + 2.5) if n_violin_actual else FIG_WIDTH

    if has_actual:
        fig, axes = plt.subplots(
            2, 2, figsize=(fig_w, _DIST_FIG_HEIGHT), dpi=DPI,
        )
        fig.subplots_adjust(left=0.07, right=0.82, top=0.88, bottom=0.20,
                            wspace=0.32, hspace=0.55)
        ax_ah, ax_eh = axes[0]   # histogram row
        ax_av, ax_ev = axes[1]   # violin row
    else:
        fig, (ax_eh, ax_ev) = plt.subplots(
            1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI,
        )
        fig.subplots_adjust(left=0.08, right=0.82, top=0.84, bottom=0.24, wspace=0.35)
        ax_ah = ax_av = None

    fig.suptitle(
        _wrap_title(f"{label} Distribution \u2014 {dataset_name}"),
        fontsize=TITLE_FS, y=0.97,
    )

    # ---- helper: violin (supports per-entry alpha and hatch) ----
    def _draw_violin(ax, data_lists, labels, clrs, alphas=None, hatches=None):
        if not data_lists:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=AXIS_FS)
            ax.set_xticks([])
            return
        parts = ax.violinplot(data_lists, positions=range(len(data_lists)),
                              showmedians=True, showextrema=True)
        for j, body in enumerate(parts["bodies"]):
            body.set_facecolor(clrs[j])
            body.set_alpha(alphas[j] if alphas else 0.6)
            if hatches and hatches[j]:
                body.set_hatch(hatches[j])
                body.set_edgecolor(clrs[j])
        for pn in ("cbars", "cmins", "cmaxes", "cmedians"):
            if pn in parts:
                parts[pn].set_color("black"); parts[pn].set_linewidth(0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=X_TICK_ROTATION, ha="right",
                           fontsize=TICK_FS)
        ax.tick_params(axis="y", labelsize=TICK_FS)
        ax.spines[["top", "right"]].set_visible(False)

    # ---- actual value panels ----
    if has_actual:
        act_data  = [per_potential_actuals.get(p, []) for p in plot_pots]
        qm_vals   = qm_ref_values or []
        act_data_all = [
            (per_potential_actuals_all.get(p, []) if per_potential_actuals_all else [])
            for p in plot_pots
        ]

        # Common bin range spanning high-error and full distributions
        all_for_bins = (
            [v for lst in act_data for v in lst]
            + qm_vals
            + (qm_ref_all or [])
            + [v for lst in act_data_all for v in lst]
        )
        if all_for_bins:
            v0, v1 = min(all_for_bins), max(all_for_bins)
            bins_a = np.linspace(v0, v1, 40) if v1 > v0 else 20
        else:
            bins_a = 20

        # Full-dataset — step/outline style (drawn first, behind)
        if has_all:
            if qm_ref_all:
                ax_ah.hist(qm_ref_all, bins=bins_a, histtype='step',
                           linewidth=1.5, linestyle='--', color='gray', alpha=0.9)
                ax_ah.axvline(float(np.mean(qm_ref_all)), color='gray',
                              linestyle=':', linewidth=0.8)
            for p, vals_all in zip(plot_pots, act_data_all):
                if vals_all:
                    ax_ah.hist(vals_all, bins=bins_a, histtype='step',
                               linewidth=1.5, linestyle='--',
                               color=pot_colors[p], alpha=0.85)
                    ax_ah.axvline(float(np.mean(vals_all)), color=pot_colors[p],
                                  linestyle=':', linewidth=0.8)

        # High-error subset — solid fill (drawn on top, current behaviour)
        for p, vals in zip(plot_pots, act_data):
            if vals:
                ax_ah.hist(vals, bins=bins_a, alpha=0.5,
                           color=pot_colors[p], edgecolor="none")
                ax_ah.axvline(float(np.mean(vals)), color=pot_colors[p],
                              linestyle="--", linewidth=1.0)
        if qm_vals:
            ax_ah.hist(qm_vals, bins=bins_a, alpha=0.3,
                       color="gray", edgecolor="none",
                       label='QM ref*' if has_all else 'QM ref')
            ax_ah.axvline(float(np.mean(qm_vals)), color="gray",
                          linestyle=":", linewidth=1.2)

        ax_ah.set_xlabel(f"Actual value ({unit})", fontsize=AXIS_FS)
        ax_ah.set_ylabel("Count", fontsize=AXIS_FS)
        hist_title = "Actual values — histogram"
        if has_all:
            hist_title += "  (dashed = all, filled = high-error*)"
        ax_ah.set_title(hist_title, fontsize=TITLE_FS - (1 if has_all else 0), pad=8)
        ax_ah.tick_params(labelsize=TICK_FS)
        ax_ah.tick_params(axis="x", labelrotation=X_TICK_ROTATION)
        ax_ah.spines[["top", "right"]].set_visible(False)

        # Build violin lists — interleave (all, high-error*) pairs when full data present
        viol_data_a: list = []
        viol_lbls_a: list[str] = []
        viol_clrs_a: list = []
        viol_alphas_a: list[float] = []
        viol_hatches_a: list = []

        if has_all:
            if qm_ref_all:
                viol_data_a.append(qm_ref_all);  viol_lbls_a.append('QM (all)')
                viol_clrs_a.append('gray');       viol_alphas_a.append(0.30); viol_hatches_a.append('///')
            if qm_vals:
                viol_data_a.append(qm_vals);     viol_lbls_a.append('QM*')
                viol_clrs_a.append('gray');       viol_alphas_a.append(0.70); viol_hatches_a.append(None)
            for p, vals_all, vals_hi in zip(plot_pots, act_data_all, act_data):
                sn = short_names[p]
                if vals_all:
                    viol_data_a.append(vals_all); viol_lbls_a.append(f'{sn} (all)')
                    viol_clrs_a.append(pot_colors[p]); viol_alphas_a.append(0.25); viol_hatches_a.append('///')
                if vals_hi:
                    viol_data_a.append(vals_hi);  viol_lbls_a.append(f'{sn}*')
                    viol_clrs_a.append(pot_colors[p]); viol_alphas_a.append(0.65); viol_hatches_a.append(None)
        else:
            viol_data_a   = ([qm_vals] if qm_vals else []) + act_data
            viol_lbls_a   = (["QM ref"] if qm_vals else []) + [short_names[p] for p in plot_pots]
            viol_clrs_a   = (["gray"]   if qm_vals else []) + [pot_colors[p] for p in plot_pots]
            viol_alphas_a = [0.6] * len(viol_data_a)
            viol_hatches_a = [None] * len(viol_data_a)

        _draw_violin(ax_av, viol_data_a, viol_lbls_a, viol_clrs_a,
                     alphas=viol_alphas_a, hatches=viol_hatches_a)
        ax_av.set_ylabel(f"Actual value ({unit})", fontsize=AXIS_FS)
        violin_title = "Actual values — violin"
        if has_all:
            violin_title += "  (hatched = all, solid = high-error*)"
        ax_av.set_title(violin_title, fontsize=TITLE_FS - (1 if has_all else 0), pad=8)

    # ---- error panels ----
    err_data  = [per_potential_errors[p] for p in plot_pots]
    all_err   = [v for lst in err_data for v in lst]
    if all_err:
        v0, v1  = min(all_err), max(all_err)
        bins_e  = np.linspace(v0, v1, 30) if v1 > v0 else 10
        for p, vals in zip(plot_pots, err_data):
            ax_eh.hist(vals, bins=bins_e, alpha=0.5,
                       color=pot_colors[p], edgecolor="none")
            ax_eh.axvline(float(np.mean(vals)), color=pot_colors[p],
                          linestyle="--", linewidth=1.0)
    ax_eh.set_xlabel(f"Error ({unit})", fontsize=AXIS_FS)
    ax_eh.set_ylabel("Count", fontsize=AXIS_FS)
    ax_eh.set_title("Errors — histogram", fontsize=TITLE_FS, pad=8)
    ax_eh.tick_params(labelsize=TICK_FS)
    ax_eh.tick_params(axis="x", labelrotation=X_TICK_ROTATION)
    ax_eh.spines[["top", "right"]].set_visible(False)

    _draw_violin(ax_ev, err_data, [short_names[p] for p in plot_pots],
                 [pot_colors[p] for p in plot_pots])
    ax_ev.set_ylabel(f"Error ({unit})", fontsize=AXIS_FS)
    ax_ev.set_title("Errors — violin", fontsize=TITLE_FS, pad=8)

    # ---- shared legend on right (full names) ----
    handles = [mpatches.Patch(facecolor=pot_colors[p], alpha=0.6, label=short_names[p])
               for p in plot_pots]
    ax_ev.legend(handles, [short_names[p] for p in plot_pots], loc="center left",
                 bbox_to_anchor=(1.02, 0.5), fontsize=TICK_FS,
                 framealpha=0.8, borderaxespad=0.0)

    return _fig_to_rl_image(fig, exact=True)


def render_group_page(
    group_name: str,
    smarts: str,
    metadata: dict[str, Any],
    per_potential_data: dict[str, dict[str, list[float]]],
    potential_names: list[str],
) -> list:
    """Render one per-functional-group page as [RLImage, caption Paragraph].

    Layout (fixed, never drifts between iterations)
    ------------------------------------------------
    y=0.97  suptitle: group_name + SMARTS  (2 lines)
    y=0.84–0.18  three violin plots: bond | angle | torsion
    y=0.04  italic metadata footer
    right 18% margin: legend (bbox_to_anchor=(1.02, 0.5))

    Parameters
    ----------
    per_potential_data : dict[potential_name → dict[metric → list[float]]]
        metric keys: 'bond', 'angle', 'torsion'
    """
    metrics  = ["bond",    "angle",     "torsion"]
    units    = ["\u00c5",  "\u00b0",    "\u00b0"]
    m_titles = ["Bond deviation", "Angle deviation", "Torsion deviation"]

    # Always the same canvas size — never changes between iterations
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
    # Fixed margins — identical on every page regardless of label lengths
    fig.subplots_adjust(**SUBPLOTS_ADJUST)

    # Suptitle: pinned at y=0.97, 2 lines, well above the subplot area (top=0.84)
    fig.suptitle(
        f"{group_name}\n{_wrap_title(_escape_mpl(smarts), max_width=100)}",
        fontsize=TITLE_FS,
        y=0.97,
        ha="center",
    )

    legend_handles: list[mpatches.Patch] = []
    legend_labels:  list[str]  = []

    for ax_idx, (ax, metric, unit, mtitle) in enumerate(
        zip(axes, metrics, units, m_titles)
    ):
        pot_data:   list[list[float]] = []
        pot_labels: list[str]         = []
        pot_clrs:   list[str]         = []

        for i, pot in enumerate(potential_names):
            vals = per_potential_data.get(pot, {}).get(metric, [])
            if vals:
                pot_data.append(vals)
                pot_labels.append(_shorten_pot_name(pot))
                pot_clrs.append(_pot_color(pot, i))

        if pot_data:
            parts = ax.violinplot(
                pot_data,
                positions=range(len(pot_data)),
                showmedians=True,
                showextrema=True,
            )
            for j, body in enumerate(parts["bodies"]):
                body.set_facecolor(pot_clrs[j])
                body.set_alpha(0.6)
            for pn in ("cbars", "cmins", "cmaxes", "cmedians"):
                if pn in parts:
                    parts[pn].set_color("black")
                    parts[pn].set_linewidth(0.8)

            ax.set_xticks(range(len(pot_labels)))
            # Rotation + bottom margin (0.18) ensures labels never hit the footer
            ax.set_xticklabels(
                pot_labels, rotation=X_TICK_ROTATION, ha="right", fontsize=TICK_FS,
            )

            # Collect legend handles once (first subplot with data)
            if ax_idx == 0 and not legend_handles:
                for lbl, clr in zip(pot_labels, pot_clrs):
                    legend_handles.append(
                        mpatches.Patch(facecolor=clr, alpha=0.6, label=lbl)
                    )
                    legend_labels.append(lbl)
        else:
            ax.text(
                0.5, 0.5, "no data",
                ha="center", va="center",
                transform=ax.transAxes, fontsize=AXIS_FS,
            )
            ax.set_xticks([])

        # pad=8: gives the title room above the axes without touching suptitle
        ax.set_title(mtitle, fontsize=TITLE_FS, pad=8)
        ax.set_ylabel(f"|error| ({unit})", fontsize=AXIS_FS)
        ax.tick_params(axis="y", labelsize=TICK_FS)
        ax.spines[["top", "right"]].set_visible(False)

    # Legend in the fixed right margin (right=0.82 → 18% space)
    # bbox_to_anchor is in axes-fraction coords of the last axes
    if legend_handles:
        axes[-1].legend(
            legend_handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=TICK_FS,
            framealpha=0.8,
            borderaxespad=0.0,
        )

    # exact=True → bbox_inches omitted → canvas stays exactly FIG_WIDTH × FIG_HEIGHT
    img = _fig_to_rl_image(fig, exact=True)

    # --- Stats table (one row per potential) ---
    _tbl_title_sty = ParagraphStyle(
        "fg_tbl_title",
        fontName="Helvetica-Bold",
        fontSize=TITLE_FS,
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    tbl_heading = Paragraph(
        f"{group_name}  <font size='{AXIS_FS}'><i>{_escape_mpl(smarts)}</i></font>",
        _tbl_title_sty,
    )

    stat_headers = [
        "Potential", "N",
        "Bond (\u00c5)\nmean\u00b1std (max)",
        "Angle (\u00b0)\nmean\u00b1std (max)",
        "Torsion (\u00b0)\nmean\u00b1std (max)",
    ]
    # Only N is truly numeric; the combined columns are pre-formatted strings
    numeric_stat_cols = {1}
    stat_rows: list[list[Any]] = []
    for pot in potential_names:
        pd_pot = per_potential_data.get(pot, {})
        bond_vals = pd_pot.get("bond", [])
        n = len(bond_vals)
        row: list[Any] = [pot, n]
        for metric, decimals in (("bond", 2), ("angle", 2), ("torsion", 2)):
            vals = pd_pot.get(metric, [])
            if not vals:
                row.append("—")
            else:
                arr = np.array(vals, dtype=float)
                d = decimals
                row.append(
                    f"{np.mean(arr):.{d}f}\u00b1{np.std(arr):.{d}f} ({np.max(arr):.{d}f})"
                )
        stat_rows.append(row)

    stat_table_flowables = _make_rl_table(
        stat_headers, stat_rows, numeric_cols=numeric_stat_cols,
    )

    # Metadata caption below the violin image
    n_mol  = metadata.get("n_molecules",  "?")
    n_conf = metadata.get("n_conformers", "?")
    hybrid = metadata.get("hybridization", "")
    geom   = metadata.get("geometry",     "")
    dset   = metadata.get("dataset",      "")
    caption_text = (
        f"<i>n_molecules={n_mol}  |  n_conformers={n_conf}  |  "
        f"hybridization={hybrid}  |  geometry={geom}  |  dataset={dset}</i>"
    )
    caption = Paragraph(
        caption_text,
        ParagraphStyle(
            "fg_caption",
            fontName="Helvetica-Oblique",
            fontSize=META_FS,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#444444"),
            spaceAfter=4,
        ),
    )
    return [tbl_heading, Spacer(1, 4)] + stat_table_flowables + [Spacer(1, 8), img, caption]


def render_groups_summary_table(
    groups: list[dict[str, Any]],
    potential_names: list[str],
) -> list:
    """Single-page table: all functional groups × all potentials.

    Rows: (group_name, potential) pairs.
    Columns: Group | Potential | N | Bond mean±std (max) | Angle … | Torsion …
    Groups are visually separated by alternating background shading.
    """
    if not groups:
        return []

    headers = [
        "Functional Group", "Potential", "N",
        "Bond (\u00c5)\nmean\u00b1std (max)",
        "Angle (\u00b0)\nmean\u00b1std (max)",
        "Torsion (\u00b0)\nmean\u00b1std (max)",
    ]
    rows: list[list[Any]] = []
    group_row_ranges: list[tuple[int, int]] = []   # (start, end) row indices per group
    for grp in groups:
        g_start = len(rows)
        grp_name = grp["group_name"]
        per_pot  = grp["per_potential_data"]
        for pi, pot in enumerate(potential_names):
            pd_pot = per_pot.get(pot, {})
            bond_v = pd_pot.get("bond", [])
            n = len(bond_v)
            row: list[Any] = [grp_name if pi == 0 else "", pot, n]
            for metric, d in (("bond", 3), ("angle", 2), ("torsion", 2)):
                vals = pd_pot.get(metric, [])
                if not vals:
                    row.append("—")
                else:
                    arr = np.array(vals, dtype=float)
                    row.append(
                        f"{np.mean(arr):.{d}f}\u00b1{np.std(arr):.{d}f} ({np.max(arr):.{d}f})"
                    )
            rows.append(row)
        group_row_ranges.append((g_start, len(rows) - 1))

    # Build table with group-level shading stripes
    _SHADES = [colors.HexColor("#FFFFFF"), colors.HexColor("#EEF3F8")]
    _hdr_sty = ParagraphStyle("fg_sum_th", fontName="Helvetica-Bold",
                               fontSize=_TABLE_FONT_SIZE, alignment=TA_CENTER,
                               leading=_TABLE_FONT_SIZE + 2)
    _td_l    = ParagraphStyle("fg_sum_td",   fontName="Helvetica",
                               fontSize=_TABLE_FONT_SIZE, alignment=TA_LEFT,
                               leading=_TABLE_FONT_SIZE + 2)
    _td_r    = ParagraphStyle("fg_sum_td_r", fontName="Helvetica",
                               fontSize=_TABLE_FONT_SIZE, alignment=TA_RIGHT,
                               leading=_TABLE_FONT_SIZE + 2)

    hdr_row = [Paragraph(h, _hdr_sty) for h in headers]
    body = []
    for row in rows:
        body.append([
            Paragraph(str(row[0]), _td_l),   # group name
            Paragraph(str(row[1]), _td_l),   # potential
            Paragraph(str(row[2]), _td_r),   # N
            Paragraph(str(row[3]), _td_l),   # bond
            Paragraph(str(row[4]), _td_l),   # angle
            Paragraph(str(row[5]), _td_l),   # torsion
        ])

    ts_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), _HDR_BG),
        ("GRID",       (0, 0), (-1, -1), 0.25, _GRID_CLR),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
    for gi, (r0, r1) in enumerate(group_row_ranges):
        bg = _SHADES[gi % 2]
        ts_cmds.append(("BACKGROUND", (0, r0 + 1), (-1, r1 + 1), bg))

    col_w = _MAX_TABLE_W / len(headers)
    tbl = Table([hdr_row] + body, colWidths=[col_w] * len(headers), repeatRows=1)
    tbl.setStyle(TableStyle(ts_cmds))

    sty = getSampleStyleSheet()
    return [
        Paragraph("<b>Per-Potential Error Summary — All Functional Groups</b>",
                  sty["Heading3"]),
        Spacer(1, 6),
        tbl,
        PageBreak(),
    ]


def render_param_pages(
    param_data: dict[str, dict[str, Any]],
    potential_names: list[str],
    dataset_name: str = "",
    min_samples: int = 30,
) -> list:
    """One distribution page per FF parameter (or element-type if no FF available).

    Pages show actual values (left, with QM ref) and errors (right) side by side,
    sorted by total error sample count descending.  Parameters with fewer than
    *min_samples* error values across all potentials are skipped.
    """
    flowables: list = []
    sty = getSampleStyleSheet()

    _PARAM_SPECS = [
        ("bond",    "\u00c5", "Bond"),
        ("angle",   "\u00b0", "Angle"),
        ("torsion", "\u00b0", "Torsion"),
    ]

    for ptype, unit, plabel in _PARAM_SPECS:
        pdict = param_data.get(ptype, {})
        if not pdict:
            continue

        sorted_keys = sorted(
            pdict.keys(),
            key=lambda k: sum(len(v) for v in pdict[k]["errors"].values()),
            reverse=True,
        )

        flowables.append(
            Paragraph(f"<b>{plabel} Parameters — per FF Parameter</b>",
                      sty["Heading2"])
        )
        flowables.append(Spacer(1, 4))

        for pkey in sorted_keys:
            entry    = pdict[pkey]
            err_dict = entry["errors"]
            act_dict = entry["actuals"]
            qm_ref   = entry["qm_ref"]
            smirks   = entry.get("smirks", "")
            elem_key = entry.get("elem_key", "")

            total_err = sum(len(v) for v in err_dict.values())
            if total_err < min_samples:
                continue

            # Title: "Bond: b14  [#6X4:1]-[#1:2]"
            if smirks:
                page_label = f"{plabel}: {pkey}  {smirks}"
            else:
                page_label = f"{plabel}: {pkey}"

            img = render_distribution_page(
                per_potential_errors=err_dict,
                potential_names=potential_names,
                label=page_label,
                unit=unit,
                dataset_name=dataset_name,
                per_potential_actuals=act_dict,
                qm_ref_values=qm_ref,
                per_potential_actuals_all=entry.get('actuals_all'),
                qm_ref_all=entry.get('qm_ref_all') or None,
            )
            if img:
                flowables.append(img)
                n_vals = {p: len(err_dict.get(p, [])) for p in potential_names}
                caption_parts = [f"{_shorten_pot_name(p)}: n={n_vals[p]}"
                                 for p in potential_names if n_vals[p]]
                caption_txt = "  |  ".join(caption_parts)
                smirks_part = f"&nbsp;·&nbsp; <tt>{smirks}</tt>" if smirks else ""
                flowables.append(
                    Paragraph(
                        f"<i>{pkey}{smirks_part} &nbsp;·&nbsp; {caption_txt}</i>",
                        ParagraphStyle("fg_pk_cap", fontName="Helvetica-Oblique",
                                       fontSize=META_FS, alignment=TA_CENTER,
                                       textColor=colors.HexColor("#444444"),
                                       spaceAfter=4),
                    )
                )
                flowables += _param_error_stats_table(err_dict, potential_names, unit)
                flowables.append(PageBreak())

    return flowables


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def build_report(
    output_path: str,
    title: str,
    potential_names: list[str],
    n_molecules: int,
    stats_df: pd.DataFrame,
    worst_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    distribution_data: dict[str, dict[str, list[float]]],
    groups: list[dict[str, Any]],
    dataset_name: str = "",
    param_data: dict | None = None,
) -> None:
    """Assemble the complete PDF using reportlab SimpleDocTemplate.

    Parameters
    ----------
    distribution_data : dict[metric, dict[str, list[float]]]
        metric keys: 'bond', 'angle', 'torsion' (errors),
        'bond_actual'/'angle_actual'/'torsion_actual' (per-potential actual values),
        'bond_qm'/'angle_qm'/'torsion_qm' (QM reference values).
    groups : list of dicts, each with keys:
        group_name, smarts, metadata, per_potential_data
    param_data : optional dict from _build_param_data; keys 'bond', 'angle', 'torsion'
        mapping element-type key → {'qm_ref': [], 'errors': {pot: []}, 'actuals': {pot: []}}
    """
    doc = SimpleDocTemplate(
        output_path,
        pagesize=PAGE_SIZE,
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )

    story: list = []

    # 1. Title page
    story += render_title_page(title, potential_names, n_molecules, dataset_name)

    # 2–4. Table pages (pure reportlab — no matplotlib)
    story += render_table_page("Overall Summary Statistics", stats_df)
    story += render_table_page("Worst-Case Molecule IDs", worst_df)
    story += render_table_page("Threshold-Crossing Subset", threshold_df)

    # 5. Functional-group summary table (one page)
    story += render_groups_summary_table(groups, potential_names)

    # 6. Three distribution pages (bond / angle / torsion) with actual + error side-by-side
    for metric_key, unit, label in [
        ("bond",    "\u00c5", "Bond"),
        ("angle",   "\u00b0", "Angle"),
        ("torsion", "\u00b0", "Torsion"),
    ]:
        per_pot_actuals = distribution_data.get(f"{metric_key}_actual")
        qm_ref_values = distribution_data.get(f"{metric_key}_qm")
        img = render_distribution_page(
            distribution_data.get(metric_key, {}),
            potential_names, label, unit, dataset_name,
            per_potential_actuals=per_pot_actuals,
            qm_ref_values=qm_ref_values,
        )
        if img:
            story.append(img)
            story.append(PageBreak())

    # 7. Per-element-type parameter distribution pages
    if param_data:
        story += render_param_pages(param_data, potential_names, dataset_name)

    # 8. One page per functional group — each gets its own page break
    for grp in groups:
        flowables = render_group_page(
            group_name=grp["group_name"],
            smarts=grp["smarts"],
            metadata=grp["metadata"],
            per_potential_data=grp["per_potential_data"],
            potential_names=potential_names,
        )
        story.extend(flowables)
        story.append(PageBreak())

    doc.build(story)
    print(f"Report written → {output_path}")


# ---------------------------------------------------------------------------
# __main__ with mock data — run as-is: python fg_report.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    random.seed(42)
    np.random.seed(42)

    POT_NAMES = [
        "openff-2.3.0",
        "aceff-2.0",
        "mace-off24-medium",
        "fennix-bio1M",
        "aimnet2",
    ]

    POTENTIAL_COLORS.update({
        "openff-2.3.0":      "#4C72B0",
        "aceff-2.0":         "#DD8452",
        "mace-off24-medium": "#55A868",
        "fennix-bio1M":      "#C44E52",
        "aimnet2":           "#8172B2",
    })

    _FG_DEFS = [
        ("Phosphate ester",  "[OX2][PX4](=O)([OX2])[OX2]",         "sp3", "tetrahedral"),
        ("Sulfoxide",        "[SX3](=O)",                            "sp3", "trigonal-pyramidal"),
        ("Carboxylate",      "[CX3](=O)[OX1-]",                     "sp2", "trigonal-planar"),
        ("Amide",            "[CX3](=O)[NX3H]",                     "sp2", "trigonal-planar"),
        ("Guanidinium",      "[NX3H2][CX3]([NX3H2])=[NX3H]",        "sp2", "trigonal-planar"),
        ("Imidazole",        "c1cn[nH]c1",                          "sp2", "aromatic"),
        ("Sulfonamide",      "[SX4](=O)(=O)[NX3]",                  "sp3", "tetrahedral"),
        ("Phosphonate",      "[PX4](=O)([OX2])([OX2])[CX4]",        "sp3", "tetrahedral"),
    ]

    def _mock_unimodal(n: int, scale: float) -> list[float]:
        return list(np.abs(np.random.normal(0.0, scale, n)))

    def _mock_bimodal(n: int, mu1: float, mu2: float) -> list[float]:
        h = n // 2
        return list(np.abs(np.concatenate([
            np.random.normal(mu1, mu1 * 0.3, h),
            np.random.normal(mu2, mu2 * 0.3, n - h),
        ])))

    # --- Stats DataFrames ---
    stats_rows = [
        {
            "Potential":      pot,
            "N_conformers":   random.randint(800, 2000),
            "RMSD_mean":      round(random.uniform(0.02, 0.2),  4),
            "RMSD_max":       round(random.uniform(0.3,  1.0),  4),
            "Bond_mean_Å":    round(random.uniform(0.005, 0.05), 4),
            "Bond_max_Å":     round(random.uniform(0.1,  0.3),  4),
            "Angle_mean_deg": round(random.uniform(1,    8),    2),
            "Angle_max_deg":  round(random.uniform(10,  40),    2),
            "Tors_mean_deg":  round(random.uniform(5,   30),    2),
            "Tors_max_deg":   round(random.uniform(50, 120),    2),
        }
        for pot in POT_NAMES
    ]
    stats_df = pd.DataFrame(stats_rows)

    worst_df = pd.DataFrame([
        {
            "Potential":    p,
            "RMSD_max_ID":  f"mol_{random.randint(1, 999)}",
            "Bond_max_ID":  f"mol_{random.randint(1, 999)}",
            "Angle_max_ID": f"mol_{random.randint(1, 999)}",
            "Tors_max_ID":  f"mol_{random.randint(1, 999)}",
        }
        for p in POT_NAMES
    ])

    threshold_df = pd.DataFrame([
        {
            "Potential":       p,
            "N_thresh_bond":   random.randint(10, 200),
            "N_thresh_angle":  random.randint(10, 200),
            "N_thresh_torsion":random.randint(10, 200),
            "RMSD_mean":       round(random.uniform(0.1, 0.5), 4),
        }
        for p in POT_NAMES
    ])

    _bond_centers    = list(np.random.uniform(1.3, 1.6, 500))
    _angle_centers   = list(np.random.uniform(100, 130, 500))
    _torsion_centers = list(np.random.uniform(-180, 180, 500))

    dist_data: dict[str, dict[str, list[float]]] = {
        "bond":         {p: _mock_unimodal(500, 0.020) for p in POT_NAMES},
        "angle":        {p: _mock_bimodal(500, 3.0, 10.0) for p in POT_NAMES},
        "torsion":      {p: _mock_bimodal(500, 10.0, 40.0) for p in POT_NAMES},
        "bond_qm":      _bond_centers,
        "angle_qm":     _angle_centers,
        "torsion_qm":   _torsion_centers,
        "bond_actual":  {p: [v + np.random.normal(0, 0.015) for v in _bond_centers]    for p in POT_NAMES},
        "angle_actual": {p: [v + np.random.normal(0, 3.0)   for v in _angle_centers]   for p in POT_NAMES},
        "torsion_actual": {p: [v + np.random.normal(0, 12.0) for v in _torsion_centers] for p in POT_NAMES},
    }

    _FF_BONDS = [
        ("b1",  "[#6:1]-[#6:2]",          1.52, 0.03, 0.012),
        ("b2",  "[#6:1]-[#8:2]",          1.43, 0.02, 0.010),
        ("b3",  "[#6:1]-[#7:2]",          1.47, 0.03, 0.011),
        ("b4",  "[#6:1]=[#8:2]",          1.23, 0.02, 0.008),
        ("b5",  "[#6:1]=[#6:2]",          1.34, 0.02, 0.009),
        ("b14", "[#6X4:1]-[#1:2]",        1.09, 0.01, 0.006),
        ("b16", "[#7:1]-[#1:2]",          1.01, 0.01, 0.007),
        ("b84", "[#15:1]-[#8:2]",         1.61, 0.03, 0.015),
    ]
    _FF_ANGLES = [
        ("a1",  "[*:1]~[#6X4:2]-[*:3]",       109.5, 6.0, 3.0),
        ("a2",  "[*:1]~[#6X3:2]~[*:3]",       120.0, 5.0, 2.5),
        ("a3",  "[#1:1]-[#6X4:2]-[#1:3]",     107.8, 2.0, 1.5),
        ("a4",  "[#6:1]-[#6:2]-[#8:3]",       111.5, 5.0, 2.8),
        ("a5",  "[#6:1]-[#7:2]-[#6:3]",       117.0, 6.0, 3.2),
        ("a27", "[#15:1]-[#8:2]-[#6:3]",      119.5, 5.0, 3.5),
        ("a31", "[#8:1]-[#15:2]-[#8:3]",      112.0, 4.0, 3.0),
    ]
    _FF_TORS = [
        ("t1",  "[*:1]-[#6X4:2]-[#6X4:3]-[*:4]",  0.0, 60.0, 10.0),
        ("t2",  "[*:1]-[#6X3:2]~[#6X3:3]-[*:4]",  0.0, 30.0,  8.0),
        ("t3",  "[#1:1]-[#6X4:2]-[#6X4:3]-[#1:4]",0.0, 50.0, 11.0),
        ("t4",  "[*:1]-[#6X4:2]-[#8:3]-[*:4]",    0.0, 55.0, 12.0),
        ("t12", "[*:1]-[#6:2]-[#7:3]-[*:4]",      0.0, 45.0,  9.0),
    ]

    def _mock_param_bucket(n: int, center: float, spread: float, err_scale: float, smirks: str):
        qm_ref = list(np.random.normal(center, spread, n))
        errors = {p: [np.random.normal(0, err_scale) for _ in range(n)] for p in POT_NAMES}
        actuals = {p: [qm_ref[i] + errors[p][i] for i in range(n)] for p in POT_NAMES}
        return {"qm_ref": qm_ref, "errors": errors, "actuals": actuals, "smirks": smirks}

    param_data: dict[str, dict] = {
        "bond":    {pid: _mock_param_bucket(random.randint(50, 200), ctr, spd, err, sm)
                    for pid, sm, ctr, spd, err in _FF_BONDS},
        "angle":   {pid: _mock_param_bucket(random.randint(50, 200), ctr, spd, err, sm)
                    for pid, sm, ctr, spd, err in _FF_ANGLES},
        "torsion": {pid: _mock_param_bucket(random.randint(50, 200), ctr, spd, err, sm)
                    for pid, sm, ctr, spd, err in _FF_TORS},
    }

    groups: list[dict[str, Any]] = []
    for fg_name, fg_smarts, hybrid, geom in _FG_DEFS:
        n_mol  = random.randint(10, 80)
        n_conf = n_mol * random.randint(3, 15)
        groups.append({
            "group_name": fg_name,
            "smarts":     fg_smarts,
            "metadata": {
                "n_molecules":   n_mol,
                "n_conformers":  n_conf,
                "hybridization": hybrid,
                "geometry":      geom,
                "dataset":       "OpenFF_NSP_v4",
            },
            "per_potential_data": {
                pot: {
                    "bond":    _mock_unimodal(n_conf, 0.015 + random.uniform(0, 0.04)),
                    "angle":   _mock_bimodal(
                        n_conf,
                        2.0 + random.uniform(0, 3),
                        8.0 + random.uniform(0, 5),
                    ),
                    "torsion": _mock_bimodal(
                        n_conf,
                        8.0  + random.uniform(0, 10),
                        35.0 + random.uniform(0, 20),
                    ),
                }
                for pot in POT_NAMES
            },
        })

    build_report(
        output_path="multi_page_report_demo.pdf",
        title="MLIP Geometry Benchmark",
        potential_names=POT_NAMES,
        n_molecules=sum(g["metadata"]["n_molecules"] for g in groups),
        stats_df=stats_df,
        worst_df=worst_df,
        threshold_df=threshold_df,
        distribution_data=dist_data,
        groups=groups,
        dataset_name="OpenFF_NSP_Optimization_Set_1_Phosphorus_v4_0",
        param_data=param_data,
    )
