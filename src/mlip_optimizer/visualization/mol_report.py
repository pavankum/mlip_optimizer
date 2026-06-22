"""Per-molecule QM benchmark PDF report.

One landscape(letter) page per molecule:

  ┌────────────────┬────────────────────────────────────────────────┐
  │  2-D structure │  mol_42  ·  3 conformers  ·  openff, aceff …  │
  │  with atom     │  SMILES: CC(=O)O                               │
  │  indices       │  QCA IDs: 12345, 12346, 12347                  │
  │                ├────────────────────────────────────────────────┤
  │                │  RMSD vs QM Reference (Å)  [table]             │
  │                ├────────────────────────────────────────────────┤
  │                │  Bond diffs > threshold  [table]               │
  │                │  Angle diffs > threshold [table]               │
  │                │  Torsion diffs > threshold [table]             │
  └────────────────┴────────────────────────────────────────────────┘

Entry point
-----------
    build_report(output_path, title, records, qm_results, potential_names)

Demo
----
    python mol_report.py
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle
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
from reportlab.platypus.flowables import KeepInFrame

# ---------------------------------------------------------------------------
# Page geometry constants
# ---------------------------------------------------------------------------

PAGE_SIZE = landscape(letter)                   # 792 × 612 pt
_PAGE_W, _PAGE_H = PAGE_SIZE

MOL_IMG_PX = (1400, 900)                        # rdMolDraw2D canvas W × H pixels

_MARGIN     = 0.35 * inch                       # 25.2 pt — all four sides
_IMG_H_FRAC = 0.42                              # image takes this fraction of available height

_AVAIL_W = _PAGE_W - 2.0 * _MARGIN
_AVAIL_H = _PAGE_H - 2.0 * _MARGIN

# Image: height-constrained, centered horizontally
_IMG_RL_H = _AVAIL_H * _IMG_H_FRAC
_IMG_RL_W = _IMG_RL_H * MOL_IMG_PX[0] / MOL_IMG_PX[1]

# Table styling
_TABLE_FONT_SIZE = 7
_HDR_BG   = colors.HexColor("#D3D3D3")
_ALT_BG   = colors.HexColor("#F0F4F8")
_GRID_CLR = colors.HexColor("#BBBBBB")
_ACCENT   = colors.HexColor("#2C5F8A")

# ---------------------------------------------------------------------------
# Paragraph styles (defined once at module level)
# ---------------------------------------------------------------------------

_HDR_STY = ParagraphStyle(
    "mol_hdr",
    fontName="Helvetica-Bold",
    fontSize=10,
    textColor=_ACCENT,
    spaceAfter=3,
    leading=13,
)
_SUBHDR_STY = ParagraphStyle(
    "mol_subhdr",
    fontName="Helvetica-Bold",
    fontSize=8,
    spaceBefore=6,
    spaceAfter=2,
    leading=10,
)
_CAPTION_STY = ParagraphStyle(
    "mol_caption",
    fontName="Helvetica",
    fontSize=6.5,
    textColor=colors.HexColor("#444444"),
    spaceAfter=3,
    leading=9,
    wordWrap="CJK",
)
_NOTE_STY = ParagraphStyle(
    "mol_note",
    fontName="Helvetica-Oblique",
    fontSize=7,
    textColor=colors.HexColor("#666666"),
    spaceAfter=2,
    leading=9,
)
_TH_STY = ParagraphStyle(
    "mol_th",
    fontName="Helvetica-Bold",
    fontSize=_TABLE_FONT_SIZE,
    alignment=TA_CENTER,
    leading=_TABLE_FONT_SIZE + 2,
)
_TD_STY = ParagraphStyle(
    "mol_td",
    fontName="Helvetica",
    fontSize=_TABLE_FONT_SIZE,
    alignment=TA_LEFT,
    leading=_TABLE_FONT_SIZE + 2,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fmt_key(key: Any) -> str:
    """Format a param-key tuple (atom indices) as '0-1', '0-1-2', etc."""
    if isinstance(key, (tuple, list)):
        return "-".join(str(k) for k in key)
    return str(key)


def _trunc(s: str, n: int = 18) -> str:
    """Truncate a string to n characters, adding '…' if clipped."""
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def _make_compact_table(
    headers: list[str],
    rows: list[list[Any]],
    max_w: float | None = None,
) -> list:
    """Build a compact reportlab Table, splitting into column groups when too wide.

    The key column (index 0) is repeated as the first column of each group so
    the user can match rows across groups.  Column widths are distributed
    proportionally to the estimated character width of each column.
    """
    if max_w is None:
        max_w = _AVAIL_W

    if not rows:
        return [Paragraph("<i>(none)</i>", _NOTE_STY)]

    n_cols = len(headers)
    _char_pt = _TABLE_FONT_SIZE * 0.55
    max_chars = [
        max(len(str(headers[c])), *(len(str(row[c])) for row in rows))
        for c in range(n_cols)
    ]
    col_pts = [_char_pt * mc for mc in max_chars]

    def _ts_cmds(nrows: int) -> list:
        cmds = [
            ("BACKGROUND",    (0, 0), (-1, 0),  _HDR_BG),
            ("GRID",          (0, 0), (-1, -1), 0.25, _GRID_CLR),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ]
        if nrows > 4:
            for ri in range(2, nrows + 1, 2):
                cmds.append(("BACKGROUND", (0, ri), (-1, ri), _ALT_BG))
        return cmds

    def _build_sub(col_indices: list[int]) -> Table:
        sub_pts = [col_pts[ci] for ci in col_indices]
        sub_total = max(sum(sub_pts), 1.0)
        col_widths = [max_w * p / sub_total for p in sub_pts]

        # Enforce a minimum that clears reportlab's default left+right padding
        # (6 + 6 = 12 pt).  If clamping pushes total over max_w, scale back
        # the excess from columns that are still above the floor.
        _MIN_W = 20.0
        col_widths = [max(w, _MIN_W) for w in col_widths]
        over = sum(col_widths) - max_w
        if over > 0:
            shrinkable = sum(w - _MIN_W for w in col_widths)
            if shrinkable > 0:
                col_widths = [
                    max(_MIN_W, w - (w - _MIN_W) / shrinkable * over)
                    for w in col_widths
                ]

        hdr_row = [Paragraph(str(headers[ci]), _TH_STY) for ci in col_indices]

        def _cell(row: list, ci: int) -> Paragraph:
            val = row[ci]
            txt = _fmt_key(val) if isinstance(val, (tuple, list)) else str(val)
            return Paragraph(txt, _TD_STY)

        body = [[_cell(row, ci) for ci in col_indices] for row in rows]
        data = [hdr_row] + body
        t = Table(data, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle(_ts_cmds(len(rows))))
        return t

    total_w = sum(col_pts)
    flowables: list = []

    if total_w <= max_w * 1.05:
        flowables.append(_build_sub(list(range(n_cols))))
    else:
        # Keep key col (0) in every group; partition remaining cols by budget
        key_w = col_pts[0]
        budget = max_w - key_w
        group: list[int] = []
        group_w = 0.0
        for ci in range(1, n_cols):
            cw = col_pts[ci]
            if group and group_w + cw > budget:
                flowables.append(_build_sub([0] + group))
                flowables.append(Spacer(1, 3))
                group = []
                group_w = 0.0
            group.append(ci)
            group_w += cw
        if group:
            flowables.append(_build_sub([0] + group))

    return flowables


# ---------------------------------------------------------------------------
# Public API — page element builders
# ---------------------------------------------------------------------------


def render_molecule_png(molecule) -> bytes:
    """Return PNG bytes for *molecule* rendered with rdMolDraw2D.MolDraw2DCairo.

    Always recomputes 2-D coordinates from topology for a clean, axis-aligned
    depiction regardless of any 3-D conformers stored on the molecule.
    Atom indices are shown as annotations.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D

    rdmol = molecule.to_rdkit()

    # Store original atom indices on heavy atoms before stripping H so the
    # annotations still show the index in the full molecule (useful for
    # cross-referencing bond/angle tables).
    for atom in rdmol.GetAtoms():
        atom.SetIntProp("_origIdx", atom.GetIdx())

    rdmol = Chem.RemoveHs(rdmol)

    # CoordGen handles complex ring systems and charged atoms better than the
    # default distance-geometry 2D layout; fall back silently if unavailable.
    try:
        rdDepictor.SetPreferCoordGen(True)
    except AttributeError:
        pass
    AllChem.Compute2DCoords(rdmol)

    # Annotate with original (pre-H-removal) indices.
    for atom in rdmol.GetAtoms():
        atom.SetProp("atomNote", str(atom.GetIntProp("_origIdx")))

    w, h = MOL_IMG_PX
    drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
    opts = drawer.drawOptions()
    opts.annotationFontScale = 0.55
    try:
        opts.baseFontSize = 0.6
    except AttributeError:
        pass
    opts.bondLineWidth = 2

    drawer.DrawMolecule(rdmol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def build_header(
    mol_label: str,
    n_conf: int,
    potential_names: list[str],
) -> list:
    """Return flowables for the compact one-line header strip."""
    confs = f"{n_conf} conformer{'s' if n_conf != 1 else ''}"
    pot_str = ", ".join(potential_names)
    text = f"<b>{mol_label}</b>  ·  {confs}  ·  {pot_str}"
    return [
        Paragraph(text, _HDR_STY),
        HRFlowable(
            width="100%", thickness=0.5,
            color=colors.HexColor("#888888"), spaceAfter=3,
        ),
    ]


def build_caption(
    smiles: str,
    qca_ids: list[int],
    fg_text: str | None = None,
) -> list:
    """Return flowables for the SMILES + QCA IDs + functional groups caption."""
    ids_str = ", ".join(str(i) for i in qca_ids) if qca_ids else "—"
    text = f"<b>SMILES:</b> {smiles}<br/><b>QCA IDs:</b> {ids_str}"
    if fg_text:
        text += f"<br/><b>Functional groups:</b> {fg_text}"
    return [Paragraph(text, _CAPTION_STY)]


def build_ring_planarity_table(
    ring_planarity_table: list[list],
    potential_names: list[str],
) -> list:
    """Return flowables for the ring planarity deviation table.

    Columns: Ring atoms | n | QM (Å) | Pot1 (Å) | Pot2 (Å) | ...
    Each cell shows mean±std across conformers.
    """
    if not ring_planarity_table:
        return [Paragraph("<i>No rings detected.</i>", _NOTE_STY)]

    headers = ["Ring atoms", "n"] + ["QM dev (Å)"] + [f"{_trunc(p, 14)} dev (Å)" for p in potential_names]
    rows = [[str(r[0]), str(r[1])] + [str(c) for c in r[2:]] for r in ring_planarity_table]
    return _make_compact_table(headers, rows)


def build_rmsd_table(
    per_potential: dict,
    potential_names: list[str],
) -> list:
    """Return flowables for the per-potential RMSD summary table.

    Columns: Potential | N conf | RMSD mean±std (max) Å
    """
    headers = ["Potential", "N conf", "RMSD mean\u00b1std (max) \u00c5"]
    rows: list[list] = []
    for pot in potential_names:
        metrics_list = per_potential.get(pot, [])
        valid = [m for m in metrics_list if not getattr(m, "opt_failed", False)]
        if not valid:
            rows.append([_trunc(pot, 28), "0", "FAILED"])
            continue
        rmsds = [m.rmsd for m in valid if not np.isnan(m.rmsd)]
        if not rmsds:
            rows.append([_trunc(pot, 28), str(len(valid)), "N/A"])
            continue
        rows.append([
            _trunc(pot, 28),
            str(len(valid)),
            f"{np.mean(rmsds):.4f}\u00b1{np.std(rmsds):.4f} ({max(rmsds):.4f})",
        ])
    return _make_compact_table(headers, rows)


def build_threshold_table(
    label: str,
    diff_table: list[list],
    potential_names: list[str],
    per_potential: dict | None = None,
    values_attr: str = "",
    unit: str = "",
    decimals: int = 3,
) -> list:
    """Return flowables for a threshold diff table (bonds / angles / torsions).

    *diff_table* rows: [param_key, QM_ref_str, pot1_diff_str, ..., [param_id, smirks]]

    When *per_potential* and *values_attr* are supplied, each potential cell
    shows ``actual (mean,max)`` — or ``actual±std (mean,max)`` when multiple
    conformers produce meaningful spread. ``mean`` is the per-key mean
    difference across conformers and ``max`` is the conformer-level max for
    that same key. Param_id / smirks columns are excluded.
    """
    if not diff_table:
        return [
            Paragraph(
                f"<i>No {label} differences exceed threshold.</i>",
                _NOTE_STY,
            )
        ]

    n_pots = len(potential_names)
    unit_str = f" ({unit})" if unit else ""
    fmt = f".{decimals}f"
    tol = 0.5 * 10 ** (-decimals)          # threshold below which std is suppressed
    diff_attr = {
        "bond": "bond_diffs",
        "angle": "angle_diffs",
        "torsion": "torsion_diffs",
    }.get(label)

    def _mean_only(s: str) -> str:
        for sep in (" ± ", "±", " +/- "):
            if sep in s:
                return s.split(sep)[0].strip()
        return s.strip()

    headers = [f"Key{unit_str}", f"QM Ref{unit_str}"]
    for p in potential_names:
        headers.append(_trunc(p, 16))

    n_base = 2 + n_pots
    rows: list[list] = []
    for row in diff_table:
        key = row[0]
        key_str = _fmt_key(key)
        if len(row) > n_base and row[n_base]:
            key_str = f"{key_str} ({row[n_base]})"
        qm_mean = _mean_only(str(row[1]))
        cells: list = [key_str, qm_mean]

        for pi, pot in enumerate(potential_names):
            diff_mean = _mean_only(str(row[2 + pi]))
            diff_summary = diff_mean

            actual_cell = None
            if per_potential and values_attr:
                metrics_list = per_potential.get(pot, [])
                diff_vals = []
                if diff_attr:
                    diff_vals = [
                        getattr(m, diff_attr, {}).get(key)
                        for m in metrics_list
                        if not getattr(m, "opt_failed", False)
                    ]
                    diff_vals = [
                        v for v in diff_vals if v is not None and not np.isnan(v)
                    ]
                if diff_vals:
                    diff_max = float(np.max(diff_vals))
                    diff_summary = f"{diff_mean},{diff_max:{fmt}}"
                vals = [
                    getattr(m, values_attr, {}).get(key)
                    for m in metrics_list
                    if not getattr(m, "opt_failed", False)
                ]
                vals = [v for v in vals if v is not None]
                if vals:
                    mean_v = float(np.mean(vals))
                    std_v  = float(np.std(vals))
                    if len(vals) > 1 and std_v >= tol:
                        actual_cell = f"{mean_v:{fmt}}±{std_v:{fmt}} ({diff_summary})"
                    else:
                        actual_cell = f"{mean_v:{fmt}} ({diff_summary})"

            cells.append(actual_cell if actual_cell is not None else diff_summary)

        rows.append(cells)

    return _make_compact_table(headers, rows)


def build_molecule_page(
    molecule,
    smiles: str,
    qca_ids: list[int],
    qm_comparison,
    potential_names: list[str],
    mol_label: str = "",
    fg_text: str | None = None,
) -> list:
    """Return a list of reportlab flowables for one molecule page.

    Layout (top → bottom):
      header strip  →  molecule image (centered)  →  caption
      →  RMSD table  →  ring planarity table
      →  bond/angle/torsion diff tables  →  PageBreak
    """
    n_conf = getattr(qm_comparison, "n_conformers", 0) if qm_comparison else 0
    label = mol_label or smiles[:30]

    story: list = []
    story.extend(build_header(label, n_conf, potential_names))
    story.extend(build_caption(smiles, qca_ids, fg_text=fg_text))
    story.append(Spacer(1, 4))

    # Molecule image — centered by wrapping in a full-width single-cell Table
    try:
        png_bytes = render_molecule_png(molecule)
        mol_img: Any = RLImage(io.BytesIO(png_bytes), width=_IMG_RL_W, height=_IMG_RL_H)
    except Exception as exc:
        mol_img = Paragraph(f"[Molecule image unavailable: {exc}]", _NOTE_STY)

    img_wrapper = Table([[mol_img]], colWidths=[_AVAIL_W])
    img_wrapper.setStyle(TableStyle([
        ("ALIGN",         (0, 0), (0, 0), "CENTER"),
        ("TOPPADDING",    (0, 0), (0, 0), 0),
        ("BOTTOMPADDING", (0, 0), (0, 0), 0),
        ("LEFTPADDING",   (0, 0), (0, 0), 0),
        ("RIGHTPADDING",  (0, 0), (0, 0), 0),
    ]))
    story.append(img_wrapper)
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=0.4, color=colors.HexColor("#BBBBBB"), spaceAfter=3))

    if qm_comparison is not None:
        story.append(Paragraph("<b>RMSD vs QM Reference (Å)</b>", _SUBHDR_STY))
        story.extend(build_rmsd_table(qm_comparison.per_potential, potential_names))

        ring_table = getattr(qm_comparison, "ring_planarity_table", [])
        if ring_table:
            story.append(Paragraph(
                "<b>Ring Planarity Deviation (Å RMSD from best-fit plane) — mean±std across conformers</b>",
                _SUBHDR_STY,
            ))
            story.extend(build_ring_planarity_table(ring_table, potential_names))

        story.append(Paragraph(
            "<b>Bond Differences (Å) — above threshold; format: actual±std (mean error, max error)</b>",
            _SUBHDR_STY,
        ))
        story.extend(build_threshold_table(
            "bond", qm_comparison.bond_diff_table, potential_names,
            qm_comparison.per_potential, "bond_values", "Å", decimals=3,
        ))

        story.append(Paragraph(
            "<b>Angle Differences (°) — above threshold; format: actual±std (mean error, max error)</b>",
            _SUBHDR_STY,
        ))
        story.extend(build_threshold_table(
            "angle", qm_comparison.angle_diff_table, potential_names,
            qm_comparison.per_potential, "angle_values", "°", decimals=2,
        ))

        story.append(Paragraph(
            "<b>Torsion Differences (°) — above threshold; format: actual±std (mean error, max error)</b>",
            _SUBHDR_STY,
        ))
        story.extend(build_threshold_table(
            "torsion", qm_comparison.torsion_diff_table, potential_names,
            qm_comparison.per_potential, "torsion_values", "°", decimals=2,
        ))
    else:
        story.append(Paragraph("<i>No comparison data available.</i>", _NOTE_STY))

    page = KeepInFrame(_AVAIL_W, _AVAIL_H, story, mode="shrink")
    return [page, PageBreak()]


# ---------------------------------------------------------------------------
# Title page
# ---------------------------------------------------------------------------


def _render_title_page(
    title: str,
    n_molecules: int,
    potential_names: list[str],
) -> list:
    _title_sty = ParagraphStyle(
        "mol_main_title",
        fontName="Helvetica-Bold",
        fontSize=20,
        alignment=TA_CENTER,
        spaceAfter=16,
        textColor=_ACCENT,
    )
    _sub_sty = ParagraphStyle(
        "mol_sub",
        fontName="Helvetica",
        fontSize=11,
        alignment=TA_CENTER,
        spaceAfter=6,
    )
    _bullet_sty = ParagraphStyle(
        "mol_bullet",
        fontName="Helvetica",
        fontSize=10,
        alignment=TA_CENTER,
        spaceAfter=3,
        leading=14,
    )

    flowables: list = [
        Spacer(1, _AVAIL_H * 0.22),
        Paragraph(title, _title_sty),
        HRFlowable(width="60%", thickness=1, color=_ACCENT, spaceAfter=14),
        Paragraph(f"Molecules: <b>{n_molecules}</b>", _sub_sty),
        Paragraph("Potentials compared against QM reference:", _sub_sty),
    ]
    for p in potential_names:
        flowables.append(Paragraph(f"  \u2022  {p}", _bullet_sty))
    flowables.append(PageBreak())
    return flowables


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_report(
    output_path: str,
    title: str,
    records,
    qm_results: list,
    potential_names: list[str],
    fg_matches_per_mol: list | None = None,
) -> None:
    """Build the per-molecule benchmark PDF.

    Parameters
    ----------
    output_path : str
        Destination PDF file path.
    title : str
        Report title shown on the cover page.
    records : sequence
        One record per molecule — must expose ``.molecule``, ``.smiles``,
        and ``.record_ids`` attributes.
    qm_results : list
        One ``QMComparisonResult`` (or ``None``) per molecule, aligned with
        *records*.
    potential_names : list[str]
        Ordered list of potential names matching ``qm_results`` columns.
    fg_matches_per_mol : list, optional
        One element per molecule: the output of
        ``functional_groups.match_and_cache()`` — a list of
        ``(fg_name, [match_tuples])`` pairs.  Pass ``None`` to omit
        functional group annotations.
    """
    from mlip_optimizer.analysis.functional_groups import format_fg_matches

    doc = SimpleDocTemplate(
        output_path,
        pagesize=PAGE_SIZE,
        leftMargin=_MARGIN,
        rightMargin=_MARGIN,
        topMargin=_MARGIN,
        bottomMargin=_MARGIN,
        title=title,
    )

    story: list = _render_title_page(title, len(records), potential_names)

    for mol_idx, (rec, qm_comp) in enumerate(zip(records, qm_results)):
        qca_ids = list(getattr(rec, "record_ids", []))
        fg_text: str | None = None
        if fg_matches_per_mol is not None and mol_idx < len(fg_matches_per_mol):
            fg_text = format_fg_matches(fg_matches_per_mol[mol_idx])
        story.extend(
            build_molecule_page(
                rec.molecule,
                rec.smiles,
                qca_ids,
                qm_comp,
                potential_names,
                mol_label=f"mol_{mol_idx}",
                fg_text=fg_text,
            )
        )

    doc.build(story)


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import dataclasses

    POTS = [
        "openff-2.3.0",
        "aceff-2.0",
        "mace-off24-medium",
        "fennix-bio1M",
        "aimnet2",
    ]

    @dataclasses.dataclass
    class _MockMetrics:
        rmsd: float
        opt_failed: bool = False
        bond_values:    dict = dataclasses.field(default_factory=dict)
        angle_values:   dict = dataclasses.field(default_factory=dict)
        torsion_values: dict = dataclasses.field(default_factory=dict)

    @dataclasses.dataclass
    class _MockResult:
        n_conformers: int
        per_potential: dict
        bond_diff_table: list = dataclasses.field(default_factory=list)
        angle_diff_table: list = dataclasses.field(default_factory=list)
        torsion_diff_table: list = dataclasses.field(default_factory=list)

    @dataclasses.dataclass
    class _MockRecord:
        molecule: object
        smiles: str
        record_ids: list

    def _mock_result(
        n_conf: int,
        n_bond: int = 0,
        n_angle: int = 0,
        n_tors: int = 0,
    ) -> _MockResult:
        rng = np.random.default_rng(42)

        def _diff_rows(n: int, ref_base: float, is_bond: bool) -> list:
            keys = [(i, i + 1) if is_bond else (i, i + 1, i + 2) for i in range(n)]
            return [
                [keys[i], f"{ref_base + i * 0.01:.3f} ± 0.001"]
                + [f"{rng.uniform(0.05, 0.25):.3f} ± {rng.uniform(0, 0.01):.3f}" for _ in POTS]
                for i in range(n)
            ]

        bond_rows  = _diff_rows(n_bond,   1.50,  True)
        angle_rows = _diff_rows(n_angle, 110.0, False)
        tors_rows  = _diff_rows(n_tors,   60.0, False)

        # Build per_potential with actual geometry values matching the diff table keys
        def _actual_vals(rows: list, ref_base: float, attr: str) -> dict:
            """dict[key → actual_value] per row, slightly offset from QM ref."""
            result = {}
            for row in rows:
                key = row[0]
                # pull mean diff back from the formatted string for the first pot
                diff_str = row[2].split(" ±")[0]
                try:
                    diff = float(diff_str)
                except ValueError:
                    diff = 0.1
                result[key] = ref_base + diff + float(rng.uniform(-0.01, 0.01))
            return result

        per_pot = {}
        for p in POTS:
            metrics = []
            for _ in range(n_conf):
                metrics.append(_MockMetrics(
                    rmsd=float(rng.uniform(0.01, 0.35)),
                    bond_values=_actual_vals(bond_rows, 1.50, "bond"),
                    angle_values=_actual_vals(angle_rows, 110.0, "angle"),
                    torsion_values=_actual_vals(tors_rows, 60.0, "torsion"),
                ))
            per_pot[p] = metrics

        return _MockResult(
            n_conformers=n_conf,
            per_potential=per_pot,
            bond_diff_table=bond_rows,
            angle_diff_table=angle_rows,
            torsion_diff_table=tors_rows,
        )

    # Three mock molecules: tiny, medium, large
    from openff.toolkit import Molecule

    _mock_specs = [
        ("CCO", [1001], 1, 0, 0, 0),
        ("CC1=CC=CC=C1CC(=O)O", [2001, 2002], 2, 3, 2, 0),
        (
            "CC(C)CC1=CC=C(C=C1)[C@@H](C)C(=O)NCCCC(=O)O",
            [3001, 3002, 3003],
            3,
            5,
            4,
            2,
        ),
    ]

    _records = []
    _results = []
    for smiles, ids, n_conf, nb, na, nt in _mock_specs:
        mol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
        mol.generate_conformers(n_conformers=1)
        _records.append(_MockRecord(molecule=mol, smiles=smiles, record_ids=ids))
        _results.append(_mock_result(n_conf, nb, na, nt))

    _out = "mol_report_demo.pdf"
    build_report(_out, "Per-Molecule QM Benchmark — Demo", _records, _results, POTS)
    print(f"Wrote {_out}")
