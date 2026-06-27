"""Triple-level last-match-wins hierarchy validator.

The central upgrade over the PDF-based skill: assignment happens at the
**atom-triple level**, not the molecule level.  Two distinct centers in the
same molecule (e.g. a ring N and a chain N both carrying parameter a20) can
land in different children.  This is possible because ``InstanceRecord.rdmol``
and ``InstanceRecord.atom_key`` share one consistent atom ordering.

For each instance the validator:

1. Iterates through the ordered hierarchy (parent first, most-specific last).
2. For each SMARTS pattern, finds ALL substructure matches in ``rdmol``.
3. Checks whether any match places the :1/:2/:3 mapped atoms at exactly
   the ``atom_key`` triple (forward or reverse).
4. The last-matching pattern wins.

Patterns without atom-map numbers fall back to molecule-level matching
(``HasSubstructMatch``), which preserves backward compatibility with
non-mapped patterns, though they should be avoided in new hierarchies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ChildStats:
    """Per-child statistics from the validator.

    Attributes
    ----------
    name : str
        Child parameter name (e.g. ``'a20a'``).
    smarts : str
    n_instances : int
    n_molecules : int
    qm_min : float
    qm_max : float
    qm_mean : float
    qm_width : float
    is_empty : bool
    is_wide : bool
        ``qm_width > target_width`` (only flagged for non-parent children).
    is_strained : bool
        Any instance in this child has a strained-ring central atom
        (ring size ≤ 4).
    instances : list[InstanceRecord]
    """

    name: str
    smarts: str
    n_instances: int
    n_molecules: int
    qm_min: float
    qm_max: float
    qm_mean: float
    qm_width: float
    is_empty: bool
    is_wide: bool
    is_strained: bool
    instances: list = field(default_factory=list, repr=False)


@dataclass
class ValidationReport:
    """Result of :func:`validate_hierarchy`.

    Attributes
    ----------
    children : list[ChildStats]
        Ordered as supplied (parent first).
    n_total : int
    n_unmatched : int
        Instances that matched no pattern (should be 0 if the parent is
        the first entry and is truly a catch-all).
    n_unparsed_mols : int
        Instances with ``rdmol=None`` (skipped).
    mean_child_width : float
        Mean ``qm_width`` across non-parent children with ≥ 1 instance.
    flags : list[str]
        Human-readable warnings (empty children, wide children, etc.).
    """

    children: list[ChildStats]
    n_total: int
    n_unmatched: int
    n_unparsed_mols: int
    mean_child_width: float
    flags: list[str]


# ---------------------------------------------------------------------------
# Internal SMARTS helpers
# ---------------------------------------------------------------------------

def _map_to_query_idx(query: Chem.Mol) -> dict[int, int]:
    """Return {map_number: query_atom_index} for all mapped atoms."""
    result: dict[int, int] = {}
    for qi in range(query.GetNumAtoms()):
        mn = query.GetAtomWithIdx(qi).GetAtomMapNum()
        if mn > 0:
            result[mn] = qi
    return result


def _triple_matches_rdmol(
    query: Chem.Mol,
    rdmol: Chem.Mol,
    atom_key: tuple,
    map_to_qi: dict[int, int],
) -> bool:
    """True if any substructure match places the :1/:2/:3 atoms at *atom_key*.

    For angle parameters the key is ``(i, j, k)`` and map numbers 1, 2, 3
    must be present.  Both the forward ``(i, j, k)`` and reverse
    ``(k, j, i)`` orientations are accepted.

    Falls back to molecule-level ``HasSubstructMatch`` when the query has
    fewer than three mapped positions (e.g. an unmapped catch-all pattern).
    """
    n_key = len(atom_key)

    if n_key == 3 and 1 in map_to_qi and 2 in map_to_qi and 3 in map_to_qi:
        qi1, qi2, qi3 = map_to_qi[1], map_to_qi[2], map_to_qi[3]
        target_fwd = atom_key
        target_rev = (atom_key[2], atom_key[1], atom_key[0])
        for match in rdmol.GetSubstructMatches(query, uniquify=False):
            matched = (match[qi1], match[qi2], match[qi3])
            if matched == target_fwd or matched == target_rev:
                return True
        return False

    if n_key == 2 and 1 in map_to_qi and 2 in map_to_qi:
        qi1, qi2 = map_to_qi[1], map_to_qi[2]
        target_fwd = atom_key
        target_rev = (atom_key[1], atom_key[0])
        for match in rdmol.GetSubstructMatches(query, uniquify=False):
            matched = (match[qi1], match[qi2])
            if matched == target_fwd or matched == target_rev:
                return True
        return False

    if n_key == 4 and all(k in map_to_qi for k in (1, 2, 3, 4)):
        qi1, qi2, qi3, qi4 = map_to_qi[1], map_to_qi[2], map_to_qi[3], map_to_qi[4]
        target_fwd = atom_key
        target_rev = tuple(reversed(atom_key))
        for match in rdmol.GetSubstructMatches(query, uniquify=False):
            matched = (match[qi1], match[qi2], match[qi3], match[qi4])
            if matched == target_fwd or matched == target_rev:
                return True
        return False

    # Fallback: molecule-level (pattern has no/insufficient map numbers)
    return bool(rdmol.HasSubstructMatch(query))


# ---------------------------------------------------------------------------
# Compile hierarchy
# ---------------------------------------------------------------------------

def _compile_hierarchy(
    hierarchy: list[tuple[str, str]],
) -> list[tuple[Chem.Mol, str, str, dict]]:
    """Compile (smarts, name) pairs → (query, name, smarts, map_to_qi).

    Raises ``ValueError`` on invalid SMARTS.
    """
    compiled = []
    for smarts, name in hierarchy:
        query = Chem.MolFromSmarts(smarts)
        if query is None:
            raise ValueError(f"Invalid SMARTS for {name!r}: {smarts!r}")
        m2qi = _map_to_query_idx(query)
        compiled.append((query, name, smarts, m2qi))
    return compiled


# ---------------------------------------------------------------------------
# Core validator
# ---------------------------------------------------------------------------

def validate_hierarchy(
    instances,
    hierarchy: list[tuple[str, str]],
    param_type: str = "angle",
    target_width: float = 10.0,
    strained_ring_max: int = 4,
) -> ValidationReport:
    """Simulate last-match-wins assignment at the **triple level**.

    Parameters
    ----------
    instances : list[InstanceRecord]
        From :func:`~.extract.extract_param_instances`.  Each record must
        have ``rdmol`` populated (use ``load_instances(reconstruct_rdmol=True)``
        if loading from JSON).
    hierarchy : list[tuple[str, str]]
        ``[(smarts, name), ...]`` in last-match-wins order — parent first,
        most-specific last.  The parent is expected to match every instance;
        if an instance matches no pattern it is counted in ``n_unmatched``.
    param_type : str
        ``'angle'``, ``'bond'``, or ``'torsion'`` (used for strained-ring
        detection via the featurizer).
    target_width : float
        Flag threshold in native units.
    strained_ring_max : int
        Ring size at or below which a cluster is considered strained.

    Returns
    -------
    ValidationReport
    """
    from .featurize import featurize_instance

    compiled = _compile_hierarchy(hierarchy)
    child_names = [name for _, name, _, _ in compiled]
    child_instances: dict[str | None, list] = {name: [] for name in child_names}
    child_instances[None] = []   # unmatched bucket

    n_unparsed = 0

    for rec in instances:
        if rec.rdmol is None:
            n_unparsed += 1
            continue

        winner: str | None = None
        for query, name, smarts, m2qi in compiled:
            if _triple_matches_rdmol(query, rec.rdmol, rec.atom_key, m2qi):
                winner = name   # last match wins → keep updating

        child_instances[winner].append(rec)

    flags: list[str] = []
    children: list[ChildStats] = []
    non_parent_widths: list[float] = []
    is_parent = True  # first entry is the parent (catch-all)

    for _, name, smarts, _ in compiled:
        group = child_instances.get(name, [])
        n_inst = len(group)
        n_mols = len({r.mol_idx for r in group})

        if n_inst == 0:
            children.append(ChildStats(
                name=name, smarts=smarts,
                n_instances=0, n_molecules=0,
                qm_min=float("nan"), qm_max=float("nan"),
                qm_mean=float("nan"), qm_width=0.0,
                is_empty=True, is_wide=False, is_strained=False,
                instances=[],
            ))
            if not is_parent:
                flags.append(
                    f"{name}: EMPTY — eaten by a later pattern or"
                    " co-occurs everywhere with another child; reorder or drop."
                )
            is_parent = False
            continue

        vals = [r.qm_mean for r in group]
        qmin, qmax = float(np.min(vals)), float(np.max(vals))
        qmean = float(np.mean(vals))
        qwidth = qmax - qmin
        is_wide = (not is_parent) and (qwidth > target_width)

        # Check strained ring membership
        is_strained = False
        for r in group:
            fs = featurize_instance(r.rdmol, r.atom_key, param_type)
            if fs and fs.center_ring_sizes and min(fs.center_ring_sizes) <= strained_ring_max:
                is_strained = True
                break

        children.append(ChildStats(
            name=name, smarts=smarts,
            n_instances=n_inst, n_molecules=n_mols,
            qm_min=qmin, qm_max=qmax, qm_mean=qmean, qm_width=qwidth,
            is_empty=False,
            is_wide=is_wide,
            is_strained=is_strained,
            instances=group,
        ))

        if not is_parent:
            non_parent_widths.append(qwidth)

        if is_wide and not is_strained:
            flags.append(
                f"{name}: width {qwidth:.1f} > target {target_width:.0f}"
                " — split further or confirm irreducible ring strain."
            )
        elif is_wide and is_strained:
            flags.append(
                f"{name}: width {qwidth:.1f} — STRAINED ring center"
                " (ring ≤ {strained_ring_max}-membered); irreducible — isolate and flag."
            )

        is_parent = False

    unmatched = child_instances.get(None, [])
    n_unmatched = len(unmatched)
    if n_unmatched:
        flags.append(
            f"{n_unmatched} instance(s) matched no pattern — "
            "parent SMARTS may be too restrictive."
        )

    mean_cw = float(np.mean(non_parent_widths)) if non_parent_widths else float("nan")

    return ValidationReport(
        children=children,
        n_total=len(instances) - n_unparsed,
        n_unmatched=n_unmatched,
        n_unparsed_mols=n_unparsed,
        mean_child_width=mean_cw,
        flags=flags,
    )


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------

def print_validation_report(
    report: ValidationReport,
    param_id: str = "",
    unit: str = "°",
) -> None:
    """Pretty-print a ValidationReport to stdout."""
    header = f"Validation: {param_id}" if param_id else "Validation report"
    print(f"\n{header}")
    print(
        f"Assigned {report.n_total} instances"
        f" ({report.n_unparsed_mols} skipped — rdmol=None,"
        f" {report.n_unmatched} unmatched)\n"
    )
    hdr = (
        f"{'#':<3} {'name':<14} {'SMARTS':<50} | {'mols':>4} | {'inst':>4} | "
        f"{'min':>7} | {'max':>7} | {'mean':>7} | {'width':>7} | flag"
    )
    print(hdr)
    print("-" * 120)

    for i, cs in enumerate(report.children):
        if cs.is_empty:
            print(
                f"{i:<3} {cs.name:<14} {cs.smarts:<50} | "
                f"{'--':>4} | {'0':>4} |  EMPTY"
            )
            continue
        flag = ""
        if cs.is_strained:
            flag = "STRAINED (irreducible)"
        elif cs.is_wide:
            flag = "<-- wide"
        print(
            f"{i:<3} {cs.name:<14} {cs.smarts:<50} | "
            f"{cs.n_molecules:>4} | {cs.n_instances:>4} | "
            f"{cs.qm_min:>7.2f}{unit} | {cs.qm_max:>7.2f}{unit} | "
            f"{cs.qm_mean:>7.2f}{unit} | {cs.qm_width:>7.2f}{unit} | {flag}"
        )

    if not np.isnan(report.mean_child_width):
        print(f"\nMean child width (excl. parent): {report.mean_child_width:.1f}{unit}")

    if report.flags:
        print("\nFlags:")
        for fl in report.flags:
            print(f"  - {fl}")
    else:
        print("\nAll children non-empty and within target width.")
    print()
