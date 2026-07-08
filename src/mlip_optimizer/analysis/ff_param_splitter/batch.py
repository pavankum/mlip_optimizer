"""Batch mode: rank all high-error parameters and auto-propose split hierarchies.

Workflow
--------
1. Discover all parameter IDs present across the dataset from the precomputed
   FF lookup tables.
2. For each parameter, extract instances and compute the QM-value distribution.
3. Score: "split warranted" when distribution is wide or multimodal AND the
   FF mean error exceeds the flag threshold.
4. For warranted parameters, run the four-axis featurizer, propose candidate
   child SMARTS based on bond-order splits (highest-value axis), validate, and
   emit a ranked report.

Auto-SMARTS generation
----------------------
The auto-proposer is intentionally conservative: it generates bond-order-only
splits (axis 1) and ring-isolation splits (for strained centers), which are
the most reliable.  Element-specific and charge splits require chemical judgment
and are flagged as "needs human review" in the report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

from .extract import extract_param_instances, InstanceRecord
from .featurize import featurize_instance, FeatureSet
from .cluster import cluster_instances, ClusterReport, choose_partition
from .validate import validate_hierarchy, ValidationReport


# ---------------------------------------------------------------------------
# Per-parameter summary
# ---------------------------------------------------------------------------

@dataclass
class ParameterSummary:
    """Distribution + error statistics for one FF parameter.

    Attributes
    ----------
    param_id : str
    param_type : str
    smirks : str
    n_instances : int
    n_molecules : int
    qm_min : float
    qm_max : float
    qm_mean : float
    qm_std : float
    qm_width : float
    is_bimodal : bool
        True when BIC favours k=2 over k=1 (suggests a split is worthwhile).
    ff_mean_error : float
        Mean abs diff (QM – FF) across all instances; ``nan`` if not supplied.
    ff_max_error : float
    split_warranted : bool
        ``True`` when width > threshold AND (is_bimodal OR ff_mean_error > flag_thresh).
    cluster_reports : list[ClusterReport]
        Feature-based clusters (four-axis grouping).
    proposed_hierarchy : list[tuple[str, str]] or None
        Auto-proposed (smarts, name) list in last-match-wins order.
        ``None`` when auto-proposal is not attempted.
    validation : ValidationReport or None
        Validation of *proposed_hierarchy* if auto-proposal was run.
    note : str
        One-line note on any bucket that cannot be narrowed and why.
    """

    param_id: str
    param_type: str
    smirks: str
    n_instances: int
    n_molecules: int
    qm_min: float
    qm_max: float
    qm_mean: float
    qm_std: float
    qm_width: float
    is_bimodal: bool
    ff_mean_error: float
    ff_max_error: float
    split_warranted: bool
    cluster_reports: list = field(default_factory=list, repr=False)
    proposed_hierarchy: list | None = None
    validation: ValidationReport | None = None
    note: str = ""


# ---------------------------------------------------------------------------
# Helper: discover all parameter IDs in a set of FF lookups
# ---------------------------------------------------------------------------

def discover_param_ids(
    ff_param_lookups: list[dict],
    param_types: list[str] | None = None,
) -> dict[str, tuple[str, str]]:
    """Return {param_id: (param_type, smirks)} for all IDs found in the lookups.

    Parameters
    ----------
    ff_param_lookups : list[dict]
        From ``_build_ff_param_lookups``.  Each entry maps
        ``atom_key → (param_id, smirks)``.
    param_types : list[str] or None
        If given, restrict to these types inferred from key length:
        2-tuple → bond, 3-tuple → angle, 4-tuple → torsion.
        Default: all types.

    Returns
    -------
    dict[str, tuple[str, str]]
        ``{param_id: (inferred_type, smirks)}``.
    """
    _len_to_type = {2: "bond", 3: "angle", 4: "torsion"}
    result: dict[str, tuple[str, str]] = {}
    for lut in ff_param_lookups:
        for atom_key, (pid, smirks) in lut.items():
            if pid in result:
                continue
            ptype = _len_to_type.get(len(atom_key), "unknown")
            if param_types and ptype not in param_types:
                continue
            result[pid] = (ptype, smirks)
    return result


# ---------------------------------------------------------------------------
# Auto-SMARTS generation from feature clusters
# ---------------------------------------------------------------------------

_BO_CHAR_TO_SMARTS = {"s": "-", "a": ":", "d": "=", "t": "#"}

# Common elements seen in NSP flank positions; extend as new elements appear.
_ELEM_TO_Z = {
    "H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "Si": 14, "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53,
}

# Tokenises a SMARTS atom-bracket primitive into its constituent constraints.
# Ordered so that longer/more-specific patterns are tried before shorter ones
# (e.g. '#\d+' before a bare letter, '+\d+' before bare '+').
# NOTE: recursive $(...) tokens are extracted by depth-tracking in
# _primitive_tokens() BEFORE this regex runs, so no $() alternative is needed
# here and nested parentheses cannot corrupt the remaining token matching.
_PRIM_TOKEN_RE = re.compile(
    r'#\d+'         # #Z  atomic number
    r'|[Xx]\d+'     # Xn  total degree / connectivity
    r'|[+-]\d+'     # \u00b1n  formal charge with explicit value (+0, +1, -2 \u2026)
    r'|[Rr]\d+'     # rN / RN  ring membership / ring count with value
    r'|[Hh]\d+'     # Hn  hydrogen count with value
    r'|[aAvV]'       # a/A aromaticity, v/V valence (bare)
    r'|[+-]'         # bare +/- (any positive / any negative charge)
    r'|[Rr]'         # bare r/R (in any ring / ring count)
    r'|[Hh]'         # bare H  (has at least one H)
)


def _center_atom_primitive(parent_smarts: str, param_type: str = "angle") -> str:
    """Extract the central atom primitive from the parent SMARTS.

    For angle ``[*:1]~[#7X3$(...):2]~[*:3]`` returns ``#7X3$(...)``
    (the content of the :2 atom bracket, minus map number).
    Falls back to ``*`` if parsing fails.

    Delegates to :func:`_parse_parent_constraints` so recursive SMARTS
    (``$([...])``) and primitives ending in digits (``+0``, ``X3``, etc.)
    are handled correctly.  The old regex approach stripped trailing digits
    and failed on nested brackets.
    """
    map_num = 2 if param_type == "angle" else 1
    atom_prims, _ = _parse_parent_constraints(parent_smarts)
    return atom_prims.get(map_num, "*") or "*"


def _propose_bond_order_splits(
    parent_smarts: str,
    cluster_reports: list[ClusterReport],
    param_id: str,
    param_type: str = "angle",
) -> list[tuple[str, str]]:
    """Generate a bond-order-only hierarchy from feature clusters.

    Groups clusters by their bond-order signature.  If a signature
    contains a strained-ring cluster, adds a ring-isolation child first.

    Returns a ``[(smarts, name), ...]`` list in last-match-wins order
    (parent first) or an empty list if no useful split is found.
    """
    center_prim = _center_atom_primitive(parent_smarts, param_type)

    # Collect distinct bo_chars signatures from clusters
    bo_groups: dict[tuple, list[ClusterReport]] = defaultdict(list)
    strained_reports: list[ClusterReport] = []
    for cr in cluster_reports:
        if cr.feature_set is None:
            continue
        if cr.is_strained:
            strained_reports.append(cr)
        else:
            bo_groups[cr.feature_set.bo_chars].append(cr)

    if len(bo_groups) + len(strained_reports) <= 1:
        return []   # no useful split

    hierarchy: list[tuple[str, str]] = [(parent_smarts, param_id)]

    child_idx = ord("a")

    # Strained ring children first (they override the parent but are broad)
    for cr in strained_reports:
        fs = cr.feature_set
        assert fs is not None
        min_r = min(fs.center_ring_sizes)
        child_smarts = (
            f"[*:1]~[{center_prim};r{min_r}:2]~[*:3]"
            if param_type == "angle"
            else f"[*:1]~[{center_prim};r{min_r}:2]"
        )
        child_name = f"{param_id}{chr(child_idx)}"
        hierarchy.append((child_smarts, child_name))
        child_idx += 1

    # Bond-order children
    for bo_sig, reports in sorted(bo_groups.items(), key=lambda kv: kv[0]):
        if param_type == "angle" and len(bo_sig) == 2:
            b1_char = _BO_CHAR_TO_SMARTS.get(bo_sig[0], "~")
            b2_char = _BO_CHAR_TO_SMARTS.get(bo_sig[1], "~")
            # If both bonds have the same order use symmetric pattern
            if b1_char == b2_char:
                child_smarts = f"[*:1]{b1_char}[{center_prim}:2]{b2_char}[*:3]"
            else:
                # Use ~ on the single side, explicit on the double/aromatic side
                single_side = b1_char if bo_sig[0] == "s" else b2_char
                special_side = b2_char if bo_sig[0] == "s" else b1_char
                child_smarts = f"[*:1]~[{center_prim}:2]{special_side}[*:3]"
        elif param_type == "bond" and len(bo_sig) == 1:
            bc = _BO_CHAR_TO_SMARTS.get(bo_sig[0], "~")
            child_smarts = f"[{center_prim}:1]{bc}[*:2]"
        else:
            child_smarts = parent_smarts  # fallback; won't add

        if child_smarts == parent_smarts:
            continue

        child_name = f"{param_id}{chr(child_idx)}"
        hierarchy.append((child_smarts, child_name))
        child_idx += 1

    return hierarchy if len(hierarchy) > 1 else []


def _flank_smarts(elements: tuple[str, ...]) -> str:
    """``#z1,#z2,...`` OR-list SMARTS fragment for a set of flank element symbols."""
    zs = sorted({_ELEM_TO_Z[e] for e in elements if e in _ELEM_TO_Z})
    if not zs:
        return "*"
    return ",".join(f"#{z}" for z in zs)


# ---------------------------------------------------------------------------
# Parent-constraint parsing and candidate subset validation
# ---------------------------------------------------------------------------

def _parse_parent_constraints(
    smarts: str,
) -> tuple[dict[int, str], dict[tuple[int, int], str]]:
    """Parse a SMARTS string to extract mapped-atom primitives and inter-atom
    bond orders for every ``[primitive:N]`` atom pair found.

    Handles recursive SMARTS primitives (e.g. ``$([#6X3])``) correctly via
    depth-tracking bracket matching, so nested ``[...]`` inside an atom clause
    do not confuse the parser.

    Returns
    -------
    atom_prims : dict[int, str]
        ``{map_num: primitive}`` — primitive is the bracket content stripped
        of its trailing ``:N`` map number.
    bond_orders : dict[tuple[int, int], str]
        ``{(min_map, max_map): bond_primitive}`` for bonds between consecutive
        mapped atoms in the (linear) SMARTS string.  The full bond-primitive
        string is preserved verbatim (e.g. ``'@'``, ``'!-'``, ``'-,=:'``) after
        stripping ring-closure annotations (bare digits and ``%NN`` forms).
        Returns ``'~'`` for empty/unparseable spans (safe default: will not
        cause false rejections in the subset check).
    """
    # Locate every mapped atom and record span + primitive
    mapped_spans: list[tuple[int, int, int, str]] = []  # (start, end, map_num, prim)
    i, n = 0, len(smarts)
    while i < n:
        if smarts[i] != "[":
            i += 1
            continue
        depth = 1
        j = i + 1
        while j < n and depth:
            if smarts[j] == "[":
                depth += 1
            elif smarts[j] == "]":
                depth -= 1
            j += 1
        atom_str = smarts[i:j]  # e.g. "[#7X3$(*~[#6]):2]"
        m = re.search(r":(\d+)\]$", atom_str)
        if m:
            map_num = int(m.group(1))
            prim = re.sub(r":\d+$", "", atom_str[1:-1]).strip()
            mapped_spans.append((i, j, map_num, prim))
        i = j

    atom_prims = {map_num: prim for _, _, map_num, prim in mapped_spans}

    # Extract bond primitive between each consecutive pair of mapped atoms.
    # Strip ring-closure annotations (bare digits like '1', or '%10' style) so
    # that only the bond-primitive characters remain.  Keep the full string —
    # including '@', '!-', '-,=:', etc. — so that ALL parent bond primitives
    # are preserved, not just the simple single-character ones.
    bond_orders: dict[tuple[int, int], str] = {}
    for k in range(len(mapped_spans) - 1):
        _, end1, map1, _ = mapped_spans[k]
        start2, _, map2, _ = mapped_spans[k + 1]
        between = re.sub(r"%?\d+", "", smarts[end1:start2]).strip()  # strip ring-closure digits
        key = (min(map1, map2), max(map1, map2))
        bond_orders[key] = between if between else "~"

    return atom_prims, bond_orders


def _elem_from_prim(prim: str) -> str | None:
    """Return ``'#Z'`` if *prim* specifies an element, ``None`` for wildcards.

    Used by :func:`propose_cluster_candidates` to decide which flank atoms
    are element-wildcards (eligible for flank-element specialisation).  For
    the full subset validity check, use :func:`_is_valid_candidate_subset`
    (which checks all primitive tokens, not just the element).

    Examples::

        _elem_from_prim("#7X2+0")          # → '#7'
        _elem_from_prim("*")               # → None  (wildcard)
        _elem_from_prim("#7X3$(*~[#6X3])") # → '#7'  (recursive SMARTS)
    """
    if not prim or prim in ("*", "a", "A"):
        return None
    m = re.search(r"#(\d+)", prim)
    return f"#{m.group(1)}" if m else None


def _primitive_tokens(prim: str) -> frozenset[str]:
    """Return the set of SMARTS constraint tokens in an atom-bracket primitive.

    ``'*'`` and empty strings return an empty frozenset (no constraints — the
    atom is a wildcard, so any candidate primitive is a valid narrowing).
    Recursive SMARTS ``$([...])`` are extracted first with a depth-tracking
    loop so that nested parentheses (e.g. ``$([!$(C=O)])``) are captured as
    a single opaque token regardless of nesting depth.  All remaining tokens
    (``#Z``, ``Xn``, ``±n``, ``rN``, etc.) are then extracted by regex.

    Examples::

        _primitive_tokens("#7X2+0")            → frozenset({'#7', 'X2', '+0'})
        _primitive_tokens("*")                 → frozenset()
        _primitive_tokens("#7X2+0;r5")         → frozenset({'#7', 'X2', '+0', 'r5'})
        _primitive_tokens("#6X3$([!$(C=O)])")  → frozenset({'#6', 'X3', '$([!$(C=O)])'})
    """
    if not prim or prim in ("*", "a", "A"):
        return frozenset()
    # Phase 1 — extract $(...) tokens with depth-tracking (handles nested parens)
    dollar_tokens: list[str] = []
    remainder_chars: list[str] = []
    i, n = 0, len(prim)
    while i < n:
        if prim[i] == '$' and i + 1 < n and prim[i + 1] == '(':
            depth, j = 1, i + 2
            while j < n and depth:
                depth += prim[j] == '('
                depth -= prim[j] == ')'
                j += 1
            dollar_tokens.append(prim[i:j])
            i = j
        else:
            remainder_chars.append(prim[i])
            i += 1
    # Phase 2 — extract remaining primitive tokens with regex
    return frozenset(dollar_tokens) | frozenset(_PRIM_TOKEN_RE.findall(''.join(remainder_chars)))


def _is_valid_candidate_subset(
    candidate_smarts: str,
    parent_atom_prims: dict[int, str],
    parent_bonds: dict[tuple[int, int], str],
) -> bool:
    """Return ``True`` iff *candidate_smarts* is a structurally valid subset of
    the parent described by *parent_atom_prims* / *parent_bonds*.

    Rule 1 — Full atom primitives: ALL SMARTS constraint tokens present on a
    mapped atom in the parent (element ``#Z``, degree ``Xn``, charge ``±n``,
    ring membership ``rN``/``Rn``, recursive ``$(...)`` etc.) must also be
    present on the same mapped atom in the candidate.  The candidate may ADD
    further constraints (narrowing is allowed), but may not CHANGE, DROP, or
    WIDEN any token the parent already specifies.

    Rule 2 — Fixed bond primitives: if the parent fixes a bond with any
    non-wildcard primitive (``-``, ``=``, ``:``, ``#``, ``@``, ``!-``,
    ``-,=:`` etc.) between two mapped atoms, the candidate must carry the
    identical bond-primitive string at that position.  Re-wildcarding to
    ``~`` or substituting any different primitive (even a valid narrowing
    such as ``-`` inside a parent ``-,=:``) fails.  Only bonds that are
    ``~`` in the parent may be narrowed in a candidate.
    """
    cand_prims, cand_bonds = _parse_parent_constraints(candidate_smarts)

    for map_num, parent_prim in parent_atom_prims.items():
        parent_tokens = _primitive_tokens(parent_prim)
        if not parent_tokens:
            continue  # parent wildcard — candidate may do anything
        cand_tokens = _primitive_tokens(cand_prims.get(map_num, "*"))
        if not parent_tokens.issubset(cand_tokens):
            return False  # candidate changed, dropped, or widened a primitive

    for bond_key, parent_bond in parent_bonds.items():
        if parent_bond == "~":
            continue  # parent wildcard — any order in candidate is fine
        cand_bond = cand_bonds.get(bond_key, "~")
        if cand_bond != parent_bond:
            return False  # changed or re-wildcarded a fixed bond order

    return True


def _propose_flank_element_splits(
    parent_smarts: str,
    cluster_reports: list[ClusterReport],
    param_id: str,
    param_type: str = "angle",
) -> list[tuple[str, str]]:
    """Bond-order axis as entry point, layered with a flank-element child per
    distinct non-C/H flank signature (idiom: ``reference/smarts_idioms.md``
    "Flank-element split"). Only meaningful for angles — bonds have no
    flanking atoms to key on, so this falls back to the bond-order proposer.
    """
    if param_type != "angle":
        return _propose_bond_order_splits(parent_smarts, cluster_reports, param_id, param_type)

    center_prim = _center_atom_primitive(parent_smarts, param_type)

    flank_groups: dict[tuple, list[ClusterReport]] = defaultdict(list)
    for cr in cluster_reports:
        if cr.feature_set is None or cr.is_strained:
            continue
        # Dedupe elements BEFORE using as a group key — otherwise a cluster with
        # both flanks the same heteroatom (e.g. ('N', 'N')) gets its own group
        # distinct from a single-flank-N cluster (('N',)), even though
        # _flank_smarts() renders both to the identical "#7" fragment, producing
        # two hierarchy entries with the same SMARTS (the second always EMPTY).
        hetero = tuple(sorted({e for e in cr.feature_set.flank_elements if e not in ("C", "H")}))
        if hetero:
            flank_groups[hetero].append(cr)

    if not flank_groups:
        return []

    hierarchy: list[tuple[str, str]] = [(parent_smarts, param_id)]
    child_idx = ord("a")
    for hetero in sorted(flank_groups):
        frag = _flank_smarts(hetero)
        child_smarts = f"[{frag}:1]~[{center_prim}:2]~[*:3]"
        hierarchy.append((child_smarts, f"{param_id}{chr(child_idx)}"))
        child_idx += 1

    return hierarchy if len(hierarchy) > 1 else []


def _propose_ring_splits(
    parent_smarts: str,
    cluster_reports: list[ClusterReport],
    param_id: str,
    param_type: str = "angle",
) -> list[tuple[str, str]]:
    """Bond-order axis as entry point, layered with a ring-size-isolation
    child per distinct center ring size (idiom: "Ring-size isolation").
    Strained (ring <= 4) clusters are isolated first, matching
    ``_propose_bond_order_splits``'s ordering.
    """
    center_prim = _center_atom_primitive(parent_smarts, param_type)

    ring_groups: dict[int, list[ClusterReport]] = defaultdict(list)
    strained_reports: list[ClusterReport] = []
    for cr in cluster_reports:
        if cr.feature_set is None:
            continue
        if cr.is_strained:
            strained_reports.append(cr)
        elif cr.feature_set.center_ring_sizes:
            ring_groups[min(cr.feature_set.center_ring_sizes)].append(cr)

    if not ring_groups and not strained_reports:
        return []

    def _ring_child_smarts(min_r: int) -> str:
        if param_type == "angle":
            return f"[*:1]~[{center_prim};r{min_r}:2]~[*:3]"
        return f"[{center_prim};r{min_r}:1]~[*:2]"

    hierarchy: list[tuple[str, str]] = [(parent_smarts, param_id)]
    child_idx = ord("a")
    for cr in strained_reports:
        min_r = min(cr.feature_set.center_ring_sizes)
        hierarchy.append((_ring_child_smarts(min_r), f"{param_id}{chr(child_idx)}"))
        child_idx += 1
    for min_r in sorted(ring_groups):
        hierarchy.append((_ring_child_smarts(min_r), f"{param_id}{chr(child_idx)}"))
        child_idx += 1

    return hierarchy if len(hierarchy) > 1 else []


def _propose_attachment_context_splits(
    parent_smarts: str,
    cluster_reports: list[ClusterReport],
    param_id: str,
    param_type: str = "angle",
) -> list[tuple[str, str]]:
    """Axis-5 (attachment-context, BESMARTS d>=1) hierarchy.

    Groups clusters by ``shell_elements`` (what the flank atoms are bonded
    to) via recursive ``$(*-[#Z])`` on the flank map atoms — the
    ``smarts_idioms.md`` "Branch atom on mapped flank" idiom. Falls back to
    ``center_substituent_elements`` (what the center's non-angle substituent
    is) if the flank-shell signature yields fewer than 2 groups. Requires
    ``cluster_reports`` to have been computed with ``depth > 0`` (i.e. via
    ``cluster_instances(..., depth=d)``) — at d=0 both signatures are always
    empty and this returns ``[]``.
    """
    center_prim = _center_atom_primitive(parent_smarts, param_type)

    def _groups_by(key_fn) -> dict[tuple, list[ClusterReport]]:
        g: dict[tuple, list[ClusterReport]] = defaultdict(list)
        for cr in cluster_reports:
            fs = cr.feature_set
            if fs is None or cr.is_strained:
                continue
            sig = key_fn(fs)
            if sig:
                g[sig].append(cr)
        return g

    flank_groups = _groups_by(lambda fs: fs.shell_elements)
    use_center_sub = len(flank_groups) < 2
    groups = _groups_by(lambda fs: fs.center_substituent_elements) if use_center_sub else flank_groups

    if len(groups) < 2:
        return []

    hierarchy: list[tuple[str, str]] = [(parent_smarts, param_id)]
    child_idx = ord("a")
    for sig in sorted(groups):
        frag = _flank_smarts(sig)
        if use_center_sub:
            # Recursive on the CENTER atom: "center has a substituent of element Z"
            if param_type == "angle":
                child_smarts = f"[*:1]~[{center_prim}$(*-[{frag}]):2]~[*:3]"
            else:
                child_smarts = f"[{center_prim}$(*-[{frag}]):1]~[*:2]"
        else:
            # Recursive on a FLANK atom: "flank atom's own neighbor is element Z"
            child_smarts = f"[*;$(*-[{frag}]):1]~[{center_prim}:2]~[*:3]"
        hierarchy.append((child_smarts, f"{param_id}{chr(child_idx)}"))
        child_idx += 1

    return hierarchy if len(hierarchy) > 1 else []


def propose_cluster_candidates(
    cr: ClusterReport,
    parent_smarts: str,
    param_id: str,
    param_type: str = "angle",
    depth: int = 0,
) -> list[tuple[str, str]]:
    """Generate 2-5 candidate ``(smarts, name)`` SMARTS for ONE wide/flagged
    cluster, walking the axis priority (bond order > flank element >
    ring size > charge > attachment-context). Only emits an axis when it is
    actually discriminating for this cluster (e.g. skipped when acyclic or
    fc0; attachment-context skipped unless ``depth > 0`` AND the cluster was
    featurized at that depth).

    Every candidate is guaranteed to be a structural subset of the parent:

    * Atom elements fixed in the parent (``[#Z:N]``) are preserved on the same
      mapped atom in every candidate — they are never changed or widened to
      ``[*]``.
    * Bond orders fixed in the parent (``-``, ``=``, ``:`` or ``#``) are
      preserved at the same bond position in every candidate — they are never
      re-wildcarded to ``~`` or changed to a different explicit order.
    * Only wildcard atoms (``[*:N]``) and wildcard bonds (``~``) in the parent
      are eligible for narrowing in a candidate.

    Candidates that fail this subset check are silently discarded so that the
    caller never receives a structurally invalid (and thus always-empty)
    candidate — which previously masked the real bug behind a misleading
    ``0 inst / EMPTY`` symptom.

    Intended for ``analyze_parameter.py --auto-candidates`` — these are
    single-cluster alternatives to sanity-check by eye or feed into
    ``validate_hierarchy.py``, not a full hierarchy by themselves.
    """
    fs = cr.feature_set
    if fs is None:
        return []
    center_prim = _center_atom_primitive(parent_smarts, param_type)

    # Parse parent constraints — used both to build valid candidates and to
    # validate them as a final safety net.
    parent_atom_prims, parent_bonds = _parse_parent_constraints(parent_smarts)

    # Preserved flank-atom primitives (fall back to '*' for genuine wildcards).
    prim_1 = parent_atom_prims.get(1, "*")
    prim_3 = parent_atom_prims.get(3, "*") if param_type == "angle" else "*"
    bond_12 = parent_bonds.get((1, 2), "~")
    bond_23 = parent_bonds.get((2, 3), "~") if param_type == "angle" else "~"

    candidates: list[tuple[str, str]] = []

    # Axis 1: bond order.
    # Only specialise bonds that are ~ in the parent; preserve fixed bonds.
    # Note: fs.bo_smarts is sorted by bond order (lower first), not positionally.
    # For the common case (bond_12 fixed, bond_23 wildcard) the sorted order
    # happens to match position order; when both are wildcard, the candidate may
    # not perfectly reflect the cluster's positional assignment but remains a
    # valid subset — validate_hierarchy.py determines actual instance coverage.
    if param_type == "angle" and len(fs.bo_smarts) == 2:
        b1_clust, b2_clust = fs.bo_smarts
        b1 = b1_clust if bond_12 == "~" else bond_12
        b2 = b2_clust if bond_23 == "~" else bond_23
        smarts = f"[{prim_1}:1]{b1}[{center_prim}:2]{b2}[{prim_3}:3]"
        candidates.append((smarts, "bond-order"))
    elif param_type == "bond" and len(fs.bo_smarts) == 1:
        prim_2 = parent_atom_prims.get(2, "*")
        b = fs.bo_smarts[0] if bond_12 == "~" else bond_12
        candidates.append((f"[{center_prim}:1]{b}[{prim_2}:2]", "bond-order"))

    # Axis 2: flank element.
    # Only specialise a flank atom that is a wildcard in the parent — never
    # change the element of an already-fixed flank atom.
    if param_type == "angle":
        elem_1 = _elem_from_prim(prim_1)  # None ↔ wildcard on :1
        elem_3 = _elem_from_prim(prim_3)  # None ↔ wildcard on :3
        hetero = tuple(sorted(e for e in fs.flank_elements if e not in ("C", "H")))
        if hetero:
            frag = _flank_smarts(hetero)
            if elem_1 is None and elem_3 is not None:
                # :3 is fixed; :1 is wildcard → specialise :1
                smarts = f"[{frag}:1]{bond_12}[{center_prim}:2]{bond_23}[{prim_3}:3]"
                candidates.append((smarts, "flank-element"))
            elif elem_3 is None and elem_1 is not None:
                # :1 is fixed; :3 is wildcard → specialise :3
                smarts = f"[{prim_1}:1]{bond_12}[{center_prim}:2]{bond_23}[{frag}:3]"
                candidates.append((smarts, "flank-element"))
            elif elem_1 is None and elem_3 is None:
                # Both wildcards → put hetero on :1 (convention, preserves existing behaviour)
                smarts = f"[{frag}:1]{bond_12}[{center_prim}:2]{bond_23}[{prim_3}:3]"
                candidates.append((smarts, "flank-element"))
            # Both fixed → no flank-element specialisation possible; skip this axis.

    # Axis 3: ring size — add narrowing ring constraint to the center atom.
    if fs.center_ring_sizes:
        min_r = min(fs.center_ring_sizes)
        if param_type == "angle":
            smarts = f"[{prim_1}:1]{bond_12}[{center_prim};r{min_r}:2]{bond_23}[{prim_3}:3]"
        else:
            smarts = f"[{center_prim};r{min_r}:1]~[*:2]"
        candidates.append((smarts, f"ring-r{min_r}"))

    # Axis 4: formal charge — add narrowing charge constraint to the center atom.
    if fs.center_formal_charge != 0:
        chg = f"{fs.center_formal_charge:+d}"
        if param_type == "angle":
            smarts = f"[{prim_1}:1]{bond_12}[{center_prim};{chg}:2]{bond_23}[{prim_3}:3]"
        else:
            smarts = f"[{center_prim};{chg}:1]~[*:2]"
        candidates.append((smarts, f"charge{chg}"))

    # Axis 5: attachment context (depth > 0 only).
    if depth > 0 and param_type == "angle":
        if fs.shell_elements:
            frag = _flank_smarts(fs.shell_elements)
            candidates.append((
                f"[{prim_1};$(*-[{frag}]):1]{bond_12}[{center_prim}:2]{bond_23}[{prim_3}:3]",
                f"attach-d{depth}-flank",
            ))
        elif fs.center_substituent_elements:
            frag = _flank_smarts(fs.center_substituent_elements)
            candidates.append((
                f"[{prim_1}:1]{bond_12}[{center_prim}$(*-[{frag}]):2]{bond_23}[{prim_3}:3]",
                f"attach-d{depth}-center",
            ))

    # Final guard: discard any candidate that is not a true structural subset of
    # the parent.  This catches positional-ambiguity edge cases from the sorted
    # bo_smarts tuple and any unexpected primitive widening.  An always-empty
    # candidate (previously the symptom that hid this bug) is now never emitted.
    valid = [
        (s, ax)
        for s, ax in candidates
        if _is_valid_candidate_subset(s, parent_atom_prims, parent_bonds)
    ]
    return [(s, f"{param_id}_{ax}") for s, ax in valid[:5]]


# ---------------------------------------------------------------------------
# Core batch runner
# ---------------------------------------------------------------------------

def _is_bimodal(vals: list[float], min_cluster_size: int = 8) -> bool:
    """True when BIC favours k=2 over k=1."""
    if len(vals) < 2 * min_cluster_size:
        return False
    try:
        parts = __import__(
            "mlip_optimizer.analysis.ff_param_splitter.cluster",
            fromlist=["optimal_1d_partitions"],
        ).optimal_1d_partitions(vals, max_k=2, min_cluster_size=min_cluster_size)
    except Exception:
        return False
    if 1 not in parts or 2 not in parts:
        return False
    return parts[2]["bic"] < parts[1]["bic"]


def batch_split_all(
    records,
    qm_results,
    ff_param_lookups: list[dict],
    potential_name: str | None = None,
    *,
    param_types: list[str] | None = None,
    target_width_angle: float = 10.0,
    target_width_bond: float = 0.03,
    ff_error_threshold_angle: float = 5.0,
    ff_error_threshold_bond: float = 0.08,
    min_instances: int = 10,
    auto_propose: bool = True,
    validate_proposals: bool = True,
) -> list[ParameterSummary]:
    """Run the full split pipeline for every parameter in the dataset.

    Parameters
    ----------
    records, qm_results, ff_param_lookups
        As returned by ``load_openff_bundle``.
    potential_name : str or None
        Name of the FF potential (e.g. ``'openff-2.3.0'``) to pull error
        data from.  ``None`` → error columns will be NaN.
    param_types : list[str] or None
        Restrict to ``'angle'``, ``'bond'``, ``'torsion'`` or any subset.
    target_width_angle / target_width_bond : float
        Bin-width targets used for "split warranted" decision.
    ff_error_threshold_angle / ff_error_threshold_bond : float
        Mean FF error above which a parameter is flagged.
    min_instances : int
        Skip parameters with fewer instances than this.
    auto_propose : bool
        Run the bond-order auto-SMARTS proposer for warranted parameters.
    validate_proposals : bool
        Run the triple-level validator on each auto-proposed hierarchy.

    Returns
    -------
    list[ParameterSummary]
        Sorted by ``split_warranted`` (True first) then descending QM width.
    """
    ptype_to_target = {"angle": target_width_angle, "bond": target_width_bond, "torsion": 40.0}
    ptype_to_err_thresh = {
        "angle": ff_error_threshold_angle,
        "bond": ff_error_threshold_bond,
        "torsion": 40.0,
    }

    all_param_ids = discover_param_ids(ff_param_lookups, param_types)

    summaries: list[ParameterSummary] = []

    for param_id, (ptype, smirks) in sorted(all_param_ids.items()):
        target_w = ptype_to_target.get(ptype, 10.0)
        err_thresh = ptype_to_err_thresh.get(ptype, 5.0)

        pot_names = [potential_name] if potential_name else []
        instances = extract_param_instances(
            records, qm_results, ff_param_lookups,
            param_id, ptype,
            potential_names=pot_names,
        )

        if len(instances) < min_instances:
            continue

        vals = [r.qm_mean for r in instances]
        n_mols = len({r.mol_idx for r in instances})

        qmin, qmax = float(np.min(vals)), float(np.max(vals))
        qmean, qstd = float(np.mean(vals)), float(np.std(vals))
        qwidth = qmax - qmin

        # Error stats from FF potential
        ff_errors: list[float] = []
        if potential_name:
            for r in instances:
                ff_errors.extend(r.errors.get(potential_name, []))
        ff_mean_err = float(np.mean(ff_errors)) if ff_errors else float("nan")
        ff_max_err  = float(np.max(ff_errors))  if ff_errors else float("nan")

        bimodal = _is_bimodal(vals)
        split_w = (
            qwidth > target_w
            and (bimodal or (not np.isnan(ff_mean_err) and ff_mean_err > err_thresh))
        )

        cluster_reps = cluster_instances(instances, ptype, target_w)

        proposed_hier: list[tuple[str, str]] | None = None
        validation: ValidationReport | None = None
        note = ""

        # Strained clusters that cannot be narrowed
        strained = [cr for cr in cluster_reps if cr.is_strained]
        if strained:
            note = (
                f"{len(strained)} strained-ring cluster(s) (ring ≤ 4): "
                f"irreducible width; isolated and flagged."
            )

        if auto_propose and split_w:
            proposed_hier = _propose_bond_order_splits(
                smirks, cluster_reps, param_id, ptype
            )
            if proposed_hier and validate_proposals:
                try:
                    validation = validate_hierarchy(
                        instances, proposed_hier, ptype, target_w
                    )
                except Exception as exc:
                    note += f" (validation failed: {exc})"

        summaries.append(
            ParameterSummary(
                param_id=param_id,
                param_type=ptype,
                smirks=smirks,
                n_instances=len(instances),
                n_molecules=n_mols,
                qm_min=qmin,
                qm_max=qmax,
                qm_mean=qmean,
                qm_std=qstd,
                qm_width=qwidth,
                is_bimodal=bimodal,
                ff_mean_error=ff_mean_err,
                ff_max_error=ff_max_err,
                split_warranted=split_w,
                cluster_reports=cluster_reps,
                proposed_hierarchy=proposed_hier,
                validation=validation,
                note=note,
            )
        )

    summaries.sort(key=lambda s: (not s.split_warranted, -s.qm_width))
    return summaries


def print_batch_report(summaries: list[ParameterSummary], unit: str = "°") -> None:
    """Print a concise ranked table of all parameter summaries."""
    print(
        f"\n{'param':<8} {'type':<7} {'mols':>4} {'inst':>5} "
        f"{'width':>7} {'bimod':>5} {'FF_err':>8} {'warranted':>10}  note"
    )
    print("-" * 90)
    for s in summaries:
        ff_err_str = f"{s.ff_mean_error:.2f}" if not np.isnan(s.ff_mean_error) else "  N/A"
        warranted = "YES" if s.split_warranted else "no"
        print(
            f"{s.param_id:<8} {s.param_type:<7} {s.n_molecules:>4} {s.n_instances:>5} "
            f"{s.qm_width:>7.2f}{unit} {'yes' if s.is_bimodal else 'no':>5} "
            f"{ff_err_str:>8} {warranted:>10}  {s.note[:60]}"
        )
    n_warranted = sum(1 for s in summaries if s.split_warranted)
    print(f"\nTotal: {len(summaries)} parameters, {n_warranted} warranted splits.\n")
