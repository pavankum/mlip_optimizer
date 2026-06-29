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


def _center_atom_primitive(parent_smarts: str, param_type: str = "angle") -> str:
    """Extract the central atom primitive from the parent SMARTS.

    For angle ``[*:1]~[#7X3$(...):2]~[*:3]`` returns ``#7X3$(...)``
    (the content of the :2 atom bracket, minus map number).
    Falls back to ``*`` if parsing fails.
    """
    # Find atom with map number :2 (angle center)
    map_num = "2" if param_type == "angle" else "1"
    # Regex: match [...:2] bracket content
    pattern = rf"\[([^\[\]]+):{map_num}\]"
    m = re.search(pattern, parent_smarts)
    if m:
        primitive = m.group(1).strip()
        # Remove the map number suffix if it got included
        primitive = re.sub(r":?\d+$", "", primitive).strip()
        return primitive or "*"
    return "*"


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
