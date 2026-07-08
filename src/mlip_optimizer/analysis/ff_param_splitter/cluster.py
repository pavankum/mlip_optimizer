"""Cluster parameter instances by local chemistry and QM-value distribution.

Two clustering modes:

1. **Feature clustering** (``group_by_features`` from ``featurize.py``) —
   groups instances by exact four-axis chemistry fingerprint.  No statistics
   needed; each group is already a chemical cluster candidate.

2. **1-D BIC partitioning** (``optimal_1d_partitions``) — for cases where
   a feature group is still wide, finds the globally-optimal split into k
   segments by minimising within-segment SSE (dynamic programming), then
   uses BIC to choose k.  Ported from ``besmarts_angle_split.py`` to remove
   the besmarts dependency.

Typical workflow::

    # Step 1: group by chemistry
    groups = group_by_features(instances, param_type="angle")

    # Step 2: report cluster stats
    reports = cluster_instances(instances, param_type="angle")
    for r in reports:
        print(r)

    # Step 3: for wide groups, try 1-D partitioning
    vals = [i.qm_mean for i in groups["s+s|C/N|r6|fc0"]]
    best, all_parts = choose_partition(vals)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .featurize import FeatureSet, featurize_instance


# ---------------------------------------------------------------------------
# ClusterReport
# ---------------------------------------------------------------------------

@dataclass
class ClusterReport:
    """Statistics for one feature-chemistry cluster.

    Attributes
    ----------
    feature_key : str
        Compact axis-priority label (from ``FeatureSet.feature_key``).
    feature_set : FeatureSet or None
        Full feature fingerprint of any representative instance.
    n_instances : int
        Number of (molecule, triple) pairs in this cluster.
    n_molecules : int
        Number of distinct molecules.
    qm_values : list[float]
        All ``qm_mean`` values (one per instance).
    qm_min : float
    qm_max : float
    qm_mean : float
    qm_std : float
    qm_width : float
        ``qm_max - qm_min``.
    is_strained : bool
        True when the central atom is in a ring of size ≤ 4 (beta-lactam /
        azetidine / cyclopropane — irreducibly wide).
    split_warranted : bool
        ``qm_width > target_width`` and not already a singleton group.
    instances : list
        The ``InstanceRecord`` objects in this cluster.
    """

    feature_key: str
    feature_set: FeatureSet | None
    n_instances: int
    n_molecules: int
    qm_values: list
    qm_min: float
    qm_max: float
    qm_mean: float
    qm_std: float
    qm_width: float
    is_strained: bool
    split_warranted: bool
    instances: list = field(default_factory=list, repr=False)


def cluster_instances(
    instances,
    param_type: str = "angle",
    target_width: float = 10.0,
    depth: int = 0,
) -> list[ClusterReport]:
    """Group instances by local chemistry and report per-cluster QM stats.

    Parameters
    ----------
    instances : list[InstanceRecord]
        From :func:`~.extract.extract_param_instances`.
    param_type : str
        ``'angle'``, ``'bond'``, or ``'torsion'``.
    target_width : float
        Bin-width target in native units (° for angles, Å for bonds).
        Clusters exceeding this are flagged ``split_warranted=True``.
    depth : int
        Graph depth (BESMARTS d). ``0`` (default) groups by the four primary
        axes only — unchanged from before this parameter existed. ``> 0``
        additionally keys on the Axis-5 attachment-context signature
        (:func:`~.featurize.featurize_instance`), splitting groups that were
        merged at d=0 into finer sub-populations.

    Returns
    -------
    list[ClusterReport]
        Sorted descending by ``n_instances``.
    """
    from .featurize import group_by_features

    groups = group_by_features(instances, param_type, depth=depth)

    reports: list[ClusterReport] = []
    for fkey, group_instances in groups.items():
        # Representative feature set (from first instance that resolves)
        rep_fs: FeatureSet | None = None
        for inst in group_instances:
            fs = featurize_instance(inst.rdmol, inst.atom_key, param_type, depth=depth)
            if fs is not None:
                rep_fs = fs
                break

        vals = [inst.qm_mean for inst in group_instances]
        n_mols = len({inst.mol_idx for inst in group_instances})
        qm_min = float(np.min(vals))
        qm_max = float(np.max(vals))
        qm_mean = float(np.mean(vals))
        qm_std = float(np.std(vals))
        qm_width = qm_max - qm_min

        is_strained = (
            rep_fs is not None
            and rep_fs.center_ring_sizes
            and min(rep_fs.center_ring_sizes) <= 4
        )

        reports.append(
            ClusterReport(
                feature_key=fkey,
                feature_set=rep_fs,
                n_instances=len(group_instances),
                n_molecules=n_mols,
                qm_values=vals,
                qm_min=qm_min,
                qm_max=qm_max,
                qm_mean=qm_mean,
                qm_std=qm_std,
                qm_width=qm_width,
                is_strained=is_strained,
                split_warranted=(qm_width > target_width and not is_strained),
                instances=group_instances,
            )
        )

    reports.sort(key=lambda r: -r.n_instances)
    return reports


def print_cluster_report(
    reports: list[ClusterReport],
    param_id: str = "",
    unit: str = "°",
) -> None:
    """Pretty-print a cluster report to stdout."""
    header = f"Param: {param_id}" if param_id else "Cluster report"
    total_inst = sum(r.n_instances for r in reports)
    total_mols = len({inst.mol_idx for r in reports for inst in r.instances})
    print(f"\n{header}  —  {total_inst} instances, {total_mols} molecules\n")
    fmt = f"{'cluster (bond|elem|ring|fc)':<48} | {'mols':>4} | {'inst':>4} | "
    fmt += f"{'min':>7} | {'max':>7} | {'mean':>7} | {'width':>7} | flag"
    print(fmt)
    print("-" * 100)
    for r in reports:
        flag = ""
        if r.is_strained:
            flag = "STRAINED (irreducible)"
        elif r.split_warranted:
            flag = "split?"
        print(
            f"{r.feature_key:<48} | {r.n_molecules:>4} | {r.n_instances:>4} | "
            f"{r.qm_min:>7.2f}{unit} | {r.qm_max:>7.2f}{unit} | "
            f"{r.qm_mean:>7.2f}{unit} | {r.qm_width:>7.2f}{unit} | {flag}"
        )
    print()


# ---------------------------------------------------------------------------
# 1-D optimal partition by BIC  (ported from besmarts_angle_split.py)
# ---------------------------------------------------------------------------


def _segment_sse(
    prefix_sum: np.ndarray,
    prefix_sq: np.ndarray,
    start: int,
    end: int,
) -> float:
    count = end - start
    if count <= 0:
        return 0.0
    total = prefix_sum[end] - prefix_sum[start]
    total_sq = prefix_sq[end] - prefix_sq[start]
    mean = total / count
    return max(0.0, total_sq - 2.0 * mean * total + count * mean * mean)


def optimal_1d_partitions(
    values,
    *,
    max_k: int = 6,
    min_cluster_size: int = 8,
) -> dict[int, dict]:
    """Find globally optimal 1-D partitions for k = 1 .. max_k.

    Uses dynamic programming (O(n² k)) to minimise total within-cluster
    SSE.  Returns a dict keyed by k; each value contains:

    - ``k`` : int
    - ``labels`` : np.ndarray of cluster labels (same order as *values*)
    - ``rss`` : float — total residual sum of squares
    - ``bic`` : float — Bayesian information criterion (lower is better)
    - ``clusters`` : list[dict] — per-cluster statistics

    Only partitions with ``n >= min_cluster_size`` per segment are returned.
    """
    import pandas as pd

    values = np.asarray(list(values), dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty 1-D array")

    n = values.size
    order = np.argsort(values)
    sorted_v = values[order]
    prefix_sum = np.concatenate(([0.0], np.cumsum(sorted_v)))
    prefix_sq  = np.concatenate(([0.0], np.cumsum(sorted_v * sorted_v)))

    actual_max_k = max(1, min(max_k, n // max(1, min_cluster_size)))

    dp   = np.full((actual_max_k + 1, n + 1), np.inf)
    prev = np.full((actual_max_k + 1, n + 1), -1, dtype=int)
    dp[0, 0] = 0.0

    for k in range(1, actual_max_k + 1):
        for end in range(k * min_cluster_size, n + 1):
            start_min = (k - 1) * min_cluster_size
            start_max = end - min_cluster_size
            best_cost, best_start = np.inf, -1
            for start in range(start_min, start_max + 1):
                cost = dp[k - 1, start] + _segment_sse(prefix_sum, prefix_sq, start, end)
                if cost < best_cost:
                    best_cost, best_start = cost, start
            dp[k, end] = best_cost
            prev[k, end] = best_start

    partitions: dict[int, dict] = {}
    for k in range(1, actual_max_k + 1):
        if not np.isfinite(dp[k, n]):
            continue

        segs: list[tuple[int, int]] = []
        end, cur_k = n, k
        while cur_k > 0:
            start = prev[cur_k, end]
            if start < 0:
                segs = []
                break
            segs.append((start, end))
            end, cur_k = start, cur_k - 1
        segs.reverse()
        if not segs:
            continue

        labels_sorted = np.empty(n, dtype=int)
        cluster_rows = []
        for label, (s, e) in enumerate(segs):
            labels_sorted[s:e] = label
            cv = sorted_v[s:e]
            cluster_rows.append({
                "cluster": label,
                "n":       int(e - s),
                "mean":    float(cv.mean()),
                "std":     float(cv.std(ddof=0)),
                "min":     float(cv.min()),
                "max":     float(cv.max()),
            })

        labels = np.empty(n, dtype=int)
        labels[order] = labels_sorted
        rss = float(dp[k, n])
        bic = n * math.log(max(rss / max(n, 1), 1e-12)) + (2 * k) * math.log(max(n, 2))

        partitions[k] = {
            "k":        k,
            "labels":   labels,
            "rss":      rss,
            "bic":      float(bic),
            "clusters": cluster_rows,
        }

    return partitions


def choose_partition(
    values,
    *,
    max_k: int = 6,
    min_cluster_size: int = 8,
) -> tuple[dict, dict]:
    """Return ``(best_partition, all_partitions)`` chosen by minimum BIC.

    ``best_partition`` is the entry from :func:`optimal_1d_partitions` with
    the lowest BIC (ties broken by smallest k).
    """
    parts = optimal_1d_partitions(values, max_k=max_k, min_cluster_size=min_cluster_size)
    if not parts:
        raise ValueError("Unable to build any valid partitions (too few instances?)")
    best = min(parts.values(), key=lambda p: (p["bic"], p["k"]))
    return best, parts
