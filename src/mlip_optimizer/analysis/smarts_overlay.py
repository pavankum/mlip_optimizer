"""SMARTS-based overlay analysis for geometry comparisons.

Provides utilities to compile indexed SMARTS patterns and collect
bond/angle value and error distributions broken down by pattern,
for use in multi-SMARTS overlay workflows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem


def normalize_patterns(raw_patterns) -> list[str]:
    """Coerce *raw_patterns* (str, list, or mixed) into a list of SMARTS strings.

    Accepts a comma- or newline-separated string, or any iterable of
    objects whose ``str()`` representation is a SMARTS pattern.
    """
    if isinstance(raw_patterns, str):
        parts = raw_patterns.replace('\n', ',').split(',')
        return [part.strip() for part in parts if part.strip()]
    return [str(p).strip() for p in raw_patterns if str(p).strip()]


def compile_patterns(raw_patterns) -> list[dict]:
    """Compile SMARTS patterns into query dicts (no atom-map requirements).

    Parameters
    ----------
    raw_patterns : str or list
        Raw SMARTS strings.

    Returns
    -------
    list[dict]
        Each dict has keys ``label``, ``smarts``, ``query``.
    """
    patterns = []
    for idx, smarts in enumerate(normalize_patterns(raw_patterns), start=1):
        query = Chem.MolFromSmarts(smarts)
        if query is None:
            raise ValueError(f'Invalid SMARTS pattern: {smarts}')
        patterns.append({'label': f'Pattern {idx}', 'smarts': smarts, 'query': query})
    return patterns


def _normalize_with_labels(raw_patterns, label_prefix: str) -> list[tuple[str, str]]:
    """Coerce *raw_patterns* into ``(smarts, label)`` pairs.

    Each item may be a plain SMARTS string (label auto-generated as
    ``'<prefix> Pattern N'``) or a two-element ``(smarts, label)`` tuple /
    list with an explicit label.  A raw string input is split on commas or
    newlines as in :func:`normalize_patterns`.
    """
    if isinstance(raw_patterns, str):
        items: list = [p.strip() for p in raw_patterns.replace('\n', ',').split(',') if p.strip()]
    else:
        items = list(raw_patterns)

    result: list[tuple[str, str]] = []
    auto_idx = 1
    for item in items:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            smarts, label = str(item[0]).strip(), str(item[1]).strip()
        else:
            smarts = str(item).strip()
            label = f'{label_prefix} Pattern {auto_idx}'
        if smarts:
            result.append((smarts, label))
            auto_idx += 1
    return result


def compile_indexed_patterns(
    raw_patterns,
    required_mapped_atoms: int,
    label_prefix: str,
) -> list[dict]:
    """Compile SMARTS patterns that require explicit atom-map numbers.

    Each pattern must contain exactly the atom-map numbers ``1`` through
    *required_mapped_atoms* (e.g. ``:1`` and ``:2`` for a bond, ``:1``,
    ``:2``, ``:3`` for an angle).

    Parameters
    ----------
    raw_patterns : str or list
        Raw SMARTS strings, or a list of ``(smarts, label)`` tuples for
        explicit per-pattern labels.
    required_mapped_atoms : int
        Number of mapped atoms required (``2`` for bond, ``3`` for angle).
    label_prefix : str
        Prefix used in auto-generated labels when no explicit label is
        supplied, e.g. ``'Bond'`` or ``'Angle'``.

    Returns
    -------
    list[dict]
        Each dict has keys ``label``, ``smarts``, ``query``,
        ``map_to_query_idx``.
    """
    patterns = []
    required = set(range(1, required_mapped_atoms + 1))

    for smarts, label in _normalize_with_labels(raw_patterns, label_prefix):
        query = Chem.MolFromSmarts(smarts)
        if query is None:
            raise ValueError(f'Invalid SMARTS pattern: {smarts}')

        map_to_query_idx: dict[int, int] = {}
        for q_idx in range(query.GetNumAtoms()):
            atom = query.GetAtomWithIdx(q_idx)
            map_num = atom.GetAtomMapNum()
            if map_num > 0:
                if map_num in map_to_query_idx:
                    raise ValueError(
                        f'Duplicate atom-map number :{map_num} in SMARTS: {smarts}'
                    )
                map_to_query_idx[map_num] = q_idx

        found = set(map_to_query_idx.keys())
        if found != required:
            raise ValueError(
                f'{label_prefix} SMARTS must contain exactly mapped atoms '
                f'{sorted(required)}. Found {sorted(found)} in: {smarts}'
            )

        patterns.append({
            'label': label,
            'smarts': smarts,
            'query': query,
            'map_to_query_idx': map_to_query_idx,
        })
    return patterns


def _extract_mapped_keys(rdmol, pattern: dict, metric: str) -> set[tuple]:
    matches = rdmol.GetSubstructMatches(pattern['query'], uniquify=True)
    keys: set[tuple] = set()
    for match in matches:
        ordered_atoms = tuple(
            match[pattern['map_to_query_idx'][map_num]]
            for map_num in sorted(pattern['map_to_query_idx'])
        )
        if metric == 'bond':
            keys.add(tuple(sorted(ordered_atoms)))
        elif metric == 'angle':
            keys.add(ordered_atoms)
    return keys


def _bond_key_in(key: tuple, selected_keys: set) -> bool:
    return tuple(sorted(key)) in selected_keys


def _angle_key_in(key: tuple, selected_keys: set) -> bool:
    return key in selected_keys or (key[2], key[1], key[0]) in selected_keys


def _ff_param_id(key: tuple, ff_lookup: dict) -> str:
    """Return the FF param_id for *key*, checking both key orderings."""
    entry = ff_lookup.get(key) or ff_lookup.get(key[::-1])
    return entry[0] if entry else ''


def metric_label(metric: str) -> str:
    """Return a display label for a geometry metric (``'bond'`` or ``'angle'``)."""
    return {'bond': 'Bond', 'angle': 'Angle'}[metric]


def metric_unit(metric: str) -> str:
    """Return the unit string for a geometry metric (``'bond'`` or ``'angle'``)."""
    return {'bond': 'Angstrom', 'angle': 'deg'}[metric]


def _assign_pattern_keys(
    rdmol,
    patterns: list[dict],
    metric: str,
    ff_param_ids: set[str] | None,
    ff_lookup: dict,
    high_error_keys: set | None,
    hierarchy: bool,
) -> tuple[dict[str, set], set]:
    """Return ``(keys_by_label, union)`` for one molecule.

    Non-hierarchy: each pattern gets all its matched keys; keys can appear
    in multiple patterns.

    Hierarchy (last-match-wins): patterns are ordered most-general → most-specific.
    Each atom-key is assigned to the *last* pattern that matches it and removed
    from all earlier patterns, exactly mirroring SMIRNOFF parameter assignment.
    """
    key_in = _bond_key_in if metric == 'bond' else _angle_key_in

    # Pass 1 — collect raw matches for every pattern, applying optional filters
    all_matches: dict[str, set] = {}
    for item in patterns:
        keys = _extract_mapped_keys(rdmol, item, metric)
        if ff_param_ids is not None:
            keys = {k for k in keys if _ff_param_id(k, ff_lookup) in ff_param_ids}
        if high_error_keys is not None:
            keys = {k for k in keys if key_in(k, high_error_keys)}
        if keys:
            all_matches[item['label']] = keys

    if not hierarchy:
        union: set = set()
        for keys in all_matches.values():
            union.update(keys)
        return all_matches, union

    # Pass 2 (hierarchy) — last-match-wins: strip keys already claimed by later patterns
    assigned: set = set()
    keys_by_label: dict[str, set] = {}
    for item in reversed(patterns):
        label = item['label']
        if label not in all_matches:
            continue
        exclusive = {k for k in all_matches[label] if not key_in(k, assigned)}
        if exclusive:
            keys_by_label[label] = exclusive
            assigned.update(exclusive)
    return keys_by_label, assigned


def collect_overlay_data(
    bundles: list[dict],
    bond_smarts_patterns,
    angle_smarts_patterns,
    potential_name: str,
    metrics: tuple[str, ...] = ('bond', 'angle'),
    forcefield_name: str | None = None,
    ff_bond_param_ids: set[str] | None = None,
    ff_angle_param_ids: set[str] | None = None,
    high_error_only: bool = False,
    hierarchy: bool = False,
) -> tuple[dict, pd.DataFrame]:
    """Collect SMARTS-filtered geometry data from comparison bundles.

    For each bundle molecule, substructure-matches every bond/angle SMARTS
    pattern, then accumulates QM reference values, potential actual values,
    and absolute errors per pattern.

    Parameters
    ----------
    bundles : list[dict]
        Bundle dicts from :func:`~mlip_optimizer.analysis.bundle.load_bundles`.
    bond_smarts_patterns : list or str
        Indexed bond SMARTS (each must have mapped atoms ``:1``, ``:2``).
    angle_smarts_patterns : list or str
        Indexed angle SMARTS (each must have atoms ``:1``, ``:2``, ``:3``).
    potential_name : str
        Name of the reference potential to pull actual values from.
    metrics : tuple[str, ...], optional
        Geometry types to collect.  Default ``('bond', 'angle')``.
    forcefield_name : str or None, optional
        OpenFF ForceField name (e.g. ``'openff-2.3.0'``) used for FF-param
        filtering.  Required when *ff_bond_param_ids* or *ff_angle_param_ids*
        are set.
    ff_bond_param_ids : set[str] or None, optional
        If given, only include bond instances whose FF parameter ID is in this
        set (e.g. ``{'b10', 'b11'}``).  Requires *forcefield_name*.
    ff_angle_param_ids : set[str] or None, optional
        If given, only include angle instances whose FF parameter ID is in this
        set (e.g. ``{'a10'}``).  Requires *forcefield_name*.
    high_error_only : bool, optional
        If ``True``, restrict to geometry instances that exceeded the error
        threshold in at least one conformer (i.e. appear in the molecule's
        ``bond_diff_table`` / ``angle_diff_table``).  This matches the subset
        shown in the per-parameter pages of the comparison report, where the
        QM reference distribution is built from high-error instances only.
        Default ``False`` (include all matched instances).
    hierarchy : bool, optional
        If ``True``, apply last-match-wins SMARTS assignment (like SMIRNOFF):
        patterns are ordered most-general → most-specific, and each atom-key
        is assigned to the *last* pattern that matches it.  Earlier (more
        general) patterns do not double-count keys already claimed by a later
        (more specific) pattern.  Default ``False`` (independent matching).

    Returns
    -------
    tuple[dict, pd.DataFrame]
        ``(analysis, summary)`` where *analysis* is a nested dict of
        collected values keyed by metric/pattern, and *summary* is a
        DataFrame of per-pattern point counts.
    """
    bond_patterns = compile_indexed_patterns(bond_smarts_patterns, 2, 'Bond')
    angle_patterns = compile_indexed_patterns(angle_smarts_patterns, 3, 'Angle')

    def _empty_pattern_buckets(items: list[dict]) -> dict:
        return {
            item['label']: {
                'smarts': item['smarts'],
                'qm': [],
                'actual': [],
                'errors': [],
                'matched_molecules': 0,
            }
            for item in items
        }

    analysis: dict = {
        'bond': {'qm': [], 'patterns': _empty_pattern_buckets(bond_patterns)},
        'angle': {'qm': [], 'patterns': _empty_pattern_buckets(angle_patterns)},
    }

    need_ff_filter = ff_bond_param_ids is not None or ff_angle_param_ids is not None
    _ff_param_lookup_fn = None
    if need_ff_filter and forcefield_name:
        try:
            from mlip_optimizer.comparison import get_ff_param_lookup as _ff_param_lookup_fn
        except ImportError:
            pass

    for bundle in bundles:
        bundle_ff_lookups = bundle.get('ff_param_lookups')
        for mol_idx, (rec, qm_comp) in enumerate(zip(bundle['records'], bundle['qm_results'])):
            if qm_comp is None:
                continue
            try:
                rdmol = rec.molecule.to_rdkit()
            except Exception:
                continue

            ff_lookup: dict = {}
            if need_ff_filter:
                if bundle_ff_lookups is not None and mol_idx < len(bundle_ff_lookups):
                    ff_lookup = bundle_ff_lookups[mol_idx] or {}
                elif _ff_param_lookup_fn is not None:
                    try:
                        ff_lookup = _ff_param_lookup_fn(rec.molecule, forcefield_name)
                    except Exception:
                        pass

            high_error_bond_keys: set | None = None
            high_error_angle_keys: set | None = None
            if high_error_only:
                high_error_bond_keys = {
                    row[0] for row in qm_comp.bond_diff_table
                    if isinstance(row, (list, tuple)) and row and isinstance(row[0], tuple)
                }
                high_error_angle_keys = {
                    row[0] for row in qm_comp.angle_diff_table
                    if isinstance(row, (list, tuple)) and row and isinstance(row[0], tuple)
                }

            bond_keys_by_label, molecule_bond_union = _assign_pattern_keys(
                rdmol, bond_patterns, 'bond',
                ff_bond_param_ids, ff_lookup, high_error_bond_keys, hierarchy,
            )
            angle_keys_by_label, molecule_angle_union = _assign_pattern_keys(
                rdmol, angle_patterns, 'angle',
                ff_angle_param_ids, ff_lookup, high_error_angle_keys, hierarchy,
            )

            if molecule_bond_union:
                for atom_key, values in qm_comp.bond_ref_values.items():
                    if _bond_key_in(atom_key, molecule_bond_union):
                        analysis['bond']['qm'].extend(
                            float(v) for v in values if not np.isnan(v)
                        )

            if molecule_angle_union:
                for atom_key, values in qm_comp.angle_ref_values.items():
                    if _angle_key_in(atom_key, molecule_angle_union):
                        analysis['angle']['qm'].extend(
                            float(v) for v in values if not np.isnan(v)
                        )

            for label, selected_keys in bond_keys_by_label.items():
                bucket = analysis['bond']['patterns'][label]
                bucket['matched_molecules'] += 1
                for atom_key, values in qm_comp.bond_ref_values.items():
                    if _bond_key_in(atom_key, selected_keys):
                        bucket['qm'].extend(float(v) for v in values if not np.isnan(v))

            for label, selected_keys in angle_keys_by_label.items():
                bucket = analysis['angle']['patterns'][label]
                bucket['matched_molecules'] += 1
                for atom_key, values in qm_comp.angle_ref_values.items():
                    if _angle_key_in(atom_key, selected_keys):
                        bucket['qm'].extend(float(v) for v in values if not np.isnan(v))

            metrics_list = qm_comp.per_potential.get(potential_name, [])

            for label, selected_keys in bond_keys_by_label.items():
                bucket = analysis['bond']['patterns'][label]
                for m in metrics_list:
                    if m.opt_failed:
                        continue
                    for atom_key, value in m.bond_values.items():
                        if _bond_key_in(atom_key, selected_keys) and not np.isnan(value):
                            bucket['actual'].append(float(value))
                    for atom_key, diff in m.bond_diffs.items():
                        if _bond_key_in(atom_key, selected_keys) and not np.isnan(diff):
                            bucket['errors'].append(abs(float(diff)))

            for label, selected_keys in angle_keys_by_label.items():
                bucket = analysis['angle']['patterns'][label]
                for m in metrics_list:
                    if m.opt_failed:
                        continue
                    for atom_key, value in m.angle_values.items():
                        if _angle_key_in(atom_key, selected_keys) and not np.isnan(value):
                            bucket['actual'].append(float(value))
                    for atom_key, diff in m.angle_diffs.items():
                        if _angle_key_in(atom_key, selected_keys) and not np.isnan(diff):
                            bucket['errors'].append(abs(float(diff)))

    summary_rows: list[dict] = []
    for metric in metrics:
        for label, bucket in analysis[metric]['patterns'].items():
            summary_rows.append({
                'metric': metric,
                'pattern': label,
                'smarts': bucket['smarts'],
                'matched_molecules': bucket['matched_molecules'],
                'qm_points': len(bucket['qm']),
                'value_points': len(bucket['actual']),
                'error_points': len(bucket['errors']),
            })

    return analysis, pd.DataFrame(summary_rows)


def collect_hierarchy_overlay_data(
    bundles: list[dict],
    bond_smarts_patterns,
    angle_smarts_patterns,
    potential_name: str,
    metrics: tuple[str, ...] = ('bond', 'angle'),
    forcefield_name: str | None = None,
    ff_bond_param_ids: set[str] | None = None,
    ff_angle_param_ids: set[str] | None = None,
    high_error_only: bool = False,
) -> tuple[dict, pd.DataFrame]:
    """Like :func:`collect_overlay_data` but with SMIRNOFF-style last-match-wins assignment.

    Patterns must be ordered **most-general → most-specific** (same convention
    as an OpenFF ForceField SMIRKS list).  Each atom-key (bond or angle
    instance) is assigned to the *last* pattern that matches it; earlier
    patterns do not count instances that a later pattern already claimed.

    This means the counts in the summary table are mutually exclusive — the
    total across all patterns equals the number of unique matched instances,
    with no double-counting.

    All other parameters are identical to :func:`collect_overlay_data`.
    """
    return collect_overlay_data(
        bundles,
        bond_smarts_patterns,
        angle_smarts_patterns,
        potential_name,
        metrics=metrics,
        forcefield_name=forcefield_name,
        ff_bond_param_ids=ff_bond_param_ids,
        ff_angle_param_ids=ff_angle_param_ids,
        high_error_only=high_error_only,
        hierarchy=True,
    )
