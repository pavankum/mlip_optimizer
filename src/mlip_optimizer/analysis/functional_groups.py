"""Functional-group substructure matching with persistent caching.

Functional group definitions live in
``src/mlip_optimizer/data/functional_groups.csv`` — a merged,
SMARTS-deduplicated table derived from the element-taxonomy CSVs
(sulfur, phosphorus, nitrogen).

Typical usage
-------------
    from mlip_optimizer.analysis.functional_groups import (
        load_functional_groups,
        match_and_cache,
    )

    fg_records = load_functional_groups()          # once at startup
    rdmol = molecule.to_rdkit()
    matches = match_and_cache(
        inchi_key, rdmol, fg_records, cache_path="outputs/fg_cache.pkl"
    )
    # matches: [("Sulfonamide", [(7, 8, 9, 10), ...]), ...]
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import NamedTuple

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_DEFAULT_FG_CSV = _DATA_DIR / "functional_groups.csv"


class FGRecord(NamedTuple):
    name: str
    smarts: str


def load_functional_groups(csv_path: str | Path | None = None) -> list[FGRecord]:
    """Load functional group (name, SMARTS) records, deduplicated by SMARTS.

    Reads the built-in ``functional_groups.csv`` by default.  Invalid or
    unparseable SMARTS patterns are silently skipped.

    Parameters
    ----------
    csv_path : path-like, optional
        Override the built-in CSV path.

    Returns
    -------
    list[FGRecord]
        Ordered list of (name, smarts) named-tuples, one per unique SMARTS.
    """
    import pandas as pd
    from rdkit import Chem

    path = Path(csv_path) if csv_path is not None else _DEFAULT_FG_CSV
    df = pd.read_csv(str(path))
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip().fillna("")

    seen: set[str] = set()
    records: list[FGRecord] = []
    for _, row in df.iterrows():
        name = str(row.get("Functional Group", "") or "").strip()
        smarts = str(row.get("SMARTS", "") or "").strip()
        if not name or not smarts or smarts.lower() == "nan":
            continue
        if smarts in seen:
            continue
        try:
            if Chem.MolFromSmarts(smarts) is None:
                continue
        except Exception:
            continue
        seen.add(smarts)
        records.append(FGRecord(name=name, smarts=smarts))
    return records


def match_molecule(
    rdmol,
    fg_records: list[FGRecord],
) -> list[tuple[str, list[tuple[int, ...]]]]:
    """Return all functional groups that match *rdmol*.

    Parameters
    ----------
    rdmol : rdkit.Chem.Mol
        RDKit molecule (with or without Hs).
    fg_records : list[FGRecord]
        Output of :func:`load_functional_groups`.

    Returns
    -------
    list of (name, match_tuples)
        Only groups with at least one match are included.
        Each *match_tuple* is a tuple of atom indices in *rdmol*.
    """
    from rdkit import Chem

    results: list[tuple[str, list[tuple[int, ...]]]] = []
    for rec in fg_records:
        try:
            query = Chem.MolFromSmarts(rec.smarts)
        except Exception:
            continue
        if query is None:
            continue
        matches = rdmol.GetSubstructMatches(query)
        if matches:
            results.append((rec.name, list(matches)))
    return results


def match_and_cache(
    cache_key: str,
    rdmol,
    fg_records: list[FGRecord],
    cache_path: str | Path,
) -> list[tuple[str, list[tuple[int, ...]]]]:
    """Cache-backed :func:`match_molecule`.

    Results are persisted in a pickle file at *cache_path*, keyed by
    *cache_key* (typically the molecule's InChI key).  A cache miss
    runs the full matching and writes back to disk.

    Parameters
    ----------
    cache_key : str
        Unique identifier for the molecule (e.g. InChI key).
    rdmol : rdkit.Chem.Mol
        Molecule to match.
    fg_records : list[FGRecord]
        Functional group definitions from :func:`load_functional_groups`.
    cache_path : path-like
        Path to the pickle cache file.  Parent directory is created if needed.

    Returns
    -------
    list of (name, match_tuples)
        Same format as :func:`match_molecule`.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache: dict = {}
    if cache_path.exists():
        try:
            with cache_path.open("rb") as fh:
                cache = pickle.load(fh)
        except Exception:
            cache = {}

    if cache_key in cache:
        return cache[cache_key]

    result = match_molecule(rdmol, fg_records)
    cache[cache_key] = result
    try:
        with cache_path.open("wb") as fh:
            pickle.dump(cache, fh)
    except Exception:
        pass
    return result


def format_fg_matches(
    matches: list[tuple[str, list[tuple[int, ...]]]],
    max_matches_per_group: int = 3,
) -> str:
    """Format functional group matches as a compact plain-text string.

    Example output::

        Sulfonamide [7,8,9,10]; Thioamide [4,5,6], [11,12,13]

    Parameters
    ----------
    matches : list of (name, match_tuples)
        Output of :func:`match_molecule` or :func:`match_and_cache`.
    max_matches_per_group : int
        Cap on how many match sites to show per group (avoids very long lines
        for highly symmetric molecules).  Defaults to 3.

    Returns
    -------
    str
        Semicolon-separated groups, each showing up to *max_matches_per_group*
        match-site index lists in brackets.  Returns ``"—"`` when empty.
    """
    if not matches:
        return "—"
    parts: list[str] = []
    for name, match_tuples in matches:
        shown = match_tuples[:max_matches_per_group]
        idx_strs = [f"[{','.join(str(i) for i in t)}]" for t in shown]
        suffix = f", +{len(match_tuples) - max_matches_per_group} more" if len(match_tuples) > max_matches_per_group else ""
        parts.append(f"{name} {', '.join(idx_strs)}{suffix}")
    return "; ".join(parts)
