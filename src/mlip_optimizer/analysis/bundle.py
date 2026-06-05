"""QM reference bundle loading and caching utilities.

Provides functions to discover QM data files, load optimized SDF results,
compute QM comparisons, and cache results to disk for fast re-use.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from mlip_optimizer import evaluate_against_qm
from mlip_optimizer.data import load_records
from mlip_optimizer.io import read_optimized_sdf


def discover_qm_files(qm_root: str | Path) -> list[Path]:
    """Return sorted list of QM data files under *qm_root*.

    Searches recursively for ``.parquet`` files first; falls back to
    ``.sdf`` files if none are found.

    Parameters
    ----------
    qm_root : str or Path
        Root directory to search.

    Returns
    -------
    list[Path]
        Sorted list of discovered file paths.
    """
    qm_root = Path(qm_root)
    if not qm_root.exists():
        raise FileNotFoundError(f'QM root not found: {qm_root}')

    parquet_files = sorted(qm_root.rglob('*.parquet'))
    if parquet_files:
        return parquet_files

    sdf_files = sorted(qm_root.rglob('*.sdf'))
    if sdf_files:
        return sdf_files

    raise FileNotFoundError(f'No parquet or sdf files found under: {qm_root}')


def _base_dataset_name(path: Path) -> str:
    stem = path.stem
    parts = stem.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 15:
        return parts[0]
    return stem


def _file_signature(path: Path) -> dict:
    stat = path.stat()
    return {
        'path': str(path.resolve()),
        'size': int(stat.st_size),
        'mtime_ns': int(stat.st_mtime_ns),
    }


def _cache_file_path(
    data_file: Path,
    dataset_name: str,
    potential_name: str,
    cache_root: Path,
    bond_threshold: float,
    angle_threshold: float,
    torsion_threshold: float,
) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    key = (
        f"{data_file.resolve()}|{dataset_name}|{potential_name}"
        f"|{bond_threshold}|{angle_threshold}|{torsion_threshold}"
    )
    digest = hashlib.sha1(key.encode('utf-8')).hexdigest()[:12]
    safe_pot = potential_name.replace('/', '_').replace(':', '_').replace('.', '_')
    return cache_root / f"{dataset_name}__{safe_pot}__{digest}.pkl.gz"


def _build_ff_param_lookups(records, forcefield_name: str) -> list[dict]:
    """Label every molecule with *forcefield_name* and return per-molecule FF param dicts.

    The ForceField is loaded once and shared across threads.  RDKit SMARTS
    matching (used internally by label_molecules) releases the GIL, so
    ThreadPoolExecutor gives real parallelism here.
    """
    try:
        from openff.toolkit import ForceField, Topology
    except ImportError:
        return [{} for _ in records]

    ff = None
    for candidate in (forcefield_name, forcefield_name + '.offxml'):
        try:
            ff = ForceField(candidate)
            break
        except Exception:
            continue
    if ff is None:
        return [{} for _ in records]

    def _label_one(rec):
        try:
            topology = Topology.from_molecules([rec.molecule])
            mol_forces = ff.label_molecules(topology)[0]
        except Exception:
            return {}
        lookup: dict = {}
        for force_tag, force_dict in mol_forces.items():
            if force_tag not in ('Bonds', 'Angles', 'ProperTorsions'):
                continue
            for atom_indices, param in force_dict.items():
                entry = (param.id, param.smirks)
                lookup[atom_indices] = entry
                lookup[atom_indices[::-1]] = entry
        return lookup

    n_workers = min(os.cpu_count() or 4, len(records), 16)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(_label_one, records))


def _ff_param_cache_file_path(
    data_file: Path,
    dataset_name: str,
    forcefield_name: str,
    cache_root: Path,
) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    key = f"{data_file.resolve()}|{dataset_name}|{forcefield_name}"
    digest = hashlib.sha1(key.encode('utf-8')).hexdigest()[:12]
    safe_ff = forcefield_name.replace('/', '_').replace(':', '_').replace('.', '_')
    return cache_root / f"{dataset_name}__ff_{safe_ff}__{digest}.pkl.gz"


def _load_cached_ff_params(cache_path: Path, signature: dict) -> list | None:
    if not cache_path.exists():
        return None
    try:
        with gzip.open(cache_path, 'rb') as fh:
            payload = pickle.load(fh)
    except Exception:
        return None
    if payload.get('version') != 1:
        return None
    if payload.get('signature') != signature:
        return None
    return payload.get('ff_param_lookups')


def _save_cached_ff_params(cache_path: Path, signature: dict, ff_param_lookups: list) -> None:
    payload = {
        'version': 1,
        'signature': signature,
        'ff_param_lookups': ff_param_lookups,
    }
    with gzip.open(cache_path, 'wb') as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _load_cached_qm_results(cache_path: Path, signature: dict) -> Any | None:
    if not cache_path.exists():
        return None
    try:
        with gzip.open(cache_path, 'rb') as fh:
            payload = pickle.load(fh)
    except Exception:
        return None
    if payload.get('version') != 1:
        return None
    if payload.get('signature') != signature:
        return None
    return payload.get('qm_results')


def _save_cached_qm_results(cache_path: Path, signature: dict, qm_results: Any) -> None:
    payload = {
        'version': 1,
        'signature': signature,
        'qm_results': qm_results,
    }
    with gzip.open(cache_path, 'wb') as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _find_openff_sdf(
    opt_dataset_dir: Path,
    data_file: Path,
    potential_name: str,
) -> Path:
    if not opt_dataset_dir.is_dir():
        raise FileNotFoundError(f'Missing optimized directory: {opt_dataset_dir}')

    sdf_files = sorted(opt_dataset_dir.glob('optimized_*.sdf'))
    if not sdf_files:
        raise FileNotFoundError(f'No optimized_*.sdf files found in {opt_dataset_dir}')

    selected_sdf = None
    for sdf_file in sdf_files:
        try:
            model_name, _ = read_optimized_sdf(sdf_file, [])
            if model_name == potential_name:
                selected_sdf = sdf_file
                break
        except Exception:
            continue

    if selected_sdf is None:
        for sdf_file in sdf_files:
            if 'openff-2_3_0' in sdf_file.name or 'openff_2_3_0' in sdf_file.name:
                selected_sdf = sdf_file
                break

    if selected_sdf is None:
        available = [f.name for f in sdf_files]
        raise ValueError(
            f'Could not find {potential_name!r} SDF in {opt_dataset_dir}. Files: {available}'
        )

    return selected_sdf


def load_openff_bundle(
    data_file: str | Path,
    potential_name: str,
    optimized_root: str | Path,
    cache_root: str | Path | None = None,
    *,
    use_cache: bool = True,
    force_rebuild_cache: bool = False,
    bond_threshold: float = 0.1,
    angle_threshold: float = 5.0,
    torsion_threshold: float = 40.0,
    label_forcefield_name: str | None = None,
) -> dict:
    """Load QM records and compute/cache comparisons for one data file.

    Parameters
    ----------
    data_file : str or Path
        QM reference data file (.parquet or .sdf).
    potential_name : str
        Name of the potential to compare against.
    optimized_root : str or Path
        Root directory containing per-dataset optimized SDF subdirectories.
    cache_root : str or Path or None, optional
        Directory for cached results.  If ``None``, defaults to a
        ``cache/qm_diffs`` subdirectory next to *optimized_root*.
    use_cache : bool, optional
        Whether to read/write the on-disk cache.  Default ``True``.
    force_rebuild_cache : bool, optional
        Ignore any existing cache and recompute.  Default ``False``.
    bond_threshold : float, optional
        Bond difference threshold in Angstrom.  Default ``0.1``.
    angle_threshold : float, optional
        Angle difference threshold in degrees.  Default ``5.0``.
    torsion_threshold : float, optional
        Torsion difference threshold in degrees.  Default ``40.0``.
    label_forcefield_name : str or None, optional
        If given, assign FF parameter IDs to every bond/angle/torsion in every
        molecule using this OpenFF ForceField and cache the results.  The
        lookups are stored in the returned bundle under ``ff_param_lookups``
        (a list parallel to ``records``), enabling fast FF-param filtering in
        :func:`~mlip_optimizer.analysis.smarts_overlay.collect_overlay_data`.

    Returns
    -------
    dict
        Keys: ``data_file``, ``dataset_name``, ``records``, ``qm_results``,
        ``optimized_sdf``, ``cache_path``, ``ff_param_lookups``,
        ``ff_param_cache_path``.  ``ff_param_lookups`` is ``None`` when
        *label_forcefield_name* was not supplied.
    """
    data_file = Path(data_file)
    optimized_root = Path(optimized_root)
    if cache_root is None:
        cache_root = optimized_root.parent / 'cache' / 'qm_diffs'
    cache_root = Path(cache_root)

    records = load_records(data_file)
    dataset_name = _base_dataset_name(data_file)
    opt_dir = optimized_root / dataset_name
    selected_sdf = _find_openff_sdf(opt_dir, data_file, potential_name)

    model_name, selected = read_optimized_sdf(selected_sdf, records)
    if model_name != potential_name:
        raise ValueError(
            f'Expected model {potential_name!r}, found {model_name!r} in {selected_sdf}'
        )

    signature = {
        'data_file': _file_signature(data_file),
        'optimized_sdf': _file_signature(selected_sdf),
        'potential_name': potential_name,
        'bond_threshold': float(bond_threshold),
        'angle_threshold': float(angle_threshold),
        'torsion_threshold': float(torsion_threshold),
        'n_records': len(records),
    }
    cache_path = _cache_file_path(
        data_file, dataset_name, potential_name, cache_root,
        bond_threshold, angle_threshold, torsion_threshold,
    )

    qm_results = None
    if use_cache and not force_rebuild_cache:
        qm_results = _load_cached_qm_results(cache_path, signature)

    if qm_results is None:
        qm_results = []
        for rec, opt_mol in zip(records, selected):
            qm_results.append(
                evaluate_against_qm(
                    rec.molecule,
                    {potential_name: opt_mol},
                    bond_threshold=bond_threshold,
                    angle_threshold=angle_threshold,
                    torsion_threshold=torsion_threshold,
                    inchi_key=rec.inchi_key,
                    smiles=rec.smiles,
                    molecule_name=rec.inchi_key or rec.smiles,
                    record_ids=rec.record_ids,
                    forcefield_name=potential_name,
                )
            )
        if use_cache:
            _save_cached_qm_results(cache_path, signature, qm_results)
            print(f'cache miss -> wrote: {cache_path}')
    else:
        print(f'cache hit  -> loaded: {cache_path}')

    ff_param_lookups = None
    ff_param_cache_path = None
    if label_forcefield_name:
        ff_sig = {
            'data_file': _file_signature(data_file),
            'forcefield_name': label_forcefield_name,
            'n_records': len(records),
        }
        ff_param_cache_path = _ff_param_cache_file_path(
            data_file, dataset_name, label_forcefield_name, cache_root
        )
        if use_cache and not force_rebuild_cache:
            ff_param_lookups = _load_cached_ff_params(ff_param_cache_path, ff_sig)
        if ff_param_lookups is None:
            ff_param_lookups = _build_ff_param_lookups(records, label_forcefield_name)
            if use_cache:
                _save_cached_ff_params(ff_param_cache_path, ff_sig, ff_param_lookups)
                print(f'ff cache miss -> wrote: {ff_param_cache_path}')
        else:
            print(f'ff cache hit  -> loaded: {ff_param_cache_path}')

    return {
        'data_file': data_file,
        'dataset_name': dataset_name,
        'records': records,
        'qm_results': qm_results,
        'optimized_sdf': selected_sdf,
        'cache_path': cache_path,
        'ff_param_lookups': ff_param_lookups,
        'ff_param_cache_path': ff_param_cache_path,
    }


def load_bundles(
    qm_root: str | Path,
    optimized_root: str | Path,
    potential_name: str,
    *,
    data_files_override=None,
    cache_root: str | Path | None = None,
    use_cache: bool = True,
    force_rebuild_cache: bool = False,
    bond_threshold: float = 0.1,
    angle_threshold: float = 5.0,
    torsion_threshold: float = 40.0,
    label_forcefield_name: str | None = None,
) -> tuple[dict, list]:
    """Discover QM files and load comparison bundles for all of them.

    Parameters
    ----------
    qm_root : str or Path
        Root directory containing QM data files.
    optimized_root : str or Path
        Root directory containing per-dataset optimized SDF subdirectories.
    potential_name : str
        Name of the potential to compare against.
    data_files_override : None, str, Path, or list, optional
        If given, use this/these file(s) instead of auto-discovery under
        *qm_root*.
    cache_root : str or Path or None, optional
        Cache directory.  See :func:`load_openff_bundle`.
    use_cache : bool, optional
        Whether to use the on-disk cache.  Default ``True``.
    force_rebuild_cache : bool, optional
        Ignore existing cache and recompute.  Default ``False``.
    bond_threshold, angle_threshold, torsion_threshold : float, optional
        Thresholds for geometry comparisons.
    label_forcefield_name : str or None, optional
        If given, compute and cache FF parameter assignments for all molecules.
        See :func:`load_openff_bundle`.

    Returns
    -------
    tuple[dict, list]
        ``(runtime_config, bundles)`` where *bundles* is a list of dicts,
        one per data file.
    """
    if data_files_override is None:
        data_files = discover_qm_files(Path(qm_root))
    elif isinstance(data_files_override, (str, Path)):
        data_files = [Path(data_files_override)]
    else:
        data_files = [Path(p) for p in data_files_override]

    bundles = [
        load_openff_bundle(
            data_file,
            potential_name,
            optimized_root,
            cache_root=cache_root,
            use_cache=use_cache,
            force_rebuild_cache=force_rebuild_cache,
            bond_threshold=bond_threshold,
            angle_threshold=angle_threshold,
            torsion_threshold=torsion_threshold,
            label_forcefield_name=label_forcefield_name,
        )
        for data_file in data_files
    ]

    runtime_config = {
        'qm_root': str(qm_root),
        'optimized_directory': str(optimized_root),
        'cache_root': str(cache_root) if cache_root else None,
        'use_cache': bool(use_cache),
        'force_rebuild_cache': bool(force_rebuild_cache),
        'n_data_files': len(data_files),
        'potential_name': potential_name,
        'bond_threshold': float(bond_threshold),
        'angle_threshold': float(angle_threshold),
        'torsion_threshold': float(torsion_threshold),
        'label_forcefield_name': label_forcefield_name,
    }
    return runtime_config, bundles
