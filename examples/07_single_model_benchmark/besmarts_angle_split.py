from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem

from besmarts.codecs.codec_rdkit import graph_codec_rdkit
from besmarts.core import configs, graphs, mapper, splits, topology


DEFAULT_ATOM_PRIMITIVES = (
    "element",
    "hydrogen",
    "connectivity_total",
    "formal_charge",
    "aromatic",
)
DEFAULT_BOND_PRIMITIVES = ("bond_order", "bond_ring")


@dataclass(slots=True)
class ParentAngleEntry:
    dataset_name: str
    molecule_name: str
    smiles_mapped: str
    angle_key: tuple[int, int, int]
    qm_values: tuple[float, ...]
    mean_value: float
    fragment: object


def _compile_parent_pattern(parent_smarts: str) -> tuple[Chem.Mol, dict[int, int]]:
    query = Chem.MolFromSmarts(parent_smarts)
    if query is None:
        raise ValueError(f"Invalid SMARTS pattern: {parent_smarts}")

    map_to_query_idx: dict[int, int] = {}
    for atom_idx in range(query.GetNumAtoms()):
        atom = query.GetAtomWithIdx(atom_idx)
        map_num = atom.GetAtomMapNum()
        if map_num > 0:
            map_to_query_idx[map_num] = atom_idx

    if set(map_to_query_idx) != {1, 2, 3}:
        raise ValueError(
            "Parent angle SMARTS must contain exactly mapped atoms :1, :2, and :3"
        )

    return query, map_to_query_idx


def _mapped_smiles_from_mol(rdmol: Chem.Mol) -> str:
    mol = Chem.Mol(rdmol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    return Chem.MolToSmiles(mol, canonical=True)


def _angle_fragment_lookup(
    smiles_mapped: str, gcd: graph_codec_rdkit
) -> dict[tuple[int, int, int], object]:
    graph = gcd.smiles_decode(smiles_mapped)
    fragments = {}
    for angle in graphs.graph_to_structure_angles(graph):
        fragment = graphs.structure_remove_unselected(angle)
        fragments[tuple(fragment.select)] = fragment
    return fragments


def extract_parent_angle_entries(
    bundles,
    parent_smarts: str,
    *,
    value_mode: str = "mean",
    atom_primitives: tuple[str, ...] = DEFAULT_ATOM_PRIMITIVES,
    bond_primitives: tuple[str, ...] = DEFAULT_BOND_PRIMITIVES,
) -> list[ParentAngleEntry]:
    query, map_to_query_idx = _compile_parent_pattern(parent_smarts)
    gcd = graph_codec_rdkit(
        (None if atom_primitives is None else atom_primitives),
        (None if bond_primitives is None else bond_primitives),
    )

    entries: list[ParentAngleEntry] = []
    fragment_cache: dict[str, dict[tuple[int, int, int], object]] = {}

    for bundle in bundles:
        dataset_name = bundle["dataset_name"]
        for rec, qm_comp in zip(bundle["records"], bundle["qm_results"]):
            if qm_comp is None:
                continue

            try:
                rdmol = rec.molecule.to_rdkit()
            except Exception:
                continue

            matches = rdmol.GetSubstructMatches(query, uniquify=True)
            if not matches:
                continue

            smiles_mapped = _mapped_smiles_from_mol(rdmol)
            fragment_lookup = fragment_cache.get(smiles_mapped)
            if fragment_lookup is None:
                fragment_lookup = _angle_fragment_lookup(smiles_mapped, gcd)
                fragment_cache[smiles_mapped] = fragment_lookup

            seen_keys: set[tuple[int, int, int]] = set()
            for match in matches:
                angle_key = tuple(
                    match[map_to_query_idx[map_num]] for map_num in (1, 2, 3)
                )
                reverse_key = (angle_key[2], angle_key[1], angle_key[0])
                if angle_key in seen_keys or reverse_key in seen_keys:
                    continue

                values = qm_comp.angle_ref_values.get(angle_key)
                if values is None:
                    values = qm_comp.angle_ref_values.get(reverse_key)
                    if values is not None:
                        angle_key = reverse_key

                if values is None:
                    continue

                clean_values = tuple(float(v) for v in values if not np.isnan(v))
                if not clean_values:
                    continue

                one_based_key = tuple(idx + 1 for idx in angle_key)
                fragment = fragment_lookup.get(one_based_key)
                if fragment is None:
                    fragment = fragment_lookup.get(
                        (one_based_key[2], one_based_key[1], one_based_key[0])
                    )
                if fragment is None:
                    continue

                if value_mode == "mean":
                    rep_value = float(np.mean(clean_values))
                elif value_mode == "median":
                    rep_value = float(np.median(clean_values))
                else:
                    raise ValueError("value_mode must be 'mean' or 'median'")

                entries.append(
                    ParentAngleEntry(
                        dataset_name=dataset_name,
                        molecule_name=rec.inchi_key or rec.smiles,
                        smiles_mapped=smiles_mapped,
                        angle_key=angle_key,
                        qm_values=clean_values,
                        mean_value=rep_value,
                        fragment=fragment,
                    )
                )
                seen_keys.add(angle_key)

    return entries


def _segment_sse(prefix_sum, prefix_sq, start: int, end: int) -> float:
    count = end - start
    if count <= 0:
        return 0.0
    total = prefix_sum[end] - prefix_sum[start]
    total_sq = prefix_sq[end] - prefix_sq[start]
    mean = total / count
    return max(0.0, total_sq - 2.0 * mean * total + count * mean * mean)


def optimal_1d_partitions(
    values: Iterable[float],
    *,
    max_k: int = 6,
    min_cluster_size: int = 8,
) -> dict[int, dict[str, object]]:
    values = np.asarray(list(values), dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty 1D array")

    n_values = values.size
    order = np.argsort(values)
    sorted_values = values[order]
    prefix_sum = np.concatenate(([0.0], np.cumsum(sorted_values)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(sorted_values * sorted_values)))

    max_k = max(1, min(max_k, n_values // max(1, min_cluster_size)))
    if max_k == 0:
        max_k = 1

    dp = np.full((max_k + 1, n_values + 1), np.inf)
    prev = np.full((max_k + 1, n_values + 1), -1, dtype=int)
    dp[0, 0] = 0.0

    for k in range(1, max_k + 1):
        for end in range(k * min_cluster_size, n_values + 1):
            start_min = (k - 1) * min_cluster_size
            start_max = end - min_cluster_size
            best_cost = np.inf
            best_start = -1
            for start in range(start_min, start_max + 1):
                cost = dp[k - 1, start] + _segment_sse(
                    prefix_sum, prefix_sq, start, end
                )
                if cost < best_cost:
                    best_cost = cost
                    best_start = start
            dp[k, end] = best_cost
            prev[k, end] = best_start

    partitions: dict[int, dict[str, object]] = {}
    for k in range(1, max_k + 1):
        if not np.isfinite(dp[k, n_values]):
            continue

        segments: list[tuple[int, int]] = []
        end = n_values
        cur_k = k
        while cur_k > 0:
            start = prev[cur_k, end]
            if start < 0:
                segments = []
                break
            segments.append((start, end))
            end = start
            cur_k -= 1
        segments.reverse()
        if not segments:
            continue

        labels_sorted = np.empty(n_values, dtype=int)
        cluster_rows = []
        for label, (start, end) in enumerate(segments):
            labels_sorted[start:end] = label
            cluster_vals = sorted_values[start:end]
            cluster_rows.append(
                {
                    "cluster": label,
                    "n": int(end - start),
                    "mean": float(cluster_vals.mean()),
                    "std": float(cluster_vals.std(ddof=0)),
                    "min": float(cluster_vals.min()),
                    "max": float(cluster_vals.max()),
                }
            )

        labels = np.empty(n_values, dtype=int)
        labels[order] = labels_sorted
        rss = float(dp[k, n_values])
        bic = n_values * math.log(max(rss / max(n_values, 1), 1e-12)) + (
            2 * k
        ) * math.log(max(n_values, 2))
        partitions[k] = {
            "k": k,
            "labels": labels,
            "rss": rss,
            "bic": float(bic),
            "clusters": pd.DataFrame(cluster_rows),
        }

    return partitions


def choose_partition(
    values: Iterable[float], *, max_k: int = 6, min_cluster_size: int = 8
):
    partitions = optimal_1d_partitions(
        values, max_k=max_k, min_cluster_size=min_cluster_size
    )
    if not partitions:
        raise ValueError("Unable to build any valid partitions")
    return min(partitions.values(), key=lambda item: (item["bic"], item["k"])), partitions


def _default_perception_config() -> configs.smarts_perception_config:
    splitter = configs.smarts_splitter_config(
        1,
        2,
        0,
        1,
        0,
        1,
        unique=False,
        return_matches=True,
        max_splits=0,
        split_general=True,
        split_specific=True,
        unique_complements=False,
        unique_complements_prefer_min=True,
    )
    extender = configs.smarts_extender_config(1, 1, True)
    return configs.smarts_perception_config(splitter, extender)


def _intersection_smarts_for_cluster(
    cluster_fragments: list,
    gcd: graph_codec_rdkit,
    topo,
    depth: int = 1,
) -> str | None:
    """Compute the SMARTS intersection (most specific common pattern) for a list
    of angle fragments.  Returns None if encoding fails."""
    if not cluster_fragments:
        return None
    # Apply topology and trim to selected atoms, then extend to `depth`.
    frags = [graphs.structure_copy(f) for f in cluster_fragments]
    for f in frags:
        f.topology = topo
        f.select = tuple(f.select[i] for i in topo.primary)

    extender_cfg = configs.smarts_extender_config(depth, depth, True)
    mapper.mapper_smarts_extend(extender_cfg, frags)
    frags = [graphs.structure_remove_unselected(f) for f in frags]

    union_cfg = configs.mapper_config(0, False, "high")
    inter_cfg = configs.mapper_config(3, False, "high")
    ref = mapper.union_list(frags, union_cfg, max_depth=depth)
    intersected = mapper.intersection_list(frags, inter_cfg, max_depth=depth, reference=ref)
    graphs.subgraph_invert_null(intersected)

    # Relabel so selected atoms start at 1.
    relabel = {x: i for i, x in enumerate(intersected.select, 1)}
    for i, x in enumerate(intersected.nodes, len(intersected.select) + 1):
        if x not in intersected.select:
            relabel[x] = i
    intersected = graphs.structure_relabel_nodes(intersected, relabel)

    try:
        return gcd.smarts_encode(intersected)
    except Exception:
        return None


def derive_child_smarts(
    entries: list[ParentAngleEntry],
    labels: Iterable[int],
    *,
    parent_smarts: str,
    atom_primitives: tuple[str, ...] = DEFAULT_ATOM_PRIMITIVES,
    bond_primitives: tuple[str, ...] = DEFAULT_BOND_PRIMITIVES,
    maxmoves: int = 0,  # kept for API compatibility; not used in intersection path
    depth: int = 1,
) -> pd.DataFrame:
    labels = np.asarray(list(labels), dtype=int)
    if labels.size != len(entries):
        raise ValueError("labels length must match entries length")

    gcd = graph_codec_rdkit(
        (None if atom_primitives is None else atom_primitives),
        (None if bond_primitives is None else bond_primitives),
    )
    rows = []

    cluster_order = sorted(
        np.unique(labels),
        key=lambda cid: float(
            np.mean([entries[i].mean_value for i in np.where(labels == cid)[0]])
        ),
    )
    for cluster_id in cluster_order:
        indices = sorted(np.where(labels == cluster_id)[0].tolist())
        cluster_frags = [entries[i].fragment for i in indices]
        target_values = [entries[i].mean_value for i in indices]

        smarts = _intersection_smarts_for_cluster(cluster_frags, gcd, topology.angle, depth=depth)

        rows.append(
            {
                "cluster": int(cluster_id),
                "n": int(len(indices)),
                "target_mean": float(np.mean(target_values)),
                "target_std": float(np.std(target_values, ddof=0)),
                "smarts": smarts,
                "parent_smarts": parent_smarts,
            }
        )

    return pd.DataFrame(rows).sort_values("target_mean").reset_index(drop=True)


def build_parent_angle_dataframe(entries: list[ParentAngleEntry]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": entry.dataset_name,
            "molecule": entry.molecule_name,
            "angle_key": entry.angle_key,
            "mean_value": entry.mean_value,
            "n_qm_values": len(entry.qm_values),
            "qm_values": [entry.qm_values for entry in entries],
        }
        for entry in entries
    )


def propose_child_angle_smarts(
    bundles,
    parent_smarts: str,
    *,
    max_k: int = 6,
    min_cluster_size: int = 8,
    value_mode: str = "mean",
    maxmoves: int = 0,
    depth: int = 1,
    atom_primitives: tuple[str, ...] = DEFAULT_ATOM_PRIMITIVES,
    bond_primitives: tuple[str, ...] = DEFAULT_BOND_PRIMITIVES,
):
    entries = extract_parent_angle_entries(
        bundles,
        parent_smarts,
        value_mode=value_mode,
        atom_primitives=atom_primitives,
        bond_primitives=bond_primitives,
    )
    if not entries:
        raise ValueError(f"No parent-angle matches found for {parent_smarts}")

    values = np.asarray([entry.mean_value for entry in entries], dtype=float)
    best_partition, all_partitions = choose_partition(
        values,
        max_k=max_k,
        min_cluster_size=min_cluster_size,
    )
    child_df = derive_child_smarts(
        entries,
        best_partition["labels"],
        parent_smarts=parent_smarts,
        atom_primitives=atom_primitives,
        bond_primitives=bond_primitives,
        maxmoves=maxmoves,
        depth=depth,
    )

    recommended = [smarts for smarts in child_df["smarts"].tolist() if smarts]
    return {
        "entries": entries,
        "parent_df": build_parent_angle_dataframe(entries),
        "partition": best_partition,
        "all_partitions": all_partitions,
        "child_df": child_df,
        "recommended_smarts": recommended,
    }
