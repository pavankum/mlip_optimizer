"""FF parameter splitter — in-process SMARTS hierarchy design toolkit.

Provides the full pipeline from per-instance QM-value extraction through
four-axis featurization, 1-D BIC clustering, triple-level last-match-wins
validation, and batch ranking of all high-error parameters.

Quick start (interactive)
-------------------------
::

    from mlip_optimizer.analysis.bundle import load_openff_bundle
    from mlip_optimizer.analysis.ff_param_splitter import (
        extract_param_instances,
        cluster_instances, print_cluster_report,
        validate_hierarchy, print_validation_report,
    )

    bundle = load_openff_bundle(
        data_file, "openff-2.3.0", opt_root,
        label_forcefield_name="openff-2.3.0",
    )

    instances = extract_param_instances(
        bundle["records"],
        bundle["qm_results"],
        bundle["ff_param_lookups"],
        param_id="a20",
        param_type="angle",
        potential_names=["openff-2.3.0"],
    )

    reports = cluster_instances(instances, param_type="angle")
    print_cluster_report(reports, param_id="a20")

    # After proposing child SMARTS:
    hierarchy = [
        ("[*:1]~[#7X3$(*~[#6X3,#6X2,#7X2+0]):2]~[*:3]", "a20"),
        ("[*:1]-[#7X3;!r:2]-[#7X2,#7X3:3]",               "a20f"),  # hydrazide
        ("[#6;r6:1]-[#7X3;r6:2]-[#6;r6:3]",               "a20a"),  # barbiturate
        ("[*;r4:1]-[#7X3;r4:2]-[*;r4:3]",                 "a20r4"), # beta-lactam
    ]
    report = validate_hierarchy(instances, hierarchy, param_type="angle")
    print_validation_report(report, param_id="a20")
"""

from .extract import (
    InstanceRecord,
    extract_param_instances,
    save_instances,
    load_instances,
    load_combined_instances,
    sample_themol_prefix,
)
from .featurize import (
    FeatureSet,
    featurize_instance,
    group_by_features,
)
from .cluster import (
    ClusterReport,
    cluster_instances,
    print_cluster_report,
    optimal_1d_partitions,
    choose_partition,
)
from .validate import (
    ChildStats,
    ValidationReport,
    validate_hierarchy,
    print_validation_report,
)
from .batch import (
    ParameterSummary,
    discover_param_ids,
    batch_split_all,
    print_batch_report,
    propose_cluster_candidates,
    _parse_parent_constraints,
    _elem_from_prim,
    _primitive_tokens,
    _is_valid_candidate_subset,
)

__all__ = [
    # extract
    "InstanceRecord",
    "extract_param_instances",
    "save_instances",
    "load_instances",
    "load_combined_instances",
    "sample_themol_prefix",
    # featurize
    "FeatureSet",
    "featurize_instance",
    "group_by_features",
    # cluster
    "ClusterReport",
    "cluster_instances",
    "print_cluster_report",
    "optimal_1d_partitions",
    "choose_partition",
    # validate
    "ChildStats",
    "ValidationReport",
    "validate_hierarchy",
    "print_validation_report",
    # batch
    "ParameterSummary",
    "discover_param_ids",
    "batch_split_all",
    "print_batch_report",
    "propose_cluster_candidates",
    "_parse_parent_constraints",
    "_elem_from_prim",
    "_primitive_tokens",
    "_is_valid_candidate_subset",
]
