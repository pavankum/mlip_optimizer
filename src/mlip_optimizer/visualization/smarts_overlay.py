"""SMARTS overlay visualization: histogram and violin plots for geometry distributions.

Three plot functions cover the three ways to look at SMARTS-filtered data:

- :func:`plot_actual_overlay`  — actual values from the potential vs QM reference
- :func:`plot_qm_split_overlay` — QM-only distributions split by pattern
- :func:`plot_error_overlay`    — absolute error distributions by pattern
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from mlip_optimizer.analysis.smarts_overlay import metric_label, metric_unit


def _assignment_tag(hierarchy: bool) -> str:
    return 'hierarchy: last-match-wins' if hierarchy else 'independent matching'


def plot_actual_overlay(
    analysis: dict,
    metric: str,
    hierarchy: bool = False,
    potential_name: str = 'MM',
) -> None:
    """Plot overlapping histograms and violins of actual geometry values by pattern.

    Shows the QM reference distribution (black) alongside per-pattern actual
    values from the optimized potential for a side-by-side visual comparison.

    Parameters
    ----------
    analysis : dict
        Analysis dict from
        :func:`~mlip_optimizer.analysis.smarts_overlay.collect_overlay_data`.
    metric : str
        ``'bond'`` or ``'angle'``.
    hierarchy : bool, optional
        Whether the analysis used hierarchy (last-match-wins) assignment.
        Shown as a tag in the plot title.  Default ``False``.
    potential_name : str, optional
        Name of the potential / model whose values are plotted.  Used in
        titles to distinguish MM from QM.  Default ``'MM'``.
    """
    qm_vals = analysis[metric]['qm']
    pattern_items = [
        (label, bucket['actual'])
        for label, bucket in analysis[metric]['patterns'].items()
        if bucket['actual']
    ]

    if not qm_vals and not pattern_items:
        print(f'No data available for {metric}')
        return

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, (ax_hist, ax_violin) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

    all_vals = list(qm_vals) + [v for _, vals in pattern_items for v in vals]
    bins = np.linspace(min(all_vals), max(all_vals), 40) if len(all_vals) > 1 and min(all_vals) < max(all_vals) else 10

    if qm_vals:
        ax_hist.hist(qm_vals, bins=bins, color='black', alpha=0.45, label='QM ref', edgecolor='none')

    violin_data = []
    violin_labels = []
    violin_colors = []

    if qm_vals:
        violin_data.append(qm_vals)
        violin_labels.append('QM ref')
        violin_colors.append('black')

    for idx, (label, vals) in enumerate(pattern_items):
        color = colors[idx % len(colors)]
        ax_hist.hist(vals, bins=bins, color=color, alpha=0.35, label=label, edgecolor='none')
        violin_data.append(vals)
        violin_labels.append(label)
        violin_colors.append(color)

    ax_hist.set_title(
        f'Overlapping histograms — {potential_name} (MM) vs QM reference ({metric_unit(metric)})'
    )
    ax_hist.set_xlabel(metric_unit(metric))
    ax_hist.set_ylabel('Count')
    ax_hist.legend(fontsize=12, frameon=False, ncol=2)

    if violin_data:
        positions = np.arange(1, len(violin_data) + 1)
        parts = ax_violin.violinplot(violin_data, positions=positions, showmedians=True, showextrema=True)
        for idx, body in enumerate(parts['bodies']):
            body.set_facecolor(violin_colors[idx])
            body.set_alpha(0.6 if violin_colors[idx] == 'black' else 0.5)
        for part_name in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            if part_name in parts:
                parts[part_name].set_color('black')
                parts[part_name].set_linewidth(1.0)
        ax_violin.set_xticks(positions)
        ax_violin.set_xticklabels(violin_labels, rotation=30, ha='right')
        ax_violin.set_ylabel(metric_unit(metric))
        ax_violin.set_xlabel('Pattern')
        ax_violin.set_title(f'{metric_label(metric)} {potential_name} (MM) vs QM per pattern')
        ax_violin.grid(axis='y', alpha=0.3)

    fig.suptitle(
        f'{metric_label(metric)} actual values: {potential_name} (MM) vs QM reference  [{_assignment_tag(hierarchy)}]',
        y=1.02,
    )
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_qm_split_overlay(analysis: dict, metric: str, hierarchy: bool = False) -> None:
    """Plot QM-only value distributions split by SMARTS pattern.

    Parameters
    ----------
    analysis : dict
        Analysis dict from
        :func:`~mlip_optimizer.analysis.smarts_overlay.collect_overlay_data`.
    metric : str
        ``'bond'`` or ``'angle'``.
    hierarchy : bool, optional
        Whether the analysis used hierarchy (last-match-wins) assignment.
        Shown as a tag in the plot title.  Default ``False``.
    """
    pattern_items = [
        (label, bucket['qm'])
        for label, bucket in analysis[metric]['patterns'].items()
        if bucket['qm']
    ]

    if not pattern_items:
        print(f'No SMARTS-split QM data available for {metric}')
        return

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, (ax_hist, ax_violin) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

    all_vals = [v for _, vals in pattern_items for v in vals]
    bins = np.linspace(min(all_vals), max(all_vals), 40) if len(all_vals) > 1 and min(all_vals) < max(all_vals) else 10

    violin_data = []
    violin_labels = []
    violin_colors = []

    for idx, (label, vals) in enumerate(pattern_items):
        color = colors[idx % len(colors)]
        ax_hist.hist(vals, bins=bins, color=color, alpha=0.35, label=label, edgecolor='none')
        violin_data.append(vals)
        violin_labels.append(label)
        violin_colors.append(color)

    ax_hist.set_title(f'Overlapping histograms — QM {metric_label(metric)} ({metric_unit(metric)})')
    ax_hist.set_xlabel(metric_unit(metric))
    ax_hist.set_ylabel('Count')
    ax_hist.legend(fontsize=12, frameon=False, ncol=2)

    positions = np.arange(1, len(violin_data) + 1)
    parts = ax_violin.violinplot(violin_data, positions=positions, showmedians=True, showextrema=True)
    for idx, body in enumerate(parts['bodies']):
        body.set_facecolor(violin_colors[idx])
        body.set_alpha(0.5)
    for part_name in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        if part_name in parts:
            parts[part_name].set_color('black')
            parts[part_name].set_linewidth(1.0)
    ax_violin.set_xticks(positions)
    ax_violin.set_xticklabels(violin_labels, rotation=30, ha='right')
    ax_violin.set_ylabel(metric_unit(metric))
    ax_violin.set_xlabel('Pattern')
    ax_violin.set_title(f'QM {metric_label(metric)} distribution per pattern')
    ax_violin.grid(axis='y', alpha=0.3)

    fig.suptitle(
        f'QM-only {metric_label(metric)} distributions by SMARTS pattern  [{_assignment_tag(hierarchy)}]',
        y=1.02,
    )
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_error_overlay(analysis: dict, metric: str, hierarchy: bool = False) -> None:
    """Plot absolute error distributions split by SMARTS pattern.

    Parameters
    ----------
    analysis : dict
        Analysis dict from
        :func:`~mlip_optimizer.analysis.smarts_overlay.collect_overlay_data`.
    metric : str
        ``'bond'`` or ``'angle'``.
    hierarchy : bool, optional
        Whether the analysis used hierarchy (last-match-wins) assignment.
        Shown as a tag in the plot title.  Default ``False``.
    """
    pattern_items = [
        (label, bucket['errors'])
        for label, bucket in analysis[metric]['patterns'].items()
        if bucket['errors']
    ]

    if not pattern_items:
        print(f'No error data available for {metric}')
        return

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, (ax_hist, ax_violin) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

    all_vals = [v for _, vals in pattern_items for v in vals]
    bins = np.linspace(min(all_vals), max(all_vals), 40) if len(all_vals) > 1 and min(all_vals) < max(all_vals) else 10

    violin_data = []
    violin_labels = []
    violin_colors = []

    for idx, (label, vals) in enumerate(pattern_items):
        color = colors[idx % len(colors)]
        ax_hist.hist(vals, bins=bins, color=color, alpha=0.35, label=label, edgecolor='none')
        ax_hist.axvline(float(np.mean(vals)), color=color, linestyle='--', linewidth=1.0)
        violin_data.append(vals)
        violin_labels.append(label)
        violin_colors.append(color)

    ax_hist.set_title(f'Overlapping histograms — absolute {metric_label(metric).lower()} error (dashed = mean)')
    ax_hist.set_xlabel(f'Absolute {metric_label(metric).lower()} error ({metric_unit(metric)})')
    ax_hist.set_ylabel('Count')
    ax_hist.legend(fontsize=12, frameon=False, ncol=2)

    positions = np.arange(1, len(violin_data) + 1)
    parts = ax_violin.violinplot(violin_data, positions=positions, showmedians=True, showextrema=True)
    for idx, body in enumerate(parts['bodies']):
        body.set_facecolor(violin_colors[idx])
        body.set_alpha(0.5)
    for part_name in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        if part_name in parts:
            parts[part_name].set_color('black')
            parts[part_name].set_linewidth(1.0)
    ax_violin.set_xticks(positions)
    ax_violin.set_xticklabels(violin_labels, rotation=30, ha='right')
    ax_violin.set_ylabel(f'Absolute {metric_label(metric).lower()} error ({metric_unit(metric)})')
    ax_violin.set_xlabel('Pattern')
    ax_violin.set_title(f'Absolute {metric_label(metric).lower()} error distribution per pattern')
    ax_violin.grid(axis='y', alpha=0.3)

    fig.suptitle(
        f'{metric_label(metric)} absolute error by SMARTS pattern  [{_assignment_tag(hierarchy)}]',
        y=1.02,
    )
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_mm_qm_paired_overlay(
    analysis: dict,
    metric: str,
    hierarchy: bool = False,
    potential_name: str = 'MM',
) -> None:
    """Plot MM and QM geometry distributions side by side per SMARTS pattern.

    Each pattern gets two adjacent violins on a shared y-axis — QM (gray,
    left) and MM (colored, right) — so over- or under-prediction of specific
    chemical environments is immediately visible.

    Parameters
    ----------
    analysis : dict
        Analysis dict from
        :func:`~mlip_optimizer.analysis.smarts_overlay.collect_overlay_data`.
    metric : str
        ``'bond'`` or ``'angle'``.
    hierarchy : bool, optional
        Whether the analysis used hierarchy (last-match-wins) assignment.
        Shown as a tag in the plot title.  Default ``False``.
    potential_name : str, optional
        Name of the potential / model whose values are plotted.  Default ``'MM'``.
    """
    pattern_items = [
        (label, bucket['qm'], bucket['actual'])
        for label, bucket in analysis[metric]['patterns'].items()
        if bucket['qm'] or bucket['actual']
    ]

    if not pattern_items:
        print(f'No paired MM/QM data for {metric}')
        return

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    n = len(pattern_items)
    offset = 0.22
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(10, n * 1.5 + 2), 6), dpi=150)

    qm_positions: list[float] = []
    qm_data: list[list] = []
    mm_positions: list[float] = []
    mm_data: list[list] = []
    mm_colors: list[str] = []

    for i, (label, qm, actual) in enumerate(pattern_items):
        base = i + 1
        if qm:
            qm_positions.append(base - offset)
            qm_data.append(qm)
        if actual:
            mm_positions.append(base + offset)
            mm_data.append(actual)
            mm_colors.append(colors[i % len(colors)])

    if qm_data:
        qm_parts = ax.violinplot(qm_data, positions=qm_positions, widths=width,
                                 showmedians=True, showextrema=True)
        for body in qm_parts['bodies']:
            body.set_facecolor('silver')
            body.set_edgecolor('gray')
            body.set_alpha(0.7)
        for part_name in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            if part_name in qm_parts:
                qm_parts[part_name].set_color('dimgray')
                qm_parts[part_name].set_linewidth(1.0)

    if mm_data:
        mm_parts = ax.violinplot(mm_data, positions=mm_positions, widths=width,
                                 showmedians=True, showextrema=True)
        for idx, body in enumerate(mm_parts['bodies']):
            body.set_facecolor(mm_colors[idx])
            body.set_alpha(0.6)
        for part_name in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            if part_name in mm_parts:
                mm_parts[part_name].set_color('black')
                mm_parts[part_name].set_linewidth(1.0)

    ax.set_xticks(np.arange(1, n + 1))
    ax.set_xticklabels([label for label, _, _ in pattern_items], rotation=30, ha='right')
    ax.set_ylabel(metric_unit(metric))
    ax.set_xlabel('Pattern')
    ax.grid(axis='y', alpha=0.3)

    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor='silver', edgecolor='gray', alpha=0.7, label='QM'),
            Patch(facecolor='steelblue', alpha=0.6, label=f'{potential_name} (MM)'),
        ],
        fontsize=12, frameon=False,
    )

    ax.set_title(
        f'{metric_label(metric)} {potential_name} (MM) vs QM per pattern  [{_assignment_tag(hierarchy)}]',
    )
    plt.tight_layout()
    plt.show()
    plt.close(fig)
