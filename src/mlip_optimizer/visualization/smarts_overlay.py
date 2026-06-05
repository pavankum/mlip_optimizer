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


def plot_actual_overlay(analysis: dict, metric: str) -> None:
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

    ax_hist.set_title(f'{metric_label(metric)} actual-value overlay')
    ax_hist.set_xlabel(metric_unit(metric))
    ax_hist.set_ylabel('Count')
    ax_hist.legend(fontsize=8, frameon=False)

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
        ax_violin.set_xlabel('Source')
        ax_violin.set_title('Violin plot')
        ax_violin.grid(axis='y', alpha=0.3)

    fig.suptitle(f'{metric_label(metric)} values overlaid on QM reference', y=1.02)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_qm_split_overlay(analysis: dict, metric: str) -> None:
    """Plot QM-only value distributions split by SMARTS pattern.

    Parameters
    ----------
    analysis : dict
        Analysis dict from
        :func:`~mlip_optimizer.analysis.smarts_overlay.collect_overlay_data`.
    metric : str
        ``'bond'`` or ``'angle'``.
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

    ax_hist.set_title(f'{metric_label(metric)} QM values split by SMARTS')
    ax_hist.set_xlabel(metric_unit(metric))
    ax_hist.set_ylabel('Count')
    ax_hist.legend(fontsize=8, frameon=False)

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
    ax_violin.set_xlabel('SMARTS pattern')
    ax_violin.set_title('Violin plot')
    ax_violin.grid(axis='y', alpha=0.3)

    fig.suptitle(f'{metric_label(metric)} QM-only split by SMARTS pattern', y=1.02)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_error_overlay(analysis: dict, metric: str) -> None:
    """Plot absolute error distributions split by SMARTS pattern.

    Parameters
    ----------
    analysis : dict
        Analysis dict from
        :func:`~mlip_optimizer.analysis.smarts_overlay.collect_overlay_data`.
    metric : str
        ``'bond'`` or ``'angle'``.
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

    ax_hist.set_title(f'{metric_label(metric)} absolute error overlay')
    ax_hist.set_xlabel(f'Absolute {metric_label(metric).lower()} error')
    ax_hist.set_ylabel('Count')
    ax_hist.legend(fontsize=8, frameon=False)

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
    ax_violin.set_ylabel(f'Absolute {metric_label(metric).lower()} error')
    ax_violin.set_xlabel('SMARTS pattern')
    ax_violin.set_title('Violin plot')
    ax_violin.grid(axis='y', alpha=0.3)

    fig.suptitle(f'{metric_label(metric)} error overlay by SMARTS pattern', y=1.02)
    plt.tight_layout()
    plt.show()
    plt.close(fig)
