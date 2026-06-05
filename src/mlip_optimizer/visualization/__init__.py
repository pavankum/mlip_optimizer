"""Visualization and reporting tools.

- :func:`draw_molecule` -- Render a 2D molecule depiction as SVG
- :func:`create_comparison_report` -- Add a pairwise comparison page to a PDF
- :func:`create_qm_comparison_report` -- Add a QM-reference comparison page to a PDF
- :func:`create_smarts_error_report` -- Add per-SMARTS functional-group error pages to a PDF
- :func:`create_title_page` -- Add a title page to a PDF
- :func:`plot_actual_overlay` -- Histogram + violin of actual values by SMARTS pattern
- :func:`plot_qm_split_overlay` -- QM-only distributions split by SMARTS pattern
- :func:`plot_error_overlay` -- Absolute error distributions by SMARTS pattern
"""

from mlip_optimizer.visualization.drawing import asciify, draw_molecule
from mlip_optimizer.visualization.reporting import (
    create_comparison_report,
    create_qm_comparison_report,
    create_smarts_error_report,
    create_statistics_report,
    create_title_page,
)
from mlip_optimizer.visualization.smarts_overlay import (
    plot_actual_overlay,
    plot_error_overlay,
    plot_qm_split_overlay,
)

__all__ = [
    "asciify",
    "draw_molecule",
    "create_comparison_report",
    "create_qm_comparison_report",
    "create_smarts_error_report",
    "create_statistics_report",
    "create_title_page",
    "plot_actual_overlay",
    "plot_qm_split_overlay",
    "plot_error_overlay",
]
