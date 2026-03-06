"""Visualization and reporting tools.

- :func:`draw_molecule` -- Render a 2D molecule depiction as SVG
- :func:`create_comparison_report` -- Add a pairwise comparison page to a PDF
- :func:`create_qm_comparison_report` -- Add a QM-reference comparison page to a PDF
- :func:`create_title_page` -- Add a title page to a PDF
"""

from mlip_optimizer.visualization.drawing import asciify, draw_molecule
from mlip_optimizer.visualization.reporting import (
    create_comparison_report,
    create_qm_comparison_report,
    create_title_page,
)

__all__ = [
    "asciify",
    "draw_molecule",
    "create_comparison_report",
    "create_qm_comparison_report",
    "create_title_page",
]
