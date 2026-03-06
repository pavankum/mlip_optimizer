#!/usr/bin/env python
"""Check the completion status of QCArchive datasets.

Demonstrates the mlip_optimizer data status-checking utilities:
1. Load a dataset configuration (JSON file or inline dict)
2. Query QCArchive for the status of each dataset
3. Print a summary of completion counts

Requirements
------------
Install the qcarchive extras first::

    pip install -e ".[qcarchive]"

Usage
-----
::

    python examples/03_check_dataset_status/check_dataset_status.py
"""

from pathlib import Path
from pprint import pprint

from mlip_optimizer.data import check_dataset_status, get_dataset_status

# Point this at any dataset config JSON file.
CONFIG_PATH = Path(__file__).parent / "inputs" / "nsp_datasets.json"


def main():
    # --- Batch status check from a config file ---
    print(f"Checking dataset status from: {CONFIG_PATH}\n")
    statuses = check_dataset_status(CONFIG_PATH)

    for name, status in statuses.items():
        print(f"  {name}:")
        pprint(status, indent=4)
        print()

    # --- Single dataset status check ---
    print("Single dataset check:")
    status = get_dataset_status(
        "OpenFF Theory Benchmarking Set v1.0",
        dataset_type="torsiondrive",
    )
    print("  torsiondrive/OpenFF Theory Benchmarking Set v1.0:")
    pprint(status, indent=4)


if __name__ == "__main__":
    main()
