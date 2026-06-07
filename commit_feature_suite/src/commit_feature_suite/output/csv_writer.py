"""CSV writing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from commit_feature_suite.output.schema import CSV_COLUMNS
from commit_feature_suite.utils import ensure_parent_dir


def write_rows_to_csv(rows: List[Dict[str, Any]], output_csv: Path) -> None:
    """Write output rows to CSV with a stable schema."""
    ensure_parent_dir(output_csv)
    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        dataframe = pd.DataFrame(columns=CSV_COLUMNS)
    else:
        dataframe = dataframe.reindex(columns=CSV_COLUMNS)
    dataframe.to_csv(output_csv, index=False)


