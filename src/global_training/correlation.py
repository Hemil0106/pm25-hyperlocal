"""Feature correlation diagnostics (Milestone 17).

Correlation is computed ONLY on complete-case rows and reported as a diagnostic
- correlation does not imply causation and no automatic feature removal is ever
performed. Writes ``global_feature_correlation.csv``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from .schema import FEATURE_COLS, TARGET_COL

logger = logging.getLogger(__name__)


def correlation_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Pearson + Spearman correlations for target/features on complete rows.

    Returns (pairs_df, report). With fewer than two complete-case rows the
    result is empty and the report says why (honest, not an error).
    """
    complete = df[df["complete_case"]].copy() if not df.empty else df
    if complete.empty:
        return pd.DataFrame(columns=["feature_a", "feature_b", "pearson",
                                     "spearman", "n"]), {
            "status": "no_data",
            "n_complete_rows": 0,
            "note": "No complete-case rows; correlation not computable.",
        }

    cols = [TARGET_COL] + FEATURE_COLS
    matrix = complete[cols].apply(pd.to_numeric, errors="coerce")
    numeric = matrix.dropna(axis=0, how="any")
    if len(numeric) < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "pearson",
                                     "spearman", "n"]), {
            "status": "insufficient",
            "n_complete_rows": int(len(complete)),
            "note": "Fewer than two complete rows; correlation not computable.",
        }

    pearson = numeric.corr(method="pearson")
    spearman = numeric.corr(method="spearman")
    rows = []
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if j <= i:
                continue
            p = pearson.loc[a, b]
            s = spearman.loc[a, b]
            if pd.isna(p) and pd.isna(s):
                continue
            rows.append({
                "feature_a": a,
                "feature_b": b,
                "pearson": None if pd.isna(p) else round(float(p), 4),
                "spearman": None if pd.isna(s) else round(float(s), 4),
                "n": int(len(numeric)),
            })
    out = pd.DataFrame(rows, columns=["feature_a", "feature_b", "pearson",
                                      "spearman", "n"])
    report = {
        "status": "ok",
        "n_complete_rows": int(len(numeric)),
        "n_pairs": int(len(out)),
        "note": "Correlation does not imply causation; no automatic feature "
                "removal is performed.",
    }
    return out, report


def write_correlation(df: pd.DataFrame, config) -> dict:
    """Compute and write the correlation CSV; returns the report dict."""
    processed_base = Path(
        config.get("global_data", {}).get("storage", {}).get(
            "processed_base", "data/processed/global")
    )
    diagnostics_dir = processed_base / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = diagnostics_dir / "global_feature_correlation.csv"
    pairs, report = correlation_frame(df)
    pairs.to_csv(csv_path, index=False)
    report["path"] = str(csv_path)
    return report
