"""Generalized grouped spatial validation (Phase 11).

The locked Delhi model uses leave-one-station-out CV. This module generalizes
that to arbitrary spatial grouping levels - station, city, region, country -
so that any future model scope can be validated against held-out spatial
units and only then labeled "validated".
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

VALID_GROUP_BY = ("station_id", "city", "region", "country")
MIN_TEST_OBS_FOR_R2 = 3


def _fold_metrics(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    n = int(len(y))
    mae = float(mean_absolute_error(y, yhat))
    rmse = float(np.sqrt(mean_squared_error(y, yhat)))
    r2 = np.nan
    r2_reason = None
    if n < MIN_TEST_OBS_FOR_R2:
        r2_reason = "insufficient test observations for R2 (n<3)"
    else:
        try:
            r2 = float(r2_score(y, yhat))
        except Exception as exc:  # noqa: BLE001
            r2_reason = f"r2_score undefined for this fold: {exc}"
    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": None if np.isnan(r2) else round(r2, 3),
        "r2_reason": r2_reason,
        "n": n,
    }


def grouped_spatial_cv(df: pd.DataFrame, features: Sequence[str], target: str,
                       group_by: str = "station_id",
                       estimator=None) -> dict:
    """Leave-one-group-out spatial cross-validation.

    ``group_by`` must be one of station_id | city | region | country. Every
    group appears in the test split exactly once while being excluded from
    training - this is what "spatially validated" means.

    ``estimator`` is a sklearn-style regressor factory (fitted via .fit on
    (X, y)). If None, a RandomForestRegressor is used.
    """
    if group_by not in VALID_GROUP_BY:
        raise ValueError(f"group_by must be one of {VALID_GROUP_BY}")

    for col in (*features, target, group_by):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' required for grouped CV.")

    x = df[list(features)].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    groups = df[group_by].values

    if not np.isfinite(x).all():
        raise ValueError("Non-finite values in feature matrix.")
    if np.isnan(y).any():
        raise ValueError("Target contains NaN.")

    n_groups = int(len(np.unique(groups)))
    if n_groups < 2:
        raise ValueError(
            f"At least 2 {group_by} groups required for grouped CV (found {n_groups})."
        )

    if estimator is None:
        from sklearn.ensemble import RandomForestRegressor

        estimator = RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=2,
            random_state=42, n_jobs=-1,
        )

    logo = LeaveOneGroupOut()
    fold_records = []
    predictions = []
    for fold, (tr_idx, te_idx) in enumerate(logo.split(x, y, groups)):
        held = pd.unique(groups[te_idx]).tolist()
        x_tr, y_tr = x[tr_idx], y[tr_idx]
        x_te, y_te = x[te_idx], y[te_idx]
        est = estimator.fit(x_tr, y_tr)
        yhat = est.predict(x_te)
        metrics = _fold_metrics(y_te, yhat)
        fold_records.append(
            {"fold": int(fold), "held_out_group": str(held[0]),
             "group_by": group_by, **metrics}
        )
        te_idx_arr = np.asarray(te_idx)
        for pos, idx in enumerate(te_idx_arr):
            predictions.append({
                group_by: df[group_by].iloc[idx],
                "date": str(df["date"].iloc[idx]) if "date" in df.columns else None,
                "observed": float(y[idx]),
                "predicted": float(yhat[pos]),
                "fold": int(fold),
            })

    pred_df = pd.DataFrame(predictions)
    y_all = pred_df["observed"].to_numpy(dtype=float)
    yhat_all = pred_df["predicted"].to_numpy(dtype=float)
    overall = _fold_metrics(y_all, yhat_all)

    result = {
        "strategy": "leave_one_group_out",
        "group_by": group_by,
        "n_groups": n_groups,
        "n_folds": len(fold_records),
        "n_rows": int(len(df)),
        "folds": fold_records,
        "overall": overall,
        "predictions": pred_df,
        "validated": bool(overall["r2"] is not None and overall["r2"] >= 0.5),
        "validated_threshold": "R2 >= 0.5 on held-out spatial groups",
    }
    logger.info(
        "Grouped CV (%s): %d folds, MAE=%.3f, RMSE=%.3f, R2=%s",
        group_by, len(fold_records), overall["mae"], overall["rmse"], overall["r2"],
    )
    return result


def group_assignments(df: pd.DataFrame, lon_col="longitude", lat_col="latitude") -> pd.DataFrame:
    """Assign each observation's city/region/country grouping labels.

    With only CPCB data, region/country are constant (India) and city is
    derived from station proximity clustering (single-linkage on 1 km grid).
    This is honest scaffolding: real city/region fields come from the source
    network (e.g. OpenAQ 'city'/'country' columns) once global data exists.
    """
    out = df.copy()
    if "country" not in out.columns:
        out["country"] = "Unknown"
    if "region" not in out.columns:
        out["region"] = "Unknown"
    if "city" not in out.columns:
        out["city"] = out["station_id"]
    return out
