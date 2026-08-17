"""Geographic representativeness (Milestone 17).

Reports how well the real observations cover countries and time, and writes
visualizations ONLY when the data actually supports them. No visuals are
produced for an empty table.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .schema import TARGET_COL

logger = logging.getLogger(__name__)

EARTH_AREA_KM2 = 510_100_000.0


def representativeness(df: pd.DataFrame) -> dict:
    """Summarise spatial + temporal coverage honestly."""
    if df.empty:
        return {
            "status": "no_data",
            "stations": 0,
            "countries": 0,
            "by_country": [],
            "date_range": None,
            "n_days": 0,
            "spatial_density": None,
        }
    by_country = (
        df.assign(_country=df["country"].fillna("(unknown)"))
        .groupby("_country", as_index=False)
        .agg(n_stations=("station_id", "nunique"),
             n_rows=("station_id", "count"))
        .rename(columns={"_country": "country"})
        .sort_values("n_stations", ascending=False)
    )
    dates = pd.to_datetime(df["date"], errors="coerce")
    n_stations = int(df["station_id"].nunique())
    n_countries = int(df["country"].dropna().nunique())
    density = None
    if n_countries > 0 and n_stations > 0:
        density = round(n_stations / (EARTH_AREA_KM2 / 1e6), 6)  # stations / 1e6 km2
    return {
        "status": "ok",
        "stations": n_stations,
        "countries": n_countries,
        "by_country": by_country.to_dict(orient="records"),
        "date_range": {
            "min": str(dates.min().date()) if not dates.isna().all() else None,
            "max": str(dates.max().date()) if not dates.isna().all() else None,
        },
        "n_days": int(dates.dt.normalize().nunique()),
        "spatial_density": density,
        "note": "Stations/observations by country and time; density is a rough "
                "global figure, not an accuracy claim.",
    }


def write_visualizations(df: pd.DataFrame, config, correlation_path=None) -> list:
    """Write PNG visuals only when the data supports them.

    Returns the list of written paths (empty when there is no data).
    """
    if df.empty or df["complete_case"].sum() == 0:
        return []

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputs_dir = Path(config.get("paths", {}).get("outputs", "data/outputs"))
    outputs_dir.mkdir(parents=True, exist_ok=True)
    written = []

    valid = df[df[TARGET_COL].notna()]
    if not valid.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(valid[TARGET_COL].astype(float), bins=30, color="#0b5b8c",
                edgecolor="#ffffff", linewidth=0.5)
        ax.set_title("Global training: PM2.5 target distribution")
        ax.set_xlabel("PM2.5 (ug/m3)")
        ax.set_ylabel("Observation rows")
        fig.tight_layout()
        path = outputs_dir / "global_target_distribution.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(str(path))

    by_country = (
        df.groupby(df["country"].fillna("(unknown)")).size()
        .sort_values(ascending=False).head(15)
    )
    if not by_country.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        by_country.plot.bar(ax=ax, color="#2f7d63")
        ax.set_title("Global training: observation rows by country")
        ax.set_ylabel("Rows")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        path = outputs_dir / "global_training_coverage.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(str(path))

    if correlation_path and Path(correlation_path).exists():
        corr = pd.read_csv(correlation_path)
        if not corr.empty:
            cols = sorted(set(corr["feature_a"].tolist() + corr["feature_b"].tolist()))
            import numpy as np
            matrix = pd.DataFrame(np.nan, index=cols, columns=cols)
            for _, row in corr.iterrows():
                if pd.notna(row["pearson"]):
                    matrix.loc[row["feature_a"], row["feature_b"]] = row["pearson"]
                    matrix.loc[row["feature_b"], row["feature_a"]] = row["pearson"]
            fig, ax = plt.subplots(figsize=(9, 7))
            im = ax.imshow(matrix.astype(float).values, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(cols)))
            ax.set_yticks(range(len(cols)))
            ax.set_xticklabels(cols, rotation=90, fontsize=8)
            ax.set_yticklabels(cols, fontsize=8)
            ax.set_title("Feature correlations (Pearson, complete-case rows)")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            path = outputs_dir / "global_feature_correlation.png"
            fig.savefig(path, dpi=120)
            plt.close(fig)
            written.append(str(path))
    return written
