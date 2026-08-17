import numpy as np
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

import src.data_pipeline as dp

RES = 500
CRS = "EPSG:32643"


def make_frame(pm25_values, aqi_values, categories):
    rows = []
    for i, (pm, aq, cat) in enumerate(zip(pm25_values, aqi_values, categories)):
        col = i % 3
        row = i // 3
        rows.append(box(col * RES, row * RES, (col + 1) * RES, (row + 1) * RES))
    return gpd.GeoDataFrame(
        {
            "target_grid_id": [f"C{i}" for i in range(len(rows))],
            "pm25": np.asarray(pm25_values, dtype=np.float64),
            "pm25_aqi": np.asarray(aqi_values, dtype=np.float64),
            "aqi_category": categories,
        },
        geometry=rows,
        crs=CRS,
    )


def make_config():
    return {
        "paths": {"processed": "processed"},
        "downscaling": {"target_resolution_m": RES},
        "hotspot": {"method": "aqi_category", "minimum_category": "VERY_POOR"},
    }


def transform_for(frame):
    minx, miny, maxx, maxy = frame.total_bounds
    return dp.rasterio.transform.from_origin(minx, maxy, RES, RES)


@pytest.fixture
def no_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(dp, "PROJECT_ROOT", tmp_path)


def test_all_low_values(no_overwrite):
    frame = make_frame([10, 20, 30, 40, 50, 60],
                       [20, 30, 40, 50, 60, 70],
                       ["GOOD", "GOOD", "GOOD", "SATISFACTORY", "SATISFACTORY", "SATISFACTORY"])
    config = make_config()
    mask = dp._hotspot_mask(frame, config)
    assert not mask.any()
    hs, *_ = dp._polygonize_hotspots(frame, mask, pd.Timestamp("2025-01-01"), config, transform_for(frame))
    assert len(hs) == 0


def test_all_high_values(no_overwrite):
    frame = make_frame([300, 320, 340, 360, 380, 400],
                       [450, 460, 470, 480, 490, 500],
                       ["SEVERE"] * 6)
    config = make_config()
    mask = dp._hotspot_mask(frame, config)
    assert mask.sum() == 6
    hs, out_path, labels, n_labels, _ = dp._polygonize_hotspots(frame, mask, pd.Timestamp("2025-01-01"), config, transform_for(frame))
    assert n_labels == 1
    assert len(hs) == 1
    assert hs["cell_count"].fillna(0).sum() == 6 if "cell_count" in hs.columns else True


def test_mixed_adjacent_and_isolated(no_overwrite):
    pm = [400, 400, 50, 50, 50, 400]
    aq = [480, 480, 60, 60, 60, 480]
    cat = ["SEVERE", "SEVERE", "GOOD", "GOOD", "GOOD", "SEVERE"]
    frame = make_frame(pm, aq, cat)
    config = make_config()
    mask = dp._hotspot_mask(frame, config)
    assert mask.sum() == 3
    hs, out_path, labels, n_labels, _ = dp._polygonize_hotspots(frame, mask, pd.Timestamp("2025-01-01"), config, transform_for(frame))
    assert n_labels == 2
    assert len(hs) == 2
    stats, _ = dp._hotspot_statistics(frame, mask, hs, config)
    assert stats["hotspot_cell_count"] == 3
    assert stats["hotspot_zone_count"] == 2
    assert stats["max_pm25_ug_m3"] == 400.0


def test_nodata_excluded(no_overwrite):
    pm = [np.nan, 400, 50]
    aq = [np.nan, 480, 60]
    cat = [None, "SEVERE", "GOOD"]
    frame = make_frame(pm, aq, cat)
    config = make_config()
    mask = dp._hotspot_mask(frame, config)
    assert mask.sum() == 1


def test_edge_cells_polygonized(no_overwrite):
    pm = [400, 400, 400, 400, 400, 400]
    aq = [480, 480, 480, 480, 480, 480]
    cat = ["SEVERE"] * 6
    frame = make_frame(pm, aq, cat)
    config = make_config()
    mask = dp._hotspot_mask(frame, config)
    hs, out_path, labels, n_labels, _ = dp._polygonize_hotspots(frame, mask, pd.Timestamp("2025-01-01"), config, transform_for(frame))
    assert n_labels == 1
    assert len(hs) == 1
    assert hs["geometry"].iloc[0].is_valid
    assert not hs["geometry"].iloc[0].is_empty
