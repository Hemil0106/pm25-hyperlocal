import math

import numpy as np
import pytest

from src.data_pipeline import (
    _load_aqi_breakpoints,
    _pm25_to_aqi,
    _aqi_category_name,
)

MINIMAL_AQI_CONFIG = {
    "aqi": {
        "breakpoints": [
            {"concentration_low": 0, "concentration_high": 30, "aqi_low": 0, "aqi_high": 50},
            {"concentration_low": 31, "concentration_high": 60, "aqi_low": 51, "aqi_high": 100},
            {"concentration_low": 61, "concentration_high": 90, "aqi_low": 101, "aqi_high": 200},
            {"concentration_low": 91, "concentration_high": 120, "aqi_low": 201, "aqi_high": 300},
            {"concentration_low": 121, "concentration_high": 250, "aqi_low": 301, "aqi_high": 400},
            {"concentration_low": 250, "concentration_high": 350, "aqi_low": 401, "aqi_high": 500},
        ],
        "categories": [
            {"name": "GOOD", "aqi_min": 0, "aqi_max": 50},
            {"name": "SATISFACTORY", "aqi_min": 51, "aqi_max": 100},
            {"name": "MODERATELY_POLLUTED", "aqi_min": 101, "aqi_max": 200},
            {"name": "POOR", "aqi_min": 201, "aqi_max": 300},
            {"name": "VERY_POOR", "aqi_min": 301, "aqi_max": 400},
            {"name": "SEVERE", "aqi_min": 401, "aqi_max": 500},
        ],
    }
}


@pytest.fixture(scope="module")
def breakpoints_categories():
    return _load_aqi_breakpoints(MINIMAL_AQI_CONFIG)


@pytest.mark.parametrize(
    "pm25,expected_aqi,expected_category",
    [
        (0.0, 0, "GOOD"),
        (30.0, 50, "GOOD"),
        (30.9, 51, "SATISFACTORY"),
        (31.0, 51, "SATISFACTORY"),
        (60.0, 100, "SATISFACTORY"),
        (61.0, 101, "MODERATELY_POLLUTED"),
        (90.0, 200, "MODERATELY_POLLUTED"),
        (91.0, 201, "POOR"),
        (120.0, 300, "POOR"),
        (121.0, 301, "VERY_POOR"),
        (250.0, 400, "VERY_POOR"),
        (250.1, 401, "SEVERE"),
        (350.0, 500, "SEVERE"),
        (400.0, 500, "SEVERE"),
    ],
)
def test_aqi_interpolation_breakpoints(breakpoints_categories, pm25, expected_aqi, expected_category):
    breakpoints, categories = breakpoints_categories
    aqi, _ = _pm25_to_aqi(pm25, breakpoints)
    assert aqi is not None
    assert int(round(aqi)) == expected_aqi
    assert _aqi_category_name(float(round(aqi)), categories) == expected_category


@pytest.mark.parametrize(
    "pm25,expected_aqi",
    [
        (15.0, 25.0),
        (45.0, 51 + 49 / 29 * 14),      # 74.66 within 31-60 -> 51-100
        (75.0, 101 + 99 / 29 * 14),     # 148.79 within 61-90 -> 101-200
        (105.0, 201 + 99 / 29 * 14),    # 248.79 within 91-120 -> 201-300
        (185.0, 301 + 99 / 129 * 64),   # 350.12 within 121-250 -> 301-400
    ],
)
def test_aqi_linear_interpolation_inside_interval(breakpoints_categories, pm25, expected_aqi):
    breakpoints, _ = breakpoints_categories
    aqi, _ = _pm25_to_aqi(pm25, breakpoints)
    assert math.isclose(aqi, float(expected_aqi), abs_tol=0.01)


def test_aqi_nodata_and_invalid(breakpoints_categories):
    breakpoints, categories = breakpoints_categories
    for bad in [np.nan, math.inf, -1.0, None]:
        aqi, _ = _pm25_to_aqi(bad, breakpoints)
        assert aqi is None
    assert _aqi_category_name(np.nan, categories) is None


def test_aqi_category_boundaries(breakpoints_categories):
    _, categories = breakpoints_categories
    assert _aqi_category_name(float(round(0.0)), categories) == "GOOD"
    assert _aqi_category_name(float(round(50.0)), categories) == "GOOD"
    assert _aqi_category_name(float(round(50.9)), categories) == "SATISFACTORY"
    assert _aqi_category_name(float(round(100.0)), categories) == "SATISFACTORY"
    assert _aqi_category_name(float(round(200.0)), categories) == "MODERATELY_POLLUTED"
    assert _aqi_category_name(float(round(300.0)), categories) == "POOR"
    assert _aqi_category_name(float(round(400.0)), categories) == "VERY_POOR"
    assert _aqi_category_name(float(round(401.0)), categories) == "SEVERE"
