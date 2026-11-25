"""Tests for the per-mile highway-authority map, on tiny synthetic frames."""
import json
import pandas as pd
import pytest

import lamiles as lm


def write_lookup(tmp_path, pairs):
    p = tmp_path / "lookup.json"
    p.write_text(json.dumps({"features": [
        {"attributes": {"LAD23CD": lad, "CTY23CD": cty}} for lad, cty in pairs]}))
    return p


def write_traffic(tmp_path, rows):
    p = tmp_path / "traffic.csv"
    cols = ["local_authority_id", "local_authority_name", "local_authority_code",
            "year", "link_length_km", "link_length_miles", "cars_and_taxis", "all_motor_vehicles"]
    pd.DataFrame(rows, columns=cols).to_csv(p, index=False)
    return p


def test_county_map_folds_districts_and_cumbria(tmp_path):
    lu = write_lookup(tmp_path, [("E07000008", "E10000003"), ("E08000001", "E11000001")])
    cmap = lm.county_map(lu)
    assert cmap["E07000008"] == "E10000003"          # district to its county
    assert cmap["E06000063"] == lm.CUMBRIA           # Cumberland to combined key
    assert cmap["E06000064"] == lm.CUMBRIA           # Westmorland and Furness likewise
    # met boroughs stay at borough level: the lookup's E11 met county has no traffic data
    assert cmap["E08000001"] == "E08000001"


def test_traffic_miles_sums_years_and_merges_cumbria(tmp_path):
    p = write_traffic(tmp_path, [
        (1, "Cumbria", "E10000006", 2021, 0, 0, 0, 500e6),
        (1, "Cumbria", "E10000006", 2022, 0, 0, 0, 500e6),
        (2, "Cumberland", "E06000063", 2023, 0, 0, 0, 300e6),
        (3, "Westmorland and Furness", "E06000064", 2023, 0, 0, 0, 200e6),
        (4, "Aberdeen", "S12000033", 2021, 0, 0, 0, 100e6),
        (4, "Aberdeen", "S12000033", 2019, 0, 0, 0, 999e6),  # outside years
    ])
    miles = lm.traffic_miles(p)
    assert miles[lm.CUMBRIA] == pytest.approx(1500e6)
    assert miles["S12000033"] == pytest.approx(100e6)


def lad_rates(rows):
    return pd.DataFrame(rows, columns=["lad23cd", "ksi", "collisions", "rate", "low", "high", "enough"])


def test_ksi_per_mile_math_and_min_n(tmp_path):
    lu = write_lookup(tmp_path, [("E07000008", "E10000003"), ("E07000009", "E10000003")])
    cmap = lm.county_map(lu)
    rates = lad_rates([
        ("E07000008", 60, 300, 0, 0, 0, True),
        ("E07000009", 40, 100, 0, 0, 0, True),   # thin district, folds up fine
        ("S12000033", 5, 50, 0, 0, 0, False),     # thin authority -> grey
    ])
    miles = {"E10000003": 100e9, "S12000033": 1e9}
    g = lm.ksi_per_mile_by_authority(rates, cmap, miles, min_n=200).set_index("authority")
    assert g.loc["E10000003", "collisions"] == 400      # districts summed
    assert g.loc["E10000003", "ksi"] == 100
    assert g.loc["E10000003", "rate"] == pytest.approx(1.0)   # 100 / 100e9 * 1e9
    assert g.loc["E10000003", "low"] < 1.0 < g.loc["E10000003", "high"]
    assert not bool(g.loc["S12000033", "enough"])             # 50 < min_n -> grey


def test_authority_without_traffic_is_dropped(tmp_path):
    lu = write_lookup(tmp_path, [])
    cmap = lm.county_map(lu)
    rates = lad_rates([("S12000033", 50, 500, 0, 0, 0, True)])
    g = lm.ksi_per_mile_by_authority(rates, cmap, {}, min_n=200)
    assert len(g) == 0


def test_draw_map_writes_png_and_combines_cumbria(tmp_path):
    boundary = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"CTYUA23CD": "E06000063", "CTYUA23NM": "Cumberland"},
         "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}},
        {"type": "Feature", "properties": {"CTYUA23CD": "E06000064", "CTYUA23NM": "Westmorland"},
         "geometry": {"type": "Polygon", "coordinates": [[[2, 2], [3, 2], [3, 3], [2, 2]]]}},
    ]}
    bpath = tmp_path / "b.geojson"
    bpath.write_text(json.dumps(boundary))
    frame = pd.DataFrame([{"authority": lm.CUMBRIA, "ksi": 100, "collisions": 500,
                           "miles": 1e9, "rate": 100.0, "low": 80.0, "high": 120.0, "enough": True}])
    out = tmp_path / "map.png"
    lm.draw_map(frame, bpath, out)
    assert out.exists() and out.stat().st_size > 5000
