"""Tests for the who-is-involved views, on tiny synthetic frames."""
import pandas as pd
import pytest

import whorisk as wh


def casualties(rows):
    return pd.DataFrame(rows, columns=["collision_index", "casualty_severity", "age_band_of_casualty"])


def test_fatal_share_by_age_band_math():
    # band 11 (over 75): 40 fatal + 60 serious -> 40% die; band 4: 5 + 95 -> 5%
    rows = [("c1", 1, 11)] * 40 + [("c1", 2, 11)] * 60 + [("c2", 1, 4)] * 5 + [("c2", 2, 4)] * 95
    g = wh.fatal_share_by_age_band(casualties(rows), min_n=50)
    g = g.set_index("band")
    assert g.loc["over 75", "rate"] == pytest.approx(0.40)
    assert g.loc["16-20", "rate"] == pytest.approx(0.05)
    assert g.loc["over 75", "low"] < g.loc["over 75", "rate"] < g.loc["over 75", "high"]


def test_slight_casualties_are_not_counted():
    rows = [("c1", 3, 11)] * 500 + [("c1", 1, 11)] * 10 + [("c1", 2, 11)] * 90
    g = wh.fatal_share_by_age_band(casualties(rows), min_n=50)
    assert g.iloc[0]["collisions"] == 100  # only the 100 KSI casualties
    assert g.iloc[0]["rate"] == pytest.approx(0.10)


def test_unknown_age_bands_excluded_and_min_n_applied():
    rows = [("c1", 1, -1)] * 500 + [("c1", 1, 3)] * 10
    g = wh.fatal_share_by_age_band(casualties(rows), min_n=50)
    assert len(g) == 0  # -1 excluded; band 3 has only 10 < min_n


def vehicles(rows):
    return pd.DataFrame(rows, columns=["collision_index", "vehicle_type"])


def test_vehicle_group_mapping_and_involvement():
    coll = pd.DataFrame({"collision_index": ["a", "b", "c"], "ksi": [1, 0, 0]})
    veh = vehicles([("a", 23), ("a", 9), ("b", 9), ("c", 1)])
    g = wh.ksi_by_vehicle_involvement(coll, veh, min_n=1)
    g = g.set_index("group")
    # code 23 is an electric motorcycle -> Motorcycle group
    assert g.loc["Motorcycle", "collisions"] == 1
    assert g.loc["Motorcycle", "rate"] == 1.0
    # collision 'a' has car + motorcycle: the car group sees 2 collisions, 1 KSI
    assert g.loc["Car", "collisions"] == 2
    assert g.loc["Car", "rate"] == 0.5
    assert g.loc["Pedal cycle", "rate"] == 0.0


def test_unmapped_vehicle_types_ignored():
    coll = pd.DataFrame({"collision_index": ["a"], "ksi": [1]})
    veh = vehicles([("a", 16), ("a", 90)])  # ridden horse, other
    g = wh.ksi_by_vehicle_involvement(coll, veh, min_n=1)
    assert len(g) == 0


def test_min_n_filters_thin_vehicle_groups():
    coll = pd.DataFrame({"collision_index": [f"c{i}" for i in range(12)], "ksi": [0] * 12})
    veh = vehicles([(f"c{i}", 1) for i in range(10)] + [("c10", 9), ("c11", 9)])
    g = wh.ksi_by_vehicle_involvement(coll, veh, min_n=5)
    assert set(g["group"]) == {"Pedal cycle"}


def test_charts_write_pngs(tmp_path):
    age = wh.fatal_share_by_age_band(
        casualties([("c1", 1, 11)] * 40 + [("c1", 2, 11)] * 60 + [("c2", 1, 4)] * 5 + [("c2", 2, 4)] * 95),
        min_n=50)
    out = tmp_path / "age.png"
    wh.chart_age(age, out)
    assert out.exists() and out.stat().st_size > 5000
    coll = pd.DataFrame({"collision_index": ["a", "b"], "ksi": [1, 0]})
    veh = vehicles([("a", 9), ("b", 1)])
    g = wh.ksi_by_vehicle_involvement(coll, veh, min_n=1)
    out2 = tmp_path / "veh.png"
    wh.chart_vehicles(g, out2)
    assert out2.exists() and out2.stat().st_size > 5000
