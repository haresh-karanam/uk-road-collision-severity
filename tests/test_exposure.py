"""Tests for the per-mile exposure view, on tiny synthetic frames."""
import pandas as pd
import pytest

import exposure as ex


def collisions(rows):
    return pd.DataFrame(rows, columns=["collision_index", "ksi"])


def vehicles(rows):
    return pd.DataFrame(rows, columns=["collision_index", "vehicle_type"])


MILES = {"Pedal cycle": 1e9, "Motorcycle": 1e9, "Car or taxi": 100e9,
         "Bus or coach": 1e9, "Van": 10e9, "Heavy goods": 2e9}


def test_gb_vehicle_miles_sums_regions_and_years(tmp_path):
    import csv
    p = tmp_path / "traffic.csv"
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["year", "region_id", "region_name", "region_ons_code",
                    "link_length_km", "link_length_miles", "pedal_cycles",
                    "two_wheeled_motor_vehicles", "cars_and_taxis",
                    "buses_and_coaches", "LGVs", "all_HGVs", "all_motor_vehicles"])
        # two regions x two years of motorcycle miles: 100m each row -> 400m
        for year in (2021, 2022):
            for reg in ("Scotland", "Wales"):
                w.writerow([year, 1, reg, "X", 0, 0, 0, 100_000_000, 0, 0, 0, 0, 0])
        w.writerow([2019, 1, "Scotland", "X", 0, 0, 0, 999_000_000, 0, 0, 0, 0, 0])  # excluded
    miles = ex.gb_vehicle_miles(p, years=(2021, 2022))
    assert miles["Motorcycle"] == pytest.approx(4e8)
    assert miles["Car or taxi"] == 0.0


def test_rate_math_and_intervals():
    # 100 KSI motorcycle collisions over 1e9 miles -> 100 per billion miles
    coll = collisions([(f"m{i}", 1) for i in range(100)] + [(f"c{i}", 0) for i in range(100)])
    veh = vehicles([(f"m{i}", 5) for i in range(100)] + [(f"c{i}", 9) for i in range(100)])
    g = ex.ksi_per_billion_miles(coll, veh, {"Motorcycle": 1e9, "Car or taxi": 100e9}, min_n=10)
    g = g.set_index("group")
    assert g.loc["Motorcycle", "rate"] == pytest.approx(100.0)
    assert g.loc["Car or taxi", "rate"] == 0.0
    assert g.loc["Motorcycle", "low"] < 100.0 < g.loc["Motorcycle", "high"]
    # zero-KSI group gets a zero lower bound but a positive upper bound
    assert g.loc["Car or taxi", "low"] == 0.0
    assert g.loc["Car or taxi", "high"] > 0.0


def test_car_and_taxi_merge_counts_distinct_collisions():
    # one collision with both a car and a taxi counts once for the merged group
    coll = collisions([("a", 1)] + [(f"x{i}", 0) for i in range(20)])
    veh = vehicles([("a", 9), ("a", 8)] + [(f"x{i}", 9) for i in range(20)])
    g = ex.ksi_per_billion_miles(coll, veh, {"Car or taxi": 1e9}, min_n=1)
    assert g.iloc[0]["collisions"] == 21  # 20 all-car + 1 car+taxi, not 22
    assert g.iloc[0]["ksi"] == 1


def test_thin_groups_and_zero_miles_dropped():
    coll = collisions([(f"c{i}", 0) for i in range(5)])
    veh = vehicles([(f"c{i}", 1) for i in range(5)])
    g = ex.ksi_per_billion_miles(coll, veh, {"Pedal cycle": 1e9, "Van": 0.0}, min_n=10)
    assert len(g) == 0  # pedal cycles below min_n, van has no miles


def test_chart_writes_png(tmp_path):
    coll = collisions([(f"m{i}", 1) for i in range(100)] + [(f"c{i}", 1) for i in range(100)])
    veh = vehicles([(f"m{i}", 5) for i in range(100)] + [(f"c{i}", 9) for i in range(100)])
    g = ex.ksi_per_billion_miles(coll, veh, {"Motorcycle": 1e9, "Car or taxi": 100e9}, min_n=10)
    out = tmp_path / "exposure.png"
    ex.chart_exposure(g, out)
    assert out.exists() and out.stat().st_size > 5000
