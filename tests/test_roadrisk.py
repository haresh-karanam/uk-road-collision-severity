"""Offline tests: use a small synthetic table shaped like the DfT file, so no download is needed."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import roadrisk as rr
import sqlstore


def raw(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    speed = rng.choice([20, 30, 60, 70], n)
    # severity truly depends on speed here, so a working pipeline must find it
    p = np.where(speed >= 60, 0.4, 0.15)
    sev = np.where(rng.random(n) < p, 2, 3)
    year = rng.choice([2021, 2022, 2023], n)
    return pd.DataFrame({
        "collision_year": year, "collision_severity": sev,
        "date": [f"{rng.integers(1,29):02d}/{rng.integers(1,13):02d}/{y}" for y in year],
        "time": [f"{rng.integers(0,24):02d}:{rng.integers(0,60):02d}" for _ in range(n)],
        "speed_limit": speed, "number_of_vehicles": rng.integers(1, 4, n),
        "day_of_week": rng.integers(1, 8, n), "road_type": rng.choice([3, 6], n),
        "light_conditions": rng.choice([1, 4], n), "weather_conditions": rng.choice([1, 2], n),
        "road_surface_conditions": rng.choice([1, 2], n), "urban_or_rural_area": rng.choice([1, 2], n),
        "junction_detail": rng.choice([0, 16], n), "first_road_class": rng.choice([3, 6], n),
        "police_force": 1})


@pytest.fixture(scope="module")
def d():
    return rr.prepare(raw())


def test_ksi_target_is_fatal_or_serious(d):
    assert set(d["ksi"].unique()) <= {0, 1}
    assert d.loc[d["ksi"] == 1].shape[0] > 0


def test_prepare_drops_bad_rows():
    r = raw(50); r.loc[0, "speed_limit"] = -1; r.loc[1, "collision_severity"] = -1; r.loc[2, "date"] = "nonsense"
    assert len(rr.prepare(r)) == 47


def test_no_outcome_columns_in_features():
    banned = {"number_of_casualties", "did_police_officer_attend_scene_of_accident", "collision_severity",
              "ksi", "enhanced_severity_collision", "collision_adjusted_severity_serious"}
    assert not banned & set(rr.FEATURES)


def test_split_is_temporal(d):
    train, test = rr.split_by_year(d)
    assert train["collision_year"].max() < test["collision_year"].min()
    assert len(train) + len(test) == len(d)


def test_wilson_interval_contains_rate(d):
    g = rr.ksi_rate_by(d, "speed_limit", min_n=10)
    assert ((g["low"] <= g["rate"]) & (g["rate"] <= g["high"])).all()
    assert g["collisions"].sum() == len(d)


def test_min_n_filters_small_groups(d):
    assert rr.ksi_rate_by(d, "speed_limit", min_n=10**9).empty


def test_model_beats_baseline_when_signal_exists(d):
    train, test = rr.split_by_year(d)
    res = rr.evaluate(train, test)
    assert res["hgb"]["roc_auc"] > 0.6 and res["logreg"]["roc_auc"] > 0.6
    assert res["hgb"]["brier"] < res["baseline"]["brier"]


def test_model_is_near_chance_on_pure_noise():
    r = raw(); r["collision_severity"] = np.random.default_rng(1).choice([2, 3], len(r), p=[.2, .8])
    d = rr.prepare(r); train, test = rr.split_by_year(d)
    assert abs(rr.evaluate(train, test)["logreg"]["roc_auc"] - 0.5) < 0.06  # guards against leakage in the pipeline


def test_scenario_frame_has_model_columns():
    assert list(rr.scenario_frame(speed_limit=60).columns) == rr.FEATURES


def test_sql_queries_run_and_agree_with_pandas(d):
    con = sqlstore.connect(d)
    names = sqlstore.queries()
    assert {"year_on_year", "worst_hours", "severity_by_speed_and_area", "dark_vs_daylight_by_limit"} <= set(names)
    yoy = sqlstore.run(con, "year_on_year")
    assert yoy["collisions"].sum() == len(d)
    assert yoy["change_in_collisions"].isna().iloc[0]  # no previous year for the first row
    sa = sqlstore.run(con, "severity_by_speed_and_area")
    assert sa["collisions"].sum() <= len(d)
    pandas_rate = d[d.speed_limit == 70]["ksi"].mean() * 100
    sql_rate = (sa[sa.speed_limit == 70]["fatal_or_serious"].sum() / sa[sa.speed_limit == 70]["collisions"].sum()) * 100
    assert abs(pandas_rate - sql_rate) < 1.0


def test_unseen_category_codes_do_not_crash(d):
    train, test = rr.split_by_year(d)
    m = rr.build_model("hgb").fit(train[rr.FEATURES], train["ksi"])
    p = m.predict_proba(rr.scenario_frame(light_conditions=99, weather_conditions=-1))[0, 1]
    assert 0 <= p <= 1


def test_empty_selection_gives_empty_table(d):
    assert rr.ksi_rate_by(d[d["collision_year"] == 1999], "hour").empty
