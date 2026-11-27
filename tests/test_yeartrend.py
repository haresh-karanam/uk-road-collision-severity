"""Tests for the three-year trend view, on tiny synthetic frames."""
import pandas as pd
import pytest

import yeartrend as yt


def write_year(tmp_path, year, severities):
    pd.DataFrame({"collision_severity": severities}).to_csv(
        tmp_path / f"collision_{year}.csv", index=False)


def test_yearly_table_math(tmp_path, monkeypatch):
    write_year(tmp_path, 2021, [1, 2, 3, 3])          # 2 KSI of 4
    write_year(tmp_path, 2022, [1, 3, 3, 3, 3, 3])    # 1 KSI of 6
    # two regions x two years; 2021 sums to 1e9, 2022 to 2e9 vehicle-miles
    traffic = pd.DataFrame([
        (2021, "A", 400e6), (2021, "B", 600e6),
        (2022, "A", 800e6), (2022, "B", 1200e6),
        (2019, "A", 999e9),                            # outside years
    ], columns=["year", "region_name", "all_motor_vehicles"])
    tpath = tmp_path / "traffic.csv"
    traffic.to_csv(tpath, index=False)
    monkeypatch.setattr(yt.ex, "download_traffic", lambda: tpath)

    g = yt.yearly_table(data_dir=tmp_path, years=(2021, 2022)).set_index("year")
    assert g.loc[2021, "collisions"] == 4 and g.loc[2021, "ksi"] == 2
    assert g.loc[2021, "share"] == pytest.approx(0.5)
    assert g.loc[2022, "share"] == pytest.approx(1 / 6)
    assert g.loc[2021, "share_low"] < 0.5 < g.loc[2021, "share_high"]
    assert g.loc[2021, "per_bvm"] == pytest.approx(2.0)   # 2 KSI / 1e9 miles
    assert g.loc[2022, "per_bvm"] == pytest.approx(0.5)   # 1 KSI / 2e9 miles
    assert g.loc[2021, "billion_miles"] == pytest.approx(1.0)
    assert g.loc[2022, "per_bvm_low"] < 0.5 < g.loc[2022, "per_bvm_high"]


def test_missing_traffic_year_raises(tmp_path, monkeypatch):
    write_year(tmp_path, 2021, [1, 3])
    traffic = pd.DataFrame([(2022, "A", 1e9)],
                           columns=["year", "region_name", "all_motor_vehicles"])
    tpath = tmp_path / "traffic.csv"
    traffic.to_csv(tpath, index=False)
    monkeypatch.setattr(yt.ex, "download_traffic", lambda: tpath)
    with pytest.raises(ValueError):
        yt.yearly_table(data_dir=tmp_path, years=(2021,))


def test_chart_writes_png(tmp_path, monkeypatch):
    write_year(tmp_path, 2021, [1, 2, 3, 3])
    write_year(tmp_path, 2022, [1, 3, 3, 3, 3, 3])
    traffic = pd.DataFrame([(2021, "A", 1e9), (2022, "A", 2e9)],
                           columns=["year", "region_name", "all_motor_vehicles"])
    tpath = tmp_path / "traffic.csv"
    traffic.to_csv(tpath, index=False)
    monkeypatch.setattr(yt.ex, "download_traffic", lambda: tpath)
    g = yt.yearly_table(data_dir=tmp_path, years=(2021, 2022))
    out = tmp_path / "trend.png"
    yt.chart_trend(g, out)
    assert out.exists() and out.stat().st_size > 5000
