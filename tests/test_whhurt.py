"""Tests for the casualty-mix view, on tiny synthetic frames."""
import pandas as pd
import pytest

import whhurt as hh


def casualties(rows):
    return pd.DataFrame(rows, columns=["collision_index", "casualty_severity",
                                       "casualty_type", "group", "london"])


def test_group_mapping_covers_the_main_types():
    assert hh.TYPE_TO_GROUP[0] == "Pedestrian"
    assert hh.TYPE_TO_GROUP[1] == "Pedal cyclist"
    assert hh.TYPE_TO_GROUP[23] == "Motorcyclist"       # electric motorcycle
    assert hh.TYPE_TO_GROUP[8] == "Car or taxi occupant"  # taxi
    assert hh.TYPE_TO_GROUP[9] == "Car or taxi occupant"  # car
    assert hh.TYPE_TO_GROUP[20] == "Goods vehicle occupant"
    assert 16 not in hh.TYPE_TO_GROUP                   # ridden horse -> Other


def test_shares_and_london_split():
    # 100 KSI casualties: 40 pedestrians in London, 60 car occupants elsewhere
    rows = ([("c1", 2, 0, "Pedestrian", True)] * 40
            + [("c2", 2, 9, "Car or taxi occupant", False)] * 60)
    g = hh.share_by_group(casualties(rows), min_n=10)
    lon = g[g["area"] == "London"].iloc[0]
    rest = g[g["area"] == "Rest of GB"].iloc[0]
    assert lon["group"] == "Pedestrian" and lon["rate"] == pytest.approx(1.0)
    assert rest["group"] == "Car or taxi occupant" and rest["rate"] == pytest.approx(1.0)
    assert lon["low"] < 1.0 and lon["casualties"] == 40


def test_min_n_drops_thin_groups():
    rows = [("c1", 2, 0, "Pedestrian", True)] * 5 + [("c2", 2, 9, "Car or taxi occupant", True)] * 100
    g = hh.share_by_group(casualties(rows), min_n=50)
    assert set(g["group"]) == {"Car or taxi occupant"}


def test_loader_groups_and_flags_london(tmp_path):
    cas = pd.DataFrame([
        ("a", 1, 0), ("b", 2, 9), ("c", 3, 9),  # severities: fatal, serious, slight
        ("d", 2, 16),                            # ridden horse -> Other
    ], columns=["collision_index", "casualty_severity", "casualty_type"])
    cas.to_csv(tmp_path / "casualty_2023.csv", index=False)
    coll = pd.DataFrame([
        ("a", "E09000001"), ("b", "E09000001"), ("c", "E06000001"), ("d", "S12000033"),
    ], columns=["collision_index", "local_authority_ons_district"])
    coll.to_csv(tmp_path / "collision_2023.csv", index=False)
    out = hh.load_casualty_groups(years=(2023,), data_dir=tmp_path)
    assert len(out) == 3  # the slight casualty is excluded
    ped = out[out["casualty_type"] == 0].iloc[0]
    assert ped["group"] == "Pedestrian" and bool(ped["london"])
    horse = out[out["casualty_type"] == 16].iloc[0]
    assert horse["group"] == "Other / unknown" and not bool(horse["london"])


def test_chart_writes_png(tmp_path):
    rows = ([("c1", 2, 0, "Pedestrian", True)] * 40
            + [("c2", 2, 9, "Car or taxi occupant", False)] * 60)
    g = hh.share_by_group(casualties(rows), min_n=10)
    out = tmp_path / "mix.png"
    hh.chart_mix(g, out)
    assert out.exists() and out.stat().st_size > 5000
