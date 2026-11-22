"""Tests for the local-authority map, on tiny synthetic data and boundaries."""
import json

import pandas as pd
import pytest

import lamap


def frame(rows):
    return pd.DataFrame(rows, columns=["local_authority_ons_district", "ksi"])


def lookup_file(tmp_path, pairs):
    p = tmp_path / "lookup.json"
    p.write_text(json.dumps({"features": [{"attributes": {"LAD22CD": a, "LAD23CD": b}} for a, b in pairs]}))
    return p


def test_code_map_includes_unchanged_authorities(tmp_path):
    cmap = lamap.current_code_map(lookup_file(tmp_path, [("E1", "E1"), ("E2", "E9")]))
    assert cmap == {"E1": "E1", "E2": "E9"}


def test_historic_codes_fold_into_successors(tmp_path):
    cmap = lamap.current_code_map(lookup_file(tmp_path, [("E1", "E1"), ("E2", "E9"), ("E9", "E9")]))
    d = frame([("E2", 1), ("E2", 0), ("E9", 0), ("E1", 0)])
    g, dropped = lamap.severity_by_authority(d, cmap, min_n=1)
    g = g.set_index("lad23cd")
    assert dropped == 0
    assert g.loc["E9", "collisions"] == 3  # the two historic E2 rows land on E9
    assert g.loc["E9", "ksi"] == 1
    assert g.loc["E1", "rate"] == 0.0


def test_unknown_codes_are_dropped_and_counted(tmp_path):
    cmap = lamap.current_code_map(lookup_file(tmp_path, [("E1", "E1")]))
    d = frame([("E1", 1), ("EHEATHROW", 1), ("EHEATHROW", 0)])
    g, dropped = lamap.severity_by_authority(d, cmap, min_n=1)
    assert dropped == 2
    assert list(g["lad23cd"]) == ["E1"]


def test_enough_flag_marks_thin_authorities(tmp_path):
    cmap = lamap.current_code_map(lookup_file(tmp_path, [("E1", "E1"), ("E2", "E2")]))
    d = frame([("E1", 1)] * 250 + [("E2", 0)] * 10)
    g, _ = lamap.severity_by_authority(d, cmap, min_n=200)
    g = g.set_index("lad23cd")
    assert bool(g.loc["E1", "enough"]) and not bool(g.loc["E2", "enough"])


def test_wilson_interval_brackets_the_rate(tmp_path):
    cmap = lamap.current_code_map(lookup_file(tmp_path, [("E1", "E1")]))
    d = frame([("E1", 1)] * 24 + [("E1", 0)] * 76)
    g, _ = lamap.severity_by_authority(d, cmap, min_n=1)
    row = g.iloc[0]
    assert row["rate"] == pytest.approx(0.24)
    assert row["low"] < row["rate"] < row["high"]


SQUARE = {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]}


def boundaries_file(tmp_path, codes):
    feats = [{"type": "Feature",
              "properties": {"LAD23CD": c, "LAD23NM": f"Area {c}"},
              "geometry": SQUARE} for c in codes]
    p = tmp_path / "b.geojson"
    p.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    return p


def test_draw_map_writes_a_png(tmp_path):
    cmap = {"E1": "E1", "E2": "E2"}
    d = frame([("E1", 1)] * 300 + [("E2", 0)] * 5)
    g, _ = lamap.severity_by_authority(d, cmap, min_n=200)
    out = tmp_path / "map.png"
    lamap.draw_map(g, boundaries_file(tmp_path, ["E1", "E2"]), out, min_n=200)
    assert out.exists() and out.stat().st_size > 5000


def test_draw_map_greys_authorities_with_no_data(tmp_path):
    g = pd.DataFrame({"lad23cd": ["E1"], "collisions": [300], "ksi": [60],
                      "rate": [0.2], "low": [0.15], "high": [0.25], "enough": [True]})
    out = tmp_path / "map2.png"
    # E2 exists in the boundaries but has no row in the frame: must not error
    lamap.draw_map(g, boundaries_file(tmp_path, ["E1", "E2"]), out)
    assert out.exists()
