"""Who is involved: casualty age and vehicle type views of the same STATS19 data.

Data: DfT Road Safety Data casualty and vehicle files for 2021-2023,
Open Government Licence v3.0, downloaded on first run and cached in data/.

These two views are DESCRIPTIVE. They are never fed into the model, so they
add no leakage to the evaluation. For the age view, casualty severity is the
outcome being described, not an input.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

import roadrisk as rr

BASE = "https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-{kind}-{y}.csv"
DATA_DIR = Path(__file__).parent / "data"

# STATS19 vehicle_type codes grouped to plain labels. 90 = other, -1 = missing.
VEHICLE_GROUPS = {
    "Pedal cycle": {1},
    "Motorcycle": {2, 3, 4, 5, 23, 97},
    "Car": {9},
    "Taxi": {8},
    "Bus or coach": {10, 11},
    "Van": {19},
    "Heavy goods": {20, 21, 98},
}
AGE_BANDS = {1: "0-5", 2: "6-10", 3: "11-15", 4: "16-20", 5: "21-25", 6: "26-35",
             7: "36-45", 8: "46-55", 9: "56-65", 10: "66-75", 11: "over 75"}


def download(kind: str, years=rr.YEARS, data_dir: Path = DATA_DIR) -> list[Path]:
    """Fetch the yearly casualty/vehicle CSVs from DfT (cached on disk)."""
    import urllib.request
    assert kind in ("casualty", "vehicle")
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for y in years:
        p = data_dir / f"{kind}_{y}.csv"
        if not p.exists():
            req = urllib.request.Request(BASE.format(kind=kind, y=y),
                                         headers={"User-Agent": "roadrisk-portfolio-project"})
            with urllib.request.urlopen(req, timeout=300) as r, open(p, "wb") as f:
                f.write(r.read())
        paths.append(p)
    return paths


def load_casualties(years=rr.YEARS, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    cols = ["collision_index", "casualty_severity", "age_band_of_casualty"]
    frames = [pd.read_csv(p, usecols=cols, low_memory=False) for p in download("casualty", years, data_dir)]
    d = pd.concat(frames, ignore_index=True)
    d["casualty_severity"] = pd.to_numeric(d["casualty_severity"], errors="coerce")
    d["age_band_of_casualty"] = pd.to_numeric(d["age_band_of_casualty"], errors="coerce").fillna(-1).astype(int)
    return d


def load_vehicles(years=rr.YEARS, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    cols = ["collision_index", "vehicle_type"]
    frames = [pd.read_csv(p, usecols=cols, low_memory=False) for p in download("vehicle", years, data_dir)]
    d = pd.concat(frames, ignore_index=True)
    d["vehicle_type"] = pd.to_numeric(d["vehicle_type"], errors="coerce").fillna(-1).astype(int)
    return d


def load_collision_labels(years=rr.YEARS, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """collision_index -> ksi label, from the already-cached collision CSVs."""
    cols = ["collision_index", "collision_severity"]
    frames = [pd.read_csv(p, usecols=cols, low_memory=False) for p in rr.download(years, data_dir)]
    d = pd.concat(frames, ignore_index=True)
    d = d[d["collision_severity"].isin([1, 2, 3])].copy()
    d["ksi"] = (d["collision_severity"] <= 2).astype(int)
    return d[["collision_index", "ksi"]]


def fatal_share_by_age_band(cas: pd.DataFrame, min_n: int = 200) -> pd.DataFrame:
    """Among fatal-or-serious casualties, the share who died, by age band.

    Same machinery as the rest of the project (Wilson intervals, min group
    size) by reusing roadrisk.ksi_rate_by with 'fatal' as the outcome."""
    ksi_cas = cas[cas["casualty_severity"].isin([1, 2])].copy()
    ksi_cas["ksi"] = (ksi_cas["casualty_severity"] == 1).astype(int)
    ksi_cas = ksi_cas[ksi_cas["age_band_of_casualty"].isin(AGE_BANDS)]
    g = rr.ksi_rate_by(ksi_cas, "age_band_of_casualty", min_n=min_n)
    g["band"] = g["age_band_of_casualty"].map(AGE_BANDS)
    return g.sort_values("age_band_of_casualty").reset_index(drop=True)


def ksi_by_vehicle_involvement(coll: pd.DataFrame, veh: pd.DataFrame, min_n: int = 500) -> pd.DataFrame:
    """Fatal/serious share of the collisions each vehicle group is involved in.

    A collision with a car and a motorcycle counts in both groups: each row
    answers 'when this group is involved, how often is the collision KSI?'."""
    code_to_group = {code: name for name, codes in VEHICLE_GROUPS.items() for code in codes}
    v = veh.copy()
    v["group"] = v["vehicle_type"].map(code_to_group)
    v = v.dropna(subset=["group"])[["collision_index", "group"]].drop_duplicates()
    long = v.merge(coll, on="collision_index", how="inner")
    g = rr.ksi_rate_by(long, "group", min_n=min_n)
    return g.sort_values("rate", ascending=False).reset_index(drop=True)


def _bar_chart(g: pd.DataFrame, label_col: str, title: str, ylabel: str, note: str, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(g[label_col].astype(str), g["rate"] * 100,
           yerr=[(g["rate"] - g["low"]) * 100, (g["high"] - g["rate"]) * 100],
           capsize=3, color="#4c72b0")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.xticks(rotation=35, ha="right")
    fig.text(0.01, 0.01, note, fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def chart_age(g: pd.DataFrame, out_path: Path) -> None:
    _bar_chart(g, "band",
               "Share of fatal/serious casualties who died, by age band, 2021-2023",
               "% of fatal/serious casualties who died",
               "Descriptive view, not a model feature. Error bars: 95% Wilson intervals. "
               "Source: DfT STATS19 casualty data 2021-2023 (OGL v3.0).",
               out_path)


def chart_vehicles(g: pd.DataFrame, out_path: Path) -> None:
    _bar_chart(g, "group",
               "Fatal/serious share of collisions involving each vehicle type, 2021-2023",
               "% of collisions fatal/serious",
               "Descriptive view, not a model feature. Error bars: 95% Wilson intervals. "
               "Source: DfT STATS19 vehicle data 2021-2023 (OGL v3.0).",
               out_path)
