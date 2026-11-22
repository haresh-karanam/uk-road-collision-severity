"""Shared logic for the UK road-collision project (used by the notebook-free
script, the Streamlit app and the tests, so numbers come from one place).

Data: DfT Road Safety Data (STATS19), police-reported injury collisions in
Great Britain, published under the Open Government Licence v3.0.
Source page: https://www.data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data
"""
from __future__ import annotations
from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd

BASE = "https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-{y}.csv"
YEARS = (2021, 2022, 2023)
DATA_DIR = Path(__file__).parent / "data"

# Fields known when a collision is first reported/located. We deliberately
# EXCLUDE number_of_casualties, did_police_officer_attend_scene_of_accident,
# enhanced/adjusted severity columns: they are recorded after, or derived from,
# the outcome and would leak the answer.
NUMERIC = ["speed_limit", "number_of_vehicles", "hour", "month"]
CATEGORICAL = ["day_of_week", "road_type", "light_conditions", "weather_conditions",
               "road_surface_conditions", "urban_or_rural_area", "junction_detail",
               "first_road_class"]
FEATURES = NUMERIC + CATEGORICAL
USECOLS = ["collision_year", "collision_severity", "date", "time", "speed_limit",
           "number_of_vehicles", "day_of_week", "road_type", "light_conditions",
           "weather_conditions", "road_surface_conditions", "urban_or_rural_area",
           "junction_detail", "first_road_class", "police_force",
           "local_authority_ons_district"]

LABELS = {
    "day_of_week": {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday", 7: "Saturday"},
    "road_type": {1: "Roundabout", 2: "One way street", 3: "Dual carriageway", 6: "Single carriageway", 7: "Slip road", 9: "Unknown", 12: "One way/slip road"},
    "light_conditions": {1: "Daylight", 4: "Dark - lights lit", 5: "Dark - lights unlit", 6: "Dark - no lighting", 7: "Dark - lighting unknown"},
    "weather_conditions": {1: "Fine, no high winds", 2: "Raining, no high winds", 3: "Snowing, no high winds", 4: "Fine, high winds", 5: "Raining, high winds", 6: "Snowing, high winds", 7: "Fog or mist", 8: "Other", 9: "Unknown"},
    "road_surface_conditions": {1: "Dry", 2: "Wet or damp", 3: "Snow", 4: "Frost or ice", 5: "Flood", 9: "Unknown"},
    "urban_or_rural_area": {1: "Urban", 2: "Rural", 3: "Unallocated"},
}


def download(years=YEARS, data_dir: Path = DATA_DIR) -> list[Path]:
    """Fetch the yearly collision CSVs from DfT (cached on disk, no sign-up)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for y in years:
        p = data_dir / f"collision_{y}.csv"
        if not p.exists():
            req = urllib.request.Request(BASE.format(y=y), headers={"User-Agent": "roadrisk-portfolio-project"})
            with urllib.request.urlopen(req, timeout=120) as r, open(p, "wb") as f:
                f.write(r.read())
        paths.append(p)
    return paths


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw STATS19 collision rows into modelling columns.

    target `ksi` = 1 if the collision was Fatal (1) or Serious (2), else 0 (Slight).
    """
    d = df.copy()
    d = d[d["collision_severity"].isin([1, 2, 3])]
    d["ksi"] = (d["collision_severity"] <= 2).astype(int)
    d["date_parsed"] = pd.to_datetime(d["date"], format="%d/%m/%Y", errors="coerce")
    d["hour"] = pd.to_numeric(d["time"].astype(str).str.slice(0, 2), errors="coerce")
    d["month"] = d["date_parsed"].dt.month
    d = d.dropna(subset=["date_parsed", "hour"])
    d["hour"] = d["hour"].astype(int)
    # -1 / 99 mean "missing/unknown" in the DfT coding: keep as their own category
    for c in CATEGORICAL:
        d[c] = d[c].fillna(-1).astype(int).astype(str)
    d["speed_limit"] = pd.to_numeric(d["speed_limit"], errors="coerce")
    d = d[d["speed_limit"] > 0]  # drop -1 / missing
    d["number_of_vehicles"] = pd.to_numeric(d["number_of_vehicles"], errors="coerce").fillna(1).clip(upper=6)
    return d.reset_index(drop=True)


def load(years=YEARS, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    frames = [pd.read_csv(p, usecols=USECOLS, low_memory=False) for p in download(years, data_dir)]
    return prepare(pd.concat(frames, ignore_index=True))


def label(col: str, code) -> str:
    try:
        return LABELS[col].get(int(code), f"code {code}")
    except (KeyError, ValueError):
        return str(code)


def ksi_rate_by(d: pd.DataFrame, col: str, min_n: int = 200) -> pd.DataFrame:
    """KSI share of reported collisions per group, with a 95% Wilson interval.

    This is a share AMONG collisions that were reported, not the risk per
    journey or per mile (no exposure data here)."""
    g = d.groupby(col)["ksi"].agg(["sum", "count"]).reset_index()
    g = g[g["count"] >= min_n].copy()
    n, k = g["count"].to_numpy(float), g["sum"].to_numpy(float)
    p = k / n
    z = 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    g["rate"], g["low"], g["high"] = p, centre - half, centre + half
    return g.rename(columns={"sum": "ksi", "count": "collisions"})


def split_by_year(d: pd.DataFrame, test_year: int | None = None):
    """Temporal split: train on earlier years, test on the latest year."""
    test_year = test_year or int(d["collision_year"].max())
    return d[d["collision_year"] < test_year], d[d["collision_year"] == test_year]


def build_model(kind: str = "hgb"):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    if kind == "logreg":
        clf = LogisticRegression(max_iter=1000)
    else:
        clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=200, random_state=0)
    return Pipeline([("pre", pre), ("clf", clf)])


def evaluate(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Fit baseline + two models on `train`, score on `test`. Returns a dict of metrics and fitted models."""
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    y_tr, y_te = train["ksi"], test["ksi"]
    base_rate = float(y_tr.mean())
    out = {"base_rate_train": base_rate, "base_rate_test": float(y_te.mean()), "n_train": len(train), "n_test": len(test), "models": {}}
    out["baseline"] = {"roc_auc": 0.5, "pr_auc": float(y_te.mean()),
                       "brier": float(brier_score_loss(y_te, np.full(len(y_te), base_rate)))}
    for kind in ("logreg", "hgb"):
        m = build_model(kind).fit(train[FEATURES], y_tr)
        p = m.predict_proba(test[FEATURES])[:, 1]
        top = np.argsort(-p)[: max(1, len(p) // 10)]
        out[kind] = {"roc_auc": float(roc_auc_score(y_te, p)), "pr_auc": float(average_precision_score(y_te, p)),
                     "brier": float(brier_score_loss(y_te, p)),
                     "top_decile_ksi_rate": float(y_te.to_numpy()[top].mean()),
                     "top_decile_lift": float(y_te.to_numpy()[top].mean() / y_te.mean())}
        out["models"][kind] = m
    return out


def calibration_table(model, test: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    p = model.predict_proba(test[FEATURES])[:, 1]
    t = pd.DataFrame({"p": p, "y": test["ksi"].to_numpy()})
    t["bin"] = pd.qcut(t["p"].rank(method="first"), bins, labels=False)
    return t.groupby("bin").agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index()


def scenario_frame(**kw) -> pd.DataFrame:
    row = {"speed_limit": 30, "number_of_vehicles": 2, "hour": 12, "month": 6, "day_of_week": 4,
           "road_type": 6, "light_conditions": 1, "weather_conditions": 1, "road_surface_conditions": 1,
           "urban_or_rural_area": 1, "junction_detail": 0, "first_road_class": 3}
    row.update(kw)
    for c in CATEGORICAL:
        row[c] = str(int(row[c]))
    return pd.DataFrame([row])[FEATURES]
