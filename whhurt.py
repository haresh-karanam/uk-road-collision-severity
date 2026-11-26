"""Who gets hurt: the fatal/serious casualty mix by road-user type.

The who-is-involved charts (whorisk) look at collisions: what was involved
and how often those collisions were fatal or serious. This view looks at the
people: of everyone killed or seriously injured, which road users are they -
and how does that mix differ between London and the rest of Great Britain?

Casualties are grouped from the DfT casualty_type code:
pedestrian (0), pedal cyclist (1), motorcyclist (2-5, 23, 97),
car or taxi occupant (8, 9), goods vehicle occupant (19-22, 98),
bus or coach occupant (10, 11); everything else (including ridden horses
and unknown) lands in "Other / unknown".

London means a collision in a London borough (local authority code E09xxxxx).
Shares are Wilson-interval proportions of all fatal/serious casualties in
each area. Descriptive only: never fed to the model.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import roadrisk as rr
import whorisk as wh

DATA_DIR = Path(__file__).parent / "data"
YEARS = rr.YEARS

CASUALTY_GROUPS = {
    "Pedestrian": {0},
    "Pedal cyclist": {1},
    "Motorcyclist": {2, 3, 4, 5, 23, 97},
    "Car or taxi occupant": {8, 9},
    "Bus or coach occupant": {10, 11},
    "Goods vehicle occupant": {19, 20, 21, 22, 98},
}
TYPE_TO_GROUP = {code: name for name, codes in CASUALTY_GROUPS.items() for code in codes}
LONDON_PREFIX = "E09"


def load_casualty_groups(years: tuple = YEARS, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """KSI casualties with road-user group and a London flag, all years."""
    frames = []
    for y in years:
        p = data_dir / f"casualty_{y}.csv"
        if not p.exists():
            wh.download("casualty", years=years)
        frames.append(pd.read_csv(p, usecols=["collision_index", "casualty_severity", "casualty_type"]))
    cas = pd.concat(frames, ignore_index=True)
    cas = cas[cas["casualty_severity"].isin([1, 2])].copy()
    cas["group"] = cas["casualty_type"].map(TYPE_TO_GROUP).fillna("Other / unknown")
    coll = pd.concat([pd.read_csv(data_dir / f"collision_{y}.csv",
                                  usecols=["collision_index", "local_authority_ons_district"])
                      for y in years], ignore_index=True)
    cas = cas.merge(coll, on="collision_index", how="left")
    cas["london"] = cas["local_authority_ons_district"].astype(str).str.startswith(LONDON_PREFIX)
    return cas


def _wilson(k: float, n: float):
    p = k / n
    z = 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, centre - half, centre + half


def share_by_group(cas: pd.DataFrame, min_n: int = 50) -> pd.DataFrame:
    """Share of KSI casualties per road-user group, split London / rest of GB.

    Returns one row per (group, area) with casualties, share, Wilson bounds."""
    rows = []
    for area, frame in (("London", cas[cas["london"]]), ("Rest of GB", cas[~cas["london"]])):
        n = len(frame)
        for group, g in frame.groupby("group"):
            k = len(g)
            if k < min_n:
                continue
            p, low, high = _wilson(k, n)
            rows.append({"group": group, "area": area, "casualties": k, "total": n,
                         "rate": p, "low": low, "high": high})
    return pd.DataFrame(rows)


def chart_mix(g: pd.DataFrame, out_path: Path) -> None:
    """Grouped horizontal bars: share of KSI casualties by road-user type,
    London against the rest of Great Britain."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = (g[g["area"] == "London"].sort_values("rate")["group"]).tolist()
    areas = ["Rest of GB", "London"]
    colors = {"Rest of GB": "#4c72b0", "London": "#c44e52"}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(order))
    h = 0.38
    for i, area in enumerate(areas):
        sub = g[g["area"] == area].set_index("group").reindex(order)
        off = (i - 0.5) * h
        ax.barh(y + off, sub["rate"] * 100, height=h, color=colors[area], label=area,
                xerr=[(sub["rate"] - sub["low"]) * 100, (sub["high"] - sub["rate"]) * 100],
                capsize=2, error_kw={"lw": 1})
    ax.set_yticks(y, order)
    ax.set_xlabel("% of fatal/serious casualties")
    ax.set_title("Who gets killed or seriously injured: casualty mix, 2021-2023")
    ax.legend()
    fig.text(0.01, 0.005,
             "Share of all fatal/serious casualties in each area. Error bars: 95% Wilson intervals.\n"
             "Descriptive view, not a model feature. Source: DfT STATS19 casualty data 2021-2023 (OGL v3.0).",
             fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
