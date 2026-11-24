"""Per-mile exposure view: fatal/serious involvement per billion vehicle-miles.

Joins two DfT sources:
- STATS19 vehicle file (via whorisk): which vehicle groups were involved in
  each collision, and whether the collision was fatal/serious.
- DfT road traffic estimates (region_traffic_by_vehicle_type.csv): annual
  vehicle miles by vehicle type, summed to Great Britain.

The result answers the caveat the rest of the project cannot: how often
collisions involving each vehicle group turn out fatal/serious *per mile
that group travels*, not just per reported collision.

Caveats, kept honest:
- This is an involvement rate. A car-motorcycle crash counts once for the
  motorcycle group and once for the car/taxi group.
- The traffic estimates combine cars and taxis, so those two STATS19 groups
  are merged here.
- Traffic figures are estimates (accredited official statistics at national
  level); pedal cycle miles include e-bikes and are the least robust column.
"""
from pathlib import Path

import pandas as pd
from scipy.stats import chi2

import roadrisk as rr
import whorisk as wh

DATA = Path(__file__).parent / "data"
TRAFFIC_URL = ("https://storage.googleapis.com/dft-statistics/road-traffic/"
               "downloads/data-gov-uk/region_traffic_by_vehicle_type.csv")
TRAFFIC_FILE = "region_traffic_by_vehicle_type.csv"

YEARS = rr.YEARS

# exposure group -> traffic column. Car and Taxi merge because the traffic
# estimates do not separate them.
TRAFFIC_COLUMNS = {
    "Pedal cycle": "pedal_cycles",
    "Motorcycle": "two_wheeled_motor_vehicles",
    "Car or taxi": "cars_and_taxis",
    "Bus or coach": "buses_and_coaches",
    "Van": "LGVs",
    "Heavy goods": "all_HGVs",
}


def download_traffic() -> Path:
    """Fetch the DfT regional traffic-by-vehicle-type CSV (cached on disk)."""
    import urllib.request
    DATA.mkdir(parents=True, exist_ok=True)
    p = DATA / TRAFFIC_FILE
    if not p.exists():
        req = urllib.request.Request(TRAFFIC_URL, headers={"User-Agent": "roadrisk-portfolio-project"})
        with urllib.request.urlopen(req, timeout=120) as r, open(p, "wb") as f:
            f.write(r.read())
    return p


def gb_vehicle_miles(path: Path | None = None, years: tuple = YEARS) -> dict[str, float]:
    """Great Britain vehicle-miles per exposure group, summed over years."""
    path = path or download_traffic()
    df = pd.read_csv(path)
    df = df[df["year"].isin(years)]
    if df.empty:
        raise ValueError(f"no traffic rows for years {years}")
    return {group: float(df[col].sum()) for group, col in TRAFFIC_COLUMNS.items()}


def ksi_per_billion_miles(coll: pd.DataFrame, veh: pd.DataFrame,
                          miles: dict[str, float], min_n: int = 500) -> pd.DataFrame:
    """KSI collisions involving each group per billion vehicle-miles it travels.

    coll: one row per collision with a boolean 'ksi' column.
    veh: vehicle rows with collision_index and vehicle_type.
    miles: output of gb_vehicle_miles.
    """
    code_to_group = {code: name for name, codes in wh.VEHICLE_GROUPS.items() for code in codes}
    v = veh.copy()
    v["group"] = v["vehicle_type"].map(code_to_group)
    # merge before dedup so a car+taxi collision counts once for the merged group
    v.loc[v["group"].isin(["Car", "Taxi"]), "group"] = "Car or taxi"
    v = v.dropna(subset=["group"])[["collision_index", "group"]].drop_duplicates()
    long = v.merge(coll, on="collision_index", how="inner")

    rows = []
    for group, total_miles in miles.items():
        n = int((long["group"] == group).sum())
        ksi = int(long.loc[long["group"] == group, "ksi"].sum())
        if n < min_n or total_miles <= 0:
            continue
        rate = ksi / total_miles * 1e9
        low = 0.5 * chi2.ppf(0.025, 2 * ksi) / total_miles * 1e9 if ksi else 0.0
        high = 0.5 * chi2.ppf(0.975, 2 * (ksi + 1)) / total_miles * 1e9
        rows.append({"group": group, "ksi": ksi, "collisions": n,
                     "billion_miles": total_miles / 1e9,
                     "rate": rate, "low": low, "high": high})
    cols = ["group", "ksi", "collisions", "billion_miles", "rate", "low", "high"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values("rate", ascending=False).reset_index(drop=True)


def chart_exposure(g: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = g.iloc[::-1]  # highest rate on top
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.barh(d["group"], d["rate"],
            xerr=[d["rate"] - d["low"], d["high"] - d["rate"]],
            capsize=3, color="#c44e52")
    ax.set_xlabel("fatal/serious collisions involving the group, per billion vehicle-miles")
    ax.set_title("Fatal/serious involvement per mile travelled, by vehicle type, 2021-2023")
    fig.text(0.01, 0.005,
             "Involvement rate: a multi-vehicle crash counts once per group involved. Intervals: 95% Poisson.\n"
             "Sources: DfT STATS19 2021-2023 and DfT road traffic estimates 2021-2023 (both OGL v3.0).",
             fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
