"""Fatal/serious collisions per billion vehicle-miles, mapped by highway authority.

The district-level map (lamap) answers "where is the fatal/serious share
high among reported collisions". This map adds the exposure denominator the
district level lacks: DfT road traffic estimates, which are published at
highway-authority level (counties, unitary and metropolitan authorities,
London boroughs, Scottish and Welsh councils), not per district.

Method: fold each collision's local-authority district up to its highway
authority (E07 districts into their E10 county through the ONS LAD 2023 to
County 2023 lookup; every other LAD code is already a highway authority),
sum DfT vehicle-miles over the same three years, and colour each authority
by KSI collisions per billion vehicle-miles.

Historic quirk: DfT publishes traffic for Cumbria (E10000006) in 2021-2022
but for its two successor unitaries (Cumberland E06000063, Westmorland and
Furness E06000064) from 2023, because the county was the highway authority
until April 2023. Both sides of the join therefore treat the two successors
as one combined area across all years, drawn as one colour over two polygons.

Caveats (also on the image): still severity among police-reported collisions;
DfT traffic estimates for small areas are less robust than the national
figures; Scotland records severity through an injury-based system that
classifies more casualties as serious.
"""
from __future__ import annotations
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

import lamap

DATA_DIR = Path(__file__).parent / "data"
TRAFFIC_URL = ("https://storage.googleapis.com/dft-statistics/road-traffic/"
               "downloads/data-gov-uk/local_authority_traffic.csv")
CTYUA_URL = ("https://services1.arcgis.com/ESMARspQHYMw9BZ9/ArcGIS/rest/services/"
             "Counties_and_Unitary_Authorities_December_2023_Boundaries_UK_BGC/FeatureServer/0/query"
             "?where=1%3D1&outFields=CTYUA23CD%2CCTYUA23NM&resultRecordCount=400&f=geojson")
COUNTY_LOOKUP_URL = ("https://services1.arcgis.com/ESMARspQHYMw9BZ9/ArcGIS/rest/services/"
                     "LAD23_CTY23_EN_LU/FeatureServer/0/query"
                     "?where=1%3D1&outFields=LAD23CD%2CCTY23CD&returnGeometry=false&resultRecordCount=500&f=json")

CUMBRIA = "E10000006"  # historic county code, used as the combined-area key
CUMBRIA_SUCCESSORS = {"E06000063", "E06000064"}  # Cumberland, Westmorland and Furness
CUMBRIA_LABEL = "Cumberland / Westmorland and Furness (historic Cumbria)"


def _cached(url: str, name: str, data_dir: Path = DATA_DIR) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    p = data_dir / name
    if not p.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "roadrisk-portfolio-project"})
        with urllib.request.urlopen(req, timeout=120) as r:
            p.write_bytes(r.read())
    return p


def download_traffic(data_dir: Path = DATA_DIR) -> Path:
    return _cached(TRAFFIC_URL, "local_authority_traffic.csv", data_dir)


def download_ctyua_boundaries(data_dir: Path = DATA_DIR) -> Path:
    return _cached(CTYUA_URL, "ctyua_boundaries_2023.geojson", data_dir)


def download_county_lookup(data_dir: Path = DATA_DIR) -> Path:
    return _cached(COUNTY_LOOKUP_URL, "lad_to_county_2023.json", data_dir)


def county_map(lookup_path: Path) -> dict[str, str]:
    """LAD23 code -> highway-authority code (E07 districts to their county,
    Cumberland/Westmorland and Furness to the combined Cumbria key,
    everything else to itself)."""
    rows = json.loads(Path(lookup_path).read_text())["features"]
    lad_to_cty = {r["attributes"]["LAD23CD"]: r["attributes"]["CTY23CD"] for r in rows}

    def highway(lad23cd: str) -> str:
        if lad23cd in CUMBRIA_SUCCESSORS:
            return CUMBRIA
        # Only E07 shire districts need folding to their county. Unitaries
        # (E06), metropolitan boroughs (E08), London boroughs (E09) and
        # Scottish/Welsh councils are their own highway authority; the ONS
        # lookup would wrongly send boroughs to met counties (E11) and
        # London regions (E13), for which DfT publishes no traffic figures.
        if lad23cd.startswith("E07"):
            return lad_to_cty.get(lad23cd, lad23cd)
        return lad23cd

    lad_codes = set(lad_to_cty) | CUMBRIA_SUCCESSORS
    return {code: highway(code) for code in lad_codes}


def traffic_miles(traffic_path: Path, years: tuple = (2021, 2022, 2023)) -> dict[str, float]:
    """Vehicle-miles per highway authority summed over the years, with the
    Cumbria successors merged into the historic-county key."""
    df = pd.read_csv(traffic_path)
    df = df[df["year"].isin(years)]
    if df.empty:
        raise ValueError(f"no traffic rows for years {years}")
    df["authority"] = df["local_authority_code"].replace(
        {code: CUMBRIA for code in CUMBRIA_SUCCESSORS})
    return df.groupby("authority")["all_motor_vehicles"].sum().to_dict()


def ksi_per_mile_by_authority(lad_rates: pd.DataFrame, cmap: dict[str, str],
                              miles: dict[str, float], min_n: int = 200) -> pd.DataFrame:
    """Fold district-level KSI counts up to highway authorities and divide
    by vehicle-miles. lad_rates is the output of lamap.severity_by_authority."""
    g = lad_rates.copy()
    g["authority"] = g["lad23cd"].map(lambda c: cmap.get(str(c), str(c)))
    g = g.groupby("authority")[["ksi", "collisions"]].sum().reset_index()
    g["miles"] = g["authority"].map(miles)
    g = g.dropna(subset=["miles"])
    g = g[g["miles"] > 0]
    n, k, m = g["collisions"], g["ksi"], g["miles"]
    g["rate"] = k / m * 1e9
    g["low"] = [0.0 if ki == 0 else 0.5 * chi2.ppf(0.025, 2 * ki) / mi * 1e9
                for ki, mi in zip(k, m)]
    g["high"] = [0.5 * chi2.ppf(0.975, 2 * (ki + 1)) / mi * 1e9 for ki, mi in zip(k, m)]
    g["enough"] = n >= min_n
    return g


def draw_map(frame: pd.DataFrame, boundaries_path: Path, out_path: Path,
             min_n: int = 200) -> None:
    """Choropleth on the December 2023 county/unitary boundaries. The combined
    Cumbria row colours both successor polygons. Northern Ireland is grey:
    STATS19 covers Great Britain only. Colour scale spans the full range of
    the coloured authorities (nothing clipped)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import PowerNorm

    fc = json.loads(Path(boundaries_path).read_text())
    rates = frame.set_index("authority")
    coloured = rates[rates["enough"]]["rate"]
    vmin, vmax = float(coloured.min()), float(coloured.max())
    if vmax <= vmin:  # one authority, or all equal: keep a valid scale
        vmin, vmax = max(0.0, vmin - 0.01), vmax + 0.01
    # sqrt stretch: central London is ~30x the rural low end, so a linear
    # scale would leave most of the country one shade. Tick labels show
    # true values; nothing is clipped off either end.
    norm = PowerNorm(0.5, vmin=vmin, vmax=vmax)
    verts, facecolors, cmap = [], [], plt.get_cmap("YlOrRd")
    grey = (0.90, 0.90, 0.90, 1.0)
    for feat in fc["features"]:
        code = feat["properties"]["CTYUA23CD"]
        key = CUMBRIA if code in CUMBRIA_SUCCESSORS else code
        polys = list(lamap._polygons(feat["geometry"]))
        if key in rates.index and bool(rates.loc[key, "enough"]):
            color = cmap(norm(float(rates.loc[key, "rate"])))
        else:
            color = grey
        for ring in polys:
            verts.append(np.asarray(ring))
            facecolors.append(color)
    fig, ax = plt.subplots(figsize=(7.5, 8.5))
    coll = PolyCollection(verts, facecolors=facecolors, edgecolors="white", linewidths=0.3)
    ax.add_collection(coll)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    ticks = [t for t in (25, 50, 100, 200, 400, 600) if vmin <= t <= vmax]
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, ticks=ticks)
    cb.set_label("fatal/serious collisions per billion vehicle-miles\n(sqrt-stretched scale, true values on ticks)")
    ax.set_title("Fatal/serious collisions per billion vehicle-miles, 2021-2023\n"
                 f"by highway authority (grey: under {min_n} collisions; NI not in STATS19)",
                 fontsize=11)
    fig.text(0.01, 0.01,
             "Severity among police-reported collisions, with traffic as the denominator. DfT traffic estimates for small\n"
             "areas are less robust than national figures. Scotland records severity through an injury-based system that\n"
             "classifies more casualties as serious. Cumberland and Westmorland and Furness are combined (historic\n"
             "Cumbria) because DfT published traffic for the county in 2021-2022. Data: DfT STATS19 and DfT road\n"
             "traffic estimates 2021-2023; boundaries: ONS Open Geography Portal (Dec 2023). Open Government Licence v3.0.",
             fontsize=7.5, color="#444444")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
