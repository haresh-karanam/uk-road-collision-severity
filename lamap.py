"""Fatal/serious share by local authority, drawn as a map of Great Britain.

Boundaries: Office for National Statistics, Local Authority Districts
(December 2023) Boundaries UK BGC, from the ONS Open Geography Portal,
Open Government Licence v3.0.

Historic codes: the collision file records the authority as it was at the
time, so 2021-2022 rows can name districts abolished in April 2023
(Cumbria, North Yorkshire, Somerset). The ONS "LAD 2022 to LAD 2023
lookup" folds them into their successor unitary authorities so every
year lands on the same December 2023 map.
"""
from __future__ import annotations
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
BOUNDARY_URL = ("https://services1.arcgis.com/ESMARspQHYMw9BZ9/ArcGIS/rest/services/"
                "Local_Authority_Districts_December_2023_Boundaries_UK_BGC/FeatureServer/0/query"
                "?where=1%3D1&outFields=LAD23CD%2CLAD23NM&resultRecordCount=400&f=geojson")
LOOKUP_URL = ("https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
              "LAD22_LAD23_UK_LU_v1/FeatureServer/0/query"
              "?where=1%3D1&outFields=LAD22CD%2CLAD23CD&returnGeometry=false&resultRecordCount=500&f=json")
UA = {"User-Agent": "roadrisk-portfolio-project"}


def _fetch(url: str) -> dict:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def download_boundaries(data_dir: Path = DATA_DIR) -> Path:
    """Fetch the ONS local-authority boundary GeoJSON (cached on disk)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    p = data_dir / "lad_boundaries_2023.geojson"
    if not p.exists():
        p.write_text(json.dumps(_fetch(BOUNDARY_URL)))
    return p


def download_lookup(data_dir: Path = DATA_DIR) -> Path:
    """Fetch the ONS LAD 2022 -> 2023 lookup (cached on disk)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    p = data_dir / "lad_lookup_2022_to_2023.json"
    if not p.exists():
        p.write_text(json.dumps(_fetch(LOOKUP_URL)))
    return p


def current_code_map(lookup_path: Path) -> dict[str, str]:
    """LAD22 code -> December 2023 code, including unchanged authorities."""
    rows = json.loads(Path(lookup_path).read_text())["features"]
    return {r["attributes"]["LAD22CD"]: r["attributes"]["LAD23CD"] for r in rows}


def severity_by_authority(d: pd.DataFrame, cmap: dict[str, str], min_n: int = 200):
    """KSI share of reported collisions per current local authority.

    Returns (frame, dropped): frame has one row per authority with
    collisions, ksi, rate, 95% Wilson low/high and an `enough` flag
    (collisions >= min_n, so thin areas can be greyed rather than
    dramatic); dropped is how many rows carried a code with no
    December 2023 authority (e.g. DfT's special Heathrow code)."""
    mapped = d["local_authority_ons_district"].map(lambda c: cmap.get(str(c), str(c)))
    known = set(cmap.values())
    dropped = int((~mapped.isin(known)).sum())
    g = d[mapped.isin(known)].groupby(mapped[mapped.isin(known)])["ksi"].agg(["sum", "count"]).reset_index()
    g = g.rename(columns={"local_authority_ons_district": "lad23cd", "sum": "ksi", "count": "collisions"})
    n, k = g["collisions"].to_numpy(float), g["ksi"].to_numpy(float)
    p = k / n
    z = 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    g["rate"], g["low"], g["high"] = p, centre - half, centre + half
    g["enough"] = g["collisions"] >= min_n
    return g, dropped


def _polygons(geometry: dict):
    """Yield (ring) coordinate lists for Polygon/MultiPolygon geometries."""
    if geometry["type"] == "Polygon":
        for ring in geometry["coordinates"][:1]:
            yield ring
    elif geometry["type"] == "MultiPolygon":
        for poly in geometry["coordinates"]:
            for ring in poly[:1]:
                yield ring


def draw_map(frame: pd.DataFrame, boundaries_path: Path, out_path: Path,
             min_n: int = 200, vmin: float | None = None, vmax: float | None = None) -> None:
    """Draw the choropleth: colour = KSI share, grey = too few collisions.

    The colour scale spans the full range of the coloured authorities by
    default, so nothing is silently clipped off the end of the scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    fc = json.loads(Path(boundaries_path).read_text())
    rates = frame.set_index("lad23cd")
    coloured = rates[rates["enough"]]["rate"]
    if vmin is None:
        vmin = float(coloured.min())
    if vmax is None:
        vmax = float(coloured.max())
    if vmax <= vmin:  # one authority, or all equal: keep a valid scale
        vmin, vmax = max(0.0, vmin - 0.01), vmax + 0.01
    verts, facecolors, cmap = [], [], plt.get_cmap("YlOrRd")
    grey = (0.90, 0.90, 0.90, 1.0)
    names = {}
    for feat in fc["features"]:
        code = feat["properties"]["LAD23CD"]
        names[code] = feat["properties"]["LAD23NM"]
        polys = list(_polygons(feat["geometry"]))
        if code in rates.index and bool(rates.loc[code, "enough"]):
            r = float(rates.loc[code, "rate"])
            color = cmap((min(max(r, vmin), vmax) - vmin) / (vmax - vmin))
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
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin * 100, vmax * 100))
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("% of reported collisions fatal/serious")
    ax.set_title("Fatal/serious share of reported collisions by local authority, 2021-2023\n"
                 f"(grey: fewer than {min_n} collisions, or no matching authority)", fontsize=11)
    fig.text(0.01, 0.01,
             "Share among police-reported collisions, not risk per mile travelled. Forces record severity differently:\n"
             "injury-based systems (as in Scotland) classify more casualties as serious. Data: DfT STATS19 2021-2023;\n"
             "boundaries: ONS Open Geography Portal (Dec 2023). Open Government Licence v3.0.",
             fontsize=7.5, color="#444444")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
