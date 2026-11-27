"""Three years of change: severity share and per-mile KSI, 2021 vs 2022 vs 2023.

Every other view in this project pools the three years. This one keeps them
separate: 2021 traffic was still COVID-depressed (297.6 billion vehicle-miles
against 330.8 in 2023, per DfT road traffic estimates), so the per-mile view
tells a different story from raw counts.

Two honest caveats, also on the image:
- 2021 was not a normal traffic year; the per-mile comparison is the fairer
  one, but even that mixes exposure with behaviour.
- Severity recording changed over the period as police forces moved to
  injury-based reporting (39 of 44 by the end of 2025, per DfT), which
  classifies more casualties as serious. Part of any upward drift in the
  serious share is recording, not roads.

Descriptive only: never fed to the model.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

import roadrisk as rr
import exposure as ex

DATA_DIR = Path(__file__).parent / "data"
YEARS = rr.YEARS


def yearly_table(data_dir: Path = DATA_DIR, years: tuple = YEARS) -> pd.DataFrame:
    """One row per year: collisions, KSI, severity share (Wilson), and KSI
    per billion vehicle-miles (Poisson) using DfT traffic estimates."""
    traffic = pd.read_csv(ex.download_traffic())
    traffic = traffic[traffic["year"].isin(years)]
    miles_by_year = traffic.groupby("year")["all_motor_vehicles"].sum().to_dict()

    rows = []
    for y in years:
        d = pd.read_csv(data_dir / f"collision_{y}.csv", usecols=["collision_severity"])
        n = len(d)
        k = int(d["collision_severity"].isin([1, 2]).sum())
        p = k / n
        z = 1.96
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        m = miles_by_year.get(y)
        if not m:
            raise ValueError(f"no traffic miles for {y}")
        rate = k / m * 1e9
        low = 0.5 * chi2.ppf(0.025, 2 * k) / m * 1e9 if k else 0.0
        high = 0.5 * chi2.ppf(0.975, 2 * (k + 1)) / m * 1e9
        rows.append({"year": y, "collisions": n, "ksi": k, "share": p,
                     "share_low": centre - half, "share_high": centre + half,
                     "billion_miles": m / 1e9, "per_bvm": rate,
                     "per_bvm_low": low, "per_bvm_high": high})
    return pd.DataFrame(rows)


def chart_trend(g: pd.DataFrame, out_path: Path) -> None:
    """Two panels: severity share of collisions, and KSI per billion miles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = g["year"].astype(str)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8))
    a1.errorbar(x, g["share"] * 100,
                yerr=[(g["share"] - g["share_low"]) * 100, (g["share_high"] - g["share"]) * 100],
                marker="o", capsize=4, color="#4c72b0")
    a1.set_title("Fatal/serious share of reported collisions")
    a1.set_ylabel("% of collisions")
    a1.set_ylim(bottom=min(0, (g["share_low"] * 100).min() * 0.9))
    a2.errorbar(x, g["per_bvm"],
                yerr=[g["per_bvm"] - g["per_bvm_low"], g["per_bvm_high"] - g["per_bvm"]],
                marker="o", capsize=4, color="#c44e52")
    a2.set_title("Fatal/serious collisions per billion vehicle-miles")
    a2.set_ylabel("per billion vehicle-miles")
    a2.set_ylim(bottom=0)
    for a in (a1, a2):
        a.set_xlabel("year")
        a.grid(axis="y", alpha=0.3)
    fig.suptitle("Three years of change, 2021-2023")
    fig.text(0.01, 0.005,
             "2021 traffic was still COVID-depressed, so the per-mile panel is the fairer comparison. Severity recording\n"
             "changed over the period (injury-based reporting classifies more casualties as serious). Intervals: 95% Wilson\n"
             "(share) and 95% Poisson (per mile). Sources: DfT STATS19 and DfT road traffic estimates 2021-2023 (OGL v3.0).",
             fontsize=7.5, color="0.35")
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
