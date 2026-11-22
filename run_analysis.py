"""Command-line run: download data, evaluate models, write charts + metrics to outputs/."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import roadrisk as rr

OUT = Path(__file__).parent / "outputs"


def main():
    OUT.mkdir(exist_ok=True)
    d = rr.load()
    train, test = rr.split_by_year(d)
    res = rr.evaluate(train, test)
    metrics = {k: v for k, v in res.items() if k != "models"}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    g = rr.ksi_rate_by(d, "speed_limit")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(g["speed_limit"].astype(str), g["rate"] * 100,
                yerr=[(g["rate"] - g["low"]) * 100, (g["high"] - g["rate"]) * 100], fmt="o", capsize=4)
    ax.set_xlabel("Speed limit (mph)"); ax.set_ylabel("% of collisions fatal/serious")
    ax.set_title("Severity share by speed limit (95% CI)"); fig.tight_layout(); fig.savefig(OUT / "ksi_by_speed.png", dpi=130)

    h = rr.ksi_rate_by(d, "hour", min_n=50)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(h["hour"], h["rate"] * 100, marker="o"); ax.fill_between(h["hour"], h["low"] * 100, h["high"] * 100, alpha=.2)
    ax.set_xlabel("Hour of day"); ax.set_ylabel("% fatal/serious"); ax.set_title("Severity share by hour")
    fig.tight_layout(); fig.savefig(OUT / "ksi_by_hour.png", dpi=130)

    cal = rr.calibration_table(res["models"]["hgb"], test)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(cal["predicted"], cal["observed"], marker="o"); m = max(cal["predicted"].max(), cal["observed"].max())
    ax.plot([0, m], [0, m], "--", color="grey"); ax.set_xlabel("Mean predicted"); ax.set_ylabel("Observed (2023)")
    ax.set_title("Calibration by decile"); fig.tight_layout(); fig.savefig(OUT / "calibration.png", dpi=130)


if __name__ == "__main__":
    main()
