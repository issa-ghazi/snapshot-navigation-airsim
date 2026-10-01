#!/usr/bin/env python3
"""fig_three_routes.py -- the two three-route result figures of the thesis.

Reads only the aggregated summary tables (no simulator, no raw results):

    <open-loop>/summary_<route>.csv          altitude sweep, 50x50 px / 10 gray
                                             levels (aggregate_results.py)
    <sensors>/summary_sensors_<route>.csv    sensor sweep, 15 variants
                                             (aggregate_sensors.py)

and writes

    fig_altitude_three_routes.png   hits, gross failures and RIDF depth over
                                    altitude (thesis Section 5.3)
    fig_sensor_three_routes.png     (a) spatial resolution at 10 gray levels,
                                    (b) gray levels at 50x50 px (Section 5.4)

With the committed tables in results_summary/ this reproduces the thesis
figures exactly:

    python scripts/fig_three_routes.py --out figures
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROUTES = [("innenstadt", "Innenstadt", "#0072B2", "o"),
          ("nordpark", "Nordpark", "#009E73", "s"),
          ("sudbrack", "Sudbrack", "#D55E00", "^")]
ALT_TICKS = [40, 50, 60, 80, 100, 140, 200, 300]
RES = [(10, "#999999", "v"), (20, "#56B4E9", "D"), (40, "#009E73", "s"),
       (50, "#000000", "o"), (80, "#D55E00", "^")]
GRAY = [(2, "#CC79A7", "v"), (10, "#000000", "o"), (100, "#E69F00", "s")]

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "lines.linewidth": 1.3,
                     "lines.markersize": 3.8, "axes.grid": True,
                     "grid.alpha": 0.3, "grid.linewidth": 0.5})


def read_rows(path: Path) -> list[dict]:
    """CSV rows with every value converted to float where possible."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def series(rows, key, **match):
    """(altitudes, values) of `key`, filtered by `match`, sorted by altitude."""
    sel = sorted((r for r in rows
                  if all(r[k] == v for k, v in match.items())),
                 key=lambda r: r["alt_m"])
    return [r["alt_m"] for r in sel], [r[key] for r in sel]


def alt_axis(ax, label=True):
    ax.set_xscale("log")
    ax.set_xticks(ALT_TICKS)
    ax.set_xticklabels([str(a) for a in ALT_TICKS])
    ax.minorticks_off()
    ax.set_xlim(37, 325)
    if label:
        ax.set_xlabel("Altitude AGL (m)")


def fig_altitude(open_loop: Path, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.55), constrained_layout=True)
    for key, name, col, mk in ROUTES:
        d = read_rows(open_loop / f"summary_{key}.csv")
        axes[0].plot(*series(d, "loc_hit_5m_pct"), color=col, marker=mk, label=name)
        axes[1].plot(*series(d, "fail_20m_pct"), color=col, marker=mk, label=name)
        axes[2].plot(*series(d, "depth_median"), color=col, marker=mk, label=name)
    axes[0].set_ylabel("Share of observations (%)")
    axes[0].set_title(r"(a) Localization hits ($e_s \leq 5$ m)")
    axes[0].set_ylim(0, 103)
    axes[1].set_ylabel("Share of observations (%)")
    axes[1].set_title(r"(b) Gross failures ($e_s > 20$ m)")
    axes[1].set_ylim(-2, 65)
    axes[1].legend(loc="upper right", frameon=True, framealpha=0.9)
    axes[2].set_ylabel(r"Median RIDF depth $\delta$")
    axes[2].set_title("(c) RIDF depth at the true scene")
    axes[2].set_ylim(0, 0.45)
    for ax in axes:
        alt_axis(ax)
    fig.savefig(out / "fig_altitude_three_routes.png", dpi=250)
    plt.close(fig)


def fig_sensor(sensors: Path, out: Path):
    fig, axes = plt.subplots(2, 3, figsize=(7.3, 4.3), sharey=True,
                             constrained_layout=True)
    for c, (key, name, _, _) in enumerate(ROUTES):
        d = read_rows(sensors / f"summary_sensors_{key}.csv")
        ax = axes[0, c]
        for px, col, mk in RES:
            ax.plot(*series(d, "loc_hit_5m_pct", sensor_px=px, gray_levels=10),
                    color=col, marker=mk, label=rf"${px}\times{px}$")
        ax.set_title(name)
        alt_axis(ax, label=False)
        ax = axes[1, c]
        for g, col, mk in GRAY:
            ax.plot(*series(d, "loc_hit_5m_pct", sensor_px=50, gray_levels=g),
                    color=col, marker=mk, label=f"{g} gray levels")
        alt_axis(ax)
    axes[0, 0].set_ylabel("(a) 10 gray levels\n" + r"hits $e_s \leq 5$ m (%)")
    axes[1, 0].set_ylabel(r"(b) $50\times50$ px" + "\n" + r"hits $e_s \leq 5$ m (%)")
    for r in (0, 1):
        axes[r, 0].set_ylim(0, 103)
    axes[0, 0].legend(loc="lower center", ncol=2, frameon=True, framealpha=0.9,
                      title="Sensor resolution (px)", title_fontsize=7)
    axes[1, 0].legend(loc="lower right", frameon=True, framealpha=0.9)
    fig.savefig(out / "fig_sensor_three_routes.png", dpi=250)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--open-loop", default="results_summary/open_loop",
                    help="folder with summary_<route>.csv")
    ap.add_argument("--sensors", default="results_summary/sensors",
                    help="folder with summary_sensors_<route>.csv")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_altitude(Path(a.open_loop), out)
    fig_sensor(Path(a.sensors), out)
    print("written to", out)


if __name__ == "__main__":
    main()
