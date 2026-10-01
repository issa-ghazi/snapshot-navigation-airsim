r"""aggregate_sensors.py -- evaluation of the sensor-variant sweep.

Reads every results folder produced by the sensor sweep (one per
altitude x resolution x gray-level combination) and answers the question the
sweep was run for: **does the lower altitude boundary move when the sensor
gets coarser?**

Output
------
    summary_sensors.csv    one row per (altitude, resolution, gray levels)
    boundary.csv           lowest altitude with zero gross failures, per variant
    fig_sensor_matrix.png  resolution x gray-levels heat map, one panel per
                           altitude, coloured by "within 5 m" share
    fig_sensor_curves.png  share within 5 m over altitude, one panel per gray
                           level, one line per resolution

Folder layout expected (as produced by the sweep loop):

    results\sensors\alt_0050m_r10_g2\results.csv
                                    \params.json

The sensor configuration is read from params.json (`sensor_px`,
`gray_levels`, `alt_m`); the folder name is only used as a fallback, so
renaming folders does not corrupt the analysis.

Usage:
    python aggregate_sensors.py --results results\sensors --out figures_sensors
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FOLDER = re.compile(r"alt_(\d+)m_r(\d+)_g(\d+)$")


def config_of(folder: Path) -> tuple[float, int, int] | None:
    """(altitude, sensor_px, gray_levels) from params.json, folder as fallback."""
    pj = folder / "params.json"
    if pj.exists():
        try:
            p = json.loads(pj.read_text())
            return (float(p["alt_m"]), int(p["sensor_px"]),
                    int(p["gray_levels"]))
        except Exception:
            pass
    m = FOLDER.search(folder.name)
    return (float(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def summarize(path: Path, alt: float, res: int, gray: int) -> dict:
    rows = [r for r in csv.DictReader(open(path, newline=""))
            if r.get("loc_err_m", "") != ""]
    if not rows:
        return {}
    n = len(rows)
    fnum = lambda k: [float(r[k]) for r in rows]
    fp = float(rows[0].get("footprint_m") or 2.0 * alt)
    loc = fnum("loc_err_m")
    yaw = [abs(v) for v in fnum("yaw_err_deg")]
    depth = fnum("ridf_depth_true")
    alias = fnum("alias_ratio")
    pct = lambda vals, pred: round(100.0 * sum(1 for v in vals if pred(v))
                                   / len(vals), 1)
    return {
        "alt_m": alt, "sensor_px": res, "gray_levels": gray, "n": n,
        "footprint_m": fp, "sensor_gsd_m": round(fp / res, 2),
        "mask_px": "", "exact_pct": pct(
            [abs(int(r["loc_err_wp"])) for r in rows], lambda v: v == 0),
        "loc_hit_5m_pct": pct(loc, lambda v: v <= 5.0),
        "fail_20m_pct": pct(loc, lambda v: v > 20.0),
        "loc_median_m": round(st.median(loc), 2),
        "loc_mean_m": round(st.mean(loc), 2),
        "loc_max_m": round(max(loc), 2),
        "yaw_hit_10deg_pct": pct(yaw, lambda v: v <= 10.0),
        "yaw_median_deg": round(st.median(yaw), 2),
        "yaw_max_deg": round(max(yaw), 2),
        "depth_median": round(st.median(depth), 4),
        "alias_median": round(st.median(alias), 4),
    }


def collect(root: Path) -> list[dict]:
    out = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        f = d / "results.csv"
        cfg = config_of(d)
        if not f.exists() or cfg is None:
            continue
        s = summarize(f, *cfg)
        if s:
            out.append(s)
    return out


def boundary_table(rows: list[dict]) -> list[dict]:
    """Lowest tested altitude at which a variant has no gross failure, and
    the lowest at which it still keeps every waypoint within 5 m."""
    out = []
    variants = sorted({(r["sensor_px"], r["gray_levels"]) for r in rows})
    for res, gray in variants:
        sub = sorted((r for r in rows
                      if r["sensor_px"] == res and r["gray_levels"] == gray),
                     key=lambda r: r["alt_m"])
        clean = [r["alt_m"] for r in sub if r["fail_20m_pct"] == 0.0]
        perfect = [r["alt_m"] for r in sub if r["loc_hit_5m_pct"] == 100.0]
        out.append({
            "sensor_px": res, "gray_levels": gray,
            "n_altitudes": len(sub),
            "min_alt_no_failure_m": min(clean) if clean else "",
            "min_alt_all_within_5m_m": min(perfect) if perfect else "",
            "best_hit_pct": max(r["loc_hit_5m_pct"] for r in sub),
            "worst_hit_pct": min(r["loc_hit_5m_pct"] for r in sub),
        })
    return out


def matrix_figure(rows: list[dict], out: Path, key="loc_hit_5m_pct",
                  label="Share within 5 m (%)"):
    alts = sorted({r["alt_m"] for r in rows})
    res_v = sorted({r["sensor_px"] for r in rows})
    gray_v = sorted({r["gray_levels"] for r in rows})
    lut = {(r["alt_m"], r["sensor_px"], r["gray_levels"]): r for r in rows}

    ncol = min(4, len(alts))
    nrow = int(np.ceil(len(alts) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.0 * nrow),
                            squeeze=False)
    fig.subplots_adjust(wspace=0.18, hspace=0.35)
    for i, alt in enumerate(alts):
        ax = axs[i // ncol][i % ncol]
        M = np.full((len(res_v), len(gray_v)), np.nan)
        for a, r in enumerate(res_v):
            for b, g in enumerate(gray_v):
                rec = lut.get((alt, r, g))
                if rec:
                    M[a, b] = rec[key]
        im = ax.imshow(M, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")
        first_col = (i % ncol == 0)
        last_row = (i // ncol == nrow - 1) or (i + ncol >= len(alts))
        ax.set_xticks(range(len(gray_v)),
                      [str(g) for g in gray_v] if last_row else [""] * len(gray_v))
        ax.set_yticks(range(len(res_v)),
                      [f"{r}x{r}" for r in res_v] if first_col
                      else [""] * len(res_v))
        ax.set_title(f"{alt:.0f} m AGL", fontsize=9)
        if first_col:
            ax.set_ylabel("Resolution")
        if last_row:
            ax.set_xlabel("Gray levels")
        for a in range(len(res_v)):
            for b in range(len(gray_v)):
                if not np.isnan(M[a, b]):
                    ax.text(b, a, f"{M[a, b]:.0f}", ha="center", va="center",
                            fontsize=7,
                            color="black" if M[a, b] > 35 else "white")
    for j in range(len(alts), nrow * ncol):
        axs[j // ncol][j % ncol].axis("off")
    fig.colorbar(im, ax=axs, shrink=0.6, label=label)
    fig.suptitle(f"Sensor variants: {label}", fontsize=11)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def curve_figure(rows: list[dict], out: Path):
    gray_v = sorted({r["gray_levels"] for r in rows})
    res_v = sorted({r["sensor_px"] for r in rows})
    fig, axs = plt.subplots(1, len(gray_v), figsize=(4.2 * len(gray_v), 3.8),
                            sharey=True, squeeze=False)
    cmap = plt.get_cmap("viridis")
    for i, g in enumerate(gray_v):
        ax = axs[0][i]
        for j, r in enumerate(res_v):
            sub = sorted((x for x in rows if x["gray_levels"] == g
                          and x["sensor_px"] == r), key=lambda x: x["alt_m"])
            if not sub:
                continue
            ax.plot([x["alt_m"] for x in sub],
                    [x["loc_hit_5m_pct"] for x in sub], "o-", markersize=4,
                    color=cmap(j / max(1, len(res_v) - 1)), label=f"{r}x{r}")
        ax.set_title(f"{g} gray levels", fontsize=10)
        ax.set_xlabel("Altitude AGL (m)")
        ax.grid(alpha=0.3)
        ax.set_ylim(-3, 103)
        if i == 0:
            ax.set_ylabel("Share within 5 m (%)")
            ax.legend(fontsize=8, title="Resolution", title_fontsize=8)
    fig.suptitle("Lower altitude limit per sensor configuration",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True,
                    help=r"folder holding alt_XXXXm_rNN_gNN subfolders")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = collect(Path(args.results))
    if not rows:
        raise SystemExit(f"no usable results under {args.results}")
    print(f"{len(rows)} runs read "
          f"({len({r['alt_m'] for r in rows})} altitudes x "
          f"{len({(r['sensor_px'], r['gray_levels']) for r in rows})} variants)")

    rows.sort(key=lambda r: (r["alt_m"], r["sensor_px"], r["gray_levels"]))
    with open(out / "summary_sensors.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    bt = boundary_table(rows)
    with open(out / "boundary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(bt[0].keys()))
        w.writeheader()
        w.writerows(bt)

    matrix_figure(rows, out / "fig_sensor_matrix.png")
    curve_figure(rows, out / "fig_sensor_curves.png")

    print(f"\nLower limit per variant (lowest altitude without gross failure):")
    print(f"{'Resolution':>10} {'Gray':>11} {'no failure':>13} "
          f"{'all < 5 m':>12} {'best hit':>12}")
    for b in bt:
        na = b["min_alt_no_failure_m"]
        pa = b["min_alt_all_within_5m_m"]
        print(f"{b['sensor_px']:>7}x{b['sensor_px']:<2} {b['gray_levels']:>11} "
              f"{(f'{na:.0f} m' if na != '' else 'never'):>13} "
              f"{(f'{pa:.0f} m' if pa != '' else 'never'):>12} "
              f"{b['best_hit_pct']:>11.1f}%")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
