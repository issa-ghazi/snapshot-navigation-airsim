r"""aggregate_results.py -- altitude curves and figures from the RIDF results.

Reads every `alt_XXXXm/results.csv` under a results root, computes the
per-altitude summary statistics, and writes:

    summary.csv          one row per altitude (the numbers for the thesis)
    fig_altitude.png     4-panel altitude curves, control overlaid if present
    fig_route_alt####m.png   error along the route per altitude
                             (shows WHERE aliasing happens, not just how much)

Success criteria (fixed before the data existed, see the methods notes):
    loc_hit_gsd   |loc_err| <= one sensor pixel (footprint / sensor_px)
    loc_hit_5m    |loc_err| <= 5 m  (the waypoint spacing; altitude-independent)
    yaw_hit       |yaw_err| <= 10 deg
    fail_20m      |loc_err| >  20 m  -- gross mislocalisation / aliasing

`alias_ratio` (true scene vs best competitor) and `ridf_depth_true` are
reported as-is; Gaffin's 0.20 threshold on `gaffin_ratio` is reported but is
not a usable criterion across a domain gap -- see the thesis notes.

Usage:
    python aggregate_results.py --results results --out figures
    python aggregate_results.py --results results --control results\control --out figures
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:                       # keeps geometry identical to both other phases
    from snapshot_lib import footprint_from_altitude
except ImportError:        # standalone use: 90 deg -> footprint = 2 * altitude
    import math

    def footprint_from_altitude(alt_m, fov_deg=90.0):
        return 2.0 * alt_m * math.tan(math.radians(fov_deg) / 2.0)

ALT_DIR = re.compile(r"alt_(\d+)m$")


def find_runs(root: Path) -> dict[float, Path]:
    """-> {altitude: results.csv} for every alt_XXXXm folder under root."""
    out = {}
    for d in sorted(root.glob("alt_*m")):
        m = ALT_DIR.search(d.name)
        f = d / "results.csv"
        if m and f.exists():
            out[float(m.group(1))] = f
    return out


def summarize(path: Path, sensor_px: int = 50, fov_deg: float = 90.0) -> dict:
    rows = list(csv.DictReader(open(path, newline="")))
    rows = [r for r in rows if r.get("loc_err_m", "") != ""]
    if not rows:
        raise SystemExit(f"{path}: no rows with ground truth (wp column)")
    n = len(rows)
    fnum = lambda k: [float(r[k]) for r in rows]
    alt = float(rows[0]["alt_m"])
    # capture_repeat.py records footprint_m; the synthetic control manifests
    # do not, so derive it from the altitude in that case.
    fp = (float(rows[0]["footprint_m"]) if rows[0].get("footprint_m")
          else footprint_from_altitude(alt, fov_deg))
    gsd = fp / sensor_px

    loc = fnum("loc_err_m")
    yaw = [abs(v) for v in fnum("yaw_err_deg")]
    alias = fnum("alias_ratio")
    depth = fnum("ridf_depth_true")
    gaf = fnum("gaffin_ratio")
    exact = sum(1 for r in rows if int(r["loc_err_wp"]) == 0)

    def share(vals, pred):
        return 100.0 * sum(1 for v in vals if pred(v)) / len(vals)

    return {
        "alt_m": alt, "n": n, "footprint_m": fp, "sensor_gsd_m": round(gsd, 2),
        "exact_pct": round(100.0 * exact / n, 1),
        "loc_hit_gsd_pct": round(share(loc, lambda v: v <= gsd), 1),
        "loc_hit_5m_pct": round(share(loc, lambda v: v <= 5.0), 1),
        "fail_20m_pct": round(share(loc, lambda v: v > 20.0), 1),
        "loc_median_m": round(st.median(loc), 2),
        "loc_mean_m": round(st.mean(loc), 2),
        "loc_p90_m": round(sorted(loc)[min(n - 1, int(0.9 * n))], 2),
        "loc_max_m": round(max(loc), 2),
        "yaw_hit_10deg_pct": round(share(yaw, lambda v: v <= 10.0), 1),
        "yaw_median_deg": round(st.median(yaw), 2),
        "yaw_max_deg": round(max(yaw), 2),
        "alias_median": round(st.median(alias), 4),
        "alias_below1_pct": round(share(alias, lambda v: v < 1.0), 1),
        "depth_median": round(st.median(depth), 4),
        "depth_min": round(min(depth), 4),
        "gaffin_median": round(st.median(gaf), 4),
        "unsettled": sum(1 for r in rows if r.get("settled", "1") == "0"),
        "max_building_m": (round(alt - min(fnum("depth_min_m")), 1)
                           if "depth_min_m" in rows[0] else ""),
    }


def route_figure(path: Path, out: Path, sensor_px: int = 50, fov_deg: float = 90.0):
    """Localisation error along the route -- reveals where aliasing sits."""
    rows = [r for r in csv.DictReader(open(path, newline=""))
            if r.get("loc_err_m", "") != ""]
    alt = float(rows[0]["alt_m"])
    fp = (float(rows[0]["footprint_m"]) if rows[0].get("footprint_m")
        else footprint_from_altitude(alt, fov_deg))
    gsd = fp / sensor_px
    s = [float(r["s_m"]) for r in rows]
    loc = [float(r["loc_err_m"]) for r in rows]
    depth = [float(r["ridf_depth_true"]) for r in rows]

    fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    ax[0].plot(s, loc, lw=1.2, color="C3")
    ax[0].axhline(gsd, ls="--", lw=0.8, color="0.5",
                  label=f"1 sensor pixel = {gsd:.1f} m")
    ax[0].axhline(5, ls=":", lw=0.8, color="0.3", label="5 m (waypoint spacing)")
    ax[0].set_ylabel("Localization error (m)")
    ax[0].set_title(f"Error along the route, {alt:.0f} m AGL "
                        f"(Footprint {fp:.0f} m)",
                    fontsize=10)
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].plot(s, depth, lw=1.2, color="C0")
    ax[1].set_ylabel("RIDF depth")
    ax[1].set_xlabel("Arc length along the route (m)")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def altitude_figure(summ: list[dict], ctrl: list[dict] | None, out: Path):
    alts = [s["alt_m"] for s in summ]
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))

    def plot(ax, key, label, ctrl_key=None, log=False):
        ax.plot(alts, [s[key] for s in summ], "o-", color="C3", label="Repeat (Sim)")
        if ctrl:
            ca = [c["alt_m"] for c in ctrl]
            ax.plot(ca, [c[ctrl_key or key] for c in ctrl], "s--", color="0.5",
                    label="Control (teach vs. teach)")
        ax.set_xlabel("Altitude AGL (m)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        if log:
            ax.set_yscale("log")
        ax.legend(fontsize=8)

    plot(axs[0][0], "loc_median_m", "Median localization error (m)")
    axs[0][0].plot(alts, [s["loc_max_m"] for s in summ], "^:", color="C1",
                   label="Maximum")
    axs[0][0].legend(fontsize=8)
    axs[0][0].set_title("Localization", fontsize=10)

    plot(axs[0][1], "loc_hit_5m_pct", "Share within 5 m (%)")
    axs[0][1].plot(alts, [s["fail_20m_pct"] for s in summ], "v:", color="C1",
                   label="Gross failures > 20 m (%)")
    axs[0][1].legend(fontsize=8)
    axs[0][1].set_ylim(-3, 103)
    axs[0][1].set_title("Hit rate", fontsize=10)

    plot(axs[1][0], "yaw_median_deg", "Median heading error (deg)")
    axs[1][0].plot(alts, [s["yaw_max_deg"] for s in summ], "^:", color="C1",
                   label="Maximum")
    axs[1][0].legend(fontsize=8)
    axs[1][0].set_title("Visual compass", fontsize=10)

    plot(axs[1][1], "depth_median", "Median RIDF depth")
    ax2 = axs[1][1].twinx()
    ax2.plot(alts, [s["alias_median"] for s in summ], "d-.", color="C2",
             label="alias_ratio Median")
    ax2.axhline(1.0, ls=":", lw=0.8, color="C2")
    ax2.set_ylabel("alias_ratio", color="C2")
    ax2.tick_params(axis="y", labelcolor="C2")
    axs[1][1].set_title("Distinctiveness", fontsize=10)

    fig.suptitle("Snapshot navigation over altitude (50x50 px / 10 gray levels)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True,
                    help="root containing alt_XXXXm/results.csv")
    ap.add_argument("--control", help="same layout for the teach-vs-teach control")
    ap.add_argument("--out", required=True, help="figure/summary output folder")
    ap.add_argument("--sensor-px", type=int, default=50)
    ap.add_argument("--fov", type=float, default=90.0,
                    help="sensor FOV in deg, used only when a results file "
                         "has no footprint_m column (control runs)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = find_runs(Path(args.results))
    if not runs:
        raise SystemExit(f"no alt_XXXXm/results.csv under {args.results}")
    summ = [summarize(runs[a], args.sensor_px, args.fov) for a in sorted(runs)]

    ctrl = None
    if args.control:
        cruns = find_runs(Path(args.control))
        if cruns:
            ctrl = [summarize(cruns[a], args.sensor_px, args.fov)
                    for a in sorted(cruns)]
            print(f"control: {len(ctrl)} altitudes")
        else:
            print(f"no control results under {args.control} -- skipping overlay")

    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)

    altitude_figure(summ, ctrl, out / "fig_altitude.png")
    for a in sorted(runs):
        route_figure(runs[a], out / f"fig_route_alt{a:04.0f}m.png", args.sensor_px, args.fov)

    print(f"\n{'Alt':>6} {'Footpr.':>9} {'GSD':>6} {'<5 m':>7} {'>20 m':>7} "
          f"{'Median':>8} {'Max':>8} {'|yaw|':>7} {'Depth':>7} {'alias':>7}")
    for s in summ:
        print(f"{s['alt_m']:5.0f}m {s['footprint_m']:8.0f}m {s['sensor_gsd_m']:5.1f}m "
              f"{s['loc_hit_5m_pct']:6.1f}% {s['fail_20m_pct']:6.1f}% "
              f"{s['loc_median_m']:7.1f}m {s['loc_max_m']:7.1f}m "
              f"{s['yaw_median_deg']:6.1f}° {s['depth_median']:7.3f} "
              f"{s['alias_median']:7.3f}")
    print(f"\n-> {out / 'summary.csv'}, {out / 'fig_altitude.png'} "
          f"and {len(runs)} route figures")


if __name__ == "__main__":
    main()
