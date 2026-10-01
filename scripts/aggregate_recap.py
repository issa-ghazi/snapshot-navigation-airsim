#!/usr/bin/env python3
r"""aggregate_recap.py -- thesis-ready figures and summary table for stage-2 runs.

Reads every run folder produced by recapitulate_route.py (params.json,
steps.csv, candidates.csv and, for completed runs, summary.json) and writes

    summary_recap.csv        one row per run
    fig_recap_<run>.png      detailed path + error plot for one run
    fig_recap_overview.png   compact overview of all supplied runs

The overview is intended for the thesis: axis labels and legends are in
English, internal folder names are not shown, and a complete routes x
altitudes set is arranged explicitly as rows = routes and columns = altitudes.

Incomplete runs are handled deliberately. If summary.json is missing, metrics
that can be reconstructed exactly from steps.csv/candidates.csv are derived,
and a run whose final logged state is still ``running`` is labelled
``incomplete`` rather than ``lost`` or ``?``. No termination outcome is
invented.

Coordinates are ENU metres relative to the first taught route point, matching
the e_m / n_m values logged by recapitulate_route.py.

Usage
-----
    python scripts/aggregate_recap.py --runs data/recap --out data/figs_recap

For the systematic 3 x 3 thesis grid (folders sorted by name, i.e. rows =
routes, columns = altitudes):
    python scripts/aggregate_recap.py --runs <nine run folders> \\
        --out data/figs_recap_grid --cols 3
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SUMMARY_FIELDS = [
    "run", "world", "route", "alt_m", "footprint_m", "step_m", "arc_n",
    "arc_span_deg", "heading_mode", "stop_ratio", "outcome", "n_steps",
    "n_captures", "captures_per_step", "path_len_m", "progress_frac",
    "s_max_m", "xtrack_mean_m", "xtrack_max_m", "xtrack_sum_m",
    "loc_hit_5m_pct", "loc_fail_20m_pct", "heading_err_mean_deg",
    "heading_err_max_deg", "gaffin_ratio_median", "backtrack_steps",
    "wall_time_min"]

# Canonical route labels used in the thesis. Unknown routes fall back to a
# readable version of the route-file stem or run-folder name.
ROUTE_LABELS = {
    "route_s": "Innenstadt",
    "route_innenstadt": "Innenstadt",
    "route_nordpark": "Nordpark",
    "route_sudbrack": "Sudbrack",
}
ROUTE_ORDER = {"Innenstadt": 0, "Nordpark": 1, "Sudbrack": 2}


def find_runs(paths):
    """Return unique run directories containing steps.csv."""
    out = []
    for p in paths:
        p = Path(p)
        if (p / "steps.csv").exists():
            out.append(p)
        elif p.exists():
            out.extend(x.parent for x in p.glob("**/steps.csv"))
    return sorted(set(out))


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _f(row, key, default=None):
    """Safe float conversion for CSV fields."""
    try:
        value = row.get(key, "")
        return float(value) if value != "" else default
    except (TypeError, ValueError):
        return default


def _i(row, key, default=None):
    try:
        value = row.get(key, "")
        return int(value) if value != "" else default
    except (TypeError, ValueError):
        return default


def infer_summary(steps, cands):
    """Reconstruct only metrics that are exactly recoverable from log CSVs.

    This is primarily for interrupted/incomplete runs that have no
    summary.json. The function deliberately does not infer heading-error or
    backtracking metrics whose exact implementation belongs to
    recapitulate_route.py.
    """
    mv = [r for r in steps if (_i(r, "step", 0) or 0) > 0]
    if not mv:
        return {"outcome": "incomplete", "n_steps": 0}

    last = mv[-1]
    last_state = str(last.get("outcome", "")).strip()
    outcome = "incomplete" if last_state in ("", "running") else last_state

    xt = [v for r in mv if (v := _f(r, "xtrack_m")) is not None]
    loc = [v for r in mv if (v := _f(r, "loc_err_m")) is not None]
    svals = [v for r in mv if (v := _f(r, "s_nearest_m")) is not None]
    step_len = [v for r in mv if (v := _f(r, "step_len_m")) is not None]
    gaffin = [v for r in mv if (v := _f(r, "gaffin_ratio")) is not None]
    times = [v for r in mv if (v := _f(r, "t_step_s")) is not None]

    n_steps = max((_i(r, "step", 0) or 0) for r in mv)
    n_captures = len(cands) if cands else ""

    def pct(vals, pred):
        return round(100.0 * sum(1 for v in vals if pred(v)) / len(vals), 1) if vals else ""

    return {
        "outcome": outcome,
        "n_steps": n_steps,
        "n_captures": n_captures,
        "captures_per_step": (round(n_captures / n_steps, 2)
                              if n_captures != "" and n_steps else ""),
        "path_len_m": round(sum(step_len), 2) if step_len else "",
        "s_max_m": round(max(svals), 2) if svals else "",
        "xtrack_mean_m": round(st.mean(xt), 2) if xt else "",
        "xtrack_max_m": round(max(xt), 2) if xt else "",
        "xtrack_sum_m": round(sum(xt), 2) if xt else "",
        "loc_hit_5m_pct": pct(loc, lambda v: v <= 5.0),
        "loc_fail_20m_pct": pct(loc, lambda v: v > 20.0),
        "gaffin_ratio_median": round(st.median(gaffin), 4) if gaffin else "",
        "wall_time_min": round(sum(times) / 60.0, 1) if times else "",
    }


def load_run(d: Path) -> dict:
    params_path = d / "params.json"
    steps_path = d / "steps.csv"
    cands_path = d / "candidates.csv"
    if not params_path.exists() or not steps_path.exists() or not cands_path.exists():
        missing = [p.name for p in (params_path, steps_path, cands_path) if not p.exists()]
        raise FileNotFoundError(f"{d}: missing required file(s): {', '.join(missing)}")

    params = json.loads(params_path.read_text(encoding="utf-8"))
    steps = read_csv(steps_path)
    cands = read_csv(cands_path)

    summary_path = d / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        # Fill only genuinely absent fields from the logs; never overwrite the
        # authoritative summary written by recapitulate_route.py.
        inferred = infer_summary(steps, cands)
        for key, value in inferred.items():
            summary.setdefault(key, value)
    else:
        summary = infer_summary(steps, cands)

    return {
        "dir": d,
        "name": d.name,
        "params": params,
        "steps": steps,
        "cands": cands,
        "summary": summary,
        "summary_file_present": summary_path.exists(),
    }


def route_label(run):
    stem = Path(run["params"].get("route", "")).stem.lower()
    if stem in ROUTE_LABELS:
        return ROUTE_LABELS[stem]

    name = run["name"].lower()
    for key, label in (("innenstadt", "Innenstadt"),
                       ("nordpark", "Nordpark"),
                       ("sudbrack", "Sudbrack")):
        if key in name:
            return label

    label = stem.replace("route_", "").replace("_", " ").strip()
    return label.title() if label else run["name"].replace("_", " ").title()


def pretty_outcome(run):
    out = str(run["summary"].get("outcome", "incomplete") or "incomplete")
    return {
        "goal": "goal",
        "lost": "lost",
        "max_steps": "max steps",
        "running": "incomplete",
        "incomplete": "incomplete",
    }.get(out, out.replace("_", " "))


def route_xy(run):
    """Regenerate the taught route via the same code path as the agent."""
    from capture_repeat import load_route_csv
    from snapshot_lib import resample_route
    from recapitulate_route import Route

    p = run["params"]
    route_file = Path(p["route"])
    if not route_file.exists():
        alt = run["dir"] / route_file.name
        route_file = alt if alt.exists() else route_file
    if not route_file.exists():
        raise FileNotFoundError(
            f"route file not found for {run['name']}: {p['route']!r}. "
            "Run the aggregator from the project directory or place the "
            "route CSV next to the run folder.")

    wps = resample_route(load_route_csv(str(route_file)), p["spacing_m"])
    r = Route(wps)
    return r.xy, r.s


def _fmt(value, digits=1):
    if value in (None, "", "?"):
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def plot_run(run, out: Path):
    """Detailed two-panel figure for one run."""
    p, steps = run["params"], run["steps"]
    summ = run["summary"]
    xy, s_route = route_xy(run)

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(11, 6.2), gridspec_kw={"width_ratios": [1.0, 1.15]})
    draw_path(ax, run, xy, show_candidates=True, show_legend=True)

    ax.set_title(
        f"{route_label(run)}, {float(p['alt_m']):.0f} m AGL — "
        f"{pretty_outcome(run)}, {_fmt(summ.get('n_steps'), 0)} steps",
        fontsize=10)

    mv = [r for r in steps if (_i(r, "step", 0) or 0) > 0]
    s = [_f(r, "s_nearest_m", 0.0) for r in mv]
    xt = [_f(r, "xtrack_m", 0.0) for r in mv]
    le = [_f(r, "loc_err_m", 0.0) for r in mv]

    ax2.plot(s, xt, "o-", ms=3, lw=1.2, color="tab:red",
             label="Cross-track distance")
    ax2.plot(s, le, "s-", ms=3, lw=1.0, color="tab:blue", alpha=0.8,
             label="Localization error")
    top = max(25.0, 1.15 * max(xt + le, default=0.0))
    ax2.axhline(5, color="k", lw=0.6, ls=":", label="5 m hit threshold")
    ax2.axhline(20, color="k", lw=0.6, ls="--", label="20 m gross-error threshold")
    lost_m = float(p.get("lost_m", math.inf))
    if lost_m <= top:
        ax2.axhline(lost_m, color="tab:red", lw=0.7, ls="--", alpha=0.55,
                    label=f"{lost_m:.0f} m lost threshold")

    ax2.set_xlim(0, float(s_route[-1]))
    ax2.set_ylim(0, top)
    ax2.set_xlabel("Arc length of nearest route point, $s$ [m]")
    ax2.set_ylabel("Error [m]")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7.5, loc="upper left")
    ax2.set_title(
        f"Mean x-track {_fmt(summ.get('xtrack_mean_m'))} m; "
        f"max {_fmt(summ.get('xtrack_max_m'))} m; "
        f"hits $\\leq 5$ m {_fmt(summ.get('loc_hit_5m_pct'))}%",
        fontsize=9)

    fig.tight_layout()
    f = out / f"fig_recap_{run['name']}.png"
    fig.savefig(f, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return f


def draw_path(ax, run, xy, *, show_candidates=True, show_legend=True,
              show_bad=True):
    """Draw one ENU route/trajectory panel."""
    p, cands = run["params"], run["cands"]

    ax.plot(xy[:, 0], xy[:, 1], "-", color="0.55", lw=2.4,
            label="Taught route")
    ax.plot(xy[::5, 0], xy[::5, 1], ".", color="0.35", ms=3)

    if show_candidates:
        ce = [_f(r, "e_m") for r in cands if r.get("chosen") == "0"]
        cn = [_f(r, "n_m") for r in cands if r.get("chosen") == "0"]
        valid = [(e, n) for e, n in zip(ce, cn) if e is not None and n is not None]
        if valid:
            ax.plot([v[0] for v in valid], [v[1] for v in valid], ".",
                    color="tab:orange", ms=2, alpha=0.45,
                    label="Sampled candidates")

    ch = [r for r in cands if r.get("chosen") == "1"]
    pe = [_f(r, "e_m") for r in ch]
    pn = [_f(r, "n_m") for r in ch]
    path = [(e, n) for e, n in zip(pe, pn) if e is not None and n is not None]
    pe = [v[0] for v in path]
    pn = [v[1] for v in path]

    if path:
        ax.plot(pe, pn, "-", color="tab:red", lw=1.5, label="Agent trajectory")
        ax.plot(pe, pn, ".", color="tab:red", ms=3.5)
        ax.plot(pe[:1], pn[:1], "o", color="tab:green", ms=7.5,
                label="Start")

    goal_m = float(p.get("goal_m", 15.0))
    goal = plt.Circle((xy[-1, 0], xy[-1, 1]), goal_m, fill=False,
                      color="tab:green", lw=1.2, ls="--")
    ax.add_patch(goal)
    ax.plot(xy[-1, 0], xy[-1, 1], "*", color="tab:green", ms=10,
            label="Goal")

    if show_bad:
        bad = [r for r in ch if (_f(r, "loc_err_m", 0.0) or 0.0) > 20.0]
        if bad:
            ax.plot([_f(r, "e_m", 0.0) for r in bad],
                    [_f(r, "n_m", 0.0) for r in bad],
                    "x", color="k", ms=5.5, mew=1.0,
                    label="Gross localization error")

    ax.set_aspect("equal")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.grid(alpha=0.3)
    if show_legend:
        ax.legend(fontsize=7, loc="upper left")

    pad = 35.0
    xmin = min(xy[:, 0].min(), min(pe, default=float(xy[:, 0].min())))
    xmax = max(xy[:, 0].max(), max(pe, default=float(xy[:, 0].max())))
    ymin = min(xy[:, 1].min(), min(pn, default=float(xy[:, 1].min())))
    ymax = max(xy[:, 1].max(), max(pn, default=float(xy[:, 1].max())))
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)


def _overview_sort_key(run):
    label = route_label(run)
    route_key = ROUTE_ORDER.get(label, 1000)
    return (route_key, label.lower(), float(run["params"]["alt_m"]), run["name"])


def _shared_legend(fig):
    handles = [
        Line2D([0], [0], color="0.55", lw=2.4, label="Taught route"),
        Line2D([0], [0], color="tab:orange", marker=".", linestyle="None",
               markersize=5, alpha=0.55, label="Sampled candidates"),
        Line2D([0], [0], color="tab:red", lw=1.5, marker=".",
               markersize=5, label="Agent trajectory"),
        Line2D([0], [0], color="tab:green", marker="o", linestyle="None",
               markersize=6, label="Start"),
        Line2D([0], [0], color="tab:green", marker="*", linestyle="None",
               markersize=8, label="Goal / goal radius"),
        Line2D([0], [0], color="k", marker="x", linestyle="None",
               markersize=6, label="Gross localization error"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.005))


def plot_overview(runs, out: Path, cols: int = 3):
    """Create a compact thesis-ready overview.

    If the supplied runs form a complete route x altitude matrix, the figure
    is laid out explicitly with routes in rows and altitudes in columns. This
    makes the panel order independent of folder naming conventions.
    """
    runs = sorted(runs, key=_overview_sort_key)
    route_names = []
    for r in runs:
        label = route_label(r)
        if label not in route_names:
            route_names.append(label)
    route_names.sort(key=lambda x: (ROUTE_ORDER.get(x, 1000), x.lower()))
    alts = sorted({float(r["params"]["alt_m"]) for r in runs})

    by_key = {(route_label(r), float(r["params"]["alt_m"])): r for r in runs}
    full_grid = len(runs) == len(route_names) * len(alts)

    if full_grid and len(route_names) > 1 and len(alts) > 1:
        rows, ncols = len(route_names), len(alts)
        fig, axes = plt.subplots(rows, ncols,
                                 figsize=(3.35 * ncols, 4.6 * rows),
                                 squeeze=False)
        for i, route_name in enumerate(route_names):
            for j, alt in enumerate(alts):
                ax = axes[i][j]
                run = by_key[(route_name, alt)]
                xy, _ = route_xy(run)
                draw_path(ax, run, xy, show_candidates=True,
                          show_legend=False, show_bad=True)

                # Altitude is a column heading; route is a row heading.
                if i == 0:
                    ax.set_title(f"{alt:.0f} m AGL", fontsize=10, pad=7)
                else:
                    ax.set_title("")
                if j == 0:
                    ax.annotate(route_name, xy=(-0.30, 0.5), xycoords="axes fraction",
                                rotation=90, va="center", ha="center",
                                fontsize=10, fontweight="bold")

                # Keep axes readable without repeating labels everywhere.
                if i < rows - 1:
                    ax.set_xlabel("")
                    ax.tick_params(axis="x", labelbottom=False)
                if j > 0:
                    ax.set_ylabel("")
                    ax.tick_params(axis="y", labelleft=False)

                outcome = pretty_outcome(run)
                mean_xt = _fmt(run["summary"].get("xtrack_mean_m"))
                ax.text(0.98, 0.02, f"{outcome}; mean x-track {mean_xt} m",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=7,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72,
                                  pad=1.5))

        # One common extent for all panels: inner panels hide their tick
        # labels, so they must share the scale of the labelled outer axes.
        x0 = min(a.get_xlim()[0] for a in axes.flat)
        x1 = max(a.get_xlim()[1] for a in axes.flat)
        y0 = min(a.get_ylim()[0] for a in axes.flat)
        y1 = max(a.get_ylim()[1] for a in axes.flat)
        for a in axes.flat:
            a.set_xlim(x0, x1)
            a.set_ylim(y0, y1)

        _shared_legend(fig)
        fig.subplots_adjust(left=0.12, right=0.99, top=0.97, bottom=0.055,
                            wspace=0.08, hspace=0.10)
    else:
        n = len(runs)
        ncols = min(n, max(1, cols))
        rows = int(math.ceil(n / ncols))
        fig, axes = plt.subplots(rows, ncols,
                                 figsize=(3.6 * ncols, 4.8 * rows),
                                 squeeze=False)
        for ax, run in zip(axes.flat, runs):
            xy, _ = route_xy(run)
            draw_path(ax, run, xy, show_candidates=True,
                      show_legend=False, show_bad=True)
            sm = run["summary"]
            ax.set_title(
                f"{route_label(run)}, {float(run['params']['alt_m']):.0f} m AGL\n"
                f"{pretty_outcome(run)}; mean x-track "
                f"{_fmt(sm.get('xtrack_mean_m'))} m",
                fontsize=8.5)
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        _shared_legend(fig)
        fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.07,
                            wspace=0.20, hspace=0.28)

    f = out / "fig_recap_overview.png"
    fig.savefig(f, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run folders or roots containing them")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=3,
                    help="panels per row for a non-matrix overview; a complete "
                         "routes x altitudes set is laid out automatically")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = [load_run(d) for d in find_runs(args.runs)]
    if not runs:
        raise SystemExit("no runs found (folders need steps.csv)")

    with open(out / "summary_recap.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        for run in sorted(runs, key=_overview_sort_key):
            p, sm = run["params"], run["summary"]
            row = dict(
                run=run["name"], world=p["world"], route=Path(p["route"]).stem,
                alt_m=p["alt_m"], footprint_m=p["footprint_m"],
                step_m=p["arc"]["step_m"], arc_n=p["arc"]["n"],
                arc_span_deg=p["arc"]["span_deg"],
                heading_mode=p["heading_mode"], stop_ratio=p.get("stop_ratio"))
            row.update({k: sm.get(k, "") for k in SUMMARY_FIELDS if k not in row})
            w.writerow(row)

            fig = plot_run(run, out)
            source_note = "" if run["summary_file_present"] else " [inferred from logs]"
            print(
                f"{run['name']:32s} {p['world']:4s} {p['alt_m']:5.0f} m  "
                f"{str(sm.get('outcome', 'incomplete')):10s} "
                f"steps {str(sm.get('n_steps', '')):>4s}  "
                f"xtrack {_fmt(sm.get('xtrack_mean_m')):>5s} / "
                f"{_fmt(sm.get('xtrack_max_m')):>5s} m  "
                f"hit5m {_fmt(sm.get('loc_hit_5m_pct')):>5s} %  "
                f"-> {fig.name}{source_note}")

    if len(runs) > 1:
        print(f"overview -> {plot_overview(runs, out, args.cols).name}")
    print(f"summary -> {out / 'summary_recap.csv'}")


if __name__ == "__main__":
    main()
