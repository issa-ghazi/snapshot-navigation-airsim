#!/usr/bin/env python3
"""fig_routes_overview.py -- overview figure of the five study routes.

Draws each route centerline (yellow) and its southern starting point over the
cached orthophoto square of the route's middle waypoint, at the altitude
whose footprint covers the whole route (default 300 m: 4243 px at 0.20 m/px).
Panels are arranged two per row (2 x 2 x 1, last panel centred).

Requires the teach libraries of the five routes, including their WMS caches
(`_cache_dop/`), as produced by extract_route_snapshots.py:

    <data-root>/<library>/alt_0300m/manifest.csv
    <data-root>/<library>/_cache_dop/<cache_key>.png

Usage (from the repository root):

    python scripts/fig_routes_overview.py --data-root data \
        --out figures/fig_routes_overview.png

Library folder names default to the ones used in the thesis and can be
overridden per route, e.g. --lib Innenstadt=teach_rgb (folder name used in the
thesis runs).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from snapshot_lib import local_xy

# (panel title, teach library folder) in panel order
ROUTES = [("Innenstadt", "teach_innenstadt"),
          ("Sudbrack", "teach_sudbrack"),
          ("Nordpark", "teach_nordpark"),
          ("Forest", "teach_wald"),
          ("Agriculture", "teach_acker")]


def draw_route(ax, lib_dir: Path, alt_dir: str, gsd: float, half_px: int,
               title: str):
    with open(lib_dir / alt_dir / "manifest.csv", newline="") as f:
        man = list(csv.DictReader(f))
    seen, rows = set(), []
    for r in man:                      # one row per waypoint, any sensor variant
        if r["wp"] not in seen:
            seen.add(r["wp"])
            rows.append(r)
    rows.sort(key=lambda r: int(r["wp"]))

    lats = [float(r["lat"]) for r in rows]
    lons = [float(r["lon"]) for r in rows]
    c_lat = (min(lats) + max(lats)) / 2
    c_lon = (min(lons) + max(lons)) / 2
    mid = min(rows, key=lambda r: (float(r["lat"]) - c_lat) ** 2
                                + (float(r["lon"]) - c_lon) ** 2)

    img = Image.open(lib_dir / "_cache_dop" / f'{mid["cache_key"]}.png')
    half = img.size[0] / 2
    ax.imshow(img)

    xs, ys = [], []
    for la, lo in zip(lats, lons):
        e, n = local_xy(float(mid["lat"]), float(mid["lon"]), la, lo)
        xs.append(half + e / gsd)
        ys.append(half - n / gsd)
    ax.plot(xs, ys, lw=1.8, color="yellow")
    ax.plot(xs[0], ys[0], "o", ms=6, color="yellow")
    ax.set_xlim(half - half_px, half + half_px)
    ax.set_ylim(half + half_px, half - half_px)
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data",
                    help="folder holding the teach libraries")
    ap.add_argument("--alt", type=float, default=300.0,
                    help="library altitude whose cache image is shown (m)")
    ap.add_argument("--gsd", type=float, default=0.20,
                    help="source GSD of the cache images (m/px)")
    ap.add_argument("--half-px", type=int, default=1200,
                    help="half width of the shown window in px (1200 px = 240 m)")
    ap.add_argument("--lib", action="append", default=[],
                    help="override a library folder: NAME=FOLDER")
    ap.add_argument("--out", default="figures/fig_routes_overview.png")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    libs = dict(ROUTES)
    for item in args.lib:
        name, _, folder = item.partition("=")
        if name not in libs or not folder:
            ap.error(f"--lib expects NAME=FOLDER with NAME in {list(libs)}")
        libs[name] = folder

    # 3 rows x 2 panels; every panel spans two grid columns so that the fifth
    # panel can sit centred in the last row
    fig = plt.figure(figsize=(7.0, 10.6))
    gs = fig.add_gridspec(3, 4, hspace=0.12, wspace=0.08,
                          left=0.02, right=0.98, top=0.97, bottom=0.01)
    slots = [gs[0, 0:2], gs[0, 2:4], gs[1, 0:2], gs[1, 2:4], gs[2, 1:3]]
    alt_dir = f"alt_{args.alt:04.0f}m"
    root = Path(args.data_root)
    for slot, (name, _) in zip(slots, ROUTES):
        draw_route(fig.add_subplot(slot), root / libs[name], alt_dir,
                   args.gsd, args.half_px, name)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
