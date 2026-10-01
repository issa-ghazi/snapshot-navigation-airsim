#!/usr/bin/env python3
"""make_s_route.py -- generate the canonical S-shaped test route.

Gaffin et al. (2015) evaluate their route-recapitulation algorithm on a
"simple S-shaped training path" (their Fig. 5). This script produces the
same shape as a geo-referenced CSV: two stacked semicircular arcs of radius
height/4, traversed bottom-to-top, centered on a given WGS84 position.

Output CSV columns: idx, s_m, lat, lon, bearing_deg
(The teach extractor resamples and recomputes bearings anyway; this file
mainly pins down the geometry so it is reproducible from parameters.)

Example:
    python make_s_route.py --center-lat 52.024 --center-lon 8.528 \
        --height 400 --step 5 --out route_s.csv
"""
from __future__ import annotations

import argparse
import csv
import math

from snapshot_lib import bearing_from_delta, offset_latlon


def s_route_xy(height_m: float, step_m: float):
    """Local ENU points (east, north) of the S, centered on (0, 0).

    Lower semicircle: center (0, -r), traversed on the +x side from
    (0, -2r) up to (0, 0). Upper semicircle: center (0, +r), traversed on
    the -x side from (0, 0) up to (0, +2r). Total arclength = 2 * pi * r.
    """
    r = height_m / 4.0
    dphi = step_m / r
    pts = []
    phi = -math.pi / 2.0
    while phi < math.pi / 2.0:                     # lower arc, +x side
        pts.append((r * math.cos(phi), -r + r * math.sin(phi)))
        phi += dphi
    phi = -math.pi / 2.0
    while phi < math.pi / 2.0:                     # upper arc, -x side
        pts.append((-r * math.cos(phi), r + r * math.sin(phi)))
        phi += dphi
    pts.append((0.0, 2.0 * r))                     # exact endpoint
    return pts


def rotate_xy(pts, heading_deg: float):
    """Rotate the whole route clockwise so its long axis points at
    `heading_deg` (compass convention, 0 = north)."""
    th = math.radians(heading_deg)
    c, s = math.cos(th), math.sin(th)
    return [(e * c + n * s, -e * s + n * c) for e, n in pts]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--center-lat", type=float, default=52.024,
                    help="route centroid latitude (default: Steinhagen area)")
    ap.add_argument("--center-lon", type=float, default=8.528,
                    help="route centroid longitude")
    ap.add_argument("--height", type=float, default=400.0,
                    help="total N-S extent of the S in meters (radius = h/4)")
    ap.add_argument("--step", type=float, default=5.0,
                    help="point spacing along the arc in meters")
    ap.add_argument("--heading", type=float, default=0.0,
                    help="compass heading of the S's long axis (deg)")
    ap.add_argument("--out", default="route_s.csv", help="output CSV path")
    args = ap.parse_args()

    pts = rotate_xy(s_route_xy(args.height, args.step), args.heading)

    rows, s = [], 0.0
    for i, (e, n) in enumerate(pts):
        if i > 0:
            s += math.dist(pts[i - 1], pts[i])
        if i < len(pts) - 1:
            de, dn = pts[i + 1][0] - e, pts[i + 1][1] - n
            brg = bearing_from_delta(de, dn)
        else:
            brg = rows[-1][4] if rows else 0.0
        lat, lon = offset_latlon(args.center_lat, args.center_lon, e, n)
        rows.append((i, round(s, 2), f"{lat:.7f}", f"{lon:.7f}",
                     round(brg, 2)))

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "s_m", "lat", "lon", "bearing_deg"])
        w.writerows(rows)

    print(f"wrote {len(rows)} points, arclength {s:.1f} m "
          f"(radius {args.height / 4:.0f} m) -> {args.out}")


if __name__ == "__main__":
    main()
