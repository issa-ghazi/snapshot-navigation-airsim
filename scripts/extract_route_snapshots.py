#!/usr/bin/env python3
"""extract_route_snapshots.py -- teach-phase snapshot libraries from real
orthographic aerial imagery (Geobasis NRW Digital Orthophotos, WMS NW DOP).

For every requested flight altitude the script builds one library
(alt_0060m/, alt_0100m/, ...) from the SAME route: it maps altitude + camera
FOV to a ground footprint, requests one WMS GetMap image per waypoint at a
CONSTANT source ground-sample-distance (default 0.20 m/px), sized to cover
the rotation-safe footprint square, rotates each crop to the route bearing,
and derives ALL requested sensor resolutions and gray depths offline via the
shared preprocessing in snapshot_lib.py (identical to the repeat phase).

Why NRW DOP instead of Google: as of 8 July 2025 the Google Maps Static API
no longer serves the satellite/hybrid map types to EEA billing accounts, so
satellite tiles are unavailable in Germany. Geobasis NRW DOP is the open
(dl-de/zero-2.0), free, orthographic, 10 cm TrueDOP alternative -- no key,
no quota, no request cap. We resample to a constant 0.20 m source GSD so that
even the largest footprint (600 m at 300 m altitude) stays within the WMS
5000 px GetMap limit while keeping the source detail identical across all
altitudes (the ONLY thing varying with altitude is the footprint itself).

Coordinates: the route is WGS84 (lat/lon); the WMS is queried in EPSG:25832
(UTM zone 32N). pyproj handles the transform.

Manifest: one CSV row per processed image with lat/lon, altitude, source GSD,
footprint, bearing, resolution, gray levels, mask size and cache provenance.

Examples
--------
Dry run (no network, prints geometry + request count):
    python extract_route_snapshots.py --route route_s.csv \
        --altitudes 30 40 50 60 80 100 140 200 300 \
        --resolutions 10 20 40 50 80 --gray-levels 2 10 100 \
        --spacing 5 --fov 90 --out data/teach --dry-run

Real run (no API key needed -- open data):
    python extract_route_snapshots.py --route route_s.csv \
        --altitudes 60 100 140 --resolutions 50 --gray-levels 10 \
        --spacing 5 --out data/teach
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from pyproj import Transformer

from snapshot_lib import (ROTATION_MARGIN, circular_mask,
                          footprint_from_altitude, offset_latlon,
                          preprocess_view, resample_route, rotate_and_crop)

# WMS NW DOP -- Geobasis NRW, open data (dl-de/zero-2.0), 10 cm TrueDOP.
# Verified via GetCapabilities 1.3.0: Layer WMS_NW_DOP, CRS EPSG:25832,
# MaxWidth/MaxHeight 5000, formats include image/png.
WMS_URL = "https://www.wms.nrw.de/geobasis/wms_nw_dop"
# nw_dop_rgb = "DOP Farbe", the visible-light RGB rendering. The group layer
# WMS_NW_DOP with STYLES="" returns a grayscale image in which vegetation is
# the brightest content -- i.e. the near-infrared band of the RGBI product.
# That is spectrally incompatible with the visible-light imagery the repeat
# phase renders, so the RGB sub-layer is requested explicitly.
WMS_LAYER = "nw_dop_rgb"
WMS_CRS = "EPSG:25832"          # UTM zone 32N, native for NRW
WMS_MAX_PX = 5000               # per-request pixel cap (both axes)

# WGS84 (lat/lon) <-> UTM32. always_xy=True => (lon, lat) / (easting, northing)
_TF_FWD = Transformer.from_crs("EPSG:4326", WMS_CRS, always_xy=True)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--route", required=True,
                    help="CSV with lat/lon columns (bearing optional; it is "
                         "recomputed after resampling anyway)")
    ap.add_argument("--altitudes", type=float, nargs="+", required=True,
                    help="flight altitudes AGL in meters, e.g. 60 100 140")
    ap.add_argument("--resolutions", type=int, nargs="+", default=[50],
                    help="sensor matrix sizes N (N x N), e.g. 10 20 40 50 80")
    ap.add_argument("--gray-levels", type=int, nargs="+", default=[10],
                    help="gray depths, e.g. 2 10 100 (2 = black/white)")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--spacing", type=float, default=5.0,
                     help="waypoint spacing in meters (default 5, as in the "
                          "paper's 5 m grid). Same waypoints for every "
                          "altitude -> recommended.")
    grp.add_argument("--overlap", type=float,
                     help="alternative: footprint overlap fraction (0..1). "
                          "NOTE: spacing then depends on altitude, so each "
                          "altitude gets a DIFFERENT waypoint set.")
    ap.add_argument("--fov", type=float, default=90.0,
                    help="camera FOV in degrees (AirSim default 90; use 22.6 "
                         "with --altitudes 250 for a literal Gaffin replica)")
    ap.add_argument("--src-gsd", type=float, default=0.20,
                    help="CONSTANT source ground sample distance in m/px for "
                         "every altitude (default 0.20). Lower = sharper but "
                         "larger requests; keep footprint*sqrt(2)/gsd below "
                         "the WMS 5000 px cap.")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--layer", default=WMS_LAYER,
                    help="WMS sub-layer (default nw_dop_rgb = visible-light "
                         "RGB, used for all reported experiments). "
                         "WMS_NW_DOP reproduces the near-infrared grayscale "
                         "comparison of the thesis (spectral representation).")
    ap.add_argument("--cache",
                    help="cache dir for raw WMS images "
                         "(default: OUT/_cache_dop)")
    ap.add_argument("--corridor", type=int, default=1,
                    help="odd number of parallel route lines (Gaffin widened "
                         "the training path by one scene per side -> 3)")
    ap.add_argument("--corridor-offset", type=float,
                    help="lateral line spacing in meters (required if "
                         "--corridor > 1; e.g. the footprint at your "
                         "reference altitude)")
    ap.add_argument("--save-fullres", action="store_true",
                    help="additionally store the full-resolution rotated "
                         "grayscale footprint crop per waypoint (useful for "
                         "figures; costs disk, not requests)")
    ap.add_argument("--rps", type=float, default=4.0,
                    help="max WMS requests per second (be a good open-data "
                         "citizen)")
    ap.add_argument("--force", action="store_true",
                    help="reprocess existing outputs")
    ap.add_argument("--dry-run", action="store_true",
                    help="print geometry and request counts, no network")
    args = ap.parse_args()

    if args.corridor % 2 == 0 or args.corridor < 1:
        ap.error("--corridor must be an odd number >= 1")
    if args.corridor > 1 and not args.corridor_offset:
        ap.error("--corridor > 1 requires --corridor-offset (meters)")
    if args.src_gsd <= 0:
        ap.error("--src-gsd must be positive")
    return args


# --------------------------------------------------------------------------
# Route handling
# --------------------------------------------------------------------------
def load_route_csv(path: str):
    """Read (lat, lon) pairs from a CSV; header names are matched loosely."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"empty route file: {path}")
    header = [c.strip().lower() for c in rows[0]]

    def col(*names):
        for n in names:
            if n in header:
                return header.index(n)
        return None

    i_lat, i_lon = col("lat", "latitude"), col("lon", "lng", "longitude")
    body = rows[1:] if i_lat is not None else rows
    if i_lat is None:                      # headerless: assume lat,lon first
        i_lat, i_lon = 0, 1
    pts = [(float(r[i_lat]), float(r[i_lon])) for r in body if r]
    if len(pts) < 2:
        raise ValueError("route needs at least 2 points")
    return pts


def corridor_lines(waypoints, n_lines: int, offset_m: float):
    """Yield (line_offset_m, waypoint) pairs. Offsets are perpendicular to
    the local bearing; line 0 is the centerline."""
    half = n_lines // 2
    for k in range(-half, half + 1):
        for wp in waypoints:
            if k == 0:
                yield 0.0, wp
            else:
                lat_dir = (wp["bearing_deg"] + 90.0) % 360.0
                d = k * offset_m
                de = d * math.sin(math.radians(lat_dir))
                dn = d * math.cos(math.radians(lat_dir))
                la, lo = offset_latlon(wp["lat"], wp["lon"], de, dn)
                yield float(d), {**wp, "lat": la, "lon": lo}


# --------------------------------------------------------------------------
# WMS geometry, fetching + caching
# --------------------------------------------------------------------------
def latlon_to_utm(lat: float, lon: float):
    """WGS84 -> EPSG:25832 easting/northing in meters."""
    e, n = _TF_FWD.transform(lon, lat)
    return e, n


def cache_key(lat: float, lon: float, src_gsd: float, crop_px: int) -> str:
    """Cache file name. The layer is part of the key: images fetched from a
    different sub-layer (e.g. the old grayscale/NIR default) must not be
    silently reused when the layer changes."""
    return f"{WMS_LAYER}_{lat:.7f}_{lon:.7f}_g{src_gsd:.3f}_p{crop_px:04d}"


def build_wms_url(lat: float, lon: float, crop_px: int,
                  src_gsd: float) -> str:
    """WMS 1.3.0 GetMap for a square footprint centered on (lat, lon).

    The BBox is a crop_px * src_gsd metre square in UTM32, requested at
    crop_px * crop_px pixels -> exactly src_gsd m/px. NOTE: in WMS 1.3.0 the
    axis order for EPSG:25832 is easting,northing (a projected CRS), so the
    BBOX order is minx,miny,maxx,maxy = E_min,N_min,E_max,N_max.
    """
    e, n = latlon_to_utm(lat, lon)
    half = (crop_px * src_gsd) / 2.0
    bbox = f"{e - half:.3f},{n - half:.3f},{e + half:.3f},{n + half:.3f}"
    q = urllib.parse.urlencode({
        "SERVICE": "WMS",
        "REQUEST": "GetMap",
        "VERSION": "1.3.0",
        "LAYERS": WMS_LAYER,
        "STYLES": "",
        "CRS": WMS_CRS,
        "BBOX": bbox,
        "WIDTH": crop_px,
        "HEIGHT": crop_px,
        "FORMAT": "image/png",
    })
    return f"{WMS_URL}?{q}"


def fetch_image(session, url: str, dest: Path, retries: int = 3):
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=60)
        except Exception:                            # network hiccup
            if attempt == retries - 1:
                raise
            time.sleep(2.0 ** attempt)
            continue
        ctype = r.headers.get("content-type", "")
        if r.status_code == 200 and ctype.startswith("image"):
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(r.content)
            # A truncated response still carries a valid PNG header, so the
            # corruption would only surface much later in rotate_and_crop.
            # Decode the whole image here and re-fetch if it is incomplete.
            try:
                with Image.open(tmp) as probe:
                    probe.load()
            except Exception as e:
                tmp.unlink(missing_ok=True)
                if attempt == retries - 1:
                    raise RuntimeError(
                        f"image from WMS is corrupt after {retries} "
                        f"attempts ({type(e).__name__}: {e}); url: {url}")
                time.sleep(2.0 ** attempt)
                continue
            tmp.rename(dest)                          # atomic-ish resume
            return
        # WMS returns errors as XML (a ServiceExceptionReport), not an image
        if r.status_code == 200 and "xml" in ctype:
            raise RuntimeError(
                f"WMS ServiceException (check layer/CRS/BBOX/size): "
                f"{r.text[:300]!r}")
        if attempt == retries - 1:
            raise RuntimeError(
                f"HTTP {r.status_code}, content-type {ctype!r}: "
                f"{r.text[:200]!r}")
        time.sleep(2.0 ** attempt)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    global WMS_LAYER
    args = parse_args()
    WMS_LAYER = args.layer      # the layer is part of the cache key
    out_root = Path(args.out)
    cache_dir = Path(args.cache) if args.cache else out_root / "_cache_dop"

    route_pts = load_route_csv(args.route)
    lat_ref = sum(p[0] for p in route_pts) / len(route_pts)

    # ---- per-altitude geometry -------------------------------------------
    plans = []
    for alt in sorted(args.altitudes):
        fp = footprint_from_altitude(alt, args.fov)
        # rotation-safe source square, at the constant source GSD
        crop_px = int(round(fp / args.src_gsd))
        req_px = int(math.ceil(crop_px * ROTATION_MARGIN))
        if req_px > WMS_MAX_PX:
            sys.exit(
                f"altitude {alt:.0f} m: rotation-safe request {req_px}px "
                f"exceeds WMS cap {WMS_MAX_PX}px at src-gsd {args.src_gsd} "
                f"m/px. Raise --src-gsd (e.g. "
                f"{fp * ROTATION_MARGIN / WMS_MAX_PX:.3f}) or lower the "
                f"altitude.")
        spacing = (args.spacing if args.overlap is None
                   else max(fp * (1.0 - args.overlap), 0.1))
        wps = resample_route(route_pts, spacing)
        samples = list(corridor_lines(wps, args.corridor,
                                      args.corridor_offset or 0.0))
        plans.append(dict(alt=alt, footprint=fp, crop_px=crop_px,
                          req_px=req_px, spacing=spacing, samples=samples))

    # ---- fetch plan (dedup: same position + same request px) -------------
    # Each altitude has its own footprint -> its own request size, so unlike
    # the Google zoom-sharing case there is little cross-altitude reuse; we
    # still dedup identical (lat, lon, req_px) requests within/against cache.
    wanted = {}
    for p in plans:
        for _, wp in p["samples"]:
            k = cache_key(wp["lat"], wp["lon"], args.src_gsd, p["req_px"])
            if k not in wanted:
                wanted[k] = (wp["lat"], wp["lon"], p["req_px"])
    cached = sum((cache_dir / f"{k}.png").exists() for k in wanted)
    new = len(wanted) - cached

    n_variants = len(args.resolutions) * len(args.gray_levels)

    # ---- report ----------------------------------------------------------
    print(f"route: {args.route} | ref lat {lat_ref:.5f} | FOV {args.fov} deg"
          f" | src-GSD {args.src_gsd} m/px | corridor {args.corridor}")
    hdr = (f"{'alt_m':>6} {'footpr_m':>9} {'req_px':>7} {'crop_px':>8} "
           f"{'sens.GSD@50':>12} {'wps':>5} {'imgs':>6}")
    print(hdr)
    for p in plans:
        n_img = len(p["samples"]) * n_variants
        print(f"{p['alt']:>6.0f} {p['footprint']:>9.1f} {p['req_px']:>7d} "
              f"{p['crop_px']:>8d} {p['footprint'] / 50:>12.2f} "
              f"{len(p['samples']):>5d} {n_img:>6d}")
    print(f"\nunique WMS images needed : {len(wanted)}")
    print(f"  already cached         : {cached}")
    print(f"  NEW requests this run  : {new}")
    print("  (Geobasis NRW DOP -- open data dl-de/zero-2.0, no key, no quota, "
          "no cost. Please keep --rps modest.)")

    if args.dry_run:
        print("\ndry run -- no network calls made.")
        return

    # ---- real run --------------------------------------------------------
    import requests
    session = requests.Session()
    session.headers.update({"User-Agent": "mav-thesis-teach/1.0 (research)"})
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    (out_root / "params.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "imagery": f"Geobasis NRW Digital Orthophotos via {WMS_URL} "
                   f"(layer {WMS_LAYER}, {WMS_CRS}, image/png)",
        "wms_layer": WMS_LAYER,
        "spectral_note": ("nw_dop_rgb = visible-light RGB. The service also "
                          "offers CIR/NIR renderings; the group layer with "
                          "STYLES='' returns near-infrared grayscale, which "
                          "does not match the repeat phase."),
        "license": "dl-de/zero-2.0 (Datenlizenz Deutschland Zero 2.0)",
        "attribution": "Geobasis NRW (dl-de/zero-2.0)",
        "source_gsd_mpp": args.src_gsd,
        "args": {k: v for k, v in vars(args).items()},
        "ref_lat": lat_ref,
        "preprocessing": "snapshot_lib.preprocess_view: grayscale -> "
                         "equalize -> BOX resize -> quantize -> circular "
                         "mask; teach crops rotated to bearing beforehand",
    }, indent=2))

    # fetch (rate-limited, resumable)
    todo = [(k, v) for k, v in wanted.items()
            if not (cache_dir / f"{k}.png").exists()]
    t_last = 0.0
    for i, (k, (la, lo, rpx)) in enumerate(todo, 1):
        wait = (1.0 / args.rps) - (time.monotonic() - t_last)
        if wait > 0:
            time.sleep(wait)
        t_last = time.monotonic()
        url = build_wms_url(la, lo, rpx, args.src_gsd)
        fetch_image(session, url, cache_dir / f"{k}.png")
        if i % 25 == 0 or i == len(todo):
            print(f"  fetched {i}/{len(todo)}")

    # process per altitude
    mask_counts = {n: int(circular_mask(n).sum()) for n in args.resolutions}
    for p in plans:
        alt_dir = out_root / f"alt_{p['alt']:04.0f}m"
        alt_dir.mkdir(parents=True, exist_ok=True)
        man_path = alt_dir / "manifest.csv"
        write_header = args.force or not man_path.exists()
        mode = "w" if write_header else "a"
        with open(man_path, mode, newline="") as mf:
            mw = csv.writer(mf)
            if write_header:
                mw.writerow(["id", "wp", "line_offset_m", "s_m", "lat",
                             "lon", "alt_m", "fov_deg", "footprint_m",
                             "src_gsd_mpp", "req_px", "crop_px", "sensor_px",
                             "gray_levels", "sensor_gsd_mpp", "bearing_deg",
                             "mask_px", "cache_key", "file"])
            for wp_i, (off, wp) in enumerate(p["samples"]):
                k = cache_key(wp["lat"], wp["lon"], args.src_gsd, p["req_px"])
                src = Image.open(cache_dir / f"{k}.png")
                patch = rotate_and_crop(src, wp["bearing_deg"], p["crop_px"])
                if args.save_fullres:
                    fr = alt_dir / "crop_full"
                    fr.mkdir(exist_ok=True)
                    fp_out = fr / f"w{wp_i:05d}.png"
                    if args.force or not fp_out.exists():
                        patch.convert("L").save(fp_out)
                for n in args.resolutions:
                    for lv in args.gray_levels:
                        sub = alt_dir / f"res{n:03d}_g{lv:03d}"
                        sub.mkdir(exist_ok=True)
                        img_out = sub / f"w{wp_i:05d}.png"
                        if img_out.exists() and not args.force:
                            continue
                        arr = preprocess_view(patch, n, lv)
                        Image.fromarray(arr).save(img_out)
                        mw.writerow([
                            f"a{p['alt']:04.0f}_w{wp_i:05d}_r{n:03d}"
                            f"_g{lv:03d}",
                            wp_i, round(off, 2), round(wp["s_m"], 2),
                            f"{wp['lat']:.7f}", f"{wp['lon']:.7f}",
                            p["alt"], args.fov,
                            round(p["footprint"], 2), args.src_gsd,
                            p["req_px"], p["crop_px"], n, lv,
                            round(p["footprint"] / n, 4),
                            round(wp["bearing_deg"], 2), mask_counts[n], k,
                            str(img_out.relative_to(out_root))])
        print(f"  {alt_dir.name}: {len(p['samples'])} waypoints x "
              f"{n_variants} variants done")

    print("finished.")


if __name__ == "__main__":
    main()