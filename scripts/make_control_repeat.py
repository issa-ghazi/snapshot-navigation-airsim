#!/usr/bin/env python3
"""make_control_repeat.py -- synthetic repeat frames from the teach cache.

Two purposes, one script:

1. Evaluator self-test. Before the simulator has produced a single frame,
   the RIDF pipeline can be exercised end-to-end on frames whose answer is
   known: a repeat frame built from the very orthophoto the teach snapshot
   came from must localise to its own waypoint with the RIDF minimum at
   theta = 0 (or at -yaw_offset if one is injected). If it does not, the
   evaluator -- not the simulator -- is broken.

2. Ideal-domain control ("teach vs teach"). Evaluating a full control set
   per altitude gives the within-domain baseline: how much localisation
   ambiguity and RIDF flattening is inherent to the sensor footprint at that
   altitude, independent of any orthographic-vs-perspective domain gap. The
   sim results are then read against this baseline.

How a frame is built
--------------------
The teach cache holds one north-up, rotation-safe orthophoto square per
waypoint (req_px = ceil(crop_px * sqrt(2)) at the source GSD). A synthetic
"DownCamera" frame is that square rotated so that (bearing + yaw_offset)
points up -- exactly what a nadir camera on a drone yawed to that heading
sees -- and rescaled to the capture resolution the real camera will use
(default 906 px for a 640 px sensor crop). The rotation leaves black corners
in the 708 px frame, but the inscribed disc is valid under any rotation and
the subsequent rotate_and_crop(..., crop_px) never leaves that disc, so no
undefined pixel enters the footprint.

Usage
-----
    python make_control_repeat.py --teach-root data/teach --alt 50 \
        --cache-dir data/teach/_cache_dop --out data/control/alt_0050m \
        [--wps 0 1 2] [--yaw-offset 0] [--capture-px 906] [--crop-px 640]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from PIL import Image

from snapshot_lib import ROTATION_MARGIN, footprint_from_altitude

# Effective sensor FOV (deg) the sim camera must reproduce after cropping.
SENSOR_FOV_DEG = 90.0


def capture_fov_deg(sensor_fov_deg: float = SENSOR_FOV_DEG) -> float:
    """FOV of a square pinhole capture whose footprint is sqrt(2) times the
    sensor footprint: tan(FOV/2) = sqrt(2) * tan(sensor_fov/2)."""
    return 2.0 * math.degrees(
        math.atan(ROTATION_MARGIN * math.tan(math.radians(sensor_fov_deg)
                                             / 2.0)))


def _resampling(name):
    return getattr(getattr(Image, "Resampling", Image), name)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teach-root", required=True)
    ap.add_argument("--alt", type=float, required=True)
    ap.add_argument("--cache-dir", required=True,
                    help="teach WMS cache (default was OUT/_cache_dop)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wps", type=int, nargs="*",
                    help="waypoint indices to build (default: all)")
    ap.add_argument("--yaw-offset", type=float, default=0.0,
                    help="injected heading error in deg (clockwise +)")
    ap.add_argument("--crop-px", type=int, default=640,
                    help="sensor crop the evaluator will take (= 90 deg "
                         "footprint in capture pixels)")
    ap.add_argument("--capture-px", type=int,
                    help="capture edge in px; default ceil(crop_px*sqrt2)")
    args = ap.parse_args()

    cap_px = args.capture_px or int(math.ceil(args.crop_px * ROTATION_MARGIN))
    teach_root = Path(args.teach_root)
    man = teach_root / f"alt_{args.alt:04.0f}m" / "manifest.csv"
    cache = Path(args.cache_dir)
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)

    # one row per waypoint (any sensor variant carries the same geometry)
    seen, rows = set(), []
    with open(man, newline="") as f:
        for r in csv.DictReader(f):
            wp = int(r["wp"])
            if float(r.get("line_offset_m", 0.0)) != 0.0 or wp in seen:
                continue
            if args.wps is not None and wp not in args.wps:
                continue
            seen.add(wp)
            rows.append(r)
    rows.sort(key=lambda r: int(r["wp"]))
    if not rows:
        raise SystemExit("no waypoints selected")

    fov = capture_fov_deg()
    (out / "params.json").write_text(json.dumps({
        "kind": "synthetic control from teach cache (ideal domain)",
        "teach_manifest": str(man), "alt_m": args.alt,
        "yaw_offset_deg": args.yaw_offset, "capture_px": cap_px,
        "capture_fov_deg": round(fov, 3), "crop_px": args.crop_px,
        "sensor_fov_deg": SENSOR_FOV_DEG,
    }, indent=2))

    fields = ["wp", "s_m", "lat", "lon", "alt_m", "bearing_deg", "yaw_deg",
              "yaw_offset_deg", "capture_px", "capture_fov_deg", "crop_px",
              "footprint_m", "source", "file"]
    footprint_m = round(footprint_from_altitude(args.alt, SENSOR_FOV_DEG), 3)
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            wp = int(r["wp"])
            src_path = cache / f"{r['cache_key']}.png"
            if not src_path.exists():
                print(f"  skip wp {wp}: cache image missing ({src_path.name})")
                continue
            brg = float(r["bearing_deg"])
            yaw = (brg + args.yaw_offset) % 360.0
            img = Image.open(src_path).convert("RGB")
            # camera yawed to `yaw` -> image rotated CCW by yaw (PIL +angle)
            frame = img.rotate(yaw, resample=_resampling("BICUBIC"))
            frame = frame.resize((cap_px, cap_px), _resampling("BICUBIC"))
            rel = f"raw/w{wp:05d}.png"
            frame.save(out / rel)
            w.writerow({
                "wp": wp, "s_m": r["s_m"], "lat": r["lat"], "lon": r["lon"],
                "alt_m": r["alt_m"], "bearing_deg": r["bearing_deg"],
                "yaw_deg": round(yaw, 2), "yaw_offset_deg": args.yaw_offset,
                "capture_px": cap_px, "capture_fov_deg": round(fov, 3),
                "crop_px": args.crop_px, "footprint_m": footprint_m,
                "source": src_path.name, "file": rel})
    print(f"wrote {len(rows)} control frames ({cap_px}px, FOV {fov:.2f} deg, "
          f"yaw offset {args.yaw_offset:+.1f} deg) -> {out}")


if __name__ == "__main__":
    main()
