#!/usr/bin/env python3
r"""selftest_recap.py -- exercise recapitulate_route.py without WMS or sim.

Builds a synthetic north-up "orthophoto" (random blocks + roads, seeded),
derives a teach library along the S-route from it with the exact teach
geometry (rotation-safe square at 0.20 m/px, rotate_and_crop, preprocess_view),
and lets the arc-scan agent navigate over the same image through a world
object that mimics WmsWorld (north-up square -> rotate to heading -> 906 px).

Optional --gap adds a blur, a brightness change and a fixed registration
offset to the agent's views, a crude stand-in for the domain gap, to see
the agent degrade instead of succeed.

    python scripts/selftest_recap.py --out selftest_recap [--gap] [--workers 4]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from snapshot_lib import (ROTATION_MARGIN, circular_mask,
                          footprint_from_altitude, preprocess_view,
                          resample_route, rotate_and_crop)
from capture_repeat import geodetic_to_enu, load_route_csv
import recapitulate_route as rr

GSD = 0.20
CENTER = (52.024, 8.528)
ALT = 50.0
SIZE_M = 800.0


def make_world_image(seed: int, size_px: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = np.full((size_px, size_px, 3), 110, np.uint8)
    # blocks of varying grey / colour (buildings, gardens)
    for _ in range(900):
        w, h = rng.integers(40, 260, 2)
        x, y = rng.integers(0, size_px - w), rng.integers(0, size_px - h)
        col = rng.integers(40, 230, 3)
        img[y:y + h, x:x + w] = col
    # roads (bright lines)
    for _ in range(25):
        if rng.random() < 0.5:
            y = rng.integers(0, size_px - 30)
            img[y:y + rng.integers(15, 40), :] = 200
        else:
            x = rng.integers(0, size_px - 30)
            img[:, x:x + rng.integers(15, 40)] = 205
    noise = rng.normal(0, 6, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img, "RGB")


class SynthOrtho:
    """North-up image centred on CENTER, GSD m/px, grey padding outside."""

    def __init__(self, img: Image.Image):
        self.img = img
        self.w = img.size[0]

    def square(self, lat, lon, px: int) -> Image.Image:
        e, n, _ = geodetic_to_enu(CENTER[0], CENTER[1], 0.0, lat, lon, 0.0)
        cx = self.w / 2.0 + e / GSD
        cy = self.w / 2.0 - n / GSD
        left, top = int(round(cx - px / 2.0)), int(round(cy - px / 2.0))
        out = Image.new("RGB", (px, px), (120, 120, 120))
        src = self.img.crop((left, top, left + px, top + px))
        out.paste(src, (0, 0))
        return out


class SynthWorld:
    """Same construction as recapitulate_route.WmsWorld, image instead of WMS."""

    def __init__(self, args):
        self.a = args
        self.ortho = SynthWorld.ortho
        self.gap = SynthWorld.gap
        self.cap_px = int(math.ceil(args.crop_px * ROTATION_MARGIN))
        self.ground_z = float("nan")
        self.n_fetched = 0

    def close(self):
        pass

    def capture(self, lat, lon, alt_m, heading_deg):
        fp = footprint_from_altitude(alt_m, 90.0)
        crop_src = int(round(fp / GSD))
        req_px = int(math.ceil(crop_src * ROTATION_MARGIN))
        if self.gap:                                  # registration offset 3 m
            lat, lon = rr.move(lat, lon, 30.0, 3.0)
        img = self.ortho.square(lat, lon, req_px)
        rs = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
        frame = img.rotate(heading_deg, resample=rs).resize(
            (self.cap_px, self.cap_px), rs)
        if self.gap:
            frame = frame.filter(ImageFilter.GaussianBlur(6))
            arr = np.asarray(frame).astype(np.float32) * 0.8 + 30
            frame = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        self.n_fetched += 1
        return np.asarray(frame), dict(yaw_deg=round(heading_deg % 360.0, 2),
                                       settled=1)


def build_teach(ortho: SynthOrtho, route_csv: Path, teach_root: Path,
                sensor=50, gray=10):
    wps = resample_route(load_route_csv(str(route_csv)), 5.0)
    fp = footprint_from_altitude(ALT, 90.0)
    crop_px = int(round(fp / GSD))
    req_px = int(math.ceil(crop_px * ROTATION_MARGIN))
    alt_dir = teach_root / f"alt_{ALT:04.0f}m"
    var = alt_dir / f"res{sensor:03d}_g{gray:03d}"
    var.mkdir(parents=True, exist_ok=True)
    mask_px = int(circular_mask(sensor).sum())
    fields = ["wp", "s_m", "lat", "lon", "alt_m", "bearing_deg", "footprint_m",
              "src_gsd_mpp", "req_px", "crop_px", "sensor_px", "gray_levels",
              "mask_px", "cache_key", "line_offset_m", "file"]
    with open(alt_dir / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, wp in enumerate(wps):
            src = ortho.square(wp["lat"], wp["lon"], req_px)
            patch = rotate_and_crop(src, wp["bearing_deg"], crop_px)
            arr = preprocess_view(patch, sensor, gray)
            rel = f"alt_{ALT:04.0f}m/res{sensor:03d}_g{gray:03d}/w{i:05d}.png"
            Image.fromarray(arr, "L").save(teach_root / rel)
            w.writerow(dict(wp=i, s_m=round(wp["s_m"], 2), lat=f"{wp['lat']:.7f}",
                            lon=f"{wp['lon']:.7f}", alt_m=ALT,
                            bearing_deg=round(wp["bearing_deg"], 2),
                            footprint_m=round(fp, 3), src_gsd_mpp=GSD,
                            req_px=req_px, crop_px=crop_px, sensor_px=sensor,
                            gray_levels=gray, mask_px=mask_px,
                            cache_key="synthetic", line_offset_m=0.0, file=rel))
    return len(wps)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="selftest_recap")
    ap.add_argument("--gap", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--start-offset-m", type=float, default=0.0)
    ap.add_argument("--extra", default="",
                    help='extra CLI args for recapitulate_route, one string, '
                         'e.g. --extra "--ridf-step 2 --max-steps 20"')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    route_csv = out / "route_s.csv"
    subprocess.run([sys.executable, str(Path(__file__).with_name("make_s_route.py")),
                    "--center-lat",
                    str(CENTER[0]), "--center-lon", str(CENTER[1]), "--out",
                    str(route_csv)], check=True)
    img = make_world_image(a.seed, int(SIZE_M / GSD))
    img.save(out / "world.png")
    ortho = SynthOrtho(img)
    teach_root = out / "teach"
    n = build_teach(ortho, route_csv, teach_root)
    print(f"synthetic teach library: {n} scenes -> {teach_root}")

    SynthWorld.ortho = ortho
    SynthWorld.gap = a.gap
    rr.WmsWorld = SynthWorld                       # swap the world backend
    run_dir = out / ("run_gap" if a.gap else "run_clean")
    sys.argv = ["recapitulate_route.py", "--world", "wms", "--route",
                str(route_csv), "--alt", str(ALT), "--teach-root",
                str(teach_root), "--cache-dir", str(out / "_cache"),
                "--workers", str(a.workers), "--out", str(run_dir),
                "--start-offset-m", str(a.start_offset_m)] + shlex.split(a.extra)
    rr.run(rr.parse_args())
    print(json.dumps(json.loads((run_dir / "summary.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
