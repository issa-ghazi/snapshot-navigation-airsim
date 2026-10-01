#!/usr/bin/env python3
r"""recapitulate_route.py -- stage 2: closed-loop route recapitulation.

Implements the arc-scan agent of Gaffin et al. (2015), PLOS ONE 10(4):
e0122077, Fig. 3 / "Recapitulating the training path". The agent is placed
near the start of the taught route and has to retrace it using nothing but
the RIDF match of live views against the stored teach library:

    1. Initial heading: one capture at the start position, RIDF against all
       stored scenes; the rotation of the best match tells the agent which
       way the route points ("first scan is based on the direction of the
       most familiar scene at that point").
    2. Arc scan: candidate positions on an arc of radius --step-m ahead of
       the agent, sampled straight ahead first, then alternately right and
       left at increasing angles (--arc-n points, +-(--arc-span-deg)).
    3. At every candidate the view is rotated through 360 deg and compared
       to ALL stored scenes (same RIDF core as stage 1, evaluate_ridf.py).
       Familiarity = d_best / mean(D), Gaffin's ratio.
    4. The agent moves to the most familiar candidate. Gaffin stops the
       scan early once the ratio drops below 0.20; across our orthophoto ->
       rendered-perspective domain gap the ratio at a CORRECT match is ~0.6
       (stage-1 finding), so the arc is scanned completely by default.
       --stop-ratio re-enables the early stop.
    5. Next scan direction = extension of the segment previous position ->
       chosen point (--heading-mode extend, Gaffin), or the visual-compass
       estimate heading + theta* of the chosen candidate (--heading-mode
       compass).
    6. Termination: within --goal-m of the route end (success), more than
       --lost-m off the route (lost), or --max-steps exceeded.

The agent never sees the route; the geometric truth (nearest route point,
cross-track distance, localisation error of best_wp) is only logged for the
evaluation, exactly like `true_wp` in stage 1.

Two "worlds", one agent
-----------------------
--world sim   Project AirSim, non-physics teleport, identical capture chain
              to capture_repeat.py (rotation-safe 906 px DownCamera frame,
              depth-based ground measurement, settle loop). This is the
              domain-gap experiment.
--world wms   A fresh NRW DOP square is fetched at every candidate position
              (same request geometry and cache format as the teach phase),
              rotated to the agent heading and resampled to the capture
              size, exactly as make_control_repeat.py builds control frames.
              This is Gaffin's original setting (teach and repeat from the
              same orthophoto) and serves as the closed-loop control: it
              exercises the whole agent without a simulator and gives the
              no-domain-gap baseline for every altitude.

Arc geometry
------------
Fixed in metres for all altitudes (like the 5 m waypoint raster): radius
--step-m 10, 7 candidates at 15 deg spacing (+-45 deg). Neighbouring
candidates are then 2.6 m apart, about one sensor pixel at 45-60 m; a finer
arc would be sub-pixel. Scaling the arc with the footprint would confound
the altitude axis with the step size. Both parameters stay adjustable.

Outputs (one folder per run)
----------------------------
params.json       every parameter, world settings, library, timing
candidates.csv    one row per capture (step, candidate, position, RIDF
                  result, geometric truth, settle/AGL info, file)
steps.csv         one row per step (state before the scan, chosen candidate,
                  heading update, cross-track, progress)
summary.json      outcome and the run-level metrics (see aggregate_recap.py)
raw/              captured frames (--save-frames chosen|all|none)
ridf/             D[theta, scene] of the chosen candidates (--save-ridf)

Usage
-----
Simulator (Innenstadt, 100 m, heading rule of the thesis grid):
    python scripts/recapitulate_route.py --world sim --route routes/route_s.csv ^
        --alt 100 --teach-root data/teach_innenstadt --heading-mode compass ^
        --origin-lat 52.040111 --origin-lon 8.495097 --origin-h 800 ^
        --sim-config sim_config --ground-z0 640 --workers 12 ^
        --out data/recap/innenstadt_alt0100m_compass

Control without simulator (same route, 50 m):
    python scripts/recapitulate_route.py --world wms --route routes/route_s.csv ^
        --alt 50 --teach-root data/teach_innenstadt --heading-mode compass ^
        --cache-dir data/recap_ctrl/_cache_dop --workers 12 ^
        --out data/recap_ctrl/innenstadt_alt0050m_compass

Note: --heading-mode defaults to `extend` (Gaffin); all systematic runs of
the thesis used `compass`.

--dry-run prints the arc geometry and the expected number of captures and
checks the teach library without touching any world. --resume continues an
interrupted run from the last completed step in steps.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from snapshot_lib import (ROTATION_MARGIN, circular_mask,
                          footprint_from_altitude, offset_latlon,
                          preprocess_view, resample_route, rotate_and_crop)
from capture_repeat import (SENSOR_FOV_DEG, SimLink, bearing_to_ned_yaw_deg,
                            capture_fov_deg, geo_to_ned_xy, geodetic_to_enu,
                            ground_from_depth, load_route_csv, settle_and_grab)
from evaluate_ridf import TeachLibrary, load_teach_library, wrap180

CAND_FIELDS = [
    "step", "cand", "order", "angle_off_deg", "lat", "lon", "e_m", "n_m",
    "alt_m", "heading_deg", "yaw_deg",
    "best_wp", "best_theta_deg", "d_best", "d_mean_all", "gaffin_ratio",
    "compass_bearing_deg",
    "s_nearest_m", "xtrack_m", "nearest_wp", "loc_err_m", "loc_err_wp",
    "route_bearing_deg", "heading_err_deg", "chosen",
    "agl_measured_m", "ground_z_ned", "ned_x", "ned_y", "ned_z",
    "gt_x", "gt_y", "gt_z", "gt_yaw_deg",
    "settle_s", "settle_iters", "settle_diff", "depth_finite", "settled",
    "t_capture_s", "t_ridf_s", "file"]

STEP_FIELDS = [
    "step", "lat", "lon", "e_m", "n_m", "heading_deg", "n_cands",
    "chosen_cand", "chosen_angle_off_deg", "best_wp", "gaffin_ratio",
    "compass_bearing_deg", "s_nearest_m", "xtrack_m", "loc_err_m",
    "dist_goal_m", "step_len_m", "next_heading_deg", "ground_z_ned",
    "t_step_s", "outcome"]


# --------------------------------------------------------------------------
# Route geometry (ENU metres relative to the route start)
# --------------------------------------------------------------------------
def bearing_of(de: float, dn: float) -> float:
    return math.degrees(math.atan2(de, dn)) % 360.0


class Route:
    """Waypoints of the taught route in a local ENU frame (origin = first
    route point) with nearest-point projection for the evaluation."""

    def __init__(self, wps: list[dict]):
        self.wps = wps
        self.lat0, self.lon0 = wps[0]["lat"], wps[0]["lon"]
        self.xy = np.array([self.enu(w["lat"], w["lon"]) for w in wps])
        self.s = np.array([w["s_m"] for w in wps])
        self.bearing = np.array([w["bearing_deg"] for w in wps])
        self.length = float(self.s[-1])

    def enu(self, lat: float, lon: float):
        e, n, _ = geodetic_to_enu(self.lat0, self.lon0, 0.0, lat, lon, 0.0)
        return e, n

    def project(self, e: float, n: float) -> dict:
        """Nearest point on the polyline -> arclength, cross-track distance,
        nearest waypoint index and route bearing there."""
        p = np.array([e, n])
        a, b = self.xy[:-1], self.xy[1:]
        ab = b - a
        ab2 = (ab ** 2).sum(axis=1)
        t = np.clip(((p - a) * ab).sum(axis=1) / np.maximum(ab2, 1e-9), 0, 1)
        q = a + t[:, None] * ab
        d = np.linalg.norm(q - p, axis=1)
        k = int(np.argmin(d))
        s = float(self.s[k] + t[k] * (self.s[k + 1] - self.s[k]))
        wp = int(k if t[k] < 0.5 else k + 1)
        return dict(s_nearest_m=round(s, 2), xtrack_m=round(float(d[k]), 2),
                    nearest_wp=wp,
                    route_bearing_deg=round(float(self.bearing[k]), 2))

    def dist_to_end(self, e: float, n: float) -> float:
        return float(math.hypot(e - self.xy[-1, 0], n - self.xy[-1, 1]))


# --------------------------------------------------------------------------
# Arc scan
# --------------------------------------------------------------------------
def arc_offsets(n: int, span_deg: float) -> list[float]:
    """Angular offsets (deg, clockwise positive) in Gaffin's sampling order:
    straight ahead, then alternately right and left at increasing angles.
    n must be odd so the arc is symmetric around the heading."""
    if n < 1 or n % 2 == 0:
        raise ValueError("--arc-n must be odd (centre + pairs)")
    if n == 1:
        return [0.0]
    d = span_deg / ((n - 1) / 2)
    out = [0.0]
    for k in range(1, (n - 1) // 2 + 1):
        out += [k * d, -k * d]
    return out


def move(lat: float, lon: float, bearing_deg: float, dist_m: float):
    b = math.radians(bearing_deg)
    return offset_latlon(lat, lon, dist_m * math.sin(b), dist_m * math.cos(b))


# --------------------------------------------------------------------------
# Worlds: where a view at (lat, lon, alt, heading) comes from
# --------------------------------------------------------------------------
class SimWorld:
    """Project AirSim via SimLink -- the capture chain of capture_repeat.py:
    place at (ground estimate - AGL), settle, measure ground from the planar
    depth (p95), correct the height, capture the 906 px RGB frame."""

    def __init__(self, args):
        self.a = args
        self.sim = SimLink(args.scene, args.sim_config, args.robot)
        self.ground_z = args.ground_z0
        self.cap_px = int(math.ceil(args.crop_px * ROTATION_MARGIN))

    def close(self):
        self.sim.close()

    def capture(self, lat, lon, alt_m, heading_deg):
        a = self.a
        x, y = geo_to_ned_xy(a.origin_lat, a.origin_lon, a.origin_h, lat, lon,
                             a.ned_x_bearing)
        yaw = bearing_to_ned_yaw_deg(heading_deg, a.ned_x_bearing)
        z = self.ground_z - alt_m
        agl = float("nan")
        for _ in range(a.agl_iters):
            self.sim.set_pose(x, y, z, yaw)
            rgb, depth, info = settle_and_grab(
                self.sim, a.settle_min, a.settle_dt, a.settle_max,
                a.settle_thresh, a.depth_min_finite)
            agl = ground_from_depth(depth, a.ground_pct)
            if not math.isfinite(agl):
                print("    WARNING: no finite depth -- keeping nominal z")
                break
            self.ground_z = z + agl
            if abs(agl - alt_m) <= a.agl_tol:
                break
            z = self.ground_z - alt_m
        gt = self.sim.ground_truth()
        meta = dict(agl_measured_m=round(agl, 2) if math.isfinite(agl) else "",
                    ground_z_ned=round(self.ground_z, 3), ned_x=round(x, 3),
                    ned_y=round(y, 3), ned_z=round(z, 3), yaw_deg=round(yaw, 2),
                    **gt, **{k: info[k] for k in ("settle_s", "settle_iters",
                                                 "settle_diff", "depth_finite",
                                                 "settled")})
        return rgb, meta


class WmsWorld:
    """NRW DOP as the world (no domain gap): one north-up rotation-safe
    square per candidate at the teach source GSD, rotated to the heading
    and resampled to the capture size -- make_control_repeat.py for
    arbitrary positions. Fetch + cache come from extract_route_snapshots."""

    def __init__(self, args):
        import requests
        from extract_route_snapshots import (build_wms_url, cache_key,
                                             fetch_image)
        self._url, self._key, self._fetch = build_wms_url, cache_key, fetch_image
        self.a = args
        self.session = requests.Session()
        self.cache = Path(args.cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.cap_px = int(math.ceil(args.crop_px * ROTATION_MARGIN))
        self.ground_z = float("nan")
        self._last = 0.0
        self.n_fetched = 0

    def close(self):
        self.session.close()

    def capture(self, lat, lon, alt_m, heading_deg):
        a = self.a
        fp = footprint_from_altitude(alt_m, SENSOR_FOV_DEG)
        crop_src = int(round(fp / a.src_gsd))
        req_px = int(math.ceil(crop_src * ROTATION_MARGIN))
        dest = self.cache / f"{self._key(lat, lon, a.src_gsd, req_px)}.png"
        if not dest.exists():
            wait = self._last + 1.0 / a.rps - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._fetch(self.session, self._url(lat, lon, req_px, a.src_gsd),
                        dest)
            self._last = time.monotonic()
            self.n_fetched += 1
        rs = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
        img = Image.open(dest).convert("RGB")
        # camera yawed to heading -> image rotated CCW by heading (PIL +)
        frame = img.rotate(heading_deg, resample=rs)
        frame = frame.resize((self.cap_px, self.cap_px), rs)
        rgb = np.asarray(frame)
        return rgb, dict(yaw_deg=round(heading_deg % 360.0, 2),
                         settled=1, source=dest.name)


# --------------------------------------------------------------------------
# RIDF of one frame, parallel over rotation angles
# --------------------------------------------------------------------------
_W: dict = {}


def _init_worker(lib_flat, mask, sensor, gray, crop_px):
    _W.update(flat32=lib_flat.astype(np.int32), mask=mask, sensor=sensor,
              gray=gray, crop_px=crop_px)


def _ridf_chunk(task):
    """(rgb HxWx3 uint8, thetas) -> D chunk (len(thetas), N) int64.
    Rotation order = teach order: rotate the RGB frame, then grayscale
    (see evaluate_ridf.py, 'Rotation order')."""
    rgb, thetas = task
    frame = Image.fromarray(rgb, "RGB")
    mask, t32 = _W["mask"], _W["flat32"]
    D = np.empty((len(thetas), t32.shape[0]), dtype=np.int64)
    for k, th in enumerate(thetas):
        patch = rotate_and_crop(frame, float(th), _W["crop_px"])
        v = preprocess_view(patch, _W["sensor"], _W["gray"])[mask]
        D[k] = np.abs(v.astype(np.int32)[None, :] - t32).sum(axis=1)
    return D


class RidfEngine:
    def __init__(self, lib: TeachLibrary, crop_px: int, step_deg: float,
                 workers: int):
        self.lib = lib
        self.thetas = np.arange(0.0, 360.0, step_deg)
        self.workers = max(1, workers)
        init = (lib.flat, lib.mask, lib.sensor_px, lib.gray_levels, crop_px)
        if self.workers > 1:
            import multiprocessing as mp
            self.pool = mp.Pool(self.workers, initializer=_init_worker,
                                initargs=init)
            self.chunks = np.array_split(self.thetas, self.workers * 2)
        else:
            self.pool = None
            _init_worker(*init)

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()

    def evaluate(self, rgb: np.ndarray) -> np.ndarray:
        if self.pool is None:
            return _ridf_chunk((rgb, self.thetas))
        parts = self.pool.map(_ridf_chunk, [(rgb, c) for c in self.chunks])
        return np.vstack(parts)

    def summarize(self, D: np.ndarray, heading_deg: float) -> dict:
        k, j = np.unravel_index(int(np.argmin(D)), D.shape)
        theta = wrap180(float(self.thetas[k]))
        d_mean = float(D.mean())
        return dict(best_wp=int(self.lib.wp[j]),
                    best_theta_deg=round(theta, 2),
                    d_best=int(D[k, j]), d_mean_all=round(d_mean, 1),
                    gaffin_ratio=round(float(D[k, j]) / d_mean, 4),
                    # the stored scene has the route bearing "up"; the frame
                    # has the heading "up"; theta* aligns them -> the route
                    # bearing according to memory is heading + theta*.
                    compass_bearing_deg=round((heading_deg + theta) % 360.0, 2))


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------
class Recorder:
    def __init__(self, out: Path, resume: bool):
        self.out = out
        (out / "raw").mkdir(parents=True, exist_ok=True)
        mode = "a" if resume and (out / "steps.csv").exists() else "w"
        self.fc = open(out / "candidates.csv", mode, newline="")
        self.fs = open(out / "steps.csv", mode, newline="")
        self.wc = csv.DictWriter(self.fc, fieldnames=CAND_FIELDS)
        self.ws = csv.DictWriter(self.fs, fieldnames=STEP_FIELDS)
        if mode == "w":
            self.wc.writeheader()
            self.ws.writeheader()

    def cand(self, row):
        self.wc.writerow({k: row.get(k, "") for k in CAND_FIELDS})
        self.fc.flush()

    def step(self, row):
        self.ws.writerow({k: row.get(k, "") for k in STEP_FIELDS})
        self.fs.flush()

    def close(self):
        self.fc.close()
        self.fs.close()


def load_resume_state(out: Path):
    rows = list(csv.DictReader(open(out / "steps.csv", newline="")))
    if not rows:
        return None
    r = rows[-1]
    if r["outcome"] not in ("", "running"):
        sys.exit(f"run in {out} already finished ({r['outcome']})")
    st = dict(step=int(r["step"]) + 1, lat=float(r["lat"]), lon=float(r["lon"]),
              heading=float(r["next_heading_deg"]),
              ground_z=float(r["ground_z_ned"]) if r["ground_z_ned"] else None,
              n_captures=sum(1 for _ in csv.DictReader(
                  open(out / "candidates.csv", newline=""))))
    # the row holds the state BEFORE the step; advance to the chosen point
    ang = float(r["chosen_angle_off_deg"])
    st["lat"], st["lon"] = move(st["lat"], st["lon"],
                                (float(r["heading_deg"]) + ang) % 360.0,
                                float(r["step_len_m"]))
    return st


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    route_pts = load_route_csv(args.route)
    wps = resample_route(route_pts, args.spacing)
    route = Route(wps)
    offsets = arc_offsets(args.arc_n, args.arc_span_deg)
    fp = footprint_from_altitude(args.alt, SENSOR_FOV_DEG)
    max_steps = args.max_steps or int(math.ceil(2.0 * route.length / args.step_m))

    print(f"route: {len(wps)} waypoints, {route.length:.1f} m; alt {args.alt:.0f} m "
          f"(footprint {fp:.0f} m, sensor GSD {fp / args.sensor:.2f} m)")
    print(f"arc: radius {args.step_m} m, {len(offsets)} candidates at "
          f"{', '.join(f'{o:+.0f}' for o in offsets)} deg; "
          f"expected ~{route.length / args.step_m:.0f} steps, "
          f"~{route.length / args.step_m * len(offsets):.0f} captures "
          f"(max {max_steps} steps)")

    lib = load_teach_library(Path(args.teach_root), args.alt, args.sensor,
                             args.gray)
    print(f"teach library: {lib.n_scenes} scenes, {lib.sensor_px}x"
          f"{lib.sensor_px}/{lib.gray_levels} gray, {lib.flat.shape[1]} px")

    params = {
        "kind": "stage-2 closed-loop recapitulation (Gaffin arc-scan agent)",
        "world": args.world, "route": str(args.route), "spacing_m": args.spacing,
        "alt_m": args.alt, "footprint_m": round(fp, 3), "sensor_px": args.sensor,
        "gray_levels": args.gray, "ridf_step_deg": args.ridf_step,
        "teach_root": str(args.teach_root), "n_scenes": int(lib.n_scenes),
        "arc": {"step_m": args.step_m, "n": args.arc_n,
                "span_deg": args.arc_span_deg, "offsets_deg": offsets},
        "stop_ratio": args.stop_ratio, "heading_mode": args.heading_mode,
        "start": {"wp": args.start_wp, "offset_m": args.start_offset_m,
                  "heading_deg": args.start_heading},
        "goal_m": args.goal_m, "lost_m": args.lost_m, "max_steps": max_steps,
        "crop_px": args.crop_px, "capture_px": int(math.ceil(args.crop_px * ROTATION_MARGIN)),
        "capture_fov_deg": round(capture_fov_deg(), 4),
        "save_frames": args.save_frames, "note": args.note,
        "route_origin": {"lat": route.lat0, "lon": route.lon0,
                         "frame": "ENU metres at route point 0 (exact ECEF)"},
    }
    if args.world == "sim":
        params["sim"] = {
            "origin_lat": args.origin_lat, "origin_lon": args.origin_lon,
            "origin_h_m": args.origin_h, "ned_x_bearing_deg": args.ned_x_bearing,
            "ground_z0": args.ground_z0, "ground_pct": args.ground_pct,
            "agl_tol_m": args.agl_tol, "agl_iters": args.agl_iters,
            "settle": {"min_s": args.settle_min, "dt_s": args.settle_dt,
                       "max_s": args.settle_max, "thresh": args.settle_thresh,
                       "depth_min_finite": args.depth_min_finite},
            "scene": args.scene, "robot": args.robot}
    else:
        params["wms"] = {"cache_dir": str(args.cache_dir),
                         "src_gsd_mpp": args.src_gsd, "rps": args.rps}
    (out / "params.json").write_text(json.dumps(params, indent=2))

    if args.dry_run:
        print("dry run: arc geometry and library ok, nothing captured")
        return

    # ---- start state ------------------------------------------------------
    st = load_resume_state(out) if args.resume and (out / "steps.csv").exists() else None
    if st is None:
        w0 = wps[args.start_wp]
        lat, lon = w0["lat"], w0["lon"]
        if args.start_offset_m:
            lat, lon = move(lat, lon, (w0["bearing_deg"] + 90.0) % 360.0,
                            args.start_offset_m)
        heading = (args.start_heading if args.start_heading is not None
                   else w0["bearing_deg"])
        step0, n_captures = 0, 0
        print(f"start: wp {args.start_wp} (offset {args.start_offset_m} m), "
              f"initial heading {heading:.1f} deg")
    else:
        lat, lon, heading = st["lat"], st["lon"], st["heading"]
        step0, n_captures = st["step"], st["n_captures"]
        print(f"resume at step {step0} (heading {heading:.1f} deg, "
              f"{n_captures} captures so far)")

    world = SimWorld(args) if args.world == "sim" else WmsWorld(args)
    if st is not None and st.get("ground_z") is not None and args.world == "sim":
        world.ground_z = st["ground_z"]
    engine = RidfEngine(lib, args.crop_px, args.ridf_step, args.workers)
    rec = Recorder(out, args.resume)
    if args.save_ridf:
        (out / "ridf").mkdir(exist_ok=True)
    t_run = time.monotonic()
    outcome = "running"

    def capture_and_match(step, cand, order, ang, la, lo, hdg):
        nonlocal n_captures
        t0 = time.monotonic()
        rgb, meta = world.capture(la, lo, args.alt, hdg)
        t1 = time.monotonic()
        D = engine.evaluate(rgb)
        m = engine.summarize(D, hdg)
        t2 = time.monotonic()
        n_captures += 1
        e, n = route.enu(la, lo)
        tr = route.project(e, n)
        j = lib.index_of_wp(m["best_wp"])
        tag = f"s{step:04d}_c{cand:02d}"
        row = dict(step=step, cand=cand, order=order, angle_off_deg=round(ang, 2),
                   lat=f"{la:.7f}", lon=f"{lo:.7f}", e_m=round(e, 2),
                   n_m=round(n, 2), alt_m=args.alt, heading_deg=round(hdg % 360.0, 2),
                   **m, **tr,
                   loc_err_m=round(abs(float(lib.s_m[j]) - tr["s_nearest_m"]), 2),
                   loc_err_wp=int(m["best_wp"] - tr["nearest_wp"]),
                   heading_err_deg=round(wrap180(hdg - tr["route_bearing_deg"]), 2),
                   chosen=0, t_capture_s=round(t1 - t0, 2),
                   t_ridf_s=round(t2 - t1, 2), file="", **meta)
        return row, rgb, D, tag

    def store(row, rgb, D, tag, chosen):
        row["chosen"] = int(chosen)
        if args.save_frames == "all" or (args.save_frames == "chosen" and chosen):
            rel = f"raw/{tag}.png"
            Image.fromarray(np.ascontiguousarray(rgb), "RGB").save(out / rel)
            row["file"] = rel
        if args.save_ridf and chosen:
            np.save(out / "ridf" / f"ridf_{tag}.npy", D)
        rec.cand(row)

    try:
        # ---- step 0: initial heading from the memory ----------------------
        if st is None:
            row, rgb, D, tag = capture_and_match(0, 0, 0, 0.0, lat, lon, heading)
            store(row, rgb, D, tag, True)
            e, n = row["e_m"], row["n_m"]
            new_heading = row["compass_bearing_deg"]
            print(f"  init: best wp {row['best_wp']} theta {row['best_theta_deg']:+.0f} "
                  f"ratio {row['gaffin_ratio']:.3f} -> heading {new_heading:.1f} deg "
                  f"(route {row['route_bearing_deg']:.1f})")
            rec.step(dict(step=0, lat=f"{lat:.7f}", lon=f"{lon:.7f}", e_m=e, n_m=n,
                          heading_deg=round(heading, 2), n_cands=1, chosen_cand=0,
                          chosen_angle_off_deg=0.0, best_wp=row["best_wp"],
                          gaffin_ratio=row["gaffin_ratio"],
                          compass_bearing_deg=row["compass_bearing_deg"],
                          s_nearest_m=row["s_nearest_m"], xtrack_m=row["xtrack_m"],
                          loc_err_m=row["loc_err_m"],
                          dist_goal_m=round(route.dist_to_end(e, n), 2),
                          step_len_m=0.0, next_heading_deg=new_heading,
                          ground_z_ned=row.get("ground_z_ned", ""),
                          t_step_s=round(row["t_capture_s"] + row["t_ridf_s"], 2),
                          outcome="running"))
            heading = new_heading
            step0 = 1

        # ---- arc-scan loop -------------------------------------------------
        for step in range(step0, max_steps + 1):
            t_step = time.monotonic()
            e, n = route.enu(lat, lon)
            cands = []
            for cand, ang in enumerate(offsets):
                la, lo = move(lat, lon, (heading + ang) % 360.0, args.step_m)
                row, rgb, D, tag = capture_and_match(step, cand, cand, ang,
                                                     la, lo, heading)
                cands.append((row, rgb, D, tag))
                if args.stop_ratio is not None and row["gaffin_ratio"] <= args.stop_ratio:
                    break
            best = min(range(len(cands)), key=lambda i: cands[i][0]["gaffin_ratio"])
            for i, c in enumerate(cands):
                store(*c, chosen=(i == best))
            brow = cands[best][0]
            ang = brow["angle_off_deg"]
            new_lat, new_lon = float(brow["lat"]), float(brow["lon"])
            move_bearing = (heading + ang) % 360.0
            if args.heading_mode == "compass":
                next_heading = brow["compass_bearing_deg"]
            else:
                next_heading = move_bearing              # Gaffin: extend the segment
            ne, nn = brow["e_m"], brow["n_m"]
            dist_goal = route.dist_to_end(ne, nn)
            if dist_goal <= args.goal_m:
                outcome = "goal"
            elif brow["xtrack_m"] > args.lost_m:
                outcome = "lost"
            elif step >= max_steps:
                outcome = "max_steps"
            rec.step(dict(step=step, lat=f"{lat:.7f}", lon=f"{lon:.7f}", e_m=round(e, 2),
                          n_m=round(n, 2), heading_deg=round(heading, 2),
                          n_cands=len(cands), chosen_cand=best,
                          chosen_angle_off_deg=ang, best_wp=brow["best_wp"],
                          gaffin_ratio=brow["gaffin_ratio"],
                          compass_bearing_deg=brow["compass_bearing_deg"],
                          s_nearest_m=brow["s_nearest_m"], xtrack_m=brow["xtrack_m"],
                          loc_err_m=brow["loc_err_m"], dist_goal_m=round(dist_goal, 2),
                          step_len_m=args.step_m, next_heading_deg=round(next_heading, 2),
                          ground_z_ned=brow.get("ground_z_ned", ""),
                          t_step_s=round(time.monotonic() - t_step, 1),
                          outcome=outcome))
            el = time.monotonic() - t_run
            print(f"  [{step:3d}] {len(cands)} cands  chose {ang:+4.0f} deg  "
                  f"wp {brow['best_wp']:3d} ratio {brow['gaffin_ratio']:.3f}  "
                  f"s {brow['s_nearest_m']:6.1f} m  xtrack {brow['xtrack_m']:5.1f} m  "
                  f"goal {dist_goal:6.1f} m  {time.monotonic() - t_step:5.1f} s "
                  f"({el / 60:.1f} min)")
            lat, lon, heading = new_lat, new_lon, next_heading
            if outcome != "running":
                break
    except KeyboardInterrupt:
        outcome = "interrupted"
        print("interrupted -- rerun with --resume to continue")
    finally:
        rec.close()
        engine.close()
        world.close()

    # ---- summary ----------------------------------------------------------
    steps = list(csv.DictReader(open(out / "steps.csv", newline="")))
    mv = [r for r in steps if int(r["step"]) > 0]
    xt = [float(r["xtrack_m"]) for r in mv]
    loc = [float(r["loc_err_m"]) for r in mv]
    s_vals = [float(r["s_nearest_m"]) for r in steps]
    chosen = [r for r in csv.DictReader(open(out / "candidates.csv", newline=""))
              if r["chosen"] == "1" and int(r["step"]) > 0]
    hd = [abs(float(r["heading_err_deg"])) for r in chosen]
    ratio = [float(r["gaffin_ratio"]) for r in chosen]
    summ = {
        "outcome": outcome, "n_steps": len(mv), "n_captures": n_captures,
        "path_len_m": round(sum(float(r["step_len_m"]) for r in mv), 1),
        "route_len_m": round(route.length, 1),
        "s_final_m": round(s_vals[-1], 1) if s_vals else "",
        "s_max_m": round(max(s_vals), 1) if s_vals else "",
        "progress_frac": round(max(s_vals) / route.length, 3) if s_vals else "",
        "xtrack_mean_m": round(float(np.mean(xt)), 2) if xt else "",
        "xtrack_max_m": round(max(xt), 2) if xt else "",
        "xtrack_sum_m": round(sum(xt), 1) if xt else "",
        "loc_hit_5m_pct": round(100.0 * sum(1 for v in loc if v <= 5.0) / len(loc), 1) if loc else "",
        "loc_fail_20m_pct": round(100.0 * sum(1 for v in loc if v > 20.0) / len(loc), 1) if loc else "",
        "heading_err_mean_deg": round(float(np.mean(hd)), 2) if hd else "",
        "heading_err_max_deg": round(max(hd), 2) if hd else "",
        "gaffin_ratio_median": round(float(np.median(ratio)), 4) if ratio else "",
        "backtrack_steps": sum(1 for a, b in zip(s_vals, s_vals[1:]) if b < a - 1.0),
        # summed over all segments of the run (survives --resume)
        "wall_time_min": round(sum(float(r["t_step_s"]) for r in steps) / 60.0, 1),
        "captures_per_step": round(n_captures / max(1, len(mv)), 2),
    }
    if args.world == "wms":
        summ["wms_fetched"] = world.n_fetched
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(f"done: {outcome}, {len(mv)} steps, {n_captures} captures, "
          f"progress {summ['progress_frac']}, xtrack mean {summ['xtrack_mean_m']} m "
          f"max {summ['xtrack_max_m']} m -> {out}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", choices=("sim", "wms"), required=True)
    ap.add_argument("--route", required=True, help="route CSV (route_s.csv)")
    ap.add_argument("--alt", type=float, required=True, help="AGL altitude in m")
    ap.add_argument("--spacing", type=float, default=5.0,
                    help="teach waypoint spacing in m (must equal the teach run)")
    ap.add_argument("--teach-root", required=True,
                    help="teach output root with alt_XXXXm/ folders")
    ap.add_argument("--sensor", type=int, default=50)
    ap.add_argument("--gray", type=int, default=10)
    ap.add_argument("--ridf-step", type=float, default=1.0,
                    help="RIDF angular step in deg (Gaffin: 1)")
    ap.add_argument("--out", required=True, help="run folder")
    # agent
    ap.add_argument("--step-m", type=float, default=10.0,
                    help="arc radius = step length in m")
    ap.add_argument("--arc-n", type=int, default=7,
                    help="candidates per arc (odd)")
    ap.add_argument("--arc-span-deg", type=float, default=45.0,
                    help="half-angle of the arc in deg (+-)")
    ap.add_argument("--stop-ratio", type=float, default=None,
                    help="early stop when gaffin_ratio <= this (Gaffin 0.20; "
                         "off by default -- see docstring)")
    ap.add_argument("--heading-mode", choices=("extend", "compass"),
                    default="extend",
                    help="next scan direction: extend the movement segment "
                         "(Gaffin) or the RIDF compass estimate")
    ap.add_argument("--start-wp", type=int, default=0)
    ap.add_argument("--start-offset-m", type=float, default=0.0,
                    help="lateral offset of the start position (right +)")
    ap.add_argument("--start-heading", type=float, default=None,
                    help="initial heading in deg (default: route bearing at "
                         "the start wp); corrected by the first RIDF anyway")
    ap.add_argument("--goal-m", type=float, default=15.0,
                    help="success when within this distance of the route end")
    ap.add_argument("--lost-m", type=float, default=60.0,
                    help="abort when farther than this from the route")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="default: 2 * route length / step")
    ap.add_argument("--workers", type=int, default=1,
                    help="processes for the per-frame RIDF (over angles)")
    ap.add_argument("--save-frames", choices=("chosen", "all", "none"),
                    default="chosen")
    ap.add_argument("--save-ridf", action="store_true",
                    help="store D[theta, scene] of every chosen candidate")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--note", default="")
    # camera / capture geometry (both worlds)
    ap.add_argument("--crop-px", type=int, default=640,
                    help="sensor crop the evaluator takes from the capture "
                         "(capture = ceil(crop*sqrt2) = 906 px)")
    # sim world
    ap.add_argument("--origin-lat", type=float, default=52.040111)
    ap.add_argument("--origin-lon", type=float, default=8.495097)
    ap.add_argument("--origin-h", type=float, default=800.0)
    ap.add_argument("--ned-x-bearing", type=float, default=90.0)
    ap.add_argument("--ground-z0", type=float, default=0.0,
                    help="initial ground NED z (640 / 663 / 656)")
    ap.add_argument("--ground-pct", type=float, default=95.0)
    ap.add_argument("--agl-tol", type=float, default=1.0,
                    help="accepted |measured - nominal| AGL in m (depth is "
                         "metre-quantised, so 0.5 never converges)")
    ap.add_argument("--agl-iters", type=int, default=2)
    ap.add_argument("--settle-min", type=float, default=1.0)
    ap.add_argument("--settle-dt", type=float, default=0.7)
    ap.add_argument("--settle-max", type=float, default=25.0)
    ap.add_argument("--settle-thresh", type=float, default=1.0,
                    help="TAA noise floor is ~0.5; 1.0 settles in ~2 s")
    ap.add_argument("--depth-min-finite", type=float, default=0.995)
    ap.add_argument("--sim-config", help="folder with the scene/robot jsonc")
    ap.add_argument("--scene", default="scene_snapshot_repeat.jsonc")
    ap.add_argument("--robot", default="Drone1")
    # wms world
    ap.add_argument("--cache-dir", help="DOP cache for the wms world")
    ap.add_argument("--src-gsd", type=float, default=0.20,
                    help="source GSD in m/px (must equal the teach run)")
    ap.add_argument("--rps", type=float, default=4.0)
    args = ap.parse_args()
    if args.world == "sim" and not args.sim_config and not args.dry_run:
        ap.error("--sim-config is required for --world sim")
    if args.world == "wms" and not args.cache_dir and not args.dry_run:
        ap.error("--cache-dir is required for --world wms")
    return args


if __name__ == "__main__":
    run(parse_args())
