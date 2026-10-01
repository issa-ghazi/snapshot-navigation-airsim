r"""probe_sim.py -- staged probe to find where Unreal crashes.

Runs the repeat-phase setup one step at a time and prints what succeeded
before the crash. Start with --stage 1 and increase until UE dies; the last
stage that printed "OK" tells you which operation is responsible.

    stage 1  connect, load scene, read pose         -> config parsing
             (add --query-lighting to test the sky/sun API separately)
    stage 2  grab one frame at the spawn pose       -> camera config
                                                       (906 px / 109.47 deg
                                                        + planar depth)
    stage 3  set_pose in place (no movement)        -> non-physics teleport
    stage 4  hop toward the route in --hops steps   -> Cesium tile streaming
    stage 5  settle and save a frame at waypoint 0  -> full capture path

Usage (from the repository root, Unreal in Play):

    python scripts/probe_sim.py --stage 1 --sim-config sim_config
    python scripts/probe_sim.py --stage 2 --sim-config sim_config
    ...
    python scripts/probe_sim.py --stage 5 --sim-config sim_config \
        --route routes/route_s.csv

Options that matter for stage 4:
    --hops N        number of intermediate steps toward the route (default 8)
    --hop-delay S   seconds to wait at each hop for tiles (default 3)
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Same route geometry as capture_repeat.py
from capture_repeat import (CAMERA_ID, IMG_DEPTH_PLANAR, IMG_SCENE,
                            bearing_to_ned_yaw_deg, geo_to_ned_xy,
                            load_route_csv)
from snapshot_lib import resample_route


def say(msg):
    print(f"  {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--sim-config", required=True)
    ap.add_argument("--scene", default="scene_snapshot_repeat.jsonc")
    ap.add_argument("--robot", default="Drone1")
    ap.add_argument("--route", default="routes/route_s.csv")
    ap.add_argument("--spacing", type=float, default=5.0)
    ap.add_argument("--wp", type=int, default=0)
    ap.add_argument("--alt", type=float, default=50.0)
    ap.add_argument("--origin-lat", type=float, default=52.040111)
    ap.add_argument("--origin-lon", type=float, default=8.495097)
    ap.add_argument("--origin-h", type=float, default=800.0)
    ap.add_argument("--ned-x-bearing", type=float, default=90.0)
    ap.add_argument("--ground-z0", type=float, default=640.0)
    ap.add_argument("--hops", type=int, default=8)
    ap.add_argument("--hop-delay", type=float, default=3.0)
    ap.add_argument("--query-lighting", action="store_true",
                    help="test the sky/sun API calls (crash suspect)")
    ap.add_argument("--out", default="probe_out")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- stage 1: connect + load scene ----------------------
    print(f"[stage 1] connect and load {args.scene}")
    from projectairsim import Drone, ProjectAirSimClient, World
    from projectairsim.types import Pose, Quaternion, Vector3
    from projectairsim.utils import rpy_to_quaternion, unpack_image

    client = ProjectAirSimClient()
    client.connect()
    say("client connected")
    world = World(client, args.scene, delay_after_load_sec=2,
                  sim_config_path=args.sim_config)
    say("scene loaded")
    drone = Drone(client, world, args.robot)
    say(f"robot '{args.robot}' bound")
    if args.query_lighting:
        say("querying lighting (this is the suspected crash point)")
        for name, fn in (("time_of_day", world.get_time_of_day),
                         ("sunlight_lux", world.get_sunlight_intensity)):
            try:
                say(f"{name}: {fn()}")
            except Exception as e:
                say(f"{name}: n/a ({type(e).__name__}: {e})")
    else:
        say("lighting query skipped (--query-lighting to test it)")
    try:
        say(f"spawn pose: {drone.get_ground_truth_pose()}")
    except Exception as e:
        say(f"pose read failed: {e}")
    print("[stage 1] OK")
    if args.stage == 1:
        client.disconnect()
        return

    def grab(tag):
        imgs = drone.get_images(CAMERA_ID, [IMG_SCENE, IMG_DEPTH_PLANAR])
        scene = imgs[IMG_SCENE]
        bgr = unpack_image(scene)
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        d_msg = imgs[IMG_DEPTH_PLANAR]
        depth = np.asarray(unpack_image(d_msg), dtype=np.float32).reshape(
            d_msg["height"], d_msg["width"])
        fin = depth[np.isfinite(depth)]
        say(f"{tag}: rgb {rgb.shape} {scene.get('encoding')}, "
            f"mean grey {rgb.mean():.1f}; depth {depth.shape} "
            f"finite {np.isfinite(depth).mean():.3f}"
            + (f", p95 {np.percentile(fin, 95):.1f} m" if fin.size else ""))
        return rgb, depth

    # ---------------- stage 2: one frame at spawn ------------------------
    print("[stage 2] grab one frame at the spawn pose")
    rgb, _ = grab("frame")
    Image.fromarray(rgb, "RGB").save(out / "stage2_spawn.png")
    say(f"saved {out / 'stage2_spawn.png'}")
    print("[stage 2] OK")
    if args.stage == 2:
        client.disconnect()
        return

    def teleport(x, y, z, yaw_deg):
        w, qx, qy, qz = rpy_to_quaternion(0.0, 0.0, math.radians(yaw_deg))
        pose = Pose({"translation": Vector3({"x": x, "y": y, "z": z}),
                     "rotation": Quaternion({"w": w, "x": qx, "y": qy,
                                             "z": qz}),
                     "frame_id": "DEFAULT_ID"})
        ok = drone.set_pose(pose, reset_kinematics=True)
        if not ok:
            raise RuntimeError("set_pose returned False -- robot is probably "
                               "not non-physics")

    # ---------------- stage 3: teleport in place -------------------------
    print("[stage 3] set_pose at the current position (no movement)")
    p = drone.get_ground_truth_pose()
    t = p["translation"]
    teleport(float(t["x"]), float(t["y"]), float(t["z"]), 0.0)
    say("set_pose accepted")
    time.sleep(1.0)
    grab("frame after in-place teleport")
    print("[stage 3] OK")
    if args.stage == 3:
        client.disconnect()
        return

    # ---------------- target waypoint ------------------------------------
    wps = resample_route(load_route_csv(args.route), args.spacing)
    wp = wps[args.wp]
    tx, ty = geo_to_ned_xy(args.origin_lat, args.origin_lon, args.origin_h,
                           wp["lat"], wp["lon"], args.ned_x_bearing)
    tz = args.ground_z0 - args.alt
    tyaw = bearing_to_ned_yaw_deg(wp["bearing_deg"], args.ned_x_bearing)

    # ---------------- stage 4: hop toward the route ----------------------
    p = drone.get_ground_truth_pose()
    t = p["translation"]
    sx, sy, sz = float(t["x"]), float(t["y"]), float(t["z"])
    dist = math.dist((sx, sy), (tx, ty))
    print(f"[stage 4] hop {dist:.0f} m to waypoint {args.wp} in {args.hops} "
          f"steps ({dist / max(args.hops, 1):.0f} m each)")
    for k in range(1, args.hops + 1):
        f = k / args.hops
        x, y, z = sx + f * (tx - sx), sy + f * (ty - sy), sz + f * (tz - sz)
        teleport(x, y, z, tyaw)
        time.sleep(args.hop_delay)
        say(f"hop {k}/{args.hops}: x {x:.0f} y {y:.0f} z {z:.0f}")
        grab("  frame")
    print("[stage 4] OK")
    if args.stage == 4:
        client.disconnect()
        return

    # ---------------- stage 5: settle and save ---------------------------
    print(f"[stage 5] settle at waypoint {args.wp} ({args.alt} m nominal AGL)")
    teleport(tx, ty, tz, tyaw)
    prev = None
    t0 = time.monotonic()
    while time.monotonic() - t0 < 30.0:
        rgb, depth = grab(f"  t={time.monotonic() - t0:5.1f}s")
        g = rgb.astype(np.float32).mean(axis=2)
        if prev is not None:
            d = float(np.abs(g - prev).mean())
            say(f"  frame-to-frame diff {d:.2f}")
            if d < 0.5:
                break
        prev = g
        time.sleep(0.7)
    Image.fromarray(rgb, "RGB").save(out / "stage5_wp.png")
    say(f"saved {out / 'stage5_wp.png'}")
    print("[stage 5] OK -- the full capture path works")
    client.disconnect()


if __name__ == "__main__":
    main()
