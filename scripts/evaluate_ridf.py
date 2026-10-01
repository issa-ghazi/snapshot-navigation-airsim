#!/usr/bin/env python3
"""evaluate_ridf.py -- RIDF core for the repeat phase (stage 1, open loop).

For every repeat frame the script builds the rotational image difference
function (RIDF) against ALL teach snapshots of the same altitude and sensor
configuration, following Gaffin et al. (2015): the current scene is rotated
through 360 degrees and each rotation is compared pixel-by-pixel (sum of
absolute differences inside the circular mask) to every stored scene.

Geometry contract (repeat side)
--------------------------------
The DownCamera captures a square frame that is ROTATION-SAFE, i.e. its
ground footprint is sqrt(2) times the sensor footprint (FOV 109.47 deg for
an effective 90 deg sensor, capture_px = ceil(crop_px * sqrt(2))). The frame
is passed through the very same `snapshot_lib.rotate_and_crop()` that the
teach phase used, so both phases share one rotation + crop implementation
and histogram equalization never sees black rotation corners. The drone is
yawed to the route bearing at capture time, so theta = 0 corresponds to the
teach orientation ("flight direction up").

Sign convention: `theta` is the image rotation passed to rotate_and_crop
(PIL: counter-clockwise for positive angles). If the drone's true yaw was
bearing + delta (clockwise), the RIDF minimum lies at theta = -delta. The
reported `yaw_err_deg` is therefore -theta_min, wrapped to (-180, 180].

Per-frame metrics (written to results.csv)
-------------------------------------------
best_wp, best_theta_deg, d_best      global minimum over all (theta, scene)
loc_err_m                            |s(best_wp) - s(true_wp)| along the route
loc_err_wp                           best_wp - true_wp (signed, in waypoints)
d_true_min, theta_true_deg           RIDF minimum against the TRUE scene
yaw_err_deg                          -theta_true_deg, wrapped
ridf_depth_true                      (mean_theta - min_theta) / mean_theta at
                                     the true scene; 0 = flat RIDF, no
                                     compass information
alias_ratio                          d_true_min / min over all OTHER scenes;
                                     > 1 means some wrong scene matched better
gaffin_ratio                         d_best / mean(D over all theta, scenes);
                                     Gaffin's acceptance threshold is 0.20

`true_wp` comes from the repeat manifest (open-loop ground truth). If a
frame has no `wp`, only the global-minimum metrics are filled.

Rotation order
--------------
By default each rotation is applied to the RGB frame and grayscaled
afterwards -- byte-for-byte the order `extract_route_snapshots.py` used on
the teach side. Preprocessing identity between the phases outranks speed.

`--gray-first` converts the frame to grayscale once and rotates the L image
instead (~2.3x faster). In exact arithmetic the two orders commute (ITU-R
601 is per-pixel linear, bicubic rotation is a linear spatial filter), but
PIL rounds to uint8 after each step, and histogram equalization plus
quantisation amplify a rounding difference of one grey level into a
different sensor value. On genuinely grey input (R=G=B, as the NRW DOP
cache happens to be) the orders are identical; on colour input they are
not. `--gray-first` therefore always runs the equivalence check on the
first frame and aborts if a single masked sensor pixel differs -- never
assume it is safe for a new data source.

Frames are independent, so `--workers N` spreads them over N processes.

Usage
-----
    python evaluate_ridf.py --teach-root data/teach --alt 50 \
        --repeat-manifest data/repeat/alt_0050m/manifest.csv \
        --sensor 50 --gray 10 --step 1 --out results/alt_0050m

Add --save-ridf to also store the full D[theta, scene] matrix per frame as
.npy (needed for volcano / RIDF plots later).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from snapshot_lib import circular_mask, preprocess_view, rotate_and_crop

# --------------------------------------------------------------------------
# Teach library
# --------------------------------------------------------------------------
@dataclass
class TeachLibrary:
    alt_m: float
    sensor_px: int
    gray_levels: int
    wp: np.ndarray          # (N,) int   waypoint index
    s_m: np.ndarray         # (N,) float arclength along route
    bearing_deg: np.ndarray  # (N,) float
    lat: np.ndarray
    lon: np.ndarray
    footprint_m: float
    mask: np.ndarray        # (n, n) bool circular mask
    flat: np.ndarray        # (N, P) int16, masked pixels only

    @property
    def n_scenes(self) -> int:
        return self.flat.shape[0]

    def index_of_wp(self, wp: int) -> int | None:
        hits = np.nonzero(self.wp == wp)[0]
        return int(hits[0]) if len(hits) else None


def _resolve(root: Path, rel: str) -> Path:
    """Manifest paths were written on Windows (backslashes) relative to the
    teach output root."""
    return root / Path(rel.replace("\\", "/"))


def load_teach_library(teach_root: Path, alt_m: float, sensor_px: int,
                       gray_levels: int, wps: list[int] | None = None
                       ) -> TeachLibrary:
    """Read one altitude's manifest and load the sensor images of the
    requested (sensor_px, gray_levels) variant into a masked pixel matrix.

    Only the centreline (line_offset_m == 0) is used; corridor lines, if
    present, are ignored so that `wp` is unique per scene.
    """
    alt_dir = teach_root / f"alt_{alt_m:04.0f}m"
    man = alt_dir / "manifest.csv"
    if not man.exists():
        sys.exit(f"teach manifest not found: {man}")
    rows = []
    with open(man, newline="") as f:
        for r in csv.DictReader(f):
            if int(r["sensor_px"]) != sensor_px:
                continue
            if int(r["gray_levels"]) != gray_levels:
                continue
            if float(r.get("line_offset_m", 0.0)) != 0.0:
                continue
            if wps is not None and int(r["wp"]) not in wps:
                continue
            rows.append(r)
    if not rows:
        sys.exit(f"no manifest rows for sensor {sensor_px} / gray "
                 f"{gray_levels} in {man}")
    rows.sort(key=lambda r: int(r["wp"]))

    mask = circular_mask(sensor_px)
    flat = np.empty((len(rows), int(mask.sum())), dtype=np.int16)
    for i, r in enumerate(rows):
        p = _resolve(teach_root, r["file"])
        arr = np.asarray(Image.open(p).convert("L"))
        if arr.shape != (sensor_px, sensor_px):
            sys.exit(f"{p}: expected {sensor_px}x{sensor_px}, got {arr.shape}")
        flat[i] = arr[mask]

    g = lambda k: np.array([float(r[k]) for r in rows])
    return TeachLibrary(
        alt_m=alt_m, sensor_px=sensor_px, gray_levels=gray_levels,
        wp=np.array([int(r["wp"]) for r in rows]),
        s_m=g("s_m"), bearing_deg=g("bearing_deg"),
        lat=g("lat"), lon=g("lon"),
        footprint_m=float(rows[0]["footprint_m"]),
        mask=mask, flat=flat)


# --------------------------------------------------------------------------
# Repeat frame -> rotation stack
# --------------------------------------------------------------------------
def rotation_stack(frame: Image.Image, crop_px: int, sensor_px: int,
                   gray_levels: int, step_deg: float = 1.0,
                   mask: np.ndarray | None = None, gray_first: bool = False):
    """Rotate the (rotation-safe) repeat frame through 360 deg and run the
    shared sensor chain on every rotation.

    Returns (thetas (K,), flat (K, P) int16) with theta in [0, 360).
    """
    if mask is None:
        mask = circular_mask(sensor_px)
    src = ImageOps.grayscale(frame) if gray_first else frame
    thetas = np.arange(0.0, 360.0, step_deg)
    flat = np.empty((len(thetas), int(mask.sum())), dtype=np.int16)
    for k, th in enumerate(thetas):
        patch = rotate_and_crop(src, float(th), crop_px)
        flat[k] = preprocess_view(patch, sensor_px, gray_levels)[mask]
    return thetas, flat


def verify_gray_first(frame: Image.Image, crop_px: int, sensor_px: int,
                      gray_levels: int, angles=(0, 17, 45, 90, 133, 200,
                                                271, 333)) -> int:
    """Number of masked sensor pixels that differ between the gray-first
    rotation order and the teach-phase order (rotate RGB, then grayscale),
    summed over the test angles. Must be 0."""
    mask = circular_mask(sensor_px)
    g = ImageOps.grayscale(frame)
    n = 0
    for th in angles:
        a = preprocess_view(rotate_and_crop(frame, th, crop_px), sensor_px,
                            gray_levels)
        b = preprocess_view(rotate_and_crop(g, th, crop_px), sensor_px,
                            gray_levels)
        n += int(((a != b) & mask).sum())
    return n


# --------------------------------------------------------------------------
# RIDF matrix
# --------------------------------------------------------------------------
def ridf_matrix(rot_flat: np.ndarray, teach_flat: np.ndarray,
                chunk: int = 32) -> np.ndarray:
    """D[k, j] = sum |rot_k - teach_j| over masked pixels. int64 (K, N).

    Exactly snapshot_lib.scene_difference(), vectorised. Chunked over
    rotations to keep the (chunk, N, P) intermediate small.
    """
    K, N = rot_flat.shape[0], teach_flat.shape[0]
    D = np.empty((K, N), dtype=np.int64)
    t32 = teach_flat.astype(np.int32)
    for a in range(0, K, chunk):
        b = min(a + chunk, K)
        diff = np.abs(rot_flat[a:b, None, :].astype(np.int32) - t32[None])
        D[a:b] = diff.sum(axis=2)
    return D


def wrap180(deg: float) -> float:
    return ((deg + 180.0) % 360.0) - 180.0


def summarize(D: np.ndarray, thetas: np.ndarray, lib: TeachLibrary,
              true_wp: int | None) -> dict:
    """Per-frame metrics from the RIDF matrix (see module docstring)."""
    k_best, j_best = np.unravel_index(int(np.argmin(D)), D.shape)
    d_mean_all = float(D.mean())
    out = {
        "best_wp": int(lib.wp[j_best]),
        "best_theta_deg": round(wrap180(float(thetas[k_best])), 2),
        "d_best": int(D[k_best, j_best]),
        "d_mean_all": round(d_mean_all, 1),
        "gaffin_ratio": round(float(D[k_best, j_best]) / d_mean_all, 4),
    }
    j_true = lib.index_of_wp(true_wp) if true_wp is not None else None
    if j_true is None:
        out.update(dict(loc_err_m="", loc_err_wp="", d_true_min="",
                        theta_true_deg="", yaw_err_deg="",
                        ridf_depth_true="", alias_ratio=""))
        return out

    col = D[:, j_true]
    k_t = int(np.argmin(col))
    theta_t = wrap180(float(thetas[k_t]))
    per_scene_min = D.min(axis=0)
    others = np.delete(per_scene_min, j_true)
    out.update({
        "loc_err_m": round(abs(float(lib.s_m[j_best] - lib.s_m[j_true])), 2),
        "loc_err_wp": int(lib.wp[j_best] - lib.wp[j_true]),
        "d_true_min": int(col[k_t]),
        "theta_true_deg": round(theta_t, 2),
        "yaw_err_deg": round(wrap180(-theta_t), 2),
        "ridf_depth_true": round(float((col.mean() - col.min()) / col.mean()),
                                 4),
        "alias_ratio": (round(float(col[k_t]) / float(others.min()), 4)
                        if len(others) else ""),
    })
    return out


# --------------------------------------------------------------------------
# Repeat manifest
# --------------------------------------------------------------------------
def load_repeat_manifest(path: Path):
    """Minimal contract: columns `file` (relative to the manifest's folder)
    and `crop_px`; `wp` optional (ground truth for open-loop metrics).
    Everything else is passed through to the results row."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"empty repeat manifest: {path}")
    for r in rows:
        if "file" not in r or "crop_px" not in r:
            sys.exit("repeat manifest needs at least `file` and `crop_px`")
    return rows


# --------------------------------------------------------------------------
# Per-frame worker (module level so multiprocessing can pickle it)
# --------------------------------------------------------------------------
_W = {}   # worker globals: lib, rep_dir, sensor, gray, step, save_ridf


def _init_worker(lib, rep_dir, sensor, gray, step, save_ridf, gray_first):
    _W.update(lib=lib, rep_dir=rep_dir, sensor=sensor, gray=gray, step=step,
              save_ridf=save_ridf, gray_first=gray_first)


def _eval_frame(r: dict):
    lib = _W["lib"]
    fpath = _W["rep_dir"] / Path(r["file"].replace("\\", "/"))
    frame = Image.open(fpath)
    thetas, rot = rotation_stack(frame, int(r["crop_px"]), _W["sensor"],
                                 _W["gray"], _W["step"], lib.mask,
                                 _W["gray_first"])
    D = ridf_matrix(rot, lib.flat)
    true_wp = int(r["wp"]) if r.get("wp", "") != "" else None
    m = summarize(D, thetas, lib, true_wp)
    return m, true_wp, (D if _W["save_ridf"] else None)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teach-root", required=True,
                    help="teach output root (contains alt_XXXXm/ folders)")
    ap.add_argument("--alt", type=float, required=True,
                    help="altitude of the library to match against")
    ap.add_argument("--repeat-manifest", required=True,
                    help="CSV listing the repeat frames (see docstring)")
    ap.add_argument("--sensor", type=int, default=50)
    ap.add_argument("--gray", type=int, default=10)
    ap.add_argument("--step", type=float, default=1.0,
                    help="RIDF angular step in degrees (Gaffin: 1)")
    ap.add_argument("--out", required=True, help="results directory")
    ap.add_argument("--save-ridf", action="store_true",
                    help="store D[theta, scene] per frame as .npy")
    ap.add_argument("--limit", type=int,
                    help="evaluate only the first N frames (smoke test)")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel processes over frames (default 1)")
    ap.add_argument("--gray-first", action="store_true",
                    help="rotate the grayscale frame instead of the RGB one "
                         "(~2.3x faster). Only valid where it is provably "
                         "identical to the teach order; the check runs "
                         "automatically and aborts on any difference.")
    return ap.parse_args()


def main():
    args = parse_args()
    teach_root = Path(args.teach_root)
    rep_man = Path(args.repeat_manifest)
    rep_dir = rep_man.parent
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = load_repeat_manifest(rep_man)
    if args.limit:
        frames = frames[:args.limit]

    lib = load_teach_library(teach_root, args.alt, args.sensor, args.gray)
    print(f"teach library: alt {lib.alt_m:.0f} m, {lib.n_scenes} scenes, "
          f"{lib.sensor_px}x{lib.sensor_px}/{lib.gray_levels} gray, "
          f"{lib.flat.shape[1]} masked px")
    print(f"repeat frames: {len(frames)} from {rep_man}")

    (out_dir / "params.json").write_text(json.dumps({
        "teach_root": str(teach_root), "alt_m": args.alt,
        "sensor_px": args.sensor, "gray_levels": args.gray,
        "step_deg": args.step, "repeat_manifest": str(rep_man),
        "n_scenes": int(lib.n_scenes), "metric": "sum |a-b| inside "
        "circular mask (snapshot_lib.scene_difference)",
        "theta_convention": "image rotation passed to rotate_and_crop "
        "(CCW positive); yaw_err_deg = -theta_true_deg",
        "rotation_order": ("grayscale then rotate" if args.gray_first
                           else "rotate RGB then grayscale (teach order)"),
    }, indent=2))

    if args.gray_first:
        r0 = frames[0]
        f0 = Image.open(rep_dir / Path(r0["file"].replace("\\", "/")))
        n_bad = verify_gray_first(f0, int(r0["crop_px"]), args.sensor,
                                  args.gray)
        print(f"gray-first check on {r0['file']}: {n_bad} differing sensor "
              f"pixels over 8 angles")
        if n_bad:
            sys.exit("gray-first is NOT identical to the teach rotation "
                     "order on this data -- drop --gray-first")
    print(f"rotation order: {'grayscale then rotate' if args.gray_first else 'rotate RGB then grayscale (teach order)'}")

    passthrough = [k for k in frames[0].keys() if k not in ("file",)]
    fieldnames = passthrough + [
        "best_wp", "best_theta_deg", "d_best", "d_mean_all", "gaffin_ratio",
        "loc_err_m", "loc_err_wp", "d_true_min", "theta_true_deg",
        "yaw_err_deg", "ridf_depth_true", "alias_ratio", "file"]

    init_args = (lib, rep_dir, args.sensor, args.gray, args.step,
                 args.save_ridf, args.gray_first)
    t0 = time.monotonic()
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        if args.workers > 1:
            import multiprocessing as mp
            pool = mp.Pool(args.workers, initializer=_init_worker,
                           initargs=init_args)
            it = pool.imap(_eval_frame, frames)
        else:
            _init_worker(*init_args)
            pool, it = None, map(_eval_frame, frames)
        try:
            for i, (r, (m, true_wp, D)) in enumerate(zip(frames, it), 1):
                row = {k: r[k] for k in passthrough}
                row.update(m)
                row["file"] = r["file"]
                w.writerow(row)
                if D is not None:
                    tag = (f"w{true_wp:05d}" if true_wp is not None
                           else f"f{i:05d}")
                    np.save(out_dir / f"ridf_{tag}.npy", D)
                if i % 10 == 0 or i == len(frames):
                    el = time.monotonic() - t0
                    print(f"  {i}/{len(frames)}  {el / i:.2f} s/frame  "
                          f"last: wp {true_wp} -> best {m['best_wp']} @ "
                          f"{m['best_theta_deg']} deg, yaw_err "
                          f"{m['yaw_err_deg']}")
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    print(f"done -> {out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
