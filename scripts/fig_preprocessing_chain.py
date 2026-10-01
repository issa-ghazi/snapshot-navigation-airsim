#!/usr/bin/env python3
"""fig_preprocessing_chain.py -- figure for Section 3.5 of the thesis.

Renders the shared preprocessing chain for ONE waypoint, side by side for the
teach observation (orthophoto) and the repeat observation (simulator frame):

    rotation-safe source -> bearing-aligned crop -> grayscale -> histeq
    -> n x n downsample -> quantized + circularly masked sensor image

Every stage is produced by the same `snapshot_lib` functions the experiment
itself uses, so the figure shows the actual pipeline and not a reconstruction
of it. As a check, the final teach panel is compared against the sensor image
stored in the teach library; a mismatch is reported on stderr.

Requirements: the teach WMS cache (`_cache_dop`) must still be present, since
the intermediate stages are not stored anywhere. If it was deleted, either
re-run `extract_route_snapshots.py` for the single altitude used here (the
cache is rebuilt automatically) or pass the source square with
`--teach-source`.

Usage
-----
    python scripts/fig_preprocessing_chain.py ^
        --teach-root data/teach_innenstadt ^
        --repeat-manifest data/repeat_innenstadt/alt_0050m/manifest.csv ^
        --alt 50 --wp 60 --sensor 50 --gray 10 ^
        --out figures/fig_preprocessing_chain.png
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from snapshot_lib import (ROTATION_MARGIN, center_crop, circular_mask,  # noqa: E402
                          preprocess_view, quantize, rotate_and_crop,
                          _resampling)

STAGES = ["rotation-safe source", "bearing-aligned crop", "grayscale",
          "histogram equalized", "sensor downsample", "quantized + masked"]


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def read_teach_row(teach_root: Path, alt_m: float, wp: int, sensor_px: int,
                   gray_levels: int) -> dict:
    """One manifest row of the requested waypoint / sensor variant."""
    man = teach_root / f"alt_{alt_m:04.0f}m" / "manifest.csv"
    if not man.exists():
        sys.exit(f"teach manifest not found: {man}")
    with open(man, newline="") as f:
        for r in csv.DictReader(f):
            if (int(r["wp"]) == wp and int(r["sensor_px"]) == sensor_px
                    and int(r["gray_levels"]) == gray_levels
                    and float(r.get("line_offset_m", 0.0)) == 0.0):
                return r
    sys.exit(f"no teach row for wp {wp}, sensor {sensor_px}, "
             f"gray {gray_levels} in {man}")


def read_repeat_row(manifest: Path, wp: int) -> dict:
    if not manifest.exists():
        sys.exit(f"repeat manifest not found: {manifest}")
    with open(manifest, newline="") as f:
        for r in csv.DictReader(f):
            if int(r["wp"]) == wp:
                return r
    sys.exit(f"waypoint {wp} not present in {manifest}")


def open_teach_source(row: dict, teach_root: Path, cache_dir: Path | None,
                      override: Path | None) -> Image.Image:
    if override is not None:
        return Image.open(override)
    cache_dir = cache_dir or (teach_root / "_cache_dop")
    p = cache_dir / f"{row['cache_key']}.png"
    if not p.exists():
        sys.exit(f"cached source image not found: {p}\n"
                 f"The intermediate stages cannot be reconstructed without "
                 f"it. Re-run extract_route_snapshots.py for altitude "
                 f"{row['alt_m']} m (the cache is rebuilt automatically) or "
                 f"pass --teach-source.")
    return Image.open(p)


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------
def chain_stages(src: Image.Image, rotation_deg: float, crop_px: int,
                 sensor_px: int, gray_levels: int) -> list[np.ndarray]:
    """The six panels of one row, in pipeline order."""
    w, h = src.size
    rot_px = min(int(math.ceil(crop_px * ROTATION_MARGIN)), w, h)

    safe = center_crop(src, rot_px)
    patch = rotate_and_crop(src, rotation_deg, crop_px)
    gray = ImageOps.grayscale(patch)
    eq = ImageOps.equalize(gray)
    small = eq.resize((sensor_px, sensor_px), _resampling("BOX"))
    final = preprocess_view(patch, sensor_px, gray_levels)

    # sanity: the separately computed intermediates must reproduce `final`
    ref = quantize(np.asarray(small), gray_levels).copy()
    ref[~circular_mask(sensor_px)] = 0
    if not np.array_equal(ref, final):
        print("  warning: intermediate stages do not reproduce the pipeline "
              "output", file=sys.stderr)

    return [np.asarray(safe), np.asarray(patch), np.asarray(gray),
            np.asarray(eq), np.asarray(small), final]


def panel_sizes(stages: list[np.ndarray]) -> list[str]:
    return [f"{a.shape[0]} px" for a in stages]


def for_display(arr: np.ndarray, max_px: int) -> np.ndarray:
    """Shrink a panel for rendering only.

    The large panels are plotted at roughly 2 in, so anything beyond a few
    hundred pixels only inflates the output file. The sensor matrices are far
    below the limit and are never touched, so the blocky appearance that
    carries the point of the last two panels is preserved.
    """
    if max_px <= 0 or arr.shape[0] <= max_px:
        return arr
    img = Image.fromarray(arr)
    return np.asarray(img.resize((max_px, max_px), _resampling("LANCZOS")))


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
def build_figure(rows: list[tuple[str, list[np.ndarray]]], out: Path,
                 dpi: int, max_display_px: int) -> None:
    n_cols = len(STAGES)
    fig, axes = plt.subplots(len(rows), n_cols,
                             figsize=(2.05 * n_cols, 2.05 * len(rows) + 0.35))
    axes = np.atleast_2d(axes)

    for i, (label, stages) in enumerate(rows):
        sizes = panel_sizes(stages)
        for j, raw in enumerate(stages):
            arr = for_display(raw, max_display_px)
            ax = axes[i, j]
            if arr.ndim == 3:
                ax.imshow(arr, interpolation="nearest")
            else:
                ax.imshow(arr, cmap="gray", vmin=0, vmax=255,
                          interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.5)
                s.set_color("0.4")
            if i == 0:
                ax.set_title(f"({chr(97 + j)}) {STAGES[j]}", fontsize=8,
                             pad=5)
            ax.set_xlabel(sizes[j], fontsize=7, labelpad=2)
            if j == 0:
                ax.set_ylabel(label, fontsize=9, labelpad=6)

    fig.tight_layout(w_pad=0.6, h_pad=0.8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} kB, {dpi} dpi)")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teach-root", required=True,
                    help=r"teach output root, e.g. data\teach_rgb")
    ap.add_argument("--repeat-manifest", required=True,
                    help=r"e.g. data\repeat\alt_0050m\manifest.csv")
    ap.add_argument("--alt", type=float, required=True)
    ap.add_argument("--wp", type=int, default=60)
    ap.add_argument("--sensor", type=int, default=50)
    ap.add_argument("--gray", type=int, default=10)
    ap.add_argument("--cache-dir", help="default: TEACH_ROOT/_cache_dop")
    ap.add_argument("--teach-source",
                    help="use this image as the teach source square instead "
                         "of the cached one")
    ap.add_argument("--out", default="figures/fig_preprocessing_chain.png")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--max-display-px", type=int, default=512,
                    help="shrink oversized panels for rendering only "
                         "(0 disables); keeps the output file small")
    return ap.parse_args()


def main():
    a = parse_args()
    teach_root = Path(a.teach_root)
    rep_man = Path(a.repeat_manifest)

    t_row = read_teach_row(teach_root, a.alt, a.wp, a.sensor, a.gray)
    r_row = read_repeat_row(rep_man, a.wp)

    t_src = open_teach_source(
        t_row, teach_root,
        Path(a.cache_dir) if a.cache_dir else None,
        Path(a.teach_source) if a.teach_source else None)
    r_src = Image.open(rep_man.parent / Path(r_row["file"].replace("\\", "/")))

    t_crop = int(t_row["crop_px"])
    r_crop = int(r_row["crop_px"])
    bearing = float(t_row["bearing_deg"])

    print(f"waypoint {a.wp} at {a.alt:.0f} m, sensor {a.sensor}x{a.sensor}/"
          f"{a.gray}")
    print(f"  teach : {t_src.size[0]} px source, crop {t_crop} px, "
          f"bearing {bearing:.2f} deg")
    print(f"  repeat: {r_src.size[0]} px frame, crop {r_crop} px, "
          f"captured at the route bearing (rotation 0)")

    # Teach: the stored scene is rotated to the route bearing.
    t_stages = chain_stages(t_src, bearing, t_crop, a.sensor, a.gray)
    # Repeat: the camera was already yawed to the bearing, so theta = 0 is the
    # orientation the RIDF reports as a match.
    r_stages = chain_stages(r_src, 0.0, r_crop, a.sensor, a.gray)

    stored = teach_root / Path(t_row["file"].replace("\\", "/"))
    if stored.exists():
        ref = np.asarray(Image.open(stored).convert("L"))
        n_diff = int((ref != t_stages[-1]).sum())
        print(f"  check : {n_diff} of {a.sensor ** 2} pixels differ from the "
              f"stored teach snapshot")
        if n_diff:
            print("          (expected 0; a non-zero count indicates a "
                  "different Pillow version)", file=sys.stderr)

    build_figure([("teach (orthophoto)", t_stages),
                  ("repeat (simulator)", r_stages)],
                 Path(a.out), a.dpi, a.max_display_px)


if __name__ == "__main__":
    main()
