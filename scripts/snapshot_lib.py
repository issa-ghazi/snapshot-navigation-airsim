"""snapshot_lib.py -- shared geometry and preprocessing for teach and repeat phases.

Teach phase: snapshots extracted from real orthographic aerial imagery
(Geobasis NRW digital orthophotos, WMS NW DOP). Repeat phase: nadir
DownCamera frames rendered from Cesium / Google Photorealistic 3D Tiles in
Project AirSim.

ground_resolution() and pick_zoom() (Web Mercator zoom selection) are unused
leftovers of an early prototype based on the Google Maps Static API; they are
kept so that this module stays byte-identical in every function the
experiments call.

The preprocessing chain replicates Gaffin et al. (2015), PLOS ONE 10(4):
e0122077 ("Visual System" section):

    grayscale -> histogram equalization ("histeq") -> downsampling to an
    N x N sensor matrix -> gray-level quantization -> circular mask.

Teach snapshots are additionally rotated to the local route bearing *before*
the final crop, mirroring the paper's "Creating the training path" step
(each stored scene is rotated by the bearing of travel).

IMPORTANT: import THIS module from BOTH phases so that every pixel operation
is bit-identical. Any change here invalidates previously generated libraries.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageOps

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
EARTH_RADIUS_M = 6378137.0  # WGS84 semi-major axis (Web Mercator sphere)
# Ground resolution of one pixel at zoom 0 for 256 px tiles, at the equator:
# 2 * pi * R / 256  =  156543.03392... m/px
WEBMERCATOR_MPP_Z0 = 2.0 * math.pi * EARTH_RADIUS_M / 256.0
# Meters per degree of latitude (spherical approx.; error < 0.1 % at 52 deg N,
# negligible against the geolocation accuracy of the imagery itself).
M_PER_DEG_LAT = 111320.0
# A crop rotated by an arbitrary bearing needs a sqrt(2) larger source square
# so that no undefined corners enter the final footprint crop.
ROTATION_MARGIN = math.sqrt(2.0)


# --------------------------------------------------------------------------
# Web Mercator / camera geometry
# --------------------------------------------------------------------------
def ground_resolution(lat_deg: float, zoom: int, scale: int = 1) -> float:
    """Meters per *returned* pixel of a Web Mercator static image.

    `scale=2` doubles the pixel density of the returned image (same ground
    coverage, twice the pixels), hence half the meters-per-pixel.
    """
    return (WEBMERCATOR_MPP_Z0 * math.cos(math.radians(lat_deg))
            / (2.0 ** zoom) / scale)


def footprint_from_altitude(altitude_m: float, fov_deg: float) -> float:
    """Ground footprint (edge length, meters) of a square nadir camera.

    footprint = 2 * h * tan(FOV / 2). For the AirSim default FOV of 90 deg
    this is simply 2 * altitude. Gaffin et al. used a 22.6 deg beam
    (100 m footprint from 250 m altitude).
    """
    return 2.0 * altitude_m * math.tan(math.radians(fov_deg) / 2.0)


def pick_zoom(footprint_m: float, lat_deg: float, usable_px_unscaled: int,
              max_zoom: int = 20, rotation_margin: float = ROTATION_MARGIN):
    """Highest Web Mercator zoom whose usable static-image canvas still
    covers the rotated footprint crop.

    `usable_px_unscaled` is the watermark-free canvas edge in *unscaled*
    pixels (e.g. 640 - 2 * 30). Ground coverage of the canvas is independent
    of the `scale` parameter, so the check is done at scale 1.

    Returns (zoom, ground_coverage_of_canvas_m).
    Raises ValueError if even zoom 1 cannot cover the footprint.
    """
    need_m = footprint_m * rotation_margin
    for z in range(max_zoom, 0, -1):
        cover_m = usable_px_unscaled * ground_resolution(lat_deg, z, scale=1)
        if cover_m >= need_m:
            return z, cover_m
    raise ValueError(
        f"footprint {footprint_m:.0f} m cannot be covered by a "
        f"{usable_px_unscaled}px static image at any zoom level")


# --------------------------------------------------------------------------
# Local planar geodesy (valid for routes up to a few km)
# --------------------------------------------------------------------------
def offset_latlon(lat: float, lon: float, d_east_m: float, d_north_m: float):
    """Offset a WGS84 position by local ENU meters (small-area approx.)."""
    dlat = d_north_m / M_PER_DEG_LAT
    dlon = d_east_m / (M_PER_DEG_LAT * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


def local_xy(lat0: float, lon0: float, lat: float, lon: float):
    """WGS84 -> local ENU meters relative to (lat0, lon0)."""
    d_north = (lat - lat0) * M_PER_DEG_LAT
    d_east = (lon - lon0) * M_PER_DEG_LAT * math.cos(math.radians(lat0))
    return d_east, d_north


def bearing_from_delta(d_east_m: float, d_north_m: float) -> float:
    """Compass bearing (deg, clockwise from north) of a local ENU vector."""
    return math.degrees(math.atan2(d_east_m, d_north_m)) % 360.0


def resample_route(points, spacing_m: float, keep_tail_frac: float = 0.25):
    """Resample a lat/lon polyline at constant arclength spacing.

    points : iterable of (lat, lon)
    Returns a list of dicts {lat, lon, s_m, bearing_deg}. The final route
    endpoint is appended if the remaining tail exceeds
    keep_tail_frac * spacing. Bearings are forward differences (the last
    point inherits the previous bearing), matching the paper's
    "bearing of travel between successive points".
    """
    pts = list(points)
    if len(pts) < 2:
        raise ValueError("route needs at least 2 points")
    lat0, lon0 = pts[0]
    xy = [local_xy(lat0, lon0, la, lo) for la, lo in pts]

    # cumulative arclength
    seg_len, cum = [], [0.0]
    for i in range(1, len(xy)):
        d = math.dist(xy[i - 1], xy[i])
        seg_len.append(d)
        cum.append(cum[-1] + d)
    total = cum[-1]

    targets = [k * spacing_m for k in range(int(total // spacing_m) + 1)]
    if total - targets[-1] > keep_tail_frac * spacing_m:
        targets.append(total)

    # interpolate xy at each target arclength
    out_xy, j = [], 0
    for s in targets:
        while j < len(seg_len) - 1 and cum[j + 1] < s:
            j += 1
        t = 0.0 if seg_len[j] == 0 else (s - cum[j]) / seg_len[j]
        x = xy[j][0] + t * (xy[j + 1][0] - xy[j][0])
        y = xy[j][1] + t * (xy[j + 1][1] - xy[j][1])
        out_xy.append((x, y))

    out = []
    for i, (x, y) in enumerate(out_xy):
        if i < len(out_xy) - 1:
            dx = out_xy[i + 1][0] - x
            dy = out_xy[i + 1][1] - y
            brg = bearing_from_delta(dx, dy)
        else:
            brg = out[-1]["bearing_deg"] if out else 0.0
        la, lo = offset_latlon(lat0, lon0, x, y)
        out.append({"lat": la, "lon": lo, "s_m": targets[i],
                    "bearing_deg": brg})
    return out


# --------------------------------------------------------------------------
# Image operations (teach-phase geometry)
# --------------------------------------------------------------------------
def _resampling(name: str):
    """Pillow >= 9.1 moved filters into Image.Resampling."""
    return getattr(getattr(Image, "Resampling", Image), name)


def rotate_and_crop(img: Image.Image, bearing_deg: float, crop_px: int,
                    rotation_margin: float = ROTATION_MARGIN) -> Image.Image:
    """Rotate a north-up source image so that `bearing` points 'up', then
    center-crop the footprint square of `crop_px` pixels.

    A positive compass bearing (clockwise from north) requires a
    counterclockwise image rotation by the same angle; PIL's rotate() is
    counterclockwise for positive angles. The rotation is applied to an
    intermediate square sqrt(2) larger than the final crop so that no
    undefined corners can enter the footprint.
    """
    w, h = img.size
    rot_px = min(int(math.ceil(crop_px * rotation_margin)), w, h)
    img = center_crop(img, rot_px)
    img = img.rotate(bearing_deg, resample=_resampling("BICUBIC"))
    return center_crop(img, crop_px)


def center_crop(img: Image.Image, edge_px: int) -> Image.Image:
    w, h = img.size
    left = (w - edge_px) // 2
    top = (h - edge_px) // 2
    return img.crop((left, top, left + edge_px, top + edge_px))


# --------------------------------------------------------------------------
# Sensor preprocessing (Gaffin et al. 2015) -- SHARED by both phases
# --------------------------------------------------------------------------
def circular_mask(n: int) -> np.ndarray:
    """Boolean mask of the inscribed circle of an n x n matrix.

    Replicates the paper's circularization (50 x 50 -> ~1963 of 2500 px kept)
    to remove corner pixels before rotational comparisons.
    """
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    return (xx - c) ** 2 + (yy - c) ** 2 <= (n / 2.0) ** 2


def quantize(gray_u8: np.ndarray, levels: int) -> np.ndarray:
    """Quantize an 8-bit grayscale array to `levels` gray levels.

    levels=2 corresponds to the paper's black/white sensor (threshold at 128,
    i.e. at the median after histogram equalization), levels=10 and
    levels=100 to its 10- and 100-gray-level sensors. Values are mapped back
    to bin centers in 0..255 for storage/inspection; the number of distinct
    values equals `levels`.
    """
    if levels < 2 or levels > 256:
        raise ValueError("levels must be in [2, 256]")
    q = np.floor(gray_u8.astype(np.float32) * levels / 256.0)
    q = np.clip(q, 0, levels - 1)
    return np.clip(np.round((q + 0.5) * (256.0 / levels)), 0,
                   255).astype(np.uint8)


def preprocess_view(img: Image.Image, sensor_px: int, gray_levels: int,
                    apply_mask: bool = True) -> np.ndarray:
    """Full Gaffin et al. sensor chain. MUST be identical in both phases.

    img : footprint view (teach: rotated satellite crop; repeat: raw
          DownCamera frame). Any size, any mode.
    Returns an (sensor_px, sensor_px) uint8 array; pixels outside the
    circular mask are set to 0 and must be EXCLUDED from any comparison via
    circular_mask(sensor_px).
    """
    g = ImageOps.grayscale(img)          # ITU-R 601 luma
    g = ImageOps.equalize(g)             # per-scene histeq, as in the paper
    g = g.resize((sensor_px, sensor_px), _resampling("BOX"))  # area average
    arr = quantize(np.asarray(g), gray_levels)
    if apply_mask:
        arr = arr.copy()
        arr[~circular_mask(sensor_px)] = 0
    return arr


def scene_difference(a: np.ndarray, b: np.ndarray,
                     mask: np.ndarray | None = None) -> int:
    """Paper's image difference score: sum of absolute pixel differences
    inside the circular mask. Use this exact metric in the repeat-phase RIDF
    so teach and repeat share one comparator."""
    if mask is None:
        mask = circular_mask(a.shape[0])
    return int(np.abs(a.astype(np.int32) - b.astype(np.int32))[mask].sum())
