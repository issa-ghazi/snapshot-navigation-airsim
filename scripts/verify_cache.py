r"""verify_cache.py -- find and remove corrupt images in the WMS cache.

A truncated WMS response still has a valid PNG header, so PIL opens it
without complaining and only fails later, deep inside rotate_and_crop. This
script decodes every cached image fully and reports the broken ones.

With --delete the bad files are removed; extract_route_snapshots.py then
re-fetches exactly those on the next run (everything else is skipped, so a
restart is cheap).

Usage:
    python verify_cache.py --cache data\teach_rgb\_cache_dop
    python verify_cache.py --cache data\teach_rgb\_cache_dop --delete
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="cache directory")
    ap.add_argument("--delete", action="store_true",
                    help="delete the corrupt files (they are re-fetched "
                         "automatically on the next run)")
    args = ap.parse_args()

    cache = Path(args.cache)
    if not cache.is_dir():
        raise SystemExit(f"not a directory: {cache}")

    files = sorted(cache.glob("*.png"))
    leftovers = sorted(cache.glob("*.part"))
    print(f"checking {len(files)} images in {cache}")

    bad = []
    for i, f in enumerate(files, 1):
        try:
            with Image.open(f) as im:
                im.load()                 # forces full decode
        except Exception as e:
            bad.append((f, f"{type(e).__name__}: {e}"))
        if i % 200 == 0:
            print(f"  {i}/{len(files)} ... {len(bad)} defekt")

    print()
    if leftovers:
        print(f"{len(leftovers)} unfinished .part files (safe to delete):")
        for f in leftovers[:10]:
            print(f"  {f.name}")

    if not bad:
        print("all images decode cleanly")
    else:
        print(f"{len(bad)} corrupt image(s):")
        for f, err in bad[:20]:
            print(f"  {f.name}  ({f.stat().st_size} bytes)  {err}")
        if len(bad) > 20:
            print(f"  ... and {len(bad) - 20} more")

    if args.delete:
        for f, _ in bad:
            f.unlink()
        for f in leftovers:
            f.unlink()
        n = len(bad) + len(leftovers)
        print(f"\ndeleted {n} file(s) -- re-run extract_route_snapshots.py "
              f"with the same arguments to re-fetch them")
    elif bad or leftovers:
        print("\nre-run with --delete to remove them")


if __name__ == "__main__":
    main()
