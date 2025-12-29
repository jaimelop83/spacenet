#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from astropy.io import fits


def parse_args():
    parser = argparse.ArgumentParser(description="Convert FITS files to PNG previews.")
    parser.add_argument("--in-root", required=True, help="Root folder containing FITS files.")
    parser.add_argument("--out-root", required=True, help="Output folder for PNG files.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional limit on number of files.")
    parser.add_argument(
        "--include-pattern",
        default=None,
        help="Only convert files whose paths contain this substring.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PNGs.")
    return parser.parse_args()


def normalize_to_uint8(arr):
    arr = np.array(arr)
    if arr.ndim > 2:
        arr = arr[0]
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(arr, (1.0, 99.0))
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        arr = np.zeros_like(arr, dtype=np.float32)
    else:
        arr = (arr - lo) / (hi - lo)
        arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def main():
    args = parse_args()
    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    fits_paths = []
    for path in in_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".fits", ".fit", ".fts"} or path.name.lower().endswith(".fits.gz"):
            if args.include_pattern and args.include_pattern not in str(path):
                continue
            fits_paths.append(path)

    if not fits_paths:
        print(f"No FITS files found under {in_root}.")
        return

    if args.max_files is not None:
        fits_paths = fits_paths[: args.max_files]

    converted = 0
    skipped = 0
    for path in fits_paths:
        rel = path.relative_to(in_root)
        out_path = out_root / rel
        out_path = out_path.with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            data = fits.getdata(path, memmap=False)
            if data is None:
                skipped += 1
                continue
            arr = normalize_to_uint8(data)
            img = Image.fromarray(arr, mode="L").convert("RGB")
            img.save(out_path)
            converted += 1
        except Exception:
            skipped += 1

    print(f"Converted {converted} files to {out_root}")
    print(f"Skipped {skipped} files")


if __name__ == "__main__":
    main()
