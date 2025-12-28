#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Create RGB composites from multi-filter FITS.")
    parser.add_argument("--r", help="Red channel FITS")
    parser.add_argument("--g", help="Green channel FITS")
    parser.add_argument("--b", help="Blue channel FITS")
    parser.add_argument("--out", help="Output PNG path")
    parser.add_argument("--pmin", type=float, default=1.0, help="Lower percentile clip")
    parser.add_argument("--pmax", type=float, default=99.0, help="Upper percentile clip")
    parser.add_argument("--stretch", choices=["linear", "asinh"], default="asinh")
    parser.add_argument("--scale", type=float, default=3.0, help="Asinh stretch scale")
    parser.add_argument("--triples-csv", default=None, help="CSV with r,g,b,out columns")
    return parser.parse_args()


def load_fits(path):
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is None:
                continue
            data = hdu.data
            if data.ndim >= 2:
                arr = data.astype(np.float32)
                if arr.ndim > 2:
                    arr = arr[0]
                return arr
    raise ValueError(f"No image data found in {path}")


def normalize_channel(arr, pmin, pmax):
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(arr, [pmin, pmax])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    arr = np.clip(arr, lo, hi)
    return (arr - lo) / (hi - lo)


def stretch_channel(arr, mode, scale):
    if mode == "asinh":
        return np.arcsinh(arr * scale) / np.arcsinh(scale)
    return arr


def make_rgb(r_path, g_path, b_path, out_path, pmin, pmax, stretch, scale):
    r = load_fits(r_path)
    g = load_fits(g_path)
    b = load_fits(b_path)

    r = stretch_channel(normalize_channel(r, pmin, pmax), stretch, scale)
    g = stretch_channel(normalize_channel(g, pmin, pmax), stretch, scale)
    b = stretch_channel(normalize_channel(b, pmin, pmax), stretch, scale)

    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(rgb).save(out_path)


def run_single(args):
    if not (args.r and args.g and args.b and args.out):
        raise SystemExit("Provide --r --g --b --out, or use --triples-csv.")
    make_rgb(args.r, args.g, args.b, args.out, args.pmin, args.pmax, args.stretch, args.scale)
    print(f"wrote {args.out}")


def run_csv(args):
    with open(args.triples_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = row.get("r")
            g = row.get("g")
            b = row.get("b")
            out = row.get("out")
            if not (r and g and b and out):
                raise SystemExit("CSV must include r,g,b,out columns.")
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            make_rgb(r, g, b, out, args.pmin, args.pmax, args.stretch, args.scale)
            print(f"wrote {out}")


def main():
    args = parse_args()
    if args.triples_csv:
        run_csv(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
