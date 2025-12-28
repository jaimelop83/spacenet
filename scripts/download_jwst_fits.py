#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

from astroquery.mast import Observations


def parse_args():
    parser = argparse.ArgumentParser(description="Download JWST calibrated FITS data from MAST.")
    parser.add_argument("--out-dir", required=True, help="Output directory for downloads.")
    parser.add_argument("--max-files", type=int, default=500)
    parser.add_argument("--max-obs", type=int, default=2000)
    parser.add_argument(
        "--instruments",
        nargs="+",
        default=["NIRCAM", "MIRI"],
        help="JWST instruments to include.",
    )
    parser.add_argument("--calib-level", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    instrument_map = {
        "NIRCAM": "NIRCAM/IMAGE",
        "MIRI": "MIRI/IMAGE",
        "NIRISS": "NIRISS/IMAGE",
        "FGS": "FGS/FGS1",
    }
    instruments = []
    for inst in args.instruments:
        key = inst.upper()
        instruments.append(instrument_map.get(key, inst))

    obs = Observations.query_criteria(
        obs_collection="JWST",
        dataproduct_type="image",
        instrument_name=instruments,
        calib_level=args.calib_level,
    )
    if len(obs) == 0:
        raise SystemExit("No JWST observations found for the given criteria.")

    if len(obs) > args.max_obs:
        obs = obs[: args.max_obs]

    products = Observations.get_product_list(obs)
    if len(products) == 0:
        raise SystemExit("No products found for the selected observations.")

    mask_ext = [
        str(name).lower().endswith((".fits", ".fits.gz"))
        for name in products["productFilename"]
    ]
    products = products[mask_ext]
    if len(products) == 0:
        raise SystemExit("No FITS products found.")

    if len(products) > args.max_files:
        products = products[: args.max_files]

    if args.dry_run:
        print(f"Would download {len(products)} FITS files to {out_dir}")
        return

    results = Observations.download_products(products, download_dir=str(out_dir))

    manifest_path = out_dir / "jwst_fits_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["productFilename", "productType", "dataURI", "local_path"])
        for row in results:
            writer.writerow(
                [
                    row.get("productFilename", ""),
                    row.get("productType", ""),
                    row.get("dataURI", ""),
                    row.get("Local Path", ""),
                ]
            )

    print(f"Downloaded {len(results)} FITS files to {out_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
