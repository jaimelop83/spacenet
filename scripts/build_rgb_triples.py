#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build RGB triples CSV from JWST FITS files.")
    parser.add_argument("--fits-root", required=True, help="Root directory containing FITS files.")
    parser.add_argument("--out-csv", required=True, help="Output CSV path.")
    parser.add_argument(
        "--filters",
        default="F090W,F200W,F444W",
        help="Comma-separated filters for B,G,R channels (order matters).",
    )
    parser.add_argument("--limit", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.fits_root)
    if not root.exists():
        raise SystemExit(f"Missing fits root: {root}")

    filters = [f.strip().upper() for f in args.filters.split(",") if f.strip()]
    if len(filters) != 3:
        raise SystemExit("Provide exactly three filters, e.g. F090W,F200W,F444W.")

    pattern = re.compile(r"(_(F\\d{3}W|F\\d{3}M))", re.IGNORECASE)

    files = list(root.rglob("*.fits")) + list(root.rglob("*.fits.gz"))
    groups = {}
    for path in files:
        name = path.name.upper()
        match = pattern.search(name)
        if not match:
            continue
        filt = match.group(2)
        stem = name.replace(match.group(1), "")
        key = (stem, path.parent)
        groups.setdefault(key, {})[filt] = path

    triples = []
    for (_, _), mapping in groups.items():
        if all(f in mapping for f in filters):
            triples.append((mapping[filters[2]], mapping[filters[1]], mapping[filters[0]]))
        if len(triples) >= args.limit:
            break

    if not triples:
        raise SystemExit("No RGB triples found with the given filters.")

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["r", "g", "b", "out"])
        for r, g, b in triples:
            out_path = out.parent / f"{r.stem}_rgb.png"
            writer.writerow([str(r), str(g), str(b), str(out_path)])

    print(f"wrote {out} with {len(triples)} triples")


if __name__ == "__main__":
    main()
