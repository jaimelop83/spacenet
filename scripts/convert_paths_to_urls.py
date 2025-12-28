#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert local paths to HTTP URLs in a CSV.")
    parser.add_argument("--in-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--root", required=True, help="Filesystem root to replace.")
    parser.add_argument("--base-url", required=True, help="Base URL for root.")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    base_url = args.base_url.rstrip("/")

    with open(args.in_csv, newline="") as f_in, open(args.out_csv, "w", newline="") as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        for row in reader:
            if not row or row[0].startswith("#") or row[0] == "path":
                writer.writerow(row)
                continue
            path = Path(row[0]).resolve()
            try:
                rel = path.relative_to(root)
                row[0] = f"{base_url}/{rel.as_posix()}"
            except ValueError:
                pass
            writer.writerow(row)

    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
