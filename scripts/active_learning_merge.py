#!/usr/bin/env python3
import argparse
import csv
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Merge labeled samples into a dataset.")
    parser.add_argument("--labels-csv", required=True, help="CSV with path,label columns.")
    parser.add_argument("--out-root", required=True, help="Output dataset root.")
    parser.add_argument("--copy", action="store_true", help="Copy images instead of symlinking.")
    return parser.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = 0
    skipped = 0
    with open(args.labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        if "path" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise SystemExit("labels CSV must include path,label columns.")
        for row in reader:
            path = row["path"].strip()
            label = row["label"].strip()
            if not path or not label:
                skipped += 1
                continue
            src = Path(path)
            if not src.exists():
                skipped += 1
                continue
            dest_dir = out_root / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / src.name
            if dest.exists() or dest.is_symlink():
                rows += 1
                continue
            if args.copy:
                shutil.copy2(src, dest)
            else:
                dest.symlink_to(src)
            rows += 1

    print(f"merged={rows} skipped={skipped} out_root={out_root}")


if __name__ == "__main__":
    main()
