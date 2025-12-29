#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build OOD example gallery from scores CSV.")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--out-dir", default="/home/jaimelop/spacenet/reports/ood_examples")
    parser.add_argument("--top-n", type=int, default=60)
    parser.add_argument("--copy", action="store_true", help="Copy images instead of symlinking.")
    parser.add_argument(
        "--html-prefix",
        default="ood_examples",
        help="Prefix for generated HTML files (default: ood_examples).",
    )
    parser.add_argument(
        "--include-pattern",
        default="",
        help="Only include files whose names contain this substring (case-insensitive).",
    )
    parser.add_argument(
        "--exclude-pattern",
        default="rate,uncal,rateints,trapsfilled",
        help="Comma-separated substrings to exclude (case-insensitive).",
    )
    return parser.parse_args()


def load_scores(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#") or row[0] == "path":
                continue
            path = row[0]
            score = float(row[1])
            pred_class = row[2]
            ood_pred = row[3]
            rows.append((path, score, pred_class, ood_pred))
    return rows


def safe_link(src, dest, copy=False):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        return
    if copy:
        dest.write_bytes(Path(src).read_bytes())
    else:
        os.symlink(src, dest)


def write_html(items, out_path, title):
    cards = []
    for item in items:
        cards.append(
            f"<article class='card'><div class='thumb'><img src='{item['rel']}'></div>"
            f"<div class='meta'><h3>{item['name']}</h3><p>{item['score']}</p></div></article>"
        )
    options = "".join(sorted({f"<option value='{item['name']}'>{item['name']}</option>" for item in items}))
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; background: #f7f8fb; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }}
    .card {{ background: #fff; border-radius: 10px; border: 1px solid #e8ecf2; overflow: hidden; }}
    .thumb {{ padding: 10px; background: #f0f4ff; }}
    .thumb img {{ width: 100%; height: auto; border-radius: 6px; }}
    .meta {{ padding: 10px 12px 14px; }}
    .meta h3 {{ margin: 0 0 6px; font-size: 14px; }}
    .meta p {{ margin: 0; font-size: 12px; color: #5b6475; }}
    .top {{ display: flex; gap: 12px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }}
    .top a {{ font-size: 12px; text-decoration: none; color: #2563eb; }}
    select {{ padding: 6px 8px; border-radius: 6px; border: 1px solid #d7dce5; font-size: 12px; }}
  </style>
</head>
<body>
  <div class="top">
    <h1 style="margin:0">{title}</h1>
    <a href="ood_gallery.html">Back to histograms</a>
    <label>
      <select id="classFilter">
        <option value="">All classes</option>
        {options}
      </select>
    </label>
  </div>
  <div class="grid">
    {''.join(cards)}
  </div>
  <script>
    const select = document.getElementById('classFilter');
    select.addEventListener('change', () => {{
      const value = select.value;
      document.querySelectorAll('.card').forEach(card => {{
        const name = card.querySelector('h3').textContent;
        card.style.display = (value === '' || name === value) ? '' : 'none';
      }});
    }});
  </script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main():
    args = parse_args()
    rows = load_scores(args.csv_path)
    if not rows:
        raise SystemExit("No scores found.")

    include = args.include_pattern.strip().lower()
    exclude = [s.strip().lower() for s in args.exclude_pattern.split(",") if s.strip()]

    def allowed(path):
        name = Path(path).name.lower()
        if include and include not in name:
            return False
        for bad in exclude:
            if bad and bad in name:
                return False
        return True

    ood_rows = [r for r in rows if r[3] == "OOD"]
    id_rows = [r for r in rows if r[3] == "ID"]
    ood_rows.sort(key=lambda r: r[1])
    id_rows.sort(key=lambda r: r[1], reverse=True)

    out_dir = Path(args.out_dir)
    ood_dir = out_dir / "ood"
    id_dir = out_dir / "id"
    ood_dir.mkdir(parents=True, exist_ok=True)
    id_dir.mkdir(parents=True, exist_ok=True)

    ood_items = []
    skipped_missing = 0
    for path, score, pred_class, _ in ood_rows[: args.top_n]:
        if not allowed(path):
            continue
        src = Path(path)
        if not src.exists():
            skipped_missing += 1
            continue
        dest = ood_dir / src.name
        safe_link(src, dest, copy=args.copy)
        ood_items.append(
            {
                "rel": dest.relative_to(out_dir.parent),
                "name": pred_class,
                "score": f"score={score:.4f}",
            }
        )

    id_items = []
    for path, score, pred_class, _ in id_rows[: args.top_n]:
        if not allowed(path):
            continue
        src = Path(path)
        if not src.exists():
            skipped_missing += 1
            continue
        dest = id_dir / src.name
        safe_link(src, dest, copy=args.copy)
        id_items.append(
            {
                "rel": dest.relative_to(out_dir.parent),
                "name": pred_class,
                "score": f"score={score:.4f}",
            }
        )

    ood_html = out_dir.parent / f"{args.html_prefix}_ood.html"
    id_html = out_dir.parent / f"{args.html_prefix}_id.html"
    write_html(ood_items, ood_html, "OOD Examples")
    write_html(id_items, id_html, "ID Examples")
    print(f"wrote {ood_html}")
    print(f"wrote {id_html}")
    if skipped_missing:
        print(f"skipped_missing={skipped_missing}")


if __name__ == "__main__":
    main()
