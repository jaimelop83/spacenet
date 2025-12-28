#!/usr/bin/env python3
from pathlib import Path


def main():
    root = Path("/home/jaimelop/spacenet/reports/ood_per_class")
    out = Path("/home/jaimelop/spacenet/reports/ood_gallery.html")
    if not root.exists():
        raise SystemExit(f"Missing directory: {root}")

    images = sorted(p for p in root.glob("*.png"))
    if not images:
        raise SystemExit("No PNGs found in ood_per_class.")

    items = []
    for img in images:
        name = img.stem.replace("_", " ")
        rel = img.relative_to(out.parent)
        items.append(f"<figure><img src='{rel}'><figcaption>{name}</figcaption></figure>")

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>OOD Per-Class Gallery</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; }}
    figure {{ margin: 0; }}
    img {{ width: 100%; height: auto; border: 1px solid #ddd; }}
    figcaption {{ text-align: center; font-size: 12px; margin-top: 6px; }}
  </style>
</head>
<body>
  <h1>OOD Per-Class Score Histograms</h1>
  <div class="grid">
    {''.join(items)}
  </div>
</body>
</html>
"""

    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
