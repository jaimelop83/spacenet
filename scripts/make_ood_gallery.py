#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime


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
        name = img.stem.replace("_", " ").title()
        rel = img.relative_to(out.parent)
        items.append(
            f"<article class='card'><div class='thumb'><img src='{rel}' alt='{name}'></div>"
            f"<div class='meta'><h3>{name}</h3><p>Score distribution</p></div></article>"
        )

    updated = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OOD Per-Class Gallery</title>
  <style>
    :root {{
      --ink: #1b1f2a;
      --muted: #5a6375;
      --card: #ffffff;
      --accent: #0ea5e9;
      --accent-2: #22c55e;
      --bg: #f3f4f7;
      --shadow: 0 12px 30px rgba(0, 0, 0, 0.08);
    }}
    body {{
      margin: 0;
      font-family: "Palatino Linotype", "Book Antiqua", Palatino, serif;
      color: var(--ink);
      background: radial-gradient(circle at 10% 10%, #e7efff, transparent 45%),
                  radial-gradient(circle at 90% 15%, #eaf7f0, transparent 40%),
                  var(--bg);
    }}
    header {{
      padding: 32px 28px 12px;
    }}
    .title {{
      font-size: 28px;
      margin: 0 0 6px;
    }}
    .subtitle {{
      margin: 0;
      color: var(--muted);
      font-size: 14px;
    }}
    .stats {{
      margin-top: 16px;
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
    }}
    .chip {{
      background: #fff;
      border: 1px solid #e2e6ef;
      border-radius: 999px;
      padding: 6px 12px;
      font-size: 12px;
      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
    }}
    main {{
      padding: 8px 28px 40px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 18px;
    }}
    .card {{
      background: var(--card);
      border-radius: 14px;
      box-shadow: var(--shadow);
      overflow: hidden;
      border: 1px solid #eef1f6;
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .card:hover {{
      transform: translateY(-4px);
      box-shadow: 0 18px 36px rgba(0, 0, 0, 0.12);
    }}
    .thumb {{
      background: linear-gradient(135deg, rgba(14, 165, 233, 0.12), rgba(34, 197, 94, 0.12));
      padding: 12px;
    }}
    .thumb img {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 8px;
      border: 1px solid #e6ebf5;
      background: #fff;
    }}
    .meta {{
      padding: 12px 14px 16px;
    }}
    .meta h3 {{
      margin: 0 0 4px;
      font-size: 16px;
    }}
    .meta p {{
      margin: 0;
      font-size: 12px;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <header>
    <h1 class="title">OOD Per-Class Score Histograms</h1>
    <p class="subtitle">JWST preview images scored against SpaceNet model.</p>
    <div class="stats">
      <span class="chip">Classes: {len(images)}</span>
      <span class="chip">Updated: {updated}</span>
      <span class="chip">Theme: OOD Preview</span>
    </div>
  </header>
  <main>
    <div class="grid">
      {''.join(items)}
    </div>
  </main>
</body>
</html>
"""

    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
