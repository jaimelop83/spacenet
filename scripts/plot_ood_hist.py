#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(description="Plot OOD score histogram without matplotlib.")
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--bins", type=int, default=50)
    return parser.parse_args()


def load_scores(csv_path):
    id_scores = []
    ood_scores = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#") or row[0] == "path":
                continue
            score = float(row[1])
            ood_pred = row[3].strip()
            if ood_pred == "ID":
                id_scores.append(score)
            elif ood_pred == "OOD":
                ood_scores.append(score)
    return id_scores, ood_scores


def hist_counts(values, bins, vmin, vmax):
    counts = [0] * bins
    if vmax <= vmin:
        return counts
    scale = bins / (vmax - vmin)
    for v in values:
        idx = int((v - vmin) * scale)
        if idx < 0:
            idx = 0
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1
    return counts


def main():
    args = parse_args()
    csv_path = Path(args.csv_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    id_scores, ood_scores = load_scores(csv_path)
    if not id_scores and not ood_scores:
        raise SystemExit("No scores found.")

    all_scores = id_scores + ood_scores
    vmin, vmax = min(all_scores), max(all_scores)
    bins = args.bins
    id_counts = hist_counts(id_scores, bins, vmin, vmax)
    ood_counts = hist_counts(ood_scores, bins, vmin, vmax)
    max_count = max(max(id_counts), max(ood_counts), 1)

    width, height = 800, 400
    margin_left, margin_right = 60, 20
    margin_top, margin_bottom = 30, 50
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    bar_w = plot_w / bins

    img = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # Axes
    x0 = margin_left
    y0 = height - margin_bottom
    draw.line([(x0, margin_top), (x0, y0), (width - margin_right, y0)], fill=(0, 0, 0))

    # Bars
    for i in range(bins):
        x_left = x0 + i * bar_w
        x_right = x_left + bar_w - 1
        id_h = int((id_counts[i] / max_count) * plot_h)
        ood_h = int((ood_counts[i] / max_count) * plot_h)
        if id_h:
            draw.rectangle(
                [x_left, y0 - id_h, x_right, y0],
                fill=(66, 133, 244, 120),
                outline=None,
            )
        if ood_h:
            draw.rectangle(
                [x_left, y0 - ood_h, x_right, y0],
                fill=(219, 68, 55, 120),
                outline=None,
            )

    # Labels
    draw.text((margin_left, 5), "OOD Score Distribution", fill=(0, 0, 0), font=font)
    draw.text((margin_left, height - 20), f"min={vmin:.3f} max={vmax:.3f}", fill=(0, 0, 0), font=font)
    draw.rectangle([width - 180, margin_top, width - 160, margin_top + 10], fill=(66, 133, 244, 120))
    draw.text((width - 155, margin_top), "ID", fill=(0, 0, 0), font=font)
    draw.rectangle([width - 180, margin_top + 15, width - 160, margin_top + 25], fill=(219, 68, 55, 120))
    draw.text((width - 155, margin_top + 15), "OOD", fill=(0, 0, 0), font=font)

    img.convert("RGB").save(out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
