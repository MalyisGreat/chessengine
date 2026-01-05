import argparse
import csv
import math
import os
from typing import Iterable, List, Tuple


def _load_points(path: str):
    train = []
    val = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = float(row["step"])
            if row.get("train_loss"):
                train.append((step, float(row["train_loss"])))
            if row.get("val_loss"):
                val.append((step, float(row["val_loss"])))
    return train, val


def _write_svg(
    train: Iterable[Tuple[float, float]],
    val: Iterable[Tuple[float, float]],
    out_path: str,
    title: str,
    width: int,
    height: int,
):
    train = sorted(train, key=lambda p: p[0])
    val = sorted(val, key=lambda p: p[0])
    points = train + val
    if not points:
        raise ValueError("No points to plot.")

    left = 70
    right = 30
    top = 50
    bottom = 60
    plot_w = width - left - right
    plot_h = height - top - bottom

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    y_pad = max(0.0005, (max_y - min_y) * 0.08)
    min_y -= y_pad
    max_y += y_pad

    def map_x(x: float) -> float:
        return left + (x - min_x) / (max_x - min_x) * plot_w

    def map_y(y: float) -> float:
        return top + (max_y - y) / (max_y - min_y) * plot_h

    y_ticks = 5
    y_tick_values = [
        min_y + i * (max_y - min_y) / (y_ticks - 1) for i in range(y_ticks)
    ]

    x_ticks = 5
    x_tick_values = [
        min_x + i * (max_x - min_x) / (x_ticks - 1) for i in range(x_ticks)
    ]

    def polyline(points: List[Tuple[float, float]]) -> str:
        return " ".join(f"{map_x(x):.1f},{map_y(y):.1f}" for x, y in points)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b0f14"/>',
        f'<text x="{width/2:.1f}" y="28" fill="#e8edf2" font-size="18" font-family="Arial" text-anchor="middle">{title}</text>',
        f'<text x="{width/2:.1f}" y="{height-16}" fill="#c7d1db" font-size="12" font-family="Arial" text-anchor="middle">Step</text>',
        f'<text x="16" y="{height/2:.1f}" fill="#c7d1db" font-size="12" font-family="Arial" text-anchor="middle" transform="rotate(-90 16,{height/2:.1f})">Loss</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#111722" stroke="#2a3542" stroke-width="1"/>',
    ]

    for y in y_tick_values:
        y_pos = map_y(y)
        lines.append(
            f'<line x1="{left}" y1="{y_pos:.1f}" x2="{left+plot_w}" y2="{y_pos:.1f}" stroke="#1b2330" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{left-10}" y="{y_pos+4:.1f}" fill="#95a3b3" font-size="11" font-family="Arial" text-anchor="end">{y:.4f}</text>'
        )

    for x in x_tick_values:
        x_pos = map_x(x)
        lines.append(
            f'<line x1="{x_pos:.1f}" y1="{top}" x2="{x_pos:.1f}" y2="{top+plot_h}" stroke="#1b2330" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x_pos:.1f}" y="{top+plot_h+18}" fill="#95a3b3" font-size="11" font-family="Arial" text-anchor="middle">{int(x)}</text>'
        )

    if train:
        lines.append(f'<polyline fill="none" stroke="#5dd3a0" stroke-width="2.5" points="{polyline(train)}"/>')
        lines.append(
            f'<text x="{left+10}" y="{top-16}" fill="#5dd3a0" font-size="12" font-family="Arial">Train</text>'
        )
    if val:
        lines.append(f'<polyline fill="none" stroke="#f4c97a" stroke-width="2" points="{polyline(val)}"/>')
        lines.append(
            f'<text x="{left+70}" y="{top-16}" fill="#f4c97a" font-size="12" font-family="Arial">Val</text>'
        )

    lines.append("</svg>")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot train/val loss curves as an SVG chart.")
    parser.add_argument("--csv", required=True, help="CSV with columns: epoch, step, train_loss, val_loss")
    parser.add_argument("--out", required=True, help="Output SVG path")
    parser.add_argument("--title", default="Loss Curve", help="Chart title")
    parser.add_argument("--width", type=int, default=960, help="SVG width")
    parser.add_argument("--height", type=int, default=540, help="SVG height")
    args = parser.parse_args()

    train, val = _load_points(args.csv)
    _write_svg(train, val, args.out, args.title, args.width, args.height)
    print(f"Wrote {args.out} (train={len(train)}, val={len(val)})")


if __name__ == "__main__":
    main()
