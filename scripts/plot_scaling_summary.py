import argparse
import csv
import json
import math
import os
from typing import Iterable, List, Tuple


def _load_points_from_summary(path: str) -> List[Tuple[float, float]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    points = []
    for item in data.get("estimated_elo_by_time", []):
        points.append((float(item["time"]), float(item["elo"])))
    return points


def _load_points_from_csv(path: str) -> List[Tuple[float, float]]:
    points = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            points.append((float(row["time"]), float(row["elo"])))
    return points


def _write_svg(
    points: Iterable[Tuple[float, float]],
    out_path: str,
    title: str,
    x_label: str,
    y_label: str,
    width: int,
    height: int,
    log2_x: bool,
) -> None:
    points = sorted(points, key=lambda p: p[0])
    if not points:
        raise ValueError("No points to plot.")

    left = 70
    right = 30
    top = 50
    bottom = 60
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x_value(t: float) -> float:
        return math.log2(t) if log2_x else t

    xs = [x_value(t) for t, _ in points]
    ys = [elo for _, elo in points]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    y_pad = max(15.0, (max_y - min_y) * 0.08)
    min_y -= y_pad
    max_y += y_pad

    def map_x(t: float) -> float:
        return left + (x_value(t) - min_x) / (max_x - min_x) * plot_w

    def map_y(y: float) -> float:
        return top + (max_y - y) / (max_y - min_y) * plot_h

    y_ticks = 5
    y_tick_values = [
        min_y + i * (max_y - min_y) / (y_ticks - 1) for i in range(y_ticks)
    ]

    x_tick_values = [t for t, _ in points]

    polyline = " ".join(f"{map_x(t):.1f},{map_y(y):.1f}" for t, y in points)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b0f14"/>',
        f'<text x="{width/2:.1f}" y="28" fill="#e8edf2" font-size="18" font-family="Arial" text-anchor="middle">{title}</text>',
        f'<text x="{width/2:.1f}" y="{height-16}" fill="#c7d1db" font-size="12" font-family="Arial" text-anchor="middle">{x_label}</text>',
        f'<text x="16" y="{height/2:.1f}" fill="#c7d1db" font-size="12" font-family="Arial" text-anchor="middle" transform="rotate(-90 16,{height/2:.1f})">{y_label}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#111722" stroke="#2a3542" stroke-width="1"/>',
    ]

    for y in y_tick_values:
        y_pos = map_y(y)
        lines.append(
            f'<line x1="{left}" y1="{y_pos:.1f}" x2="{left+plot_w}" y2="{y_pos:.1f}" stroke="#1b2330" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{left-10}" y="{y_pos+4:.1f}" fill="#95a3b3" font-size="11" font-family="Arial" text-anchor="end">{y:.0f}</text>'
        )

    for t in x_tick_values:
        x_pos = map_x(t)
        lines.append(
            f'<line x1="{x_pos:.1f}" y1="{top}" x2="{x_pos:.1f}" y2="{top+plot_h}" stroke="#1b2330" stroke-width="1"/>'
        )
        label = f"{t:g}s"
        lines.append(
            f'<text x="{x_pos:.1f}" y="{top+plot_h+18}" fill="#95a3b3" font-size="11" font-family="Arial" text-anchor="middle">{label}</text>'
        )

    lines.append(f'<polyline fill="none" stroke="#5dd3a0" stroke-width="2.5" points="{polyline}"/>')
    for t, y in points:
        lines.append(
            f'<circle cx="{map_x(t):.1f}" cy="{map_y(y):.1f}" r="4" fill="#5dd3a0" stroke="#0b0f14" stroke-width="1"/>'
        )

    lines.append("</svg>")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot scaling summary as an SVG line chart.")
    parser.add_argument("--summary", help="Path to scaling summary.json")
    parser.add_argument("--csv", help="CSV with columns: time, elo")
    parser.add_argument("--out", required=True, help="Output SVG path")
    parser.add_argument("--title", default="Search-Time Scaling", help="Chart title")
    parser.add_argument("--x-label", default="Time per move (seconds)", help="X-axis label")
    parser.add_argument("--y-label", default="Estimated Elo", help="Y-axis label")
    parser.add_argument("--width", type=int, default=960, help="SVG width")
    parser.add_argument("--height", type=int, default=540, help="SVG height")
    parser.add_argument("--linear-x", action="store_true", help="Use linear x scale (default is log2)")
    args = parser.parse_args()

    if bool(args.summary) == bool(args.csv):
        raise SystemExit("Provide exactly one of --summary or --csv.")

    if args.summary:
        points = _load_points_from_summary(args.summary)
    else:
        points = _load_points_from_csv(args.csv)

    _write_svg(
        points=points,
        out_path=args.out,
        title=args.title,
        x_label=args.x_label,
        y_label=args.y_label,
        width=args.width,
        height=args.height,
        log2_x=not args.linear_x,
    )
    print(f"Wrote {args.out} ({len(points)} points)")


if __name__ == "__main__":
    main()
