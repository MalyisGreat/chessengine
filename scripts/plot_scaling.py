#!/usr/bin/env python3
"""Generate SVG plot from scaling CSV data."""

import argparse
import csv
from pathlib import Path

def generate_svg(times, elos, output_path, title="Search-Time Scaling",
                 extra_times=None, extra_elos=None, extra_label=None):
    """Generate an SVG plot of Elo vs search time."""
    # SVG dimensions
    width = 600
    height = 400
    margin = {"top": 40, "right": 30, "bottom": 60, "left": 70}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]

    # Data ranges (include extra points if present)
    all_times = times + (extra_times or [])
    all_elos = elos + (extra_elos or [])
    min_time = min(all_times)
    max_time = max(all_times)
    min_elo = min(all_elos) - 50
    max_elo = max(all_elos) + 50

    # Scale functions (log scale for time)
    import math
    def scale_x(t):
        log_min = math.log10(min_time)
        log_max = math.log10(max_time)
        return margin["left"] + (math.log10(t) - log_min) / (log_max - log_min) * plot_width

    def scale_y(e):
        return margin["top"] + plot_height - (e - min_elo) / (max_elo - min_elo) * plot_height

    # Build SVG
    svg_parts = []
    svg_parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">')
    svg_parts.append('<style>')
    svg_parts.append('  .title { font: bold 16px sans-serif; }')
    svg_parts.append('  .axis-label { font: 12px sans-serif; }')
    svg_parts.append('  .tick-label { font: 10px sans-serif; }')
    svg_parts.append('  .grid { stroke: #e0e0e0; stroke-width: 1; }')
    svg_parts.append('  .line { fill: none; stroke: #2196F3; stroke-width: 2; }')
    svg_parts.append('  .point { fill: #2196F3; }')
    svg_parts.append('  .point-label { font: 9px sans-serif; fill: #333; }')
    svg_parts.append('  .extra-point { fill: #FF9800; }')
    svg_parts.append('  .extra-label { font: 9px sans-serif; fill: #E65100; }')
    svg_parts.append('  .legend { font: 10px sans-serif; }')
    svg_parts.append('</style>')

    # Background
    svg_parts.append(f'<rect width="{width}" height="{height}" fill="white"/>')

    # Title
    svg_parts.append(f'<text x="{width/2}" y="25" text-anchor="middle" class="title">{title}</text>')

    # Grid lines (horizontal)
    elo_ticks = range(int(min_elo // 100) * 100, int(max_elo // 100 + 1) * 100, 100)
    for elo in elo_ticks:
        if min_elo <= elo <= max_elo:
            y = scale_y(elo)
            svg_parts.append(f'<line x1="{margin["left"]}" y1="{y}" x2="{width - margin["right"]}" y2="{y}" class="grid"/>')
            svg_parts.append(f'<text x="{margin["left"] - 5}" y="{y + 4}" text-anchor="end" class="tick-label">{elo}</text>')

    # Grid lines (vertical) - use actual data times as ticks
    time_ticks = sorted(set(times + (extra_times or [])))  # All data points
    for t in time_ticks:
        if min_time <= t <= max_time:
            x = scale_x(t)
            svg_parts.append(f'<line x1="{x}" y1="{margin["top"]}" x2="{x}" y2="{height - margin["bottom"]}" class="grid"/>')
            svg_parts.append(f'<text x="{x}" y="{height - margin["bottom"] + 15}" text-anchor="middle" class="tick-label">{t}s</text>')

    # Axes
    svg_parts.append(f'<line x1="{margin["left"]}" y1="{height - margin["bottom"]}" x2="{width - margin["right"]}" y2="{height - margin["bottom"]}" stroke="black" stroke-width="1"/>')
    svg_parts.append(f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{height - margin["bottom"]}" stroke="black" stroke-width="1"/>')

    # Axis labels
    svg_parts.append(f'<text x="{width/2}" y="{height - 15}" text-anchor="middle" class="axis-label">Search Time (seconds, log scale)</text>')
    svg_parts.append(f'<text x="15" y="{height/2}" text-anchor="middle" transform="rotate(-90, 15, {height/2})" class="axis-label">Estimated Elo</text>')

    # Line path
    points = [(scale_x(t), scale_y(e)) for t, e in zip(times, elos)]
    path_d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    svg_parts.append(f'<path d="{path_d}" class="line"/>')

    # Data points with labels
    for t, e in zip(times, elos):
        x, y = scale_x(t), scale_y(e)
        svg_parts.append(f'<circle cx="{x}" cy="{y}" r="5" class="point"/>')
        svg_parts.append(f'<text x="{x}" y="{y - 10}" text-anchor="middle" class="point-label">{e}</text>')

    # Extra data points (different color)
    if extra_times and extra_elos:
        for t, e in zip(extra_times, extra_elos):
            x, y = scale_x(t), scale_y(e)
            svg_parts.append(f'<circle cx="{x}" cy="{y}" r="5" class="extra-point"/>')
            svg_parts.append(f'<text x="{x}" y="{y - 10}" text-anchor="middle" class="extra-label">{e}</text>')

        # Legend
        legend_x = width - margin["right"] - 120
        legend_y = margin["top"] + 10
        svg_parts.append(f'<circle cx="{legend_x}" cy="{legend_y}" r="4" class="point"/>')
        svg_parts.append(f'<text x="{legend_x + 8}" y="{legend_y + 4}" class="legend">20 games/point</text>')
        svg_parts.append(f'<circle cx="{legend_x}" cy="{legend_y + 15}" r="4" class="extra-point"/>')
        svg_parts.append(f'<text x="{legend_x + 8}" y="{legend_y + 19}" class="legend">{extra_label or "5 games/point"}</text>')

    svg_parts.append('</svg>')

    # Write SVG
    with open(output_path, 'w') as f:
        f.write('\n'.join(svg_parts))

    print(f"Generated: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot scaling results from CSV")
    parser.add_argument("--csv", required=True, help="Input CSV file (time,elo)")
    parser.add_argument("--out", required=True, help="Output SVG file")
    parser.add_argument("--title", default="Search-Time Scaling (Epoch 16 NNUE)", help="Plot title")
    parser.add_argument("--extra-csv", help="Secondary CSV file for overlay points")
    parser.add_argument("--extra-times", help="Comma-separated times to include from extra CSV (e.g., '3.0,5.0')")
    parser.add_argument("--extra-label", default="5 games/point", help="Label for extra points in legend")
    args = parser.parse_args()

    # Read primary CSV
    times = []
    elos = []
    with open(args.csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row['time']))
            elos.append(float(row['elo']))

    # Read extra CSV if provided
    extra_times = None
    extra_elos = None
    if args.extra_csv and args.extra_times:
        filter_times = set(float(t) for t in args.extra_times.split(','))
        extra_times = []
        extra_elos = []
        with open(args.extra_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = float(row['time'])
                if t in filter_times:
                    extra_times.append(t)
                    extra_elos.append(float(row['elo']))

    generate_svg(times, elos, args.out, args.title, extra_times, extra_elos, args.extra_label)

if __name__ == "__main__":
    main()
