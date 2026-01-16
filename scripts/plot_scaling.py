#!/usr/bin/env python3
"""Generate SVG plot from scaling CSV data."""

import argparse
import csv
import math
from pathlib import Path

def generate_svg(times, elos, output_path, title="Search-Time Scaling",
                 extra_times=None, extra_elos=None, extra_label=None):
    """Generate an SVG plot of Elo vs search time."""
    # SVG dimensions
    width = 700
    height = 420
    margin = {"top": 50, "right": 40, "bottom": 65, "left": 75}
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
    def scale_x(t):
        log_min = math.log10(min_time)
        log_max = math.log10(max_time)
        return margin["left"] + (math.log10(t) - log_min) / (log_max - log_min) * plot_width

    def scale_y(e):
        return margin["top"] + plot_height - (e - min_elo) / (max_elo - min_elo) * plot_height

    # Build SVG
    svg_parts = []
    svg_parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">')

    # Definitions (gradients, filters)
    svg_parts.append('<defs>')
    svg_parts.append('  <linearGradient id="bgGrad" x1="0%" y1="0%" x2="0%" y2="100%">')
    svg_parts.append('    <stop offset="0%" style="stop-color:#fafbfc"/>')
    svg_parts.append('    <stop offset="100%" style="stop-color:#f0f2f5"/>')
    svg_parts.append('  </linearGradient>')
    svg_parts.append('  <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">')
    svg_parts.append('    <feDropShadow dx="1" dy="1" stdDeviation="1" flood-opacity="0.15"/>')
    svg_parts.append('  </filter>')
    svg_parts.append('</defs>')

    # Styles
    svg_parts.append('<style>')
    svg_parts.append('  .title { font: bold 18px "Segoe UI", sans-serif; fill: #1a1a2e; }')
    svg_parts.append('  .subtitle { font: 11px "Segoe UI", sans-serif; fill: #666; }')
    svg_parts.append('  .axis-label { font: 12px "Segoe UI", sans-serif; fill: #444; }')
    svg_parts.append('  .tick-label { font: 10px "Segoe UI", sans-serif; fill: #555; }')
    svg_parts.append('  .grid { stroke: #dde1e6; stroke-width: 1; stroke-dasharray: 4,3; }')
    svg_parts.append('  .axis { stroke: #888; stroke-width: 1.5; }')
    svg_parts.append('  .line { fill: none; stroke: #2563eb; stroke-width: 2.5; stroke-linecap: round; stroke-linejoin: round; }')
    svg_parts.append('  .extra-line { fill: none; stroke: #f59e0b; stroke-width: 2.5; stroke-linecap: round; stroke-linejoin: round; stroke-dasharray: 8,4; }')
    svg_parts.append('  .point { fill: #2563eb; stroke: white; stroke-width: 2; }')
    svg_parts.append('  .point-label { font: bold 10px "Segoe UI", sans-serif; fill: #1e40af; }')
    svg_parts.append('  .extra-point { fill: #f59e0b; stroke: white; stroke-width: 2; }')
    svg_parts.append('  .extra-label { font: bold 10px "Segoe UI", sans-serif; fill: #b45309; }')
    svg_parts.append('  .legend-box { fill: white; stroke: #ddd; stroke-width: 1; rx: 4; }')
    svg_parts.append('  .legend { font: 11px "Segoe UI", sans-serif; fill: #333; }')
    svg_parts.append('  .legend-line { stroke-width: 2.5; stroke-linecap: round; }')
    svg_parts.append('</style>')

    # Background with gradient
    svg_parts.append(f'<rect width="{width}" height="{height}" fill="url(#bgGrad)"/>')

    # Plot area background (white)
    svg_parts.append(f'<rect x="{margin["left"]}" y="{margin["top"]}" width="{plot_width}" height="{plot_height}" fill="white" rx="2"/>')

    # Title
    svg_parts.append(f'<text x="{width/2}" y="28" text-anchor="middle" class="title">{title}</text>')

    # Grid lines (horizontal)
    elo_ticks = range(int(min_elo // 100) * 100, int(max_elo // 100 + 1) * 100, 100)
    for elo in elo_ticks:
        if min_elo <= elo <= max_elo:
            y = scale_y(elo)
            svg_parts.append(f'<line x1="{margin["left"]}" y1="{y}" x2="{width - margin["right"]}" y2="{y}" class="grid"/>')
            svg_parts.append(f'<text x="{margin["left"] - 8}" y="{y + 4}" text-anchor="end" class="tick-label">{elo}</text>')

    # Grid lines (vertical) - use actual data times as ticks
    time_ticks = sorted(set(times + (extra_times or [])))
    for t in time_ticks:
        if min_time <= t <= max_time:
            x = scale_x(t)
            svg_parts.append(f'<line x1="{x}" y1="{margin["top"]}" x2="{x}" y2="{height - margin["bottom"]}" class="grid"/>')
            svg_parts.append(f'<text x="{x}" y="{height - margin["bottom"] + 18}" text-anchor="middle" class="tick-label">{t}s</text>')

    # Axes
    svg_parts.append(f'<line x1="{margin["left"]}" y1="{height - margin["bottom"]}" x2="{width - margin["right"]}" y2="{height - margin["bottom"]}" class="axis"/>')
    svg_parts.append(f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{height - margin["bottom"]}" class="axis"/>')

    # Axis labels
    svg_parts.append(f'<text x="{width/2}" y="{height - 18}" text-anchor="middle" class="axis-label">Search Time (seconds, log scale)</text>')
    svg_parts.append(f'<text x="18" y="{height/2}" text-anchor="middle" transform="rotate(-90, 18, {height/2})" class="axis-label">Estimated Elo</text>')

    # Main line path (blue)
    points = [(scale_x(t), scale_y(e)) for t, e in zip(times, elos)]
    path_d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    svg_parts.append(f'<path d="{path_d}" class="line"/>')

    # Data points with labels (blue)
    for t, e in zip(times, elos):
        x, y = scale_x(t), scale_y(e)
        svg_parts.append(f'<circle cx="{x}" cy="{y}" r="6" class="point" filter="url(#shadow)"/>')
        svg_parts.append(f'<text x="{x}" y="{y - 12}" text-anchor="middle" class="point-label">{int(e)}</text>')

    # Extra data points (orange) with connecting line
    if extra_times and extra_elos:
        # Sort extra points by time
        extra_sorted = sorted(zip(extra_times, extra_elos))

        # Draw dashed line connecting extra points
        if len(extra_sorted) >= 2:
            extra_points = [(scale_x(t), scale_y(e)) for t, e in extra_sorted]
            extra_path_d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in extra_points)
            svg_parts.append(f'<path d="{extra_path_d}" class="extra-line"/>')

        # Draw extra points
        for t, e in extra_sorted:
            x, y = scale_x(t), scale_y(e)
            svg_parts.append(f'<circle cx="{x}" cy="{y}" r="6" class="extra-point" filter="url(#shadow)"/>')
            svg_parts.append(f'<text x="{x}" y="{y - 12}" text-anchor="middle" class="extra-label">{int(e)}</text>')

        # Legend box
        legend_x = margin["left"] + 15
        legend_y = margin["top"] + 12
        legend_w = 135
        legend_h = 50
        svg_parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="{legend_w}" height="{legend_h}" class="legend-box"/>')

        # Legend items
        ly1 = legend_y + 18
        ly2 = legend_y + 38
        svg_parts.append(f'<line x1="{legend_x + 10}" y1="{ly1}" x2="{legend_x + 35}" y2="{ly1}" stroke="#2563eb" class="legend-line"/>')
        svg_parts.append(f'<circle cx="{legend_x + 22}" cy="{ly1}" r="4" fill="#2563eb"/>')
        svg_parts.append(f'<text x="{legend_x + 42}" y="{ly1 + 4}" class="legend">20 games/point</text>')

        svg_parts.append(f'<line x1="{legend_x + 10}" y1="{ly2}" x2="{legend_x + 35}" y2="{ly2}" stroke="#f59e0b" stroke-dasharray="6,3" class="legend-line"/>')
        svg_parts.append(f'<circle cx="{legend_x + 22}" cy="{ly2}" r="4" fill="#f59e0b"/>')
        svg_parts.append(f'<text x="{legend_x + 42}" y="{ly2 + 4}" class="legend">{extra_label or "5 games/point"}</text>')

    svg_parts.append('</svg>')

    # Write SVG
    with open(output_path, 'w') as f:
        f.write('\n'.join(svg_parts))

    print(f"Generated: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot scaling results from CSV")
    parser.add_argument("--csv", required=True, help="Input CSV file (time,elo)")
    parser.add_argument("--out", required=True, help="Output SVG file")
    parser.add_argument("--title", default="Search-Time Scaling", help="Plot title")
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
