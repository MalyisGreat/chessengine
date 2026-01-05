import argparse
import csv
import os
from typing import Iterable, List, Tuple, Optional, Dict


def _load_series(
    path: str,
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]], List[Dict[str, Optional[float]]]]:
    train = []
    val = []
    val_meta = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            step = row.get("step")
            if not step:
                continue
            step_value = int(float(step))
            epoch_value = None
            if row.get("epoch") not in (None, ""):
                epoch_value = int(float(row["epoch"]))
            train_loss = row.get("train_loss") or ""
            val_loss = row.get("val_loss") or ""
            if train_loss.strip():
                train.append((step_value, float(train_loss)))
            if val_loss.strip():
                val_value = float(val_loss)
                val.append((step_value, val_value))
                val_meta.append(
                    {
                        "step": step_value,
                        "epoch": epoch_value,
                        "loss": val_value,
                    }
                )
    return train, val, val_meta


def _format_step(value: float) -> str:
    if value >= 1000000:
        return f"{value/1000000:.1f}M"
    if value >= 1000:
        return f"{value/1000:.0f}k"
    return f"{int(value)}"


def _write_svg(
    train: Iterable[Tuple[int, float]],
    val: Iterable[Tuple[int, float]],
    val_meta: Iterable[Dict[str, Optional[float]]],
    overfit_step: Optional[int],
    out_path: str,
    title: str,
    x_label: str,
    y_label: str,
    width: int,
    height: int,
    shade_overfit: bool,
    show_best_marker: bool,
) -> None:
    train = list(train)
    val = list(val)
    val_meta = list(val_meta)
    if not train and not val:
        raise ValueError("No data points to plot.")

    left = 70
    right = 30
    top = 50
    bottom = 60
    plot_w = width - left - right
    plot_h = height - top - bottom

    all_points = train + val
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    y_pad = max(0.0005, (max_y - min_y) * 0.08)
    min_y -= y_pad
    max_y += y_pad

    def map_x(x: float) -> float:
        if max_x == min_x:
            return left + plot_w / 2
        return left + (x - min_x) / (max_x - min_x) * plot_w

    def map_y(y: float) -> float:
        if max_y == min_y:
            return top + plot_h / 2
        return top + (max_y - y) / (max_y - min_y) * plot_h

    y_ticks = 5
    y_tick_values = [
        min_y + i * (max_y - min_y) / (y_ticks - 1) for i in range(y_ticks)
    ]
    x_ticks = 6
    x_tick_values = [
        min_x + i * (max_x - min_x) / (x_ticks - 1) for i in range(x_ticks)
    ]

    def polyline(points: Iterable[Tuple[int, float]]) -> str:
        return " ".join(f"{map_x(x):.1f},{map_y(y):.1f}" for x, y in points)

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
            f'<text x="{left-10}" y="{y_pos+4:.1f}" fill="#95a3b3" font-size="11" font-family="Arial" text-anchor="end">{y:.4f}</text>'
        )

    for x in x_tick_values:
        x_pos = map_x(x)
        lines.append(
            f'<line x1="{x_pos:.1f}" y1="{top}" x2="{x_pos:.1f}" y2="{top+plot_h}" stroke="#1b2330" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x_pos:.1f}" y="{top+plot_h+18}" fill="#95a3b3" font-size="11" font-family="Arial" text-anchor="middle">{_format_step(x)}</text>'
        )

    if shade_overfit and overfit_step is not None:
        x_start = map_x(overfit_step)
        if x_start < left:
            x_start = left
        lines.append(
            f'<rect x="{x_start:.1f}" y="{top}" width="{left+plot_w-x_start:.1f}" height="{plot_h}" fill="#7a1c1c" opacity="0.18"/>'
        )

    if train:
        lines.append(
            f'<polyline fill="none" stroke="#5dd3a0" stroke-width="2.2" points="{polyline(train)}"/>'
        )
    if val:
        lines.append(
            f'<polyline fill="none" stroke="#f0b46e" stroke-width="2.2" points="{polyline(val)}"/>'
        )

    if show_best_marker and val_meta:
        best_point = min(val_meta, key=lambda p: p["loss"])
        best_step = int(best_point["step"])
        best_loss = float(best_point["loss"])
        x_best = map_x(best_step)
        y_best = map_y(best_loss)
        lines.append(
            f'<line x1="{x_best:.1f}" y1="{top}" x2="{x_best:.1f}" y2="{top+plot_h}" stroke="#ff7b7b" stroke-width="2" stroke-dasharray="4,4"/>'
        )
        lines.append(
            f'<circle cx="{x_best:.1f}" cy="{y_best:.1f}" r="4.5" fill="#ff7b7b" stroke="#0b0f14" stroke-width="1"/>'
        )
        label = f"best val @ step {best_step}"
        if best_point.get("epoch") is not None:
            label = f"best val @ epoch {int(best_point['epoch'])}, step {best_step}"
        lines.append(
            f'<text x="{x_best+6:.1f}" y="{top+14}" fill="#ff9b9b" font-size="11" font-family="Arial">{label}</text>'
        )

    legend_x = left + 10
    legend_y = top + 12
    lines.append(f'<rect x="{legend_x}" y="{legend_y}" width="10" height="10" fill="#5dd3a0"/>')
    lines.append(
        f'<text x="{legend_x+16}" y="{legend_y+9}" fill="#c7d1db" font-size="11" font-family="Arial">Train</text>'
    )
    lines.append(f'<rect x="{legend_x+70}" y="{legend_y}" width="10" height="10" fill="#f0b46e"/>')
    lines.append(
        f'<text x="{legend_x+86}" y="{legend_y+9}" fill="#c7d1db" font-size="11" font-family="Arial">Val</text>'
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
    parser.add_argument("--x-label", default="Step", help="X-axis label")
    parser.add_argument("--y-label", default="Loss", help="Y-axis label")
    parser.add_argument("--width", type=int, default=960, help="SVG width")
    parser.add_argument("--height", type=int, default=540, help="SVG height")
    parser.add_argument("--overfit-step", type=int, help="Step at which overfitting begins (default: best val loss)")
    parser.add_argument("--no-overfit-shade", action="store_true", help="Disable overfit shading")
    parser.add_argument("--no-best-marker", action="store_true", help="Disable best validation marker")
    args = parser.parse_args()

    train, val, val_meta = _load_series(args.csv)
    best_step = None
    if val_meta:
        best_step = int(min(val_meta, key=lambda p: p["loss"])["step"])
    overfit_step = args.overfit_step if args.overfit_step is not None else best_step
    _write_svg(
        train=train,
        val=val,
        val_meta=val_meta,
        overfit_step=overfit_step,
        out_path=args.out,
        title=args.title,
        x_label=args.x_label,
        y_label=args.y_label,
        width=args.width,
        height=args.height,
        shade_overfit=not args.no_overfit_shade,
        show_best_marker=not args.no_best_marker,
    )
    print(f"Wrote {args.out} (train={len(train)}, val={len(val)})")


if __name__ == "__main__":
    main()
