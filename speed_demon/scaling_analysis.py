import argparse
import json
import math
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _parse_floats(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _parse_ints(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _expected_score(test_elo: float, base_elo: float) -> float:
    return 1.0 / (1.0 + 10 ** ((base_elo - test_elo) / 400.0))


def _estimate_elo(base_elos: List[int], scores: List[float]) -> Optional[float]:
    if not base_elos or not scores or len(base_elos) != len(scores):
        return None
    lo = min(base_elos) - 800
    hi = max(base_elos) + 800
    best_elo = None
    best_err = None
    for elo in range(int(lo), int(hi) + 1):
        err = 0.0
        for base_elo, score in zip(base_elos, scores):
            pred = _expected_score(elo, base_elo)
            err += (score - pred) ** 2
        if best_err is None or err < best_err:
            best_err = err
            best_elo = float(elo)
    return best_elo


def _fit_log_scaling(times: List[float], elos: List[float]) -> Optional[Tuple[float, float]]:
    if not times or not elos or len(times) != len(elos):
        return None
    xs = [math.log2(t) for t in times]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(elos) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x == 0:
        return None
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, elos))
    slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x
    return intercept, slope


def _run_eval(task: Dict) -> Optional[Dict]:
    import eval_vs_stockfish

    metrics = eval_vs_stockfish.evaluate(
        nnue_path=task["nnue"],
        stockfish_path=task["stockfish"],
        games=task["games"],
        time_per_move=task["base_time"],
        time_per_move_test=task["test_time"],
        max_moves=task["max_moves"],
        threads=task["threads"],
        hash_mb=task["hash_mb"],
        csv_path=None,
        json_path=None,
        epoch=None,
        base_nnue=task.get("base_nnue"),
        force_classical=task.get("force_classical", False),
        debug_dir=task.get("debug_dir"),
        base_elo=task["base_elo"],
        base_skill=task.get("base_skill"),
    )
    if metrics is None:
        return None
    metrics["search_time"] = task["test_time"]
    metrics["base_time"] = task["base_time"]
    metrics["run_id"] = task["run_id"]
    return metrics


def _write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _summarize_by_time(results: List[Dict]) -> Dict[float, Dict]:
    grouped: Dict[float, Dict] = {}
    for row in results:
        t = row["search_time"]
        grouped.setdefault(t, {"scores": [], "base_elos": [], "rows": []})
        grouped[t]["scores"].append(row["score"])
        grouped[t]["base_elos"].append(row["base_elo"])
        grouped[t]["rows"].append(row)
    return grouped


def _format_table(rows: List[Tuple[str, str]]) -> str:
    if not rows:
        return ""
    left_width = max(len(left) for left, _ in rows)
    lines = []
    for left, right in rows:
        lines.append(f"{left.ljust(left_width)}  {right}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search scaling law runner for NNUE.")
    parser.add_argument("--nnue", required=True, help="Path to test NNUE")
    parser.add_argument("--stockfish", required=True, help="Path to Stockfish binary")
    parser.add_argument(
        "--times",
        type=str,
        default="0.05,0.1,0.2,0.4,0.8,1.5,3.0,5.0",
        help="Comma-separated time-per-move list (seconds)",
    )
    parser.add_argument(
        "--base-elos",
        type=str,
        default="2600,2750,2900,3050,3190",
        help="Comma-separated base Elo list",
    )
    parser.add_argument("--games", type=int, default=5, help="Games per test point")
    parser.add_argument("--threads", type=int, default=4, help="Threads per Stockfish engine")
    parser.add_argument("--hash-mb", type=int, default=128, help="Hash MB per engine")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    parser.add_argument("--max-moves", type=int, default=200, help="Max moves per game")
    parser.add_argument(
        "--base-time",
        type=float,
        default=None,
        help="Fixed base time per move (seconds). If unset, base uses the same time as test.",
    )
    parser.add_argument("--base-nnue", type=str, default=None, help="Optional base NNUE for Stockfish")
    parser.add_argument("--base-skill", type=int, default=None, help="Optional Stockfish skill level")
    parser.add_argument(
        "--classical-base",
        action="store_true",
        help="Disable NNUE for the base Stockfish engine",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/speed_demon/eval/scaling_runs",
        help="Output directory root",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier (defaults to timestamp)",
    )
    parser.add_argument(
        "--diminish-threshold",
        type=float,
        default=50.0,
        help="Elo per doubling threshold for diminishing returns",
    )
    parser.add_argument(
        "--extrapolate-times",
        type=str,
        default="",
        help="Comma-separated times (seconds) to extrapolate",
    )
    parser.add_argument(
        "--debug-dir",
        type=str,
        default=None,
        help="Optional debug log directory",
    )
    args = parser.parse_args()

    nnue_path = Path(args.nnue)
    if not nnue_path.is_file():
        raise SystemExit(f"NNUE file not found: {nnue_path}")

    stockfish_path = Path(args.stockfish)
    if not stockfish_path.is_file():
        resolved = shutil.which(args.stockfish)
        if resolved:
            stockfish_path = Path(resolved)
        else:
            raise SystemExit(
                f"Stockfish binary not found: {args.stockfish}\n"
                "Tip: pass the full path to the binary, e.g. "
                "/root/.stockfish/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"
            )

    times = _parse_floats(args.times)
    base_elos = _parse_ints(args.base_elos)
    extrapolate_times = _parse_floats(args.extrapolate_times) if args.extrapolate_times else []
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_dir) / run_id
    raw_path = out_root / "raw_results.jsonl"
    summary_path = out_root / "summary.json"
    summary_txt = out_root / "summary.txt"

    tasks = []
    for t in times:
        for base_elo in base_elos:
            base_time = args.base_time if args.base_time is not None else t
            tasks.append(
                {
                    "nnue": str(nnue_path),
                    "stockfish": str(stockfish_path),
                    "games": args.games,
                    "test_time": t,
                    "base_time": base_time,
                    "base_elo": base_elo,
                    "threads": args.threads,
                    "hash_mb": args.hash_mb,
                    "max_moves": args.max_moves,
                    "base_nnue": args.base_nnue,
                    "base_skill": args.base_skill,
                    "force_classical": args.classical_base,
                    "run_id": run_id,
                    "debug_dir": args.debug_dir,
                }
            )

    total_games = len(tasks) * args.games
    print(f"Scaling run {run_id}")
    print(f"Test points: {len(tasks)} | Games per point: {args.games} | Total games: {total_games}")
    print(f"Workers: {args.workers} | Threads per engine: {args.threads}")

    start = time.time()
    results: List[Dict] = []
    failures = 0

    if args.workers <= 1:
        for task in tasks:
            metrics = _run_eval(task)
            if metrics is None:
                failures += 1
                continue
            results.append(metrics)
            _write_jsonl(raw_path, [metrics])
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_run_eval, task) for task in tasks]
            for future in as_completed(futures):
                metrics = future.result()
                if metrics is None:
                    failures += 1
                    continue
                results.append(metrics)
                _write_jsonl(raw_path, [metrics])

    elapsed = time.time() - start
    grouped = _summarize_by_time(results)
    time_points = sorted(grouped.keys())

    time_elos = []
    for t in time_points:
        scores = grouped[t]["scores"]
        base_elo_list = grouped[t]["base_elos"]
        est = _estimate_elo(base_elo_list, scores)
        if est is not None:
            time_elos.append((t, est))

    per_doubling = []
    for (t1, e1), (t2, e2) in zip(time_elos, time_elos[1:]):
        ratio = t2 / t1
        if ratio <= 0:
            continue
        gain = e2 - e1
        gain_per_doubling = gain / math.log2(ratio)
        per_doubling.append((t1, t2, gain_per_doubling))

    diminish_at = None
    for t1, t2, gain in per_doubling:
        if gain < args.diminish_threshold:
            diminish_at = (t1, t2, gain)
            break

    fit = _fit_log_scaling([t for t, _ in time_elos], [e for _, e in time_elos])
    extrapolated = []
    if fit and extrapolate_times:
        intercept, slope = fit
        for t in extrapolate_times:
            extrapolated.append({"time": t, "elo": intercept + slope * math.log2(t)})

    summary = {
        "run_id": run_id,
        "nnue": os.path.abspath(args.nnue),
        "stockfish": os.path.abspath(args.stockfish),
        "times": times,
        "base_elos": base_elos,
        "games_per_point": args.games,
        "total_points": len(tasks),
        "total_games": total_games,
        "workers": args.workers,
        "threads": args.threads,
        "hash_mb": args.hash_mb,
        "base_time": args.base_time,
        "elapsed_sec": elapsed,
        "failures": failures,
        "estimated_elo_by_time": [{"time": t, "elo": e} for t, e in time_elos],
        "elo_per_doubling": [
            {"from_time": t1, "to_time": t2, "elo_per_double": gain}
            for t1, t2, gain in per_doubling
        ],
        "diminishing_returns": None
        if diminish_at is None
        else {"from_time": diminish_at[0], "to_time": diminish_at[1], "elo_per_double": diminish_at[2]},
        "log2_fit": None if fit is None else {"intercept": fit[0], "slope": fit[1]},
        "extrapolated": extrapolated,
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = []
    lines.append("Elo by search time")
    lines.append(_format_table([(f"{t:.2f}s", f"{e:.1f}") for t, e in time_elos]))
    lines.append("")
    lines.append("Elo gain per doubling")
    lines.append(
        _format_table(
            [(f"{t1:.2f}->{t2:.2f}", f"{gain:.1f} Elo/doubling") for t1, t2, gain in per_doubling]
        )
    )
    lines.append("")
    if diminish_at:
        lines.append(
            f"Diminishing returns: {diminish_at[0]:.2f}s to {diminish_at[1]:.2f}s "
            f"({diminish_at[2]:.1f} Elo/doubling)"
        )
    else:
        lines.append("Diminishing returns: not detected under threshold")
    if fit:
        lines.append(f"Extrapolation model: elo = {fit[0]:.1f} + {fit[1]:.1f} * log2(time)")
    if extrapolated:
        lines.append("")
        lines.append("Extrapolated points")
        lines.append(_format_table([(f"{p['time']:.2f}s", f"{p['elo']:.1f}") for p in extrapolated]))

    summary_txt.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nRaw results: {raw_path}")
    print(f"Summary JSON: {summary_path}")
    print(f"Summary TXT: {summary_txt}")


if __name__ == "__main__":
    main()
