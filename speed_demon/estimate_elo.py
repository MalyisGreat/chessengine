import argparse
import csv
import os
import statistics
from typing import Optional

from eval_vs_stockfish import evaluate

MIN_BASE_ELO = 1400


def _parse_levels(levels_arg: Optional[str], min_elo: int, max_elo: int, step: int) -> list[int]:
    if levels_arg:
        parts = [p.strip() for p in levels_arg.split(",") if p.strip()]
        levels = [int(p) for p in parts]
        for level in levels:
            if level < MIN_BASE_ELO:
                raise ValueError(f"Base Elo must be >= {MIN_BASE_ELO}. Got {level}.")
        return levels
    if step <= 0:
        raise ValueError("step must be positive")
    if min_elo > max_elo:
        raise ValueError("min-elo must be <= max-elo")
    if min_elo < MIN_BASE_ELO:
        raise ValueError(f"min-elo must be >= {MIN_BASE_ELO}.")
    return list(range(min_elo, max_elo + 1, step))


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate NNUE Elo by laddering vs Stockfish")
    parser.add_argument("--nnue", required=True, help="Path to .nnue file")
    parser.add_argument("--stockfish", required=True, help="Path to Stockfish binary")
    parser.add_argument("--levels", type=str, default=None, help="Comma-separated ELO levels")
    parser.add_argument(
        "--min-elo",
        type=int,
        default=MIN_BASE_ELO,
        help=f"Minimum ELO level (>= {MIN_BASE_ELO})",
    )
    parser.add_argument("--max-elo", type=int, default=2600, help="Maximum ELO level")
    parser.add_argument("--step", type=int, default=200, help="ELO step size")
    parser.add_argument("--games", type=int, default=8, help="Games per level")
    parser.add_argument("--time-per-move", type=float, default=0.05, help="Seconds per move")
    parser.add_argument("--max-moves", type=int, default=200, help="Max moves per game")
    parser.add_argument("--threads", type=int, default=1, help="Stockfish threads")
    parser.add_argument("--hash-mb", type=int, default=128, help="Stockfish hash size (MB)")
    parser.add_argument(
        "--stockfish-base-skill",
        type=int,
        default=None,
        help="Optional Skill Level for base engine (0-20).",
    )
    parser.add_argument(
        "--debug-dir",
        type=str,
        default=None,
        help="Directory to write Stockfish debug logs (base.log/test.log).",
    )
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV output path")
    args = parser.parse_args()

    levels = _parse_levels(args.levels, args.min_elo, args.max_elo, args.step)
    results = []

    for base_elo in levels:
        metrics = evaluate(
            nnue_path=args.nnue,
            stockfish_path=args.stockfish,
            games=args.games,
            time_per_move=args.time_per_move,
            max_moves=args.max_moves,
            threads=args.threads,
            hash_mb=args.hash_mb,
            csv_path=None,
            epoch=None,
            base_nnue=None,
            force_classical=False,
            debug_dir=args.debug_dir,
            base_elo=base_elo,
            base_skill=args.stockfish_base_skill,
        )
        if metrics is None:
            continue

        estimated = base_elo + metrics["elo"]
        row = {
            "base_elo": base_elo,
            "games": metrics["games"],
            "wins": metrics["wins"],
            "draws": metrics["draws"],
            "losses": metrics["losses"],
            "score": metrics["score"],
            "win_rate": metrics["win_rate"],
            "elo_diff": metrics["elo"],
            "estimated_elo": estimated,
        }
        results.append(row)
        print(
            "Base ELO {base_elo}: W/D/L {wins}/{draws}/{losses} | "
            "Score {score:.3f} | Est {estimated_elo:.1f}".format(**row)
        )

    if args.csv and results:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "base_elo",
                    "games",
                    "wins",
                    "draws",
                    "losses",
                    "score",
                    "win_rate",
                    "elo_diff",
                    "estimated_elo",
                ],
            )
            writer.writeheader()
            writer.writerows(results)

    if not results:
        print("No eval results collected.")
        return

    estimates = [row["estimated_elo"] for row in results]
    median_est = statistics.median(estimates)
    mean_est = sum(estimates) / len(estimates)
    print(f"Estimated Elo (median): {median_est:.1f}")
    print(f"Estimated Elo (mean):   {mean_est:.1f}")


if __name__ == "__main__":
    main()
