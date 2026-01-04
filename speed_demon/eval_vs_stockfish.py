import argparse
import csv
import math
import os
from datetime import datetime
from typing import Optional

import chess
import chess.engine


def _elo_from_score(score: float) -> float:
    epsilon = 1e-6
    score = max(epsilon, min(1.0 - epsilon, score))
    return -400.0 * math.log10(1.0 / score - 1.0)


def _configure_engine(engine: chess.engine.SimpleEngine, threads: int, hash_mb: int) -> None:
    options = {}
    if "Threads" in engine.options:
        options["Threads"] = threads
    if "Hash" in engine.options:
        options["Hash"] = hash_mb
    if options:
        engine.configure(options)


def _play_game(
    engine_white: chess.engine.SimpleEngine,
    engine_black: chess.engine.SimpleEngine,
    time_per_move: float,
    max_moves: int,
) -> str:
    board = chess.Board()
    limit = chess.engine.Limit(time=time_per_move)

    for _ in range(max_moves):
        if board.is_game_over():
            break
        engine = engine_white if board.turn == chess.WHITE else engine_black
        result = engine.play(board, limit)
        if result.move is None:
            break
        board.push(result.move)

    return board.result()


def _configure_base_engine(
    engine: chess.engine.SimpleEngine,
    base_nnue: Optional[str],
    threads: int,
    hash_mb: int,
    force_classical: bool,
) -> None:
    _configure_engine(engine, threads, hash_mb)
    if base_nnue:
        if "EvalFile" not in engine.options:
            raise RuntimeError("Stockfish does not support EvalFile option.")
        engine.configure({"EvalFile": base_nnue})
        if "Use NNUE" in engine.options:
            engine.configure({"Use NNUE": True})
    elif force_classical:
        if "Use NNUE" in engine.options:
            engine.configure({"Use NNUE": False})


def evaluate(
    nnue_path: str,
    stockfish_path: str,
    games: int,
    time_per_move: float,
    max_moves: int,
    threads: int,
    hash_mb: int,
    csv_path: Optional[str],
    epoch: Optional[int],
    base_nnue: Optional[str],
    force_classical: bool,
    debug_dir: Optional[str],
) -> Optional[dict]:
    nnue_path = os.path.abspath(nnue_path)
    stockfish_path = os.path.abspath(stockfish_path)
    stockfish_cwd = os.path.dirname(stockfish_path)
    if base_nnue:
        base_nnue = os.path.abspath(base_nnue)

    engine_base = chess.engine.SimpleEngine.popen_uci(stockfish_path, cwd=stockfish_cwd)
    engine_test = chess.engine.SimpleEngine.popen_uci(stockfish_path, cwd=stockfish_cwd)

    try:
        _configure_base_engine(engine_base, base_nnue, threads, hash_mb, force_classical)
        _configure_engine(engine_test, threads, hash_mb)

        if debug_dir and "Debug Log File" in engine_base.options:
            os.makedirs(debug_dir, exist_ok=True)
            engine_base.configure({"Debug Log File": os.path.join(debug_dir, "base.log")})
        if debug_dir and "Debug Log File" in engine_test.options:
            os.makedirs(debug_dir, exist_ok=True)
            engine_test.configure({"Debug Log File": os.path.join(debug_dir, "test.log")})

        if "EvalFile" not in engine_test.options:
            raise RuntimeError("Stockfish binary does not support EvalFile option.")
        engine_test.configure({"EvalFile": nnue_path})
        if "Use NNUE" in engine_test.options:
            engine_test.configure({"Use NNUE": True})

        engine_base.ping()
        engine_test.ping()

        wins = 0
        draws = 0
        losses = 0

        for game_idx in range(games):
            test_is_white = (game_idx % 2 == 0)
            if test_is_white:
                result = _play_game(engine_test, engine_base, time_per_move, max_moves)
            else:
                result = _play_game(engine_base, engine_test, time_per_move, max_moves)

            if result == "1-0":
                if test_is_white:
                    wins += 1
                else:
                    losses += 1
            elif result == "0-1":
                if test_is_white:
                    losses += 1
                else:
                    wins += 1
            else:
                draws += 1

        score = (wins + 0.5 * draws) / max(games, 1)
        win_rate = wins / max(games, 1)
        elo = _elo_from_score(score)

        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "epoch": epoch,
            "games": games,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "score": score,
            "win_rate": win_rate,
            "elo": elo,
        }

        if csv_path:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            file_exists = os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "timestamp",
                        "epoch",
                        "games",
                        "wins",
                        "draws",
                        "losses",
                        "score",
                        "win_rate",
                        "elo",
                    ],
                )
                if not file_exists:
                    writer.writeheader()
                writer.writerow(metrics)

        return metrics
    except Exception as exc:
        print(f"Eval failed: {exc}")
        return None
    finally:
        try:
            engine_base.quit()
        except Exception:
            pass
        try:
            engine_test.quit()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NNUE vs Stockfish")
    parser.add_argument("--nnue", required=True, help="Path to .nnue file")
    parser.add_argument("--stockfish", required=True, help="Path to stockfish binary")
    parser.add_argument("--games", type=int, default=8, help="Number of games")
    parser.add_argument("--time-per-move", type=float, default=0.05, help="Seconds per move")
    parser.add_argument("--max-moves", type=int, default=200, help="Max moves per game")
    parser.add_argument("--threads", type=int, default=1, help="Stockfish threads")
    parser.add_argument("--hash-mb", type=int, default=128, help="Stockfish hash size (MB)")
    parser.add_argument(
        "--stockfish-classical-base",
        action="store_true",
        help="Disable NNUE for the base Stockfish engine.",
    )
    parser.add_argument(
        "--stockfish-base-nnue",
        type=str,
        default=None,
        help="Optional baseline NNUE file for Stockfish base engine",
    )
    parser.add_argument(
        "--debug-dir",
        type=str,
        default=None,
        help="Directory to write Stockfish debug logs (base.log/test.log).",
    )
    parser.add_argument("--csv", type=str, default=None, help="CSV log path")
    parser.add_argument("--epoch", type=int, default=None, help="Epoch number")
    args = parser.parse_args()

    metrics = evaluate(
        nnue_path=args.nnue,
        stockfish_path=args.stockfish,
        games=args.games,
        time_per_move=args.time_per_move,
        max_moves=args.max_moves,
        threads=args.threads,
        hash_mb=args.hash_mb,
        csv_path=args.csv,
        epoch=args.epoch,
        base_nnue=args.stockfish_base_nnue,
        force_classical=args.stockfish_classical_base,
        debug_dir=args.debug_dir,
    )

    if metrics is None:
        return

    print("Eval results:")
    print(
        f"  W/D/L: {metrics['wins']}/{metrics['draws']}/{metrics['losses']}"
        f" | Score: {metrics['score']:.3f}"
        f" | Elo: {metrics['elo']:.1f}"
    )


if __name__ == "__main__":
    main()
