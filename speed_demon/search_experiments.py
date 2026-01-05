import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import chess
import chess.engine


@dataclass(frozen=True)
class SearchConfig:
    name: str
    test_uci: Dict[str, object]
    base_uci: Dict[str, object]


DEFAULT_CONFIGS = [
    SearchConfig(name="baseline", test_uci={}, base_uci={}),
    SearchConfig(name="contempt_0", test_uci={"Contempt": 0}, base_uci={}),
    SearchConfig(name="contempt_20", test_uci={"Contempt": 20}, base_uci={}),
    SearchConfig(name="contempt_-20", test_uci={"Contempt": -20}, base_uci={}),
    SearchConfig(name="slow_mover_80", test_uci={"Slow Mover": 80}, base_uci={}),
    SearchConfig(name="slow_mover_120", test_uci={"Slow Mover": 120}, base_uci={}),
    SearchConfig(name="move_overhead_30", test_uci={"Move Overhead": 30}, base_uci={}),
    SearchConfig(name="move_overhead_100", test_uci={"Move Overhead": 100}, base_uci={}),
]


def _load_configs(path: Optional[str]) -> List[SearchConfig]:
    if not path:
        return DEFAULT_CONFIGS
    config_path = Path(path)
    if not config_path.is_file():
        raise SystemExit(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    configs = []
    for entry in raw:
        configs.append(
            SearchConfig(
                name=entry["name"],
                test_uci=entry.get("test_uci", {}) or {},
                base_uci=entry.get("base_uci", {}) or {},
            )
        )
    return configs


def _detect_stockfish_options(stockfish_path: str) -> Dict[str, str]:
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        return {name.lower(): name for name in engine.options}
    finally:
        try:
            engine.quit()
        except Exception:
            pass


def _filter_uci_options(options: Dict[str, object], option_map: Dict[str, str]) -> Dict[str, object]:
    filtered: Dict[str, object] = {}
    for name, value in options.items():
        key = option_map.get(name.lower())
        if key is None:
            continue
        filtered[key] = value
    return filtered


def _write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _elo_from_score(score: float) -> float:
    score = max(1e-6, min(1.0 - 1e-6, score))
    return -400.0 * math.log10(1.0 / score - 1.0)

def _configure_engine(engine: chess.engine.SimpleEngine, threads: int, hash_mb: int) -> None:
    options = {}
    if "Threads" in engine.options:
        options["Threads"] = threads
    if "Hash" in engine.options:
        options["Hash"] = hash_mb
    if options:
        engine.configure(options)


def _configure_strength(
    engine: chess.engine.SimpleEngine, base_elo: Optional[int], base_skill: Optional[int]
) -> None:
    options = {}
    if base_elo is not None:
        if "UCI_LimitStrength" in engine.options:
            options["UCI_LimitStrength"] = True
        if "UCI_Elo" in engine.options:
            options["UCI_Elo"] = base_elo
    if base_skill is not None and "Skill Level" in engine.options:
        options["Skill Level"] = base_skill
    if options:
        engine.configure(options)


def _apply_uci_options(
    engine: chess.engine.SimpleEngine,
    options: Dict[str, object],
    skip: Optional[Iterable[str]] = None,
) -> None:
    if not options:
        return
    skip_set = {name.lower() for name in skip or []}
    name_map = {name.lower(): name for name in engine.options}
    to_apply: Dict[str, object] = {}
    for name, value in options.items():
        key = name_map.get(name.lower())
        if key is None or key.lower() in skip_set:
            continue
        to_apply[key] = value
    if to_apply:
        engine.configure(to_apply)


def _play_game(
    engine_white: chess.engine.SimpleEngine,
    engine_black: chess.engine.SimpleEngine,
    time_per_move_white: float,
    time_per_move_black: float,
    max_moves: int,
) -> str:
    board = chess.Board()
    limit_white = chess.engine.Limit(time=time_per_move_white)
    limit_black = chess.engine.Limit(time=time_per_move_black)

    for _ in range(max_moves):
        if board.is_game_over():
            break
        if board.turn == chess.WHITE:
            engine = engine_white
            limit = limit_white
        else:
            engine = engine_black
            limit = limit_black
        result = engine.play(board, limit)
        if result.move is None:
            break
        board.push(result.move)

    return board.result()


def _run_config(task: Dict) -> Optional[Dict]:
    nnue_path = os.path.abspath(task["nnue"])
    stockfish_path = os.path.abspath(task["stockfish"])
    stockfish_cwd = os.path.dirname(stockfish_path)

    engine_base = chess.engine.SimpleEngine.popen_uci(stockfish_path, cwd=stockfish_cwd)
    engine_test = chess.engine.SimpleEngine.popen_uci(stockfish_path, cwd=stockfish_cwd)

    try:
        _configure_engine(engine_base, task["threads"], task["hash_mb"])
        _configure_strength(engine_base, task["base_elo"], task.get("base_skill"))
        _apply_uci_options(engine_base, task.get("base_uci", {}))

        _configure_engine(engine_test, task["threads"], task["hash_mb"])
        if "EvalFile" not in engine_test.options:
            raise RuntimeError("Stockfish binary does not support EvalFile option.")
        engine_test.configure({"EvalFile": nnue_path})
        if "Use NNUE" in engine_test.options:
            engine_test.configure({"Use NNUE": True})
        _apply_uci_options(engine_test, task.get("test_uci", {}), skip=("EvalFile", "Use NNUE"))

        engine_base.ping()
        engine_test.ping()

        wins = 0
        draws = 0
        losses = 0

        time_base = task["base_time"]
        time_test = task["test_time"] if task["test_time"] is not None else time_base

        for game_idx in range(task["games"]):
            test_is_white = (game_idx % 2 == 0)
            if test_is_white:
                result = _play_game(engine_test, engine_base, time_test, time_base, task["max_moves"])
            else:
                result = _play_game(engine_base, engine_test, time_base, time_test, task["max_moves"])

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

        score = (wins + 0.5 * draws) / max(task["games"], 1)
        elo = _elo_from_score(score)
        estimated_elo = task["base_elo"] + elo if task["base_elo"] is not None else None

        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": task["config"],
            "run_id": task["run_id"],
            "nnue": nnue_path,
            "stockfish": stockfish_path,
            "base_elo": task["base_elo"],
            "games": task["games"],
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "score": score,
            "elo": elo,
            "estimated_elo": estimated_elo,
            "time_per_move": time_base,
            "time_per_move_test": time_test,
            "threads": task["threads"],
            "hash_mb": task["hash_mb"],
            "test_uci": task.get("test_uci", {}),
            "base_uci": task.get("base_uci", {}),
        }

        return metrics
    except Exception as exc:
        print(f"Eval failed for {task['config']}: {exc}")
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
    parser = argparse.ArgumentParser(description="Search option experiment runner.")
    parser.add_argument("--nnue", required=True, help="Path to NNUE file")
    parser.add_argument("--stockfish", required=True, help="Path to Stockfish binary")
    parser.add_argument("--base-elo", type=int, default=2800, help="Base Stockfish Elo")
    parser.add_argument("--games", type=int, default=8, help="Games per config")
    parser.add_argument("--time-per-move", type=float, default=0.1, help="Base engine time per move")
    parser.add_argument(
        "--time-per-move-test",
        type=float,
        default=None,
        help="Test engine time per move (defaults to base time)",
    )
    parser.add_argument("--threads", type=int, default=4, help="Threads per Stockfish engine")
    parser.add_argument("--hash-mb", type=int, default=128, help="Hash size per engine")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    parser.add_argument("--max-moves", type=int, default=200, help="Max moves per game")
    parser.add_argument("--config-file", type=str, default=None, help="JSON config file for experiments")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/speed_demon/eval/search_experiments",
        help="Output directory root",
    )
    parser.add_argument("--debug-dir", type=str, default=None, help="Optional debug log directory")
    parser.add_argument("--run-id", type=str, default=None, help="Run identifier")
    args = parser.parse_args()

    nnue_path = Path(args.nnue)
    if not nnue_path.is_file():
        raise SystemExit(f"NNUE file not found: {nnue_path}")

    stockfish_path = Path(args.stockfish)
    if not stockfish_path.is_file():
        raise SystemExit(f"Stockfish binary not found: {stockfish_path}")

    configs = _load_configs(args.config_file)
    option_map = _detect_stockfish_options(str(stockfish_path))
    filtered_configs = []
    for cfg in configs:
        filtered_configs.append(
            SearchConfig(
                name=cfg.name,
                test_uci=_filter_uci_options(cfg.test_uci, option_map),
                base_uci=_filter_uci_options(cfg.base_uci, option_map),
            )
        )

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_dir) / run_id
    raw_path = out_root / "raw_results.jsonl"
    summary_path = out_root / "summary.json"
    summary_txt = out_root / "summary.txt"

    print(f"Search experiment run {run_id}")
    print(f"Configs: {len(filtered_configs)} | Games per config: {args.games}")
    print(f"Workers: {args.workers} | Threads per engine: {args.threads}")

    base_time = args.time_per_move
    test_time = args.time_per_move_test

    tasks = []
    for cfg in filtered_configs:
        tasks.append(
            {
                "config": cfg.name,
                "nnue": str(nnue_path),
                "stockfish": str(stockfish_path),
                "games": args.games,
                "base_time": base_time,
                "test_time": test_time,
                "max_moves": args.max_moves,
                "threads": args.threads,
                "hash_mb": args.hash_mb,
                "base_elo": args.base_elo,
                "test_uci": cfg.test_uci,
                "base_uci": cfg.base_uci,
                "debug_dir": args.debug_dir,
                "run_id": run_id,
            }
        )

    results: List[Dict] = []
    if args.workers <= 1:
        for task in tasks:
            metrics = _run_config(task)
            if metrics is None:
                continue
            results.append(metrics)
            _write_jsonl(raw_path, [metrics])
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_run_config, task) for task in tasks]
            for future in as_completed(futures):
                metrics = future.result()
                if metrics is None:
                    continue
                results.append(metrics)
                _write_jsonl(raw_path, [metrics])

    results.sort(key=lambda r: (r.get("estimated_elo") or -99999), reverse=True)
    summary = {
        "run_id": run_id,
        "nnue": os.path.abspath(args.nnue),
        "stockfish": os.path.abspath(args.stockfish),
        "base_elo": args.base_elo,
        "games": args.games,
        "threads": args.threads,
        "hash_mb": args.hash_mb,
        "time_per_move": args.time_per_move,
        "time_per_move_test": args.time_per_move_test,
        "results": results,
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = ["Config results (sorted by Est Elo)", ""]
    for row in results:
        line = (
            f"{row['config']}: "
            f"W/D/L {row['wins']}/{row['draws']}/{row['losses']} "
            f"| score {row['score']:.3f} | est {row.get('estimated_elo'):.1f}"
        )
        lines.append(line)

    summary_txt.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nRaw results: {raw_path}")
    print(f"Summary JSON: {summary_path}")
    print(f"Summary TXT: {summary_txt}")


if __name__ == "__main__":
    main()
