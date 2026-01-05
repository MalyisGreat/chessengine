#!/usr/bin/env python3
"""Debug script to test Stockfish + NNUE integration step by step."""

import argparse
import subprocess
import sys
import time


def run_uci_commands(stockfish_path: str, commands: list[str], timeout: float = 10.0) -> str:
    """Run UCI commands and return output."""
    input_str = "\n".join(commands) + "\n"
    try:
        result = subprocess.run(
            [stockfish_path],
            input=input_str,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n\nRETURN CODE: {result.returncode}"
    except subprocess.TimeoutExpired:
        return "TIMEOUT - command took too long"
    except Exception as e:
        return f"ERROR: {e}"


def main():
    parser = argparse.ArgumentParser(description="Debug Stockfish NNUE loading")
    parser.add_argument("--stockfish", required=True, help="Path to stockfish binary")
    parser.add_argument("--nnue", required=True, help="Path to .nnue file")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 1: Basic UCI handshake (no NNUE)")
    print("=" * 60)
    output = run_uci_commands(args.stockfish, ["uci", "quit"])
    print(output)

    print("\n" + "=" * 60)
    print("STEP 2: Load NNUE and check ready")
    print("=" * 60)
    output = run_uci_commands(args.stockfish, [
        "uci",
        f"setoption name EvalFile value {args.nnue}",
        "setoption name Use NNUE value true",
        "isready",
        "quit"
    ])
    print(output)

    print("\n" + "=" * 60)
    print("STEP 3: Load NNUE and evaluate starting position")
    print("=" * 60)
    output = run_uci_commands(args.stockfish, [
        "uci",
        f"setoption name EvalFile value {args.nnue}",
        "setoption name Use NNUE value true",
        "isready",
        "position startpos",
        "eval",
        "quit"
    ], timeout=30.0)
    print(output)

    print("\n" + "=" * 60)
    print("STEP 4: Load NNUE and make a move (go movetime 100)")
    print("=" * 60)
    output = run_uci_commands(args.stockfish, [
        "uci",
        f"setoption name EvalFile value {args.nnue}",
        "setoption name Use NNUE value true",
        "isready",
        "position startpos",
        "go movetime 100",
        "quit"
    ], timeout=30.0)
    print(output)

    print("\n" + "=" * 60)
    print("STEP 5: Test UCI_LimitStrength (what eval_vs_stockfish.py uses)")
    print("=" * 60)
    output = run_uci_commands(args.stockfish, [
        "uci",
        "setoption name UCI_LimitStrength value true",
        "setoption name UCI_Elo value 2800",
        "isready",
        "position startpos",
        "go movetime 100",
        "quit"
    ], timeout=30.0)
    print(output)

    print("\n" + "=" * 60)
    print("STEP 6: Combined - NNUE on test engine, LimitStrength on base")
    print("=" * 60)
    print("This simulates what eval_vs_stockfish.py does")
    print("Testing base engine (Stockfish with LimitStrength)...")
    output = run_uci_commands(args.stockfish, [
        "uci",
        "setoption name UCI_LimitStrength value true",
        "setoption name UCI_Elo value 2800",
        "setoption name Threads value 1",
        "setoption name Hash value 128",
        "isready",
        "position startpos",
        "go movetime 100",
        "quit"
    ], timeout=30.0)
    print(output)

    print("\nTesting test engine (Stockfish with custom NNUE)...")
    output = run_uci_commands(args.stockfish, [
        "uci",
        f"setoption name EvalFile value {args.nnue}",
        "setoption name Use NNUE value true",
        "setoption name Threads value 1",
        "setoption name Hash value 128",
        "isready",
        "position startpos",
        "go movetime 100",
        "quit"
    ], timeout=30.0)
    print(output)

    print("\n" + "=" * 60)
    print("STEP 7: Test with python-chess (same method as eval script)")
    print("=" * 60)
    try:
        import chess
        import chess.engine

        print("Opening engine with python-chess...")
        engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
        print(f"Engine opened successfully")
        print(f"Available options: {list(engine.options.keys())[:10]}...")

        print("\nConfiguring EvalFile...")
        engine.configure({"EvalFile": args.nnue})
        print("EvalFile configured")

        if "Use NNUE" in engine.options:
            print("Enabling Use NNUE...")
            engine.configure({"Use NNUE": True})

        print("Pinging engine...")
        engine.ping()
        print("Ping successful")

        print("\nPlaying a test move...")
        board = chess.Board()
        result = engine.play(board, chess.engine.Limit(time=0.1))
        print(f"Move result: {result.move}")

        print("\nClosing engine...")
        engine.quit()
        print("SUCCESS - python-chess integration works!")

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
