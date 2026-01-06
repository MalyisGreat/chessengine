#!/usr/bin/env python3
"""Compare two NNUEs by having them play against each other."""
import chess
import chess.engine
import argparse
from pathlib import Path

def play_game(engine1, engine2, time_per_move, max_moves=200):
    """Play a single game, engine1 as white."""
    board = chess.Board()
    limit = chess.engine.Limit(time=time_per_move)

    for _ in range(max_moves):
        if board.is_game_over():
            break
        engine = engine1 if board.turn == chess.WHITE else engine2
        result = engine.play(board, limit)
        if result.move is None:
            break
        board.push(result.move)

    return board.result()

def main():
    parser = argparse.ArgumentParser(description="Compare two NNUEs")
    parser.add_argument("--nnue1", required=True, help="First NNUE file")
    parser.add_argument("--nnue2", required=True, help="Second NNUE file")
    parser.add_argument("--stockfish", required=True, help="Path to Stockfish")
    parser.add_argument("--games", type=int, default=10, help="Number of games")
    parser.add_argument("--time", type=float, default=0.1, help="Time per move in seconds")
    args = parser.parse_args()

    print(f"NNUE 1: {args.nnue1}")
    print(f"NNUE 2: {args.nnue2}")
    print(f"Games: {args.games}, Time/move: {args.time}s\n")

    # Start engines
    engine1 = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    engine2 = chess.engine.SimpleEngine.popen_uci(args.stockfish)

    # Configure with respective NNUEs
    engine1.configure({"EvalFile": args.nnue1, "Threads": 1, "Hash": 64})
    engine2.configure({"EvalFile": args.nnue2, "Threads": 1, "Hash": 64})

    wins1, wins2, draws = 0, 0, 0

    for i in range(args.games):
        # Alternate colors
        if i % 2 == 0:
            white, black = engine1, engine2
            white_is_1 = True
        else:
            white, black = engine2, engine1
            white_is_1 = False

        result = play_game(white, black, args.time)

        if result == "1-0":
            if white_is_1:
                wins1 += 1
                print(f"Game {i+1}: NNUE1 wins (white)")
            else:
                wins2 += 1
                print(f"Game {i+1}: NNUE2 wins (white)")
        elif result == "0-1":
            if white_is_1:
                wins2 += 1
                print(f"Game {i+1}: NNUE2 wins (black)")
            else:
                wins1 += 1
                print(f"Game {i+1}: NNUE1 wins (black)")
        else:
            draws += 1
            print(f"Game {i+1}: Draw")

    engine1.quit()
    engine2.quit()

    print(f"\n{'='*40}")
    print(f"Results: NNUE1 {wins1} - {draws} - {wins2} NNUE2")
    print(f"NNUE1 score: {(wins1 + draws*0.5) / args.games * 100:.1f}%")

    if wins1 == wins2 and abs(wins1 - wins2) <= 1:
        print("\nConclusion: NNUEs appear to be EQUIVALENT")
    elif wins1 > wins2:
        print(f"\nConclusion: NNUE1 is stronger (+{wins1 - wins2})")
    else:
        print(f"\nConclusion: NNUE2 is stronger (+{wins2 - wins1})")

if __name__ == "__main__":
    main()
