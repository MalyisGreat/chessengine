"""
Interactive Chess Play Interface

Play against your trained chess engine in the terminal.

Usage:
    python -m engine.play --model ./outputs/checkpoint_best.pt
    python -m engine.play --model ./outputs/checkpoint_best.pt --uci  # UCI mode
"""

import sys
import argparse
import chess

from .search import ChessEngine, UCIEngine


def print_board(board: chess.Board, perspective: chess.Color = chess.WHITE):
    """Print board with nice formatting"""
    ranks = range(8) if perspective == chess.WHITE else range(7, -1, -1)
    files = range(8) if perspective == chess.WHITE else range(7, -1, -1)

    piece_symbols = {
        'P': '\u2659', 'N': '\u2658', 'B': '\u2657', 'R': '\u2656', 'Q': '\u2655', 'K': '\u2654',
        'p': '\u265F', 'n': '\u265E', 'b': '\u265D', 'r': '\u265C', 'q': '\u265B', 'k': '\u265A',
    }

    print("\n")
    for rank in ranks:
        print(f"  {rank + 1} ", end="")
        for file in files:
            square = chess.square(file, rank)
            piece = board.piece_at(square)

            # Background color
            is_light = (rank + file) % 2 == 1
            bg = "\033[48;5;180m" if is_light else "\033[48;5;94m"

            if piece:
                symbol = piece_symbols.get(piece.symbol(), piece.symbol())
                # Piece color
                fg = "\033[97m" if piece.color == chess.WHITE else "\033[30m"
                print(f"{bg}{fg} {symbol} \033[0m", end="")
            else:
                print(f"{bg}   \033[0m", end="")

        print()

    print("\n    ", end="")
    file_labels = "abcdefgh" if perspective == chess.WHITE else "hgfedcba"
    for f in file_labels:
        print(f" {f} ", end="")
    print("\n")


def play_interactive(engine: ChessEngine, player_color: chess.Color):
    """Play an interactive game against the engine"""
    board = chess.Board()

    print("\n" + "=" * 50)
    print("CHESS ENGINE - Interactive Play")
    print("=" * 50)
    print("\nCommands:")
    print("  - Enter moves in UCI format (e.g., e2e4)")
    print("  - Type 'quit' to exit")
    print("  - Type 'undo' to take back a move")
    print("  - Type 'hint' for a move suggestion")
    print("  - Type 'eval' to see position evaluation")
    print()

    while not board.is_game_over():
        print_board(board, player_color)

        if board.turn == player_color:
            # Player's turn
            print(f"Your turn ({('White' if player_color else 'Black')})")
            print(f"Legal moves: {', '.join(m.uci() for m in list(board.legal_moves)[:10])}...")

            while True:
                try:
                    cmd = input("\nYour move: ").strip().lower()

                    if cmd == "quit":
                        print("Thanks for playing!")
                        return

                    elif cmd == "undo":
                        if len(board.move_stack) >= 2:
                            board.pop()
                            board.pop()
                            print("Moves undone!")
                            break
                        else:
                            print("No moves to undo")
                            continue

                    elif cmd == "hint":
                        result = engine.search(board, time_limit=2.0)
                        print(f"Hint: {result.best_move.uci()} (eval: {result.score:.1f})")
                        continue

                    elif cmd == "eval":
                        _, value = engine.evaluate(board)
                        print(f"Evaluation: {value * 100:.1f} (+ = White advantage)")
                        continue

                    else:
                        move = chess.Move.from_uci(cmd)
                        if move in board.legal_moves:
                            board.push(move)
                            break
                        else:
                            print("Illegal move! Try again.")
                            continue

                except ValueError:
                    print("Invalid input. Enter a move like 'e2e4'")
                    continue

        else:
            # Engine's turn
            print(f"Engine thinking...")
            result = engine.search(board, time_limit=3.0)

            if result and result.best_move:
                print(f"\nEngine plays: {result.best_move.uci()}")
                print(f"  Evaluation: {result.score:.1f}")
                print(f"  Nodes: {result.nodes}")
                print(f"  PV: {' '.join(m.uci() for m in result.pv[:5])}")
                board.push(result.best_move)
            else:
                print("Engine couldn't find a move!")
                break

    # Game over
    print_board(board, player_color)
    print("\n" + "=" * 50)
    print(f"GAME OVER: {board.result()}")

    if board.is_checkmate():
        winner = "Black" if board.turn else "White"
        print(f"{winner} wins by checkmate!")
    elif board.is_stalemate():
        print("Draw by stalemate")
    elif board.is_insufficient_material():
        print("Draw by insufficient material")
    elif board.is_fifty_moves():
        print("Draw by 50-move rule")
    elif board.is_repetition():
        print("Draw by repetition")

    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Play chess against your engine")

    parser.add_argument(
        "--model",
        type=str,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--color",
        type=str,
        choices=["white", "black"],
        default="white",
        help="Color to play as (default: white)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Search depth (default: 4)",
    )
    parser.add_argument(
        "--uci",
        action="store_true",
        help="Run in UCI mode for GUI integration",
    )
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="Use pure policy network (no search)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (default: cuda)",
    )

    args = parser.parse_args()

    if args.uci:
        # UCI mode for chess GUIs
        uci = UCIEngine(args.model)
        uci.run()
    else:
        # Interactive mode
        engine = ChessEngine(
            model_path=args.model,
            device=args.device,
            search_depth=args.depth,
            use_search=not args.no_search,
        )

        player_color = chess.WHITE if args.color == "white" else chess.BLACK
        play_interactive(engine, player_color)


if __name__ == "__main__":
    main()
