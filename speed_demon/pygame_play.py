import argparse
import os
import sys
from pathlib import Path

import chess
import chess.engine
import pygame


LIGHT_SQUARE = (238, 224, 196)
DARK_SQUARE = (153, 102, 68)
HIGHLIGHT = (240, 179, 90)
LAST_MOVE = (90, 200, 250)
TEXT_COLOR = (20, 18, 22)
BG_COLOR = (20, 18, 22)


def _configure_engine(engine: chess.engine.SimpleEngine, nnue_path: str, threads: int, hash_mb: int) -> None:
    options = {}
    if "Threads" in engine.options:
        options["Threads"] = threads
    if "Hash" in engine.options:
        options["Hash"] = hash_mb
    if nnue_path and "EvalFile" in engine.options:
        options["EvalFile"] = nnue_path
    if "Use NNUE" in engine.options:
        options["Use NNUE"] = True
    if options:
        engine.configure(options)


def _find_stockfish() -> str | None:
    env_path = os.environ.get("STOCKFISH_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    candidates = [
        "/root/.stockfish/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2",
        "/usr/games/stockfish",
        "/usr/bin/stockfish",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _find_latest_nnue(repo_root: Path) -> Path | None:
    nnue_dir = repo_root / "outputs" / "speed_demon" / "nnue"
    candidates: list[Path] = []
    if nnue_dir.exists():
        candidates = list(nnue_dir.glob("*.nnue"))
    if not candidates:
        fallback = repo_root / "speed_demon.nnue"
        if fallback.exists():
            return fallback
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _board_coords(flip: bool):
    files = ["a", "b", "c", "d", "e", "f", "g", "h"]
    ranks = ["8", "7", "6", "5", "4", "3", "2", "1"]
    if flip:
        files = list(reversed(files))
        ranks = list(reversed(ranks))
    return files, ranks


def _parse_square(square: str) -> int:
    if hasattr(chess, "parse_square"):
        return chess.parse_square(square)
    file_idx = ord(square[0]) - ord("a")
    rank_idx = int(square[1]) - 1
    return chess.square(file_idx, rank_idx)


def _square_from_mouse(x: int, y: int, square_size: int, flip: bool) -> str | None:
    if x < 0 or y < 0:
        return None
    file_idx = x // square_size
    rank_idx = y // square_size
    if file_idx > 7 or rank_idx > 7:
        return None
    files, ranks = _board_coords(flip)
    return f"{files[file_idx]}{ranks[rank_idx]}"


def _render_board(
    screen,
    font,
    game: chess.Board,
    square_size: int,
    flip: bool,
    selected: str | None,
    legal_targets: list[str],
    last_move: chess.Move | None,
):
    files, ranks = _board_coords(flip)
    for rank_idx, rank in enumerate(ranks):
        for file_idx, file in enumerate(files):
            square = f"{file}{rank}"
            is_light = (file_idx + rank_idx) % 2 == 0
            rect = pygame.Rect(
                file_idx * square_size,
                rank_idx * square_size,
                square_size,
                square_size,
            )
            color = LIGHT_SQUARE if is_light else DARK_SQUARE
            pygame.draw.rect(screen, color, rect)

            if last_move and (
                square == chess.square_name(last_move.from_square)
                or square == chess.square_name(last_move.to_square)
            ):
                pygame.draw.rect(screen, LAST_MOVE, rect, 4)
            if square == selected:
                pygame.draw.rect(screen, HIGHLIGHT, rect, 4)
            if square in legal_targets:
                pygame.draw.rect(screen, HIGHLIGHT, rect, 2)

            piece = game.piece_at(_parse_square(square))
            if piece:
                label = piece.symbol()
                text = font.render(label, True, TEXT_COLOR)
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)


def main() -> None:
    parser = argparse.ArgumentParser(description="Play against NNUE in pygame.")
    parser.add_argument("--stockfish", default=None, help="Path to Stockfish binary")
    parser.add_argument("--nnue", default=None, help="Path to .nnue file")
    parser.add_argument("--side", choices=["white", "black"], default="white")
    parser.add_argument("--time", type=float, default=0.05, help="Seconds per move")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--hash-mb", type=int, default=128)
    parser.add_argument("--square-size", type=int, default=80)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    stockfish_path = os.path.abspath(args.stockfish) if args.stockfish else _find_stockfish()
    if not stockfish_path:
        print("Stockfish not found. Pass --stockfish /path/to/stockfish.")
        sys.exit(2)

    nnue_path = None
    if args.nnue:
        nnue_path = os.path.abspath(args.nnue)
    else:
        latest_nnue = _find_latest_nnue(repo_root)
        if latest_nnue is not None:
            nnue_path = str(latest_nnue)
    if not nnue_path or not os.path.exists(nnue_path):
        print("NNUE not found. Pass --nnue /path/to/nnue.")
        sys.exit(2)

    print(f"Using Stockfish: {stockfish_path}")
    print(f"Using NNUE:      {nnue_path}")

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    _configure_engine(engine, nnue_path, args.threads, args.hash_mb)

    pygame.init()
    square_size = args.square_size
    board_px = square_size * 8
    status_height = 80
    screen = pygame.display.set_mode((board_px, board_px + status_height))
    pygame.display.set_caption("Speed Demon NNUE")
    font = pygame.font.Font(None, int(square_size * 0.6))
    status_font = pygame.font.Font(None, 22)

    game = chess.Board()
    player_is_white = args.side == "white"
    flip = not player_is_white
    selected = None
    legal_targets: list[str] = []
    last_move = None
    engine_thinking = False
    status = "Your move."

    def update_status(text: str) -> None:
        nonlocal status
        status = text

    def draw() -> None:
        screen.fill(BG_COLOR)
        _render_board(screen, font, game, square_size, flip, selected, legal_targets, last_move)
        status_rect = pygame.Rect(0, board_px, board_px, status_height)
        pygame.draw.rect(screen, BG_COLOR, status_rect)
        text = status_font.render(status, True, (230, 230, 230))
        screen.blit(text, (10, board_px + 20))
        pygame.display.flip()

    def engine_move() -> None:
        nonlocal last_move, engine_thinking
        if game.is_game_over():
            update_status(f"Game over: {game.result()}")
            return
        if game.turn == chess.WHITE and player_is_white:
            update_status("Your move.")
            return
        if game.turn == chess.BLACK and not player_is_white:
            update_status("Your move.")
            return
        engine_thinking = True
        update_status("Engine thinking...")
        draw()
        result = engine.play(game, chess.engine.Limit(time=args.time))
        if result.move:
            game.push(result.move)
            last_move = result.move
        engine_thinking = False
        update_status("Your move.")

    draw()
    engine_move()

    clock = pygame.time.Clock()
    running = True
    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    game.reset()
                    selected = None
                    legal_targets = []
                    last_move = None
                    update_status("New game.")
                    engine_move()
                elif event.key == pygame.K_u:
                    if len(game.move_stack) >= 2:
                        game.pop()
                        game.pop()
                        last_move = None
                        update_status("Move undone.")
                elif event.key == pygame.K_f:
                    flip = not flip
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if engine_thinking:
                    continue
                mx, my = event.pos
                if my >= board_px:
                    continue
                square = _square_from_mouse(mx, my, square_size, flip)
                if not square:
                    continue
                piece = game.piece_at(_parse_square(square))
                if selected:
                    move = chess.Move(
                        from_square=_parse_square(selected),
                        to_square=_parse_square(square),
                        promotion=chess.QUEEN,
                    )
                    if move in game.legal_moves:
                        game.push(move)
                        last_move = move
                        selected = None
                        legal_targets = []
                        update_status("Engine thinking...")
                        engine_move()
                    else:
                        selected = None
                        legal_targets = []
                else:
                    if piece and ((piece.color == chess.WHITE) == player_is_white) and piece.color == game.turn:
                        selected = square
                        legal_targets = [
                            chess.square_name(m.to_square)
                            for m in game.legal_moves
                            if m.from_square == _parse_square(square)
                        ]
        draw()

    engine.quit()
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
