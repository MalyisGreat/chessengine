"""
Data Download and Processing Script

Downloads chess training data from Hugging Face (pre-evaluated positions).
This is MUCH faster than processing PGN files.

Usage:
    python download_data.py                          # Downloads 10M positions (~5 min)
    python download_data.py --positions 50000000     # Downloads 50M positions
    python download_data.py --dataset lichess_pgn    # Old PGN method (slower)
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import shutil

# Global chess import (only used for streaming fallback)
import chess


# Fast FEN parsing - avoids chess.Board() overhead
PIECE_TO_PLANE = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White pieces
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,  # Black pieces
}


def fast_parse_fen(fen: str, tensor: np.ndarray) -> bool:
    """
    Parse FEN string directly into pre-allocated tensor (18x8x8).
    Returns True on success, False on error.

    Much faster than chess.Board(fen) - avoids object creation overhead.
    """
    try:
        parts = fen.split(' ')
        if len(parts) < 4:
            return False

        position, turn, castling, ep = parts[0], parts[1], parts[2], parts[3]

        # Parse piece positions (planes 0-11)
        rank = 7  # Start from rank 8 (index 7)
        file = 0

        for char in position:
            if char == '/':
                rank -= 1
                file = 0
            elif char.isdigit():
                file += int(char)
            elif char in PIECE_TO_PLANE:
                plane = PIECE_TO_PLANE[char]
                tensor[plane, rank, file] = 1.0
                file += 1
            else:
                return False

        # Side to move (plane 12)
        if turn == 'w':
            tensor[12, :, :] = 1.0

        # Castling rights (planes 13-16)
        if 'K' in castling:
            tensor[13, :, :] = 1.0
        if 'Q' in castling:
            tensor[14, :, :] = 1.0
        if 'k' in castling:
            tensor[15, :, :] = 1.0
        if 'q' in castling:
            tensor[16, :, :] = 1.0

        # En passant (plane 17)
        if ep != '-' and len(ep) == 2:
            ep_file = ord(ep[0]) - ord('a')
            ep_rank = int(ep[1]) - 1
            if 0 <= ep_file < 8 and 0 <= ep_rank < 8:
                tensor[17, ep_rank, ep_file] = 1.0

        return True
    except:
        return False


def _fen_to_tensor(fen: str) -> np.ndarray:
    """Convert FEN string to 18x8x8 tensor (module-level for multiprocessing)"""
    board = chess.Board(fen)
    tensor = np.zeros((18, 8, 8), dtype=np.float32)

    piece_to_plane = {
        (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2, (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8, (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
    }

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            plane = piece_to_plane[(piece.piece_type, piece.color)]
            tensor[plane, sq // 8, sq % 8] = 1.0

    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1.0
    if board.ep_square is not None:
        tensor[17, board.ep_square // 8, board.ep_square % 8] = 1.0

    return tensor


def _normalize_eval(cp: Optional[int], mate: Optional[int]) -> float:
    """Convert centipawn/mate to [-1, 1] value"""
    if mate is not None:
        return 1.0 if mate > 0 else -1.0
    elif cp is not None:
        return float(np.tanh(cp / 1000.0))
    return 0.0


def _process_example(args: Tuple[str, Optional[int], Optional[int]]) -> Optional[Tuple[np.ndarray, float]]:
    """Process a single example (for multiprocessing)"""
    fen, cp, mate = args
    try:
        tensor = _fen_to_tensor(fen)
        value = _normalize_eval(cp, mate)
        return (tensor, value)
    except Exception:
        return None


def download_lichess_evaluations(
    output_dir: str = "./data/train",
    num_positions: int = 10_000_000,
    batch_size: int = 100_000,
) -> str:
    """
    Download pre-evaluated chess positions from Lichess/Hugging Face.

    This is the FAST method - positions already have Stockfish evaluations.
    No PGN parsing needed!

    Args:
        output_dir: Where to save processed data
        num_positions: Number of positions to download (default 10M)
        batch_size: Positions per .npz file

    Returns:
        Path to output directory
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        os.system("pip install datasets")
        from datasets import load_dataset

    import chess

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"DOWNLOADING LICHESS POSITION EVALUATIONS")
    print(f"{'='*60}")
    print(f"Positions: {num_positions:,}")
    print(f"Source: Hugging Face (Lichess/chess-position-evaluations)")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # Download from Hugging Face
    print("Connecting to Hugging Face...")

    # Try non-streaming first (uses cached data if available - MUCH faster!)
    try:
        print(f"Loading {num_positions:,} positions from cache (fast!)...")
        ds = load_dataset(
            "Lichess/chess-position-evaluations",
            split=f"train[:{num_positions}]",
            streaming=False,  # Use cached parquet files
        )
        use_streaming = False
        print(f"Loaded {len(ds):,} positions from cache\n")
    except Exception as e:
        # Fall back to streaming if cache not available
        print(f"Cache not available, using streaming mode...")
        ds = load_dataset(
            "Lichess/chess-position-evaluations",
            split="train",
            streaming=True,
        )
        ds = ds.take(num_positions)
        use_streaming = True
        print(f"Streaming {num_positions:,} positions...\n")

    print(f"Converting to training format (optimized single-thread)...")

    # Get actual policy size from encoder (must match model)
    from data.encoder import BoardEncoder
    encoder = BoardEncoder()
    policy_size = encoder.num_moves
    print(f"Policy size: {policy_size} (from encoder)")

    def save_chunk(boards_arr, values_arr, idx):
        """Save a chunk of data"""
        # NOTE: policies are zeros because this dataset has no move data
        # Training will only work for value head, not policy head
        policies = np.zeros((len(boards_arr), policy_size), dtype=np.float32)
        chunk_path = os.path.join(output_dir, f"chunk_{idx:06d}.npz")
        np.savez_compressed(chunk_path, boards=boards_arr, policies=policies, values=values_arr)

    chunk_idx = 0

    if not use_streaming:
        # FAST PATH: Load all data into memory first, then process
        total = len(ds)

        print("Loading all FEN strings into memory...")
        # Convert to pandas for faster access (loads fully into RAM)
        df = ds.to_pandas()
        all_fens = df['fen'].tolist()
        all_cps = df['cp'].tolist() if 'cp' in df.columns else [None] * total
        all_mates = df['mate'].tolist() if 'mate' in df.columns else [None] * total
        del df  # Free pandas dataframe
        print(f"Loaded {len(all_fens):,} positions into memory")

        process_batch_size = batch_size

        pbar = tqdm(total=total, desc="Processing positions")

        for start in range(0, total, process_batch_size):
            end = min(start + process_batch_size, total)
            batch_len = end - start

            # Pre-allocate numpy arrays
            boards_arr = np.zeros((batch_len, 18, 8, 8), dtype=np.float32)
            values_arr = np.zeros(batch_len, dtype=np.float32)

            valid_count = 0

            for i in range(batch_len):
                idx = start + i
                fen = all_fens[idx]
                cp = all_cps[idx]
                mate = all_mates[idx]

                # Fast FEN parsing (no chess.Board overhead)
                if not fast_parse_fen(fen, boards_arr[valid_count]):
                    boards_arr[valid_count] = 0  # Clear partial writes
                    continue

                # Value
                if mate is not None:
                    values_arr[valid_count] = 1.0 if mate > 0 else -1.0
                elif cp is not None:
                    values_arr[valid_count] = np.tanh(cp / 1000.0)

                valid_count += 1

            # Save chunk
            if valid_count > 0:
                save_chunk(boards_arr[:valid_count], values_arr[:valid_count], chunk_idx)
                chunk_idx += 1

            pbar.update(end - start)

        pbar.close()

    else:
        # SLOW PATH: Streaming mode (network bound)
        pbar = tqdm(total=num_positions, desc="Processing positions (streaming)")
        boards_list = []
        values_list = []

        for example in ds:
            fen = example['fen']
            cp = example.get('cp')
            mate = example.get('mate')

            # Fast FEN parsing
            tensor = np.zeros((18, 8, 8), dtype=np.float32)
            if not fast_parse_fen(fen, tensor):
                pbar.update(1)
                continue

            # Value
            if mate is not None:
                value = 1.0 if mate > 0 else -1.0
            elif cp is not None:
                value = float(np.tanh(cp / 1000.0))
            else:
                value = 0.0

            boards_list.append(tensor)
            values_list.append(value)
            pbar.update(1)

            # Save chunks as we go
            if len(boards_list) >= batch_size:
                boards_arr = np.array(boards_list[:batch_size])
                values_arr = np.array(values_list[:batch_size])
                save_chunk(boards_arr, values_arr, chunk_idx)
                chunk_idx += 1
                boards_list = boards_list[batch_size:]
                values_list = values_list[batch_size:]

        pbar.close()

        # Save remaining
        if boards_list:
            boards_arr = np.array(boards_list)
            values_arr = np.array(values_list)
            save_chunk(boards_arr, values_arr, chunk_idx)
            chunk_idx += 1

    # Count total positions saved
    total_positions = 0
    for f in os.listdir(output_dir):
        if f.endswith('.npz'):
            data = np.load(os.path.join(output_dir, f))
            total_positions += len(data['boards'])

    print(f"\n{'='*60}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"Total positions: {total_positions:,}")
    print(f"Saved to: {output_dir}")
    print(f"Chunks: {chunk_idx}")
    print(f"{'='*60}\n")

    return output_dir


def download_lichess_games(
    output_dir: str = "./data/train",
    num_positions: int = 10_000_000,
    batch_size: int = 100_000,
    min_elo: int = 2200,
) -> str:
    """
    Download Lichess Elite games with MOVES for policy training.

    This extracts actual moves played, giving proper policy targets.
    Uses game outcome as value target.

    Args:
        output_dir: Where to save processed data
        num_positions: Target number of positions
        batch_size: Positions per .npz file
        min_elo: Minimum player ELO

    Returns:
        Path to output directory
    """
    import chess
    import chess.pgn
    import requests
    import zipfile
    import io

    from data.encoder import BoardEncoder

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"DOWNLOADING LICHESS ELITE GAMES (WITH MOVES)")
    print(f"{'='*60}")
    print(f"Target positions: {num_positions:,}")
    print(f"Min ELO: {min_elo}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    encoder = BoardEncoder()
    print(f"Policy size: {encoder.num_moves} (from encoder)")

    # Download Lichess Elite PGN files
    LICHESS_ELITE_BASE = "https://database.nikonoel.fr/"
    MONTHLY_FILES = [
        "lichess_elite_2023-01.zip", "lichess_elite_2023-02.zip",
        "lichess_elite_2023-03.zip", "lichess_elite_2023-04.zip",
        "lichess_elite_2023-05.zip", "lichess_elite_2023-06.zip",
        "lichess_elite_2022-01.zip", "lichess_elite_2022-02.zip",
        "lichess_elite_2022-03.zip", "lichess_elite_2022-04.zip",
        "lichess_elite_2022-05.zip", "lichess_elite_2022-06.zip",
        "lichess_elite_2021-01.zip", "lichess_elite_2021-02.zip",
    ]

    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # Collect positions
    boards_list = []
    policies_list = []
    values_list = []
    chunk_idx = 0
    total_positions = 0
    total_games = 0

    def save_chunk():
        nonlocal chunk_idx, boards_list, policies_list, values_list
        if not boards_list:
            return
        chunk_path = os.path.join(output_dir, f"chunk_{chunk_idx:06d}.npz")
        np.savez_compressed(
            chunk_path,
            boards=np.array(boards_list, dtype=np.float32),
            policies=np.array(policies_list, dtype=np.float32),
            values=np.array(values_list, dtype=np.float32),
        )
        chunk_idx += 1
        boards_list = []
        policies_list = []
        values_list = []

    pbar = tqdm(total=num_positions, desc="Extracting positions")

    for pgn_file in MONTHLY_FILES:
        if total_positions >= num_positions:
            break

        zip_path = os.path.join(raw_dir, pgn_file)

        # Download if needed
        if not os.path.exists(zip_path):
            print(f"\nDownloading {pgn_file}...")
            try:
                url = f"{LICHESS_ELITE_BASE}{pgn_file}"
                response = requests.get(url, stream=True, timeout=120)
                response.raise_for_status()
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
            except Exception as e:
                print(f"  Failed to download: {e}")
                continue

        # Extract and process PGN
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for name in zf.namelist():
                    if not name.endswith('.pgn'):
                        continue

                    with zf.open(name) as pgn_bytes:
                        pgn_text = io.TextIOWrapper(pgn_bytes, encoding='utf-8', errors='ignore')

                        while total_positions < num_positions:
                            try:
                                game = chess.pgn.read_game(pgn_text)
                                if game is None:
                                    break
                            except Exception:
                                continue

                            # Check ELO
                            try:
                                white_elo = int(game.headers.get('WhiteElo', '0'))
                                black_elo = int(game.headers.get('BlackElo', '0'))
                                if white_elo < min_elo or black_elo < min_elo:
                                    continue
                            except ValueError:
                                continue

                            # Get result
                            result = game.headers.get('Result', '*')
                            if result == '1-0':
                                white_value, black_value = 1.0, -1.0
                            elif result == '0-1':
                                white_value, black_value = -1.0, 1.0
                            elif result == '1/2-1/2':
                                white_value, black_value = 0.0, 0.0
                            else:
                                continue  # Unknown result

                            # Process all positions
                            board = game.board()
                            moves = list(game.mainline_moves())

                            for i, move in enumerate(moves[:-1]):  # Skip last move
                                # Encode board
                                board_tensor = encoder.encode_board(board)

                                # Encode move as policy target
                                policy = np.zeros(encoder.num_moves, dtype=np.float32)
                                move_idx = encoder.encode_move(move)
                                if move_idx >= 0:
                                    policy[move_idx] = 1.0
                                else:
                                    board.push(move)
                                    continue  # Skip moves we can't encode

                                # Value from current player's perspective
                                value = white_value if board.turn == chess.WHITE else black_value

                                boards_list.append(board_tensor)
                                policies_list.append(policy)
                                values_list.append(value)
                                total_positions += 1
                                pbar.update(1)

                                board.push(move)

                                # Save chunk if full
                                if len(boards_list) >= batch_size:
                                    save_chunk()

                                if total_positions >= num_positions:
                                    break

                            total_games += 1

                            if total_positions >= num_positions:
                                break

        except Exception as e:
            print(f"  Error processing {pgn_file}: {e}")
            continue

    pbar.close()

    # Save remaining
    save_chunk()

    print(f"\n{'='*60}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"Total games: {total_games:,}")
    print(f"Total positions: {total_positions:,}")
    print(f"Saved to: {output_dir}")
    print(f"Chunks: {chunk_idx}")
    print(f"{'='*60}\n")

    return output_dir


def download_lichess_pgn_parallel(
    output_dir: str,
    max_files: Optional[int] = None,
    num_threads: int = 8,
) -> List[str]:
    """
    OLD METHOD: Download Lichess Elite PGN files.
    This is slower - use download_lichess_games() instead.
    """
    import requests
    import zipfile
    import threading
    from concurrent.futures import as_completed

    LICHESS_ELITE_BASE = "https://database.nikonoel.fr/"

    MONTHLY_FILES = [
        "lichess_elite_2020-06.zip", "lichess_elite_2020-07.zip",
        "lichess_elite_2020-08.zip", "lichess_elite_2020-09.zip",
        "lichess_elite_2020-10.zip", "lichess_elite_2020-11.zip",
        "lichess_elite_2020-12.zip", "lichess_elite_2021-01.zip",
        "lichess_elite_2021-02.zip", "lichess_elite_2021-03.zip",
        "lichess_elite_2021-04.zip", "lichess_elite_2021-05.zip",
        "lichess_elite_2021-06.zip", "lichess_elite_2021-07.zip",
        "lichess_elite_2021-08.zip", "lichess_elite_2021-09.zip",
        "lichess_elite_2021-10.zip", "lichess_elite_2021-11.zip",
        "lichess_elite_2021-12.zip", "lichess_elite_2022-01.zip",
    ]

    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    files_to_download = MONTHLY_FILES[:max_files] if max_files else MONTHLY_FILES

    def download_and_extract(filename):
        url = f"{LICHESS_ELITE_BASE}{filename}"
        zip_path = os.path.join(raw_dir, filename)

        if os.path.exists(zip_path):
            pass
        else:
            try:
                response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
            except Exception as e:
                return False, filename, []

        try:
            extracted = []
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for name in zf.namelist():
                    if name.endswith('.pgn'):
                        zf.extract(name, raw_dir)
                        extracted.append(os.path.join(raw_dir, name))
            return True, filename, extracted
        except Exception:
            return False, filename, []

    print(f"Downloading {len(files_to_download)} files with {num_threads} threads...")

    all_pgn_files = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(download_and_extract, f): f for f in files_to_download}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            success, name, extracted = future.result()
            if success:
                all_pgn_files.extend(extracted)

    return all_pgn_files


def main():
    # Preset configurations
    PRESETS = {
        "test": 2_000_000,      # Quick test run
        "small": 10_000_000,    # 10M - good baseline
        "medium": 50_000_000,   # 50M - strong performance
        "large": 100_000_000,   # 100M - superhuman
        "full": 200_000_000,    # 200M - maximum strength
    }

    parser = argparse.ArgumentParser(
        description="Download chess training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_data.py --preset test                          # 2M positions with moves
  python download_data.py --preset small                         # 10M positions (baseline)
  python download_data.py --preset medium                        # 50M positions (strong)
  python download_data.py --dataset lichess_eval --preset test   # Value-only (no moves)
        """
    )

    parser.add_argument(
        "--preset",
        type=str,
        choices=list(PRESETS.keys()),
        help="Use a preset size: test(2M), small(10M), medium(50M), large(100M), full(200M)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["lichess_eval", "lichess_games", "lichess_pgn", "sample"],
        default="lichess_games",
        help="Dataset source: lichess_games (with moves, best for training), lichess_eval (value only)",
    )
    parser.add_argument(
        "--positions",
        type=int,
        default=10_000_000,
        help="Number of positions to download (default: 10M, overridden by --preset)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/train",
        help="Output directory (default: ./data/train)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100_000,
        help="Positions per chunk file (default: 100K)",
    )

    args = parser.parse_args()

    # Preset overrides --positions
    if args.preset:
        args.positions = PRESETS[args.preset]
        print(f"Using preset '{args.preset}': {args.positions:,} positions")

    if args.dataset == "lichess_games":
        # RECOMMENDED: Games with moves for policy training
        download_lichess_games(
            output_dir=args.output,
            num_positions=args.positions,
            batch_size=args.batch_size,
        )

    elif args.dataset == "lichess_eval":
        # Fast download from HuggingFace (value training only)
        print("WARNING: lichess_eval has no move data - policy training won't work!")
        print("Use --dataset lichess_games for proper policy+value training.\n")
        download_lichess_evaluations(
            output_dir=args.output,
            num_positions=args.positions,
            batch_size=args.batch_size,
        )

    elif args.dataset == "lichess_pgn":
        # OLD METHOD: Download and process PGN files
        print("Using old PGN method (slower). Consider using --dataset lichess_eval instead.")
        pgn_files = download_lichess_pgn_parallel(
            output_dir="./data",
            max_files=20,
        )
        if pgn_files:
            from data.encoder import BoardEncoder
            from data.dataset import process_pgn_to_npz
            encoder = BoardEncoder()
            for i, pgn in enumerate(pgn_files):
                out_dir = os.path.join(args.output, f"file_{i:03d}")
                os.makedirs(out_dir, exist_ok=True)
                process_pgn_to_npz(pgn, out_dir, encoder, min_elo=2300, max_games=5000)

    elif args.dataset == "sample":
        # Generate random sample data for testing
        import chess
        os.makedirs(args.output, exist_ok=True)
        print("Generating sample data...")

        boards = []
        values = []

        for _ in tqdm(range(10000), desc="Generating"):
            board = chess.Board()
            for _ in range(np.random.randint(5, 50)):
                legal = list(board.legal_moves)
                if not legal:
                    break
                board.push(np.random.choice(legal))

            tensor = np.zeros((12, 8, 8), dtype=np.float32)
            for sq in chess.SQUARES:
                piece = board.piece_at(sq)
                if piece:
                    plane = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
                    tensor[plane, sq // 8, sq % 8] = 1.0

            boards.append(tensor)
            values.append(np.random.uniform(-1, 1))

        np.savez(
            os.path.join(args.output, "sample.npz"),
            boards=np.array(boards),
            policies=np.zeros((len(boards), 1858)),
            values=np.array(values),
        )
        print(f"Saved sample data to {args.output}")

    print("\nDone! Now run:")
    print("  python train.py --data ./data/train")


if __name__ == "__main__":
    main()
