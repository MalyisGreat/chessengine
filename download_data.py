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

# Global chess import for multiprocessing workers
import chess


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

    def save_chunk(boards_arr, values_arr, idx):
        """Save a chunk of data"""
        policies = np.zeros((len(boards_arr), 1858), dtype=np.float32)
        chunk_path = os.path.join(output_dir, f"chunk_{idx:06d}.npz")
        np.savez_compressed(chunk_path, boards=boards_arr, policies=policies, values=values_arr)

    chunk_idx = 0

    if not use_streaming:
        # FAST PATH: Direct batch processing (no multiprocessing overhead)
        total = len(ds)
        process_batch_size = batch_size  # Process one chunk at a time

        pbar = tqdm(total=total, desc="Processing positions")

        for start in range(0, total, process_batch_size):
            end = min(start + process_batch_size, total)
            batch = ds[start:end]
            batch_len = len(batch['fen'])

            # Pre-allocate numpy arrays
            boards_arr = np.zeros((batch_len, 18, 8, 8), dtype=np.float32)
            values_arr = np.zeros(batch_len, dtype=np.float32)

            valid_count = 0
            for i in range(batch_len):
                try:
                    fen = batch['fen'][i]
                    cp = batch['cp'][i] if 'cp' in batch and batch['cp'][i] is not None else None
                    mate = batch['mate'][i] if 'mate' in batch and batch['mate'][i] is not None else None

                    # Inline FEN parsing for speed
                    board = chess.Board(fen)

                    # Piece planes (0-11)
                    for sq in chess.SQUARES:
                        piece = board.piece_at(sq)
                        if piece:
                            plane = (piece.piece_type - 1) + (6 if piece.color == chess.BLACK else 0)
                            boards_arr[valid_count, plane, sq // 8, sq % 8] = 1.0

                    # Side to move (plane 12)
                    if board.turn == chess.WHITE:
                        boards_arr[valid_count, 12, :, :] = 1.0

                    # Castling (planes 13-16)
                    if board.has_kingside_castling_rights(chess.WHITE):
                        boards_arr[valid_count, 13, :, :] = 1.0
                    if board.has_queenside_castling_rights(chess.WHITE):
                        boards_arr[valid_count, 14, :, :] = 1.0
                    if board.has_kingside_castling_rights(chess.BLACK):
                        boards_arr[valid_count, 15, :, :] = 1.0
                    if board.has_queenside_castling_rights(chess.BLACK):
                        boards_arr[valid_count, 16, :, :] = 1.0

                    # En passant (plane 17)
                    if board.ep_square is not None:
                        boards_arr[valid_count, 17, board.ep_square // 8, board.ep_square % 8] = 1.0

                    # Value
                    if mate is not None:
                        values_arr[valid_count] = 1.0 if mate > 0 else -1.0
                    elif cp is not None:
                        values_arr[valid_count] = float(np.tanh(cp / 1000.0))

                    valid_count += 1
                except Exception:
                    continue

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
            try:
                fen = example['fen']
                cp = example.get('cp')
                mate = example.get('mate')

                # Parse FEN
                board = chess.Board(fen)
                tensor = np.zeros((18, 8, 8), dtype=np.float32)

                # Piece planes
                for sq in chess.SQUARES:
                    piece = board.piece_at(sq)
                    if piece:
                        plane = (piece.piece_type - 1) + (6 if piece.color == chess.BLACK else 0)
                        tensor[plane, sq // 8, sq % 8] = 1.0

                # Side to move
                if board.turn == chess.WHITE:
                    tensor[12, :, :] = 1.0

                # Castling
                if board.has_kingside_castling_rights(chess.WHITE):
                    tensor[13, :, :] = 1.0
                if board.has_queenside_castling_rights(chess.WHITE):
                    tensor[14, :, :] = 1.0
                if board.has_kingside_castling_rights(chess.BLACK):
                    tensor[15, :, :] = 1.0
                if board.has_queenside_castling_rights(chess.BLACK):
                    tensor[16, :, :] = 1.0

                # En passant
                if board.ep_square is not None:
                    tensor[17, board.ep_square // 8, board.ep_square % 8] = 1.0

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

            except Exception:
                pbar.update(1)
                continue

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


def download_lichess_pgn_parallel(
    output_dir: str,
    max_files: Optional[int] = None,
    num_threads: int = 8,
) -> List[str]:
    """
    OLD METHOD: Download Lichess Elite PGN files.
    This is slower - use download_lichess_evaluations() instead.
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
  python download_data.py --preset test           # 2M positions (quick test)
  python download_data.py --preset small          # 10M positions (baseline)
  python download_data.py --preset medium         # 50M positions (strong)
  python download_data.py --preset large          # 100M positions (superhuman)
  python download_data.py --positions 25000000    # Custom: 25M positions
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
        choices=["lichess_eval", "lichess_pgn", "sample"],
        default="lichess_eval",
        help="Dataset source (default: lichess_eval - fast HuggingFace download)",
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

    if args.dataset == "lichess_eval":
        # RECOMMENDED: Fast download from HuggingFace
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
