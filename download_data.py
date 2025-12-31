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
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
import shutil


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
    split = f"train[:{num_positions}]"

    print(f"Downloading {num_positions:,} positions using streaming mode (FAST!)...")

    # Use streaming to avoid loading entire 784M dataset
    ds = load_dataset(
        "Lichess/chess-position-evaluations",
        split="train",
        streaming=True,
    )

    # Take only what we need
    ds = ds.take(num_positions)
    print(f"Streaming {num_positions:,} positions...\n")

    def fen_to_tensor(fen: str) -> np.ndarray:
        """Convert FEN string to 18x8x8 tensor

        Planes:
        0-5: White pieces (P, N, B, R, Q, K)
        6-11: Black pieces (p, n, b, r, q, k)
        12: Side to move (all 1s if white to move)
        13: White kingside castling
        14: White queenside castling
        15: Black kingside castling
        16: Black queenside castling
        17: En passant square
        """
        board = chess.Board(fen)
        tensor = np.zeros((18, 8, 8), dtype=np.float32)

        piece_to_plane = {
            (chess.PAWN, chess.WHITE): 0,
            (chess.KNIGHT, chess.WHITE): 1,
            (chess.BISHOP, chess.WHITE): 2,
            (chess.ROOK, chess.WHITE): 3,
            (chess.QUEEN, chess.WHITE): 4,
            (chess.KING, chess.WHITE): 5,
            (chess.PAWN, chess.BLACK): 6,
            (chess.KNIGHT, chess.BLACK): 7,
            (chess.BISHOP, chess.BLACK): 8,
            (chess.ROOK, chess.BLACK): 9,
            (chess.QUEEN, chess.BLACK): 10,
            (chess.KING, chess.BLACK): 11,
        }

        # Piece positions (planes 0-11)
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece:
                plane = piece_to_plane[(piece.piece_type, piece.color)]
                rank = sq // 8
                file = sq % 8
                tensor[plane, rank, file] = 1.0

        # Side to move (plane 12)
        if board.turn == chess.WHITE:
            tensor[12, :, :] = 1.0

        # Castling rights (planes 13-16)
        if board.has_kingside_castling_rights(chess.WHITE):
            tensor[13, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            tensor[14, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            tensor[15, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            tensor[16, :, :] = 1.0

        # En passant square (plane 17)
        if board.ep_square is not None:
            ep_rank = board.ep_square // 8
            ep_file = board.ep_square % 8
            tensor[17, ep_rank, ep_file] = 1.0

        return tensor

    def normalize_eval(cp: Optional[int], mate: Optional[int]) -> float:
        """Convert centipawn/mate to [-1, 1] value"""
        if mate is not None:
            # Mate score: positive = white wins, negative = black wins
            return 1.0 if mate > 0 else -1.0
        elif cp is not None:
            # Centipawn score: normalize with tanh
            # cp=1000 (~10 pawns) -> ~0.76
            # cp=300 (~3 pawns) -> ~0.29
            return float(np.tanh(cp / 1000.0))
        else:
            return 0.0

    print("Converting to training format...")

    # Process streaming dataset in batches
    boards = []
    values = []
    chunk_idx = 0
    total_processed = 0

    pbar = tqdm(total=num_positions, desc="Processing positions")

    for example in ds:
        try:
            tensor = fen_to_tensor(example['fen'])
            value = normalize_eval(example.get('cp'), example.get('mate'))
            boards.append(tensor)
            values.append(value)
            total_processed += 1
            pbar.update(1)
        except Exception:
            continue

        # Save chunk when batch is full
        if len(boards) >= batch_size:
            boards_arr = np.array(boards, dtype=np.float32)
            values_arr = np.array(values, dtype=np.float32)
            policies = np.zeros((len(boards_arr), 1858), dtype=np.float32)

            chunk_path = os.path.join(output_dir, f"chunk_{chunk_idx:06d}.npz")
            np.savez_compressed(
                chunk_path,
                boards=boards_arr,
                policies=policies,
                values=values_arr,
            )
            chunk_idx += 1
            boards = []
            values = []

    pbar.close()

    # Save remaining positions
    if boards:
        boards_arr = np.array(boards, dtype=np.float32)
        values_arr = np.array(values, dtype=np.float32)
        policies = np.zeros((len(boards_arr), 1858), dtype=np.float32)

        chunk_path = os.path.join(output_dir, f"chunk_{chunk_idx:06d}.npz")
        np.savez_compressed(
            chunk_path,
            boards=boards_arr,
            policies=policies,
            values=values_arr,
        )
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
    parser = argparse.ArgumentParser(
        description="Download chess training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_data.py                         # Download 10M positions (recommended)
  python download_data.py --positions 50000000    # Download 50M positions
  python download_data.py --dataset lichess_pgn   # Use old PGN method (slower)
        """
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
        help="Number of positions to download (default: 10M)",
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
