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
import json
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


def _save_npz_chunk(output_dir: str, name: str, boards: np.ndarray,
                    policies: np.ndarray, values: np.ndarray, compress: bool) -> str:
    chunk_path = os.path.join(output_dir, name)
    if compress:
        np.savez_compressed(chunk_path, boards=boards, policies=policies, values=values)
    else:
        np.savez(chunk_path, boards=boards, policies=policies, values=values)
    return chunk_path


def _maybe_write_metadata_cache(output_dir: str, files_info: List[dict],
                                board_shape: Tuple[int, int, int],
                                policy_shape: Tuple[int]) -> None:
    if not files_info:
        return

    disk_files = [f for f in os.listdir(output_dir) if f.endswith('.npz')]
    cached_files = {f['name'] for f in files_info}
    if set(disk_files) != cached_files:
        print("Skipping metadata cache (output directory contains other .npz files).")
        return

    cache_path = os.path.join(output_dir, "_metadata.json")
    cache = {
        'files': files_info,
        'board_shape': list(board_shape),
        'policy_shape': list(policy_shape),
    }
    try:
        with open(cache_path, 'w') as f:
            json.dump(cache, f)
    except Exception:
        print("Warning: Failed to write metadata cache.")


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
    compress: bool = True,
) -> str:
    """
    Download pre-evaluated chess positions from Lichess/Hugging Face.

    This is the FAST method - positions already have Stockfish evaluations.
    No PGN parsing needed!

    Args:
        output_dir: Where to save processed data
        num_positions: Number of positions to download (default 10M)
        batch_size: Positions per .npz file
        compress: Use compressed .npz files (smaller, slower)

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

    files_info = []
    total_positions = 0

    def save_chunk(boards_arr, values_arr, idx):
        """Save a chunk of data"""
        # NOTE: policies are zeros because this dataset has no move data
        # Training will only work for value head, not policy head
        policies = np.zeros((len(boards_arr), policy_size), dtype=np.float32)
        chunk_name = f"chunk_{idx:06d}.npz"
        chunk_path = _save_npz_chunk(output_dir, chunk_name, boards_arr, policies, values_arr, compress)
        files_info.append({
            'name': chunk_name,
            'size': len(boards_arr),
            'mtime': os.path.getmtime(chunk_path),
        })

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
                total_positions += valid_count

            pbar.update(end - start)

        pbar.close()

    else:
        # SLOW PATH: Streaming mode (network bound)
        pbar = tqdm(total=num_positions, desc="Processing positions (streaming)")
        boards_arr = np.zeros((batch_size, 18, 8, 8), dtype=np.float32)
        values_arr = np.zeros(batch_size, dtype=np.float32)
        count = 0

        for example in ds:
            fen = example['fen']
            cp = example.get('cp')
            mate = example.get('mate')

            # Fast FEN parsing
            if not fast_parse_fen(fen, boards_arr[count]):
                boards_arr[count] = 0
                pbar.update(1)
                continue

            # Value
            if mate is not None:
                value = 1.0 if mate > 0 else -1.0
            elif cp is not None:
                value = float(np.tanh(cp / 1000.0))
            else:
                value = 0.0

            values_arr[count] = value
            count += 1
            pbar.update(1)

            # Save chunks as we go
            if count >= batch_size:
                save_chunk(boards_arr, values_arr, chunk_idx)
                chunk_idx += 1
                total_positions += count
                boards_arr = np.zeros((batch_size, 18, 8, 8), dtype=np.float32)
                values_arr = np.zeros(batch_size, dtype=np.float32)
                count = 0

        pbar.close()

        # Save remaining
        if count > 0:
            save_chunk(boards_arr[:count], values_arr[:count], chunk_idx)
            chunk_idx += 1
            total_positions += count

    _maybe_write_metadata_cache(
        output_dir,
        files_info,
        board_shape=(18, 8, 8),
        policy_shape=(policy_size,),
    )

    print(f"\n{'='*60}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"Total positions: {total_positions:,}")
    print(f"Saved to: {output_dir}")
    print(f"Chunks: {chunk_idx}")
    print(f"{'='*60}\n")

    return output_dir


def _process_game_batch(args):
    """
    Process a batch of games in a worker process.
    Returns (boards, policies, values, num_games) tuple.
    """
    import chess
    import chess.pgn
    import io
    from data.encoder import BoardEncoder

    game_texts, min_elo = args
    encoder = BoardEncoder()

    boards = []
    policies = []
    values = []
    games_processed = 0

    for game_text in game_texts:
        try:
            pgn_io = io.StringIO(game_text)
            game = chess.pgn.read_game(pgn_io)
            if game is None:
                continue
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
            continue

        # Process positions
        board = game.board()
        moves_list = list(game.mainline_moves())

        for move in moves_list[:-1]:
            board_tensor = encoder.encode_board(board)

            policy = np.zeros(encoder.num_moves, dtype=np.float32)
            move_idx = encoder.encode_move(move)
            if move_idx >= 0:
                policy[move_idx] = 1.0
            else:
                board.push(move)
                continue

            value = white_value if board.turn == chess.WHITE else black_value

            boards.append(board_tensor)
            policies.append(policy)
            values.append(value)

            board.push(move)

        games_processed += 1

    if boards:
        return (
            np.array(boards, dtype=np.float32),
            np.array(policies, dtype=np.float32),
            np.array(values, dtype=np.float32),
            games_processed
        )
    return None


def _split_pgn_into_games(pgn_content: str) -> List[str]:
    """Split PGN content into individual game strings."""
    games = []
    current_game = []
    in_moves = False

    for line in pgn_content.split('\n'):
        if line.startswith('[Event '):
            if current_game:
                games.append('\n'.join(current_game))
                current_game = []
            in_moves = False
        current_game.append(line)

        # Detect end of game (result token)
        if not line.startswith('[') and line.strip():
            in_moves = True
        if in_moves and ('1-0' in line or '0-1' in line or '1/2-1/2' in line or '*' in line):
            if current_game:
                games.append('\n'.join(current_game))
                current_game = []
            in_moves = False

    if current_game:
        games.append('\n'.join(current_game))

    return games


def _process_zip_file(args):
    """
    Process an entire zip file in a worker process.
    Returns list of (boards, policies, values) arrays.

    This runs in a separate process so it can be parallelized across zip files.
    """
    import zipfile
    import numpy as np
    import chess
    import chess.pgn
    import io
    from data.encoder import BoardEncoder

    zip_path, min_elo, positions_per_chunk = args
    encoder = BoardEncoder()

    all_results = []
    total_games = 0

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for name in zf.namelist():
                if not name.endswith('.pgn'):
                    continue

                # Process all games in this PGN file
                boards = []
                policies = []
                values = []

                with zf.open(name) as pgn_bytes:
                    pgn_text = io.TextIOWrapper(pgn_bytes, encoding='utf-8', errors='ignore')

                    while True:
                        try:
                            game = chess.pgn.read_game(pgn_text)
                        except Exception:
                            break

                        if game is None:
                            break

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
                            continue

                        # Process positions
                        board = game.board()
                        prev_move = None

                        for move in game.mainline_moves():
                            if prev_move is not None:
                                board_tensor = encoder.encode_board(board)

                                policy = np.zeros(encoder.num_moves, dtype=np.float32)
                                move_idx = encoder.encode_move(prev_move)
                                if move_idx >= 0:
                                    policy[move_idx] = 1.0

                                    value = white_value if board.turn == chess.WHITE else black_value

                                    boards.append(board_tensor)
                                    policies.append(policy)
                                    values.append(value)

                                board.push(prev_move)

                            prev_move = move

                        total_games += 1

                        if positions_per_chunk and len(boards) >= positions_per_chunk:
                            all_results.append((
                                np.array(boards, dtype=np.float32),
                                np.array(policies, dtype=np.float32),
                                np.array(values, dtype=np.float32),
                            ))
                            boards = []
                            policies = []
                            values = []

                if boards:
                    all_results.append((
                        np.array(boards, dtype=np.float32),
                        np.array(policies, dtype=np.float32),
                        np.array(values, dtype=np.float32),
                    ))

    except Exception as e:
        return None, 0, str(e)

    return all_results, total_games, None


def download_lichess_games(
    output_dir: str = "./data/train",
    num_positions: int = 10_000_000,
    batch_size: int = 100_000,
    min_elo: int = 2200,
    num_workers: int = 0,  # 0 = auto-detect
    use_ram_disk: bool = False,  # Use /dev/shm for faster I/O
    compress: bool = True,
) -> str:
    """
    Download Lichess Elite games with MOVES for policy training.

    Uses FULLY PARALLEL processing:
    - Downloads and processes zip files SIMULTANEOUSLY
    - Multiple zip files processed in parallel
    - Each zip processed by a dedicated worker

    Args:
        output_dir: Where to save processed data
        num_positions: Target number of positions
        batch_size: Positions per .npz file
        min_elo: Minimum player ELO
        num_workers: Number of parallel workers (0 = auto)
        use_ram_disk: Use /dev/shm (RAM) for temp files (Linux only)
        compress: Use compressed .npz files (smaller, slower)

    Returns:
        Path to output directory
    """
    import requests
    import zipfile
    import multiprocessing as mp
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
    import threading
    import queue

    from data.encoder import BoardEncoder

    os.makedirs(output_dir, exist_ok=True)

    # Auto-detect workers
    if num_workers == 0:
        num_workers = min(mp.cpu_count(), 16)

    # Setup RAM disk if requested
    ram_disk_path = None
    if use_ram_disk:
        ram_disk_path = "/dev/shm/chess_temp"
        if os.path.exists("/dev/shm"):
            os.makedirs(ram_disk_path, exist_ok=True)
            print(f"Using RAM disk: {ram_disk_path}")
        else:
            print("WARNING: /dev/shm not available, falling back to disk")
            use_ram_disk = False
            ram_disk_path = None

    print(f"\n{'='*60}")
    print(f"DOWNLOADING LICHESS ELITE GAMES (FULLY PARALLEL)")
    print(f"{'='*60}")
    print(f"Target positions: {num_positions:,}")
    print(f"Min ELO: {min_elo}")
    print(f"Output: {output_dir}")
    print(f"Parallel zip processors: {min(num_workers, 4)}")
    if use_ram_disk:
        print(f"RAM disk: ENABLED ({ram_disk_path})")
    if not compress:
        print("Compression: DISABLED (faster writes, larger files)")
    print(f"{'='*60}\n")

    encoder = BoardEncoder()
    print(f"Policy size: {encoder.num_moves} (from encoder)")

    # Lichess Elite PGN files
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

    raw_dir = ram_disk_path if use_ram_disk else os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # Thread-safe queue for downloaded files ready to process
    download_queue = queue.Queue()
    download_done = threading.Event()

    # =========================================================================
    # THREAD 1: Download files and put in queue (overlaps with processing)
    # =========================================================================
    def download_worker():
        """Downloads files and queues them for processing."""
        session = requests.Session()
        for filename in MONTHLY_FILES:
            zip_path = os.path.join(raw_dir, filename)

            if not os.path.exists(zip_path):
                try:
                    url = f"{LICHESS_ELITE_BASE}{filename}"
                    response = session.get(url, stream=True, timeout=120)
                    response.raise_for_status()
                    with open(zip_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                except Exception as e:
                    print(f"  Failed to download {filename}: {e}")
                    continue

            download_queue.put(zip_path)

        download_done.set()

    # Start download thread
    download_thread = threading.Thread(target=download_worker, daemon=True)
    download_thread.start()
    print("Started download thread (downloading while processing)...")

    # =========================================================================
    # MAIN: Process zip files in parallel as they become available
    # =========================================================================
    all_boards = []
    all_policies = []
    all_values = []
    total_positions = 0
    total_games = 0
    chunk_idx = 0
    files_info = []

    def save_chunk():
        nonlocal chunk_idx, all_boards, all_policies, all_values
        if not all_boards:
            return
        chunk_name = f"chunk_{chunk_idx:06d}.npz"
        chunk_size = sum(len(b) for b in all_boards)
        chunk_path = _save_npz_chunk(
            output_dir,
            chunk_name,
            np.concatenate(all_boards) if len(all_boards) > 1 else all_boards[0],
            np.concatenate(all_policies) if len(all_policies) > 1 else all_policies[0],
            np.concatenate(all_values) if len(all_values) > 1 else all_values[0],
            compress,
        )
        files_info.append({
            'name': chunk_name,
            'size': chunk_size,
            'mtime': os.path.getmtime(chunk_path),
        })
        chunk_idx += 1
        all_boards = []
        all_policies = []
        all_values = []

    pbar = tqdm(total=num_positions, desc="Extracting positions")
    positions_per_chunk = min(batch_size, 10000)

    # Process multiple zip files in parallel
    num_zip_workers = min(num_workers, 4)  # Max 4 parallel zip processors

    with ProcessPoolExecutor(max_workers=num_zip_workers) as executor:
        pending_futures = {}
        files_submitted = 0

        while total_positions < num_positions:
            # Submit new work as downloads complete
            while not download_queue.empty() and len(pending_futures) < num_zip_workers:
                try:
                    zip_path = download_queue.get_nowait()
                    future = executor.submit(_process_zip_file, (zip_path, min_elo, positions_per_chunk))
                    pending_futures[future] = zip_path
                    files_submitted += 1
                except queue.Empty:
                    break

            # If no pending work and downloads are done, we're finished
            if not pending_futures and download_done.is_set() and download_queue.empty():
                break

            # If nothing to do yet, wait a bit
            if not pending_futures:
                import time
                time.sleep(0.1)
                continue

            # Wait for any result
            done_futures = []
            for future in list(pending_futures.keys()):
                if future.done():
                    done_futures.append(future)

            if not done_futures:
                import time
                time.sleep(0.1)
                continue

            for future in done_futures:
                zip_path = pending_futures.pop(future)

                try:
                    results, games_count, error = future.result()

                    if error:
                        print(f"  Error processing {zip_path}: {error}")
                        continue

                    if results is None:
                        continue

                    total_games += games_count

                    for boards, policies, values in results:
                        remaining = num_positions - total_positions
                        if remaining <= 0:
                            break

                        if len(boards) > remaining:
                            boards = boards[:remaining]
                            policies = policies[:remaining]
                            values = values[:remaining]

                        all_boards.append(boards)
                        all_policies.append(policies)
                        all_values.append(values)
                        total_positions += len(boards)
                        pbar.update(len(boards))

                        # Save chunk if needed
                        current_size = sum(len(b) for b in all_boards)
                        if current_size >= batch_size:
                            save_chunk()

                except Exception as e:
                    print(f"  Exception processing {zip_path}: {e}")

    pbar.close()

    # Wait for download thread
    download_thread.join(timeout=5)

    # Save remaining
    save_chunk()

    _maybe_write_metadata_cache(
        output_dir,
        files_info,
        board_shape=(18, 8, 8),
        policy_shape=(encoder.num_moves,),
    )

    # Cleanup RAM disk
    if use_ram_disk and ram_disk_path and os.path.exists(ram_disk_path):
        print("Cleaning up RAM disk...")
        try:
            shutil.rmtree(ram_disk_path)
            print(f"  Removed {ram_disk_path}")
        except Exception as e:
            print(f"  Warning: Could not clean up RAM disk: {e}")

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
  python download_data.py --preset small --ram-disk              # Use RAM for faster I/O (Linux)
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
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers (default: 0 = auto-detect CPU cores)",
    )
    parser.add_argument(
        "--ram-disk",
        action="store_true",
        help="Use /dev/shm (RAM) for temp files - faster I/O (Linux only)",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Save .npz without compression for faster writes (larger files)",
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
            num_workers=args.workers,
            use_ram_disk=args.ram_disk,
            compress=not args.no_compress,
        )

    elif args.dataset == "lichess_eval":
        # Fast download from HuggingFace (value training only)
        print("WARNING: lichess_eval has no move data - policy training won't work!")
        print("Use --dataset lichess_games for proper policy+value training.\n")
        download_lichess_evaluations(
            output_dir=args.output,
            num_positions=args.positions,
            batch_size=args.batch_size,
            compress=not args.no_compress,
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
