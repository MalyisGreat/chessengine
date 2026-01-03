"""
Download Lichess Evaluated Positions Dataset

This dataset has 316M positions with:
- FEN (board state)
- Centipawn evaluation (VALUE TARGET - what we need!)
- Best move line (POLICY TARGET)

Dataset fields:
- fen: Chess position in FEN notation
- line: Principal variation in UCI format (e.g., "e2e4 e7e5 g1f3")
- cp: Centipawn evaluation (-20000 to 20000), None if mate
- mate: Mate in N moves, None if not mate
- depth: Search depth
- knodes: Kilo-nodes searched

This fixes the value head training problem - instead of noisy game outcomes,
we get direct Stockfish evaluations for each position.
"""

import os
import json
import time
import numpy as np
from tqdm import tqdm
import chess
from datasets import load_dataset
import multiprocessing as mp

from data.encoder import BoardEncoder


def load_dataset_with_retry(dataset_name, split="train", streaming=True, max_retries=5):
    """Load HuggingFace dataset with retry logic for timeouts"""
    import httpx

    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt + 1}/{max_retries}...")
            # Increase timeout for large datasets
            os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
            dataset = load_dataset(
                dataset_name,
                split=split,
                streaming=streaming,
                trust_remote_code=True,
            )
            print("  Dataset loaded successfully!")
            return dataset
        except (httpx.ReadTimeout, httpx.ConnectTimeout, Exception) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8, 16 seconds
                print(f"  Timeout/error, retrying in {wait_time}s... ({e.__class__.__name__})")
                time.sleep(wait_time)
            else:
                print(f"  Failed after {max_retries} attempts")
                raise


def cp_to_winrate(cp: int) -> float:
    """
    Convert centipawn evaluation to win probability [-1, 1]

    Uses the standard formula: winrate = 2 / (1 + 10^(-cp/400)) - 1
    This maps:
    - cp = 0 -> 0.0 (equal)
    - cp = 100 -> ~0.14 (slight advantage)
    - cp = 300 -> ~0.36 (clear advantage)
    - cp = 1000 -> ~0.85 (winning)
    - cp = 10000 -> ~1.0 (mate)
    """
    # Clamp to avoid overflow
    cp = max(-10000, min(10000, cp))
    winrate = 2.0 / (1.0 + 10.0 ** (-cp / 400.0)) - 1.0
    return winrate


def _maybe_write_metadata_cache(output_dir: str, files_info: list,
                                board_shape: tuple, policy_shape: tuple) -> None:
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


def parse_uci_move(move_str: str) -> chess.Move:
    """Parse a UCI move string like 'e2e4' or 'e7e8q'"""
    try:
        return chess.Move.from_uci(move_str)
    except:
        return None


def process_batch(batch, encoder):
    """Process a batch of positions from the dataset

    Dataset fields:
    - fen: position
    - line: PV string like "e2e4 e7e5 g1f3"
    - cp: centipawn eval (None if mate)
    - mate: mate in N (None if not mate)
    """
    boards = []
    policies = []
    values = []

    for i in range(len(batch['fen'])):
        fen = batch['fen'][i]
        line = batch['line'][i]
        cp = batch['cp'][i]
        mate = batch['mate'][i]

        # Get centipawn evaluation or mate score
        if cp is not None:
            eval_cp = cp
        elif mate is not None:
            # Convert mate to high centipawn value
            eval_cp = 10000 if mate > 0 else -10000
        else:
            # No evaluation available
            continue

        # Get best move from the line
        if not line:
            continue

        moves = line.split()
        if not moves:
            continue

        best_move_uci = moves[0]

        try:
            # FEN from this dataset is partial (no move counters)
            # Add default move counters
            if fen.count(' ') == 3:
                fen = fen + ' 0 1'

            board = chess.Board(fen)

            # Parse the best move
            best_move = parse_uci_move(best_move_uci)
            if best_move is None or best_move not in board.legal_moves:
                continue

            # Encode board
            board_tensor = encoder.encode_board(board)

            # Encode policy (one-hot for best move)
            policy = np.zeros(encoder.num_moves, dtype=np.float32)
            move_idx = encoder.encode_move(best_move)
            if move_idx < 0:
                continue
            policy[move_idx] = 1.0

            # Convert centipawn to win probability
            # Note: eval is from side-to-move's perspective in this dataset
            value = cp_to_winrate(eval_cp)

            boards.append(board_tensor)
            policies.append(policy)
            values.append(value)

        except Exception as e:
            continue

    if boards:
        return (
            np.array(boards, dtype=np.float32),
            np.array(policies, dtype=np.float32),
            np.array(values, dtype=np.float32),
        )
    return None


def download_lichess_evaluated(
    output_dir: str = "./data/lichess_eval",
    num_positions: int = 50_000_000,
    batch_size: int = 100_000,
    streaming: bool = True,
    compress: bool = True,
):
    """
    Download and process Lichess evaluated positions from HuggingFace

    Args:
        output_dir: Where to save processed .npz files
        num_positions: Target number of positions (default 50M)
        batch_size: Positions per output file
        streaming: Use streaming mode (recommended for large dataset)
        compress: Use compressed .npz files (smaller, slower)
    """
    os.makedirs(output_dir, exist_ok=True)

    encoder = BoardEncoder()

    print(f"\n{'='*60}")
    print("DOWNLOADING LICHESS EVALUATED POSITIONS")
    print(f"{'='*60}")
    print(f"Dataset: Lichess/chess-position-evaluations (316M positions)")
    print(f"Target: {num_positions:,} positions")
    print(f"Output: {output_dir}")
    print(f"Policy size: {encoder.num_moves}")
    print(f"{'='*60}\n")

    print("Loading dataset from HuggingFace (streaming mode)...")

    # Load dataset with retry logic for timeouts
    dataset = load_dataset_with_retry(
        "Lichess/chess-position-evaluations",
        split="train",
        streaming=streaming,
    )

    all_boards = []
    all_policies = []
    all_values = []
    total_positions = 0
    chunk_idx = 0
    files_info = []

    # Process in batches - now with correct field names
    batch_data = {
        'fen': [],
        'line': [],
        'cp': [],
        'mate': [],
    }

    pbar = tqdm(total=num_positions, desc="Processing positions")

    for item in dataset:
        if total_positions >= num_positions:
            break

        batch_data['fen'].append(item['fen'])
        batch_data['line'].append(item['line'])
        batch_data['cp'].append(item['cp'])
        batch_data['mate'].append(item['mate'])

        # Process when batch is full
        if len(batch_data['fen']) >= 10000:
            result = process_batch(batch_data, encoder)

            if result is not None:
                boards, policies, values = result
                all_boards.append(boards)
                all_policies.append(policies)
                all_values.append(values)

                new_positions = len(boards)
                total_positions += new_positions
                pbar.update(new_positions)

            # Reset batch
            batch_data = {'fen': [], 'line': [], 'cp': [], 'mate': []}

            # Save chunk if we have enough
            current_size = sum(len(b) for b in all_boards)
            if current_size >= batch_size:
                # Concatenate and save
                chunk_name = f"chunk_{chunk_idx:04d}.npz"
                chunk_path = os.path.join(output_dir, chunk_name)
                if compress:
                    np.savez_compressed(
                        chunk_path,
                        boards=np.concatenate(all_boards),
                        policies=np.concatenate(all_policies),
                        values=np.concatenate(all_values),
                    )
                else:
                    np.savez(
                        chunk_path,
                        boards=np.concatenate(all_boards),
                        policies=np.concatenate(all_policies),
                        values=np.concatenate(all_values),
                    )
                print(f"\n  Saved {chunk_path} ({current_size:,} positions)")

                files_info.append({
                    'name': chunk_name,
                    'size': current_size,
                    'mtime': os.path.getmtime(chunk_path),
                })

                chunk_idx += 1
                all_boards = []
                all_policies = []
                all_values = []

    pbar.close()

    # Process remaining batch
    if batch_data['fen']:
        result = process_batch(batch_data, encoder)
        if result is not None:
            boards, policies, values = result
            all_boards.append(boards)
            all_policies.append(policies)
            all_values.append(values)

    # Save remaining data
    if all_boards:
        chunk_name = f"chunk_{chunk_idx:04d}.npz"
        chunk_path = os.path.join(output_dir, chunk_name)
        current_size = sum(len(b) for b in all_boards)
        if compress:
            np.savez_compressed(
                chunk_path,
                boards=np.concatenate(all_boards),
                policies=np.concatenate(all_policies),
                values=np.concatenate(all_values),
            )
        else:
            np.savez(
                chunk_path,
                boards=np.concatenate(all_boards),
                policies=np.concatenate(all_policies),
                values=np.concatenate(all_values),
            )
        print(f"\n  Saved {chunk_path} ({current_size:,} positions)")
        files_info.append({
            'name': chunk_name,
            'size': current_size,
            'mtime': os.path.getmtime(chunk_path),
        })

    _maybe_write_metadata_cache(
        output_dir,
        files_info,
        board_shape=(18, 8, 8),
        policy_shape=(encoder.num_moves,),
    )

    print(f"\n{'='*60}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"Total positions: {total_positions:,}")
    print(f"Chunks saved: {len(files_info)}")
    print(f"Output directory: {output_dir}")
    print(f"\nTo train:")
    print(f"  python train.py --data {output_dir}")
    print(f"{'='*60}\n")

    return output_dir


def verify_dataset(data_dir: str, num_samples: int = 5):
    """Verify the downloaded dataset looks correct"""
    import glob

    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not files:
        print(f"No .npz files found in {data_dir}")
        return

    print(f"\nVerifying dataset in {data_dir}")
    print(f"Found {len(files)} chunk files")

    # Load first file
    data = np.load(files[0])
    boards = data['boards']
    policies = data['policies']
    values = data['values']

    print(f"\nFirst chunk stats:")
    print(f"  Boards shape: {boards.shape}")
    print(f"  Policies shape: {policies.shape}")
    print(f"  Values shape: {values.shape}")

    print(f"\nValue distribution (should be spread, not all near 0):")
    print(f"  Min: {values.min():.3f}")
    print(f"  Max: {values.max():.3f}")
    print(f"  Mean: {values.mean():.3f}")
    print(f"  Std: {values.std():.3f}")

    # Count value ranges
    winning = (values > 0.5).sum()
    losing = (values < -0.5).sum()
    equal = ((values >= -0.5) & (values <= 0.5)).sum()

    print(f"\nValue breakdown:")
    print(f"  Winning (>0.5): {winning} ({100*winning/len(values):.1f}%)")
    print(f"  Equal (-0.5 to 0.5): {equal} ({100*equal/len(values):.1f}%)")
    print(f"  Losing (<-0.5): {losing} ({100*losing/len(values):.1f}%)")

    print(f"\nSample positions:")
    encoder = BoardEncoder()

    for i in range(min(num_samples, len(boards))):
        # Find the move from policy
        move_idx = np.argmax(policies[i])
        move = encoder.decode_move(move_idx)
        value = values[i]

        print(f"  Position {i}: best_move={move.uci()}, value={value:.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Lichess evaluated positions")
    parser.add_argument("--output", type=str, default="./data/lichess_eval",
                        help="Output directory")
    parser.add_argument("--positions", type=int, default=50_000_000,
                        help="Number of positions to download")
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing dataset")
    parser.add_argument("--batch-size", type=int, default=100_000,
                        help="Positions per output file")
    parser.add_argument("--no-compress", action="store_true",
                        help="Save .npz without compression for faster writes (larger files)")

    args = parser.parse_args()

    if args.verify:
        verify_dataset(args.output)
    else:
        download_lichess_evaluated(
            output_dir=args.output,
            num_positions=args.positions,
            batch_size=args.batch_size,
            compress=not args.no_compress,
        )
