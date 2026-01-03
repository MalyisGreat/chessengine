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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

from data.encoder import BoardEncoder


# Global encoder for multiprocessing
_encoder = None

def _init_worker():
    """Initialize worker with encoder"""
    global _encoder
    _encoder = BoardEncoder()


def load_dataset_with_retry(dataset_name, split="train", streaming=True, max_retries=5):
    """Load HuggingFace dataset with retry logic for timeouts"""
    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt + 1}/{max_retries}...")
            os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
            dataset = load_dataset(
                dataset_name,
                split=split,
                streaming=streaming,
            )
            print("  Dataset loaded successfully!")
            return dataset
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"  Timeout/error, retrying in {wait_time}s... ({e.__class__.__name__})")
                time.sleep(wait_time)
            else:
                print(f"  Failed after {max_retries} attempts")
                raise


def cp_to_winrate(cp: int) -> float:
    """Convert centipawn evaluation to win probability [-1, 1]"""
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
        pass


def process_single_position(args):
    """Process a single position - for multiprocessing"""
    fen, line, cp, mate = args
    global _encoder

    if _encoder is None:
        _encoder = BoardEncoder()

    # Get centipawn evaluation
    if cp is not None:
        eval_cp = cp
    elif mate is not None:
        eval_cp = 10000 if mate > 0 else -10000
    else:
        return None

    if not line:
        return None

    moves = line.split()
    if not moves:
        return None

    best_move_uci = moves[0]

    try:
        if fen.count(' ') == 3:
            fen = fen + ' 0 1'

        board = chess.Board(fen)
        best_move = chess.Move.from_uci(best_move_uci)

        if best_move not in board.legal_moves:
            return None

        board_tensor = _encoder.encode_board(board)

        policy = np.zeros(_encoder.num_moves, dtype=np.float32)
        move_idx = _encoder.encode_move(best_move)
        if move_idx < 0:
            return None
        policy[move_idx] = 1.0

        value = cp_to_winrate(eval_cp)

        return (board_tensor, policy, value)
    except:
        return None


def process_batch_fast(batch_data):
    """Process a batch using the global encoder"""
    global _encoder
    if _encoder is None:
        _encoder = BoardEncoder()

    boards = []
    policies = []
    values = []

    for i in range(len(batch_data['fen'])):
        result = process_single_position((
            batch_data['fen'][i],
            batch_data['line'][i],
            batch_data['cp'][i],
            batch_data['mate'][i],
        ))
        if result is not None:
            boards.append(result[0])
            policies.append(result[1])
            values.append(result[2])

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
    num_workers: int = None,
):
    """
    Download and process Lichess evaluated positions from HuggingFace

    Uses multiprocessing for ~10x speedup.
    """
    os.makedirs(output_dir, exist_ok=True)

    if num_workers is None:
        num_workers = min(mp.cpu_count(), 16)

    encoder = BoardEncoder()

    print(f"\n{'='*60}")
    print("DOWNLOADING LICHESS EVALUATED POSITIONS")
    print(f"{'='*60}")
    print(f"Dataset: Lichess/chess-position-evaluations (316M positions)")
    print(f"Target: {num_positions:,} positions")
    print(f"Output: {output_dir}")
    print(f"Workers: {num_workers}")
    print(f"Policy size: {encoder.num_moves}")
    print(f"{'='*60}\n")

    print("Loading dataset from HuggingFace (streaming mode)...")

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

    # Larger batch for multiprocessing efficiency
    process_batch_size = 50000
    batch_data = {'fen': [], 'line': [], 'cp': [], 'mate': []}

    pbar = tqdm(total=num_positions, desc="Processing positions")

    # Initialize encoder in main process
    global _encoder
    _encoder = encoder

    with ProcessPoolExecutor(max_workers=num_workers, initializer=_init_worker) as executor:
        for item in dataset:
            if total_positions >= num_positions:
                break

            batch_data['fen'].append(item['fen'])
            batch_data['line'].append(item['line'])
            batch_data['cp'].append(item['cp'])
            batch_data['mate'].append(item['mate'])

            # Process larger batches for efficiency
            if len(batch_data['fen']) >= process_batch_size:
                # Split into chunks for parallel processing
                chunk_size = process_batch_size // num_workers
                futures = []

                for i in range(0, len(batch_data['fen']), chunk_size):
                    chunk = {
                        'fen': batch_data['fen'][i:i+chunk_size],
                        'line': batch_data['line'][i:i+chunk_size],
                        'cp': batch_data['cp'][i:i+chunk_size],
                        'mate': batch_data['mate'][i:i+chunk_size],
                    }
                    futures.append(executor.submit(process_batch_fast, chunk))

                # Collect results
                for future in futures:
                    result = future.result()
                    if result is not None:
                        boards, policies, values = result
                        all_boards.append(boards)
                        all_policies.append(policies)
                        all_values.append(values)
                        total_positions += len(boards)
                        pbar.update(len(boards))

                batch_data = {'fen': [], 'line': [], 'cp': [], 'mate': []}

                # Save chunk if we have enough
                current_size = sum(len(b) for b in all_boards)
                if current_size >= batch_size:
                    chunk_name = f"chunk_{chunk_idx:04d}.npz"
                    chunk_path = os.path.join(output_dir, chunk_name)

                    concat_boards = np.concatenate(all_boards)
                    concat_policies = np.concatenate(all_policies)
                    concat_values = np.concatenate(all_values)

                    if compress:
                        np.savez_compressed(chunk_path, boards=concat_boards,
                                          policies=concat_policies, values=concat_values)
                    else:
                        np.savez(chunk_path, boards=concat_boards,
                                policies=concat_policies, values=concat_values)

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
        result = process_batch_fast(batch_data)
        if result is not None:
            all_boards.append(result[0])
            all_policies.append(result[1])
            all_values.append(result[2])

    # Save remaining data
    if all_boards:
        chunk_name = f"chunk_{chunk_idx:04d}.npz"
        chunk_path = os.path.join(output_dir, chunk_name)
        current_size = sum(len(b) for b in all_boards)

        concat_boards = np.concatenate(all_boards)
        concat_policies = np.concatenate(all_policies)
        concat_values = np.concatenate(all_values)

        if compress:
            np.savez_compressed(chunk_path, boards=concat_boards,
                              policies=concat_policies, values=concat_values)
        else:
            np.savez(chunk_path, boards=concat_boards,
                    policies=concat_policies, values=concat_values)

        print(f"\n  Saved {chunk_path} ({current_size:,} positions)")
        files_info.append({
            'name': chunk_name,
            'size': current_size,
            'mtime': os.path.getmtime(chunk_path),
        })

    _maybe_write_metadata_cache(output_dir, files_info,
                                board_shape=(18, 8, 8),
                                policy_shape=(encoder.num_moves,))

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

    winning = (values > 0.5).sum()
    losing = (values < -0.5).sum()
    equal = ((values >= -0.5) & (values <= 0.5)).sum()

    print(f"\nValue breakdown:")
    print(f"  Winning (>0.5): {winning} ({100*winning/len(values):.1f}%)")
    print(f"  Equal (-0.5 to 0.5): {equal} ({100*equal/len(values):.1f}%)")
    print(f"  Losing (<-0.5): {losing} ({100*losing/len(values):.1f}%)")

    encoder = BoardEncoder()
    print(f"\nSample positions:")
    for i in range(min(num_samples, len(boards))):
        move_idx = np.argmax(policies[i])
        move = encoder.decode_move(move_idx)
        value = values[i]
        print(f"  Position {i}: best_move={move.uci()}, value={value:.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Lichess evaluated positions")
    parser.add_argument("--output", type=str, default="./data/lichess_eval")
    parser.add_argument("--positions", type=int, default=50_000_000)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--no-compress", action="store_true")
    parser.add_argument("--workers", type=int, default=None)

    args = parser.parse_args()

    if args.verify:
        verify_dataset(args.output)
    else:
        download_lichess_evaluated(
            output_dir=args.output,
            num_positions=args.positions,
            batch_size=args.batch_size,
            compress=not args.no_compress,
            num_workers=args.workers,
        )
