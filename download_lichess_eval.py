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
from concurrent.futures import ProcessPoolExecutor
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


PIECE_TO_PLANE = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White pieces
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,  # Black pieces
}

PROMO_MAP = {
    'q': chess.QUEEN,
    'r': chess.ROOK,
    'b': chess.BISHOP,
    'n': chess.KNIGHT,
}


def fast_parse_fen(fen: str, tensor: np.ndarray) -> bool:
    """Parse FEN directly into a pre-allocated tensor."""
    try:
        tensor.fill(0.0)
        parts = fen.split(' ')
        if len(parts) < 4:
            return False

        position, turn, castling, ep = parts[0], parts[1], parts[2], parts[3]

        rank = 7
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

        if turn == 'w':
            tensor[12, :, :] = 1.0

        if 'K' in castling:
            tensor[13, :, :] = 1.0
        if 'Q' in castling:
            tensor[14, :, :] = 1.0
        if 'k' in castling:
            tensor[15, :, :] = 1.0
        if 'q' in castling:
            tensor[16, :, :] = 1.0

        if ep != '-' and len(ep) == 2:
            ep_file = ord(ep[0]) - ord('a')
            ep_rank = int(ep[1]) - 1
            if 0 <= ep_file < 8 and 0 <= ep_rank < 8:
                tensor[17, ep_rank, ep_file] = 1.0

        return True
    except Exception:
        return False


def fast_parse_uci_move(move_str: str):
    """Parse UCI move into (from_sq, to_sq, promotion)."""
    if len(move_str) < 4:
        return None

    from_file = ord(move_str[0]) - ord('a')
    from_rank = ord(move_str[1]) - ord('1')
    to_file = ord(move_str[2]) - ord('a')
    to_rank = ord(move_str[3]) - ord('1')

    if not (0 <= from_file < 8 and 0 <= from_rank < 8 and 0 <= to_file < 8 and 0 <= to_rank < 8):
        return None

    promotion = None
    if len(move_str) >= 5:
        promotion = PROMO_MAP.get(move_str[4])
        if promotion is None:
            return None

    from_sq = from_rank * 8 + from_file
    to_sq = to_rank * 8 + to_file
    return from_sq, to_sq, promotion


def _maybe_write_metadata_cache(output_dir: str, files_info: list,
                                board_shape: tuple, policy_shape: tuple,
                                policy_mode: str) -> None:
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
        'policy_mode': policy_mode,
    }
    try:
        with open(cache_path, 'w') as f:
            json.dump(cache, f)
    except Exception:
        pass


def process_batch_fast(batch_data):
    """Process a batch using the global encoder"""
    global _encoder
    if _encoder is None:
        _encoder = BoardEncoder()

    batch_len = len(batch_data['fen'])
    boards = np.zeros((batch_len, 18, 8, 8), dtype=np.float32)
    policy_idx = np.zeros(batch_len, dtype=np.int32)
    values = np.zeros(batch_len, dtype=np.float32)
    count = 0

    for i in range(batch_len):
        fen = batch_data['fen'][i]
        line = batch_data['line'][i]
        cp = batch_data['cp'][i]
        mate = batch_data['mate'][i]

        if cp is not None:
            eval_cp = cp
        elif mate is not None:
            eval_cp = 10000 if mate > 0 else -10000
        else:
            continue

        if not line:
            continue

        moves = line.split()
        if not moves:
            continue

        # Validate move legality by trying to parse with python-chess
        # This catches corrupted FENs and invalid PV lines
        try:
            board = chess.Board(fen)
            move = board.parse_uci(moves[0])
            if move not in board.legal_moves:
                continue
        except (ValueError, chess.InvalidMoveError, chess.AmbiguousMoveError):
            continue

        move_parsed = fast_parse_uci_move(moves[0])
        if move_parsed is None:
            continue

        from_sq, to_sq, promotion = move_parsed
        move_idx = _encoder.move_to_idx.get((from_sq, to_sq, promotion), -1)
        if move_idx < 0:
            continue

        if not fast_parse_fen(fen, boards[count]):
            continue

        policy_idx[count] = move_idx
        values[count] = cp_to_winrate(eval_cp)
        count += 1

    if count > 0:
        return (
            boards[:count],
            policy_idx[:count],
            values[:count],
        )
    return None


def download_lichess_evaluated(
    output_dir: str = "./data/lichess_eval",
    num_positions: int = 50_000_000,
    batch_size: int = 100_000,
    streaming: bool = True,
    compress: bool = True,
    num_workers: int = None,
    compact_policy: bool = True,
):
    """
    Download and process Lichess evaluated positions from HuggingFace

    Uses multiprocessing for speed.
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
    if compact_policy:
        print("Policy format: indices (compact)")
    print(f"{'='*60}\n")

    print("Loading dataset from HuggingFace (streaming mode)...")

    dataset = load_dataset_with_retry(
        "Lichess/chess-position-evaluations",
        split="train",
        streaming=streaming,
    )

    all_boards = []
    all_policy_idx = []
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
                        boards, policy_idx, values = result
                        all_boards.append(boards)
                        all_policy_idx.append(policy_idx)
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
                    concat_policy_idx = np.concatenate(all_policy_idx)
                    concat_values = np.concatenate(all_values)

                    if compact_policy:
                        if compress:
                            np.savez_compressed(
                                chunk_path,
                                boards=concat_boards,
                                policy_idx=concat_policy_idx,
                                values=concat_values,
                            )
                        else:
                            np.savez(
                                chunk_path,
                                boards=concat_boards,
                                policy_idx=concat_policy_idx,
                                values=concat_values,
                            )
                    else:
                        policies = np.zeros((len(concat_policy_idx), encoder.num_moves), dtype=np.float32)
                        policies[np.arange(len(concat_policy_idx)), concat_policy_idx] = 1.0

                        if compress:
                            np.savez_compressed(
                                chunk_path,
                                boards=concat_boards,
                                policies=policies,
                                values=concat_values,
                            )
                        else:
                            np.savez(
                                chunk_path,
                                boards=concat_boards,
                                policies=policies,
                                values=concat_values,
                            )

                    print(f"\n  Saved {chunk_path} ({current_size:,} positions)")

                    files_info.append({
                        'name': chunk_name,
                        'size': current_size,
                        'mtime': os.path.getmtime(chunk_path),
                    })

                    chunk_idx += 1
                    all_boards = []
                    all_policy_idx = []
                    all_values = []

    pbar.close()

    # Process remaining batch
    if batch_data['fen']:
        result = process_batch_fast(batch_data)
        if result is not None:
            all_boards.append(result[0])
            all_policy_idx.append(result[1])
            all_values.append(result[2])

    # Save remaining data
    if all_boards:
        chunk_name = f"chunk_{chunk_idx:04d}.npz"
        chunk_path = os.path.join(output_dir, chunk_name)
        current_size = sum(len(b) for b in all_boards)

        concat_boards = np.concatenate(all_boards)
        concat_policy_idx = np.concatenate(all_policy_idx)
        concat_values = np.concatenate(all_values)

        if compact_policy:
            if compress:
                np.savez_compressed(
                    chunk_path,
                    boards=concat_boards,
                    policy_idx=concat_policy_idx,
                    values=concat_values,
                )
            else:
                np.savez(
                    chunk_path,
                    boards=concat_boards,
                    policy_idx=concat_policy_idx,
                    values=concat_values,
                )
        else:
            policies = np.zeros((len(concat_policy_idx), encoder.num_moves), dtype=np.float32)
            policies[np.arange(len(concat_policy_idx)), concat_policy_idx] = 1.0

            if compress:
                np.savez_compressed(
                    chunk_path,
                    boards=concat_boards,
                    policies=policies,
                    values=concat_values,
                )
            else:
                np.savez(
                    chunk_path,
                    boards=concat_boards,
                    policies=policies,
                    values=concat_values,
                )

        print(f"\n  Saved {chunk_path} ({current_size:,} positions)")
        files_info.append({
            'name': chunk_name,
            'size': current_size,
            'mtime': os.path.getmtime(chunk_path),
        })

    policy_mode = 'index' if compact_policy else 'one_hot'
    policy_shape = () if compact_policy else (encoder.num_moves,)
    _maybe_write_metadata_cache(
        output_dir,
        files_info,
        board_shape=(18, 8, 8),
        policy_shape=policy_shape,
        policy_mode=policy_mode,
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

    data = np.load(files[0])
    boards = data['boards']
    policy_idx = data['policy_idx'] if 'policy_idx' in data else None
    policies = data['policies'] if 'policies' in data else None
    values = data['values']

    print(f"\nFirst chunk stats:")
    print(f"  Boards shape: {boards.shape}")
    if policy_idx is not None:
        print(f"  Policy idx shape: {policy_idx.shape}")
    if policies is not None:
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
        if policy_idx is not None:
            move_idx = int(policy_idx[i])
        else:
            move_idx = int(np.argmax(policies[i]))
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
    parser.add_argument("--compact-policy", action="store_true",
                        help="Store policy as move indices (smaller, faster)")
    parser.add_argument("--full-policy", action="store_true",
                        help="Store full one-hot policy vectors (very large)")

    args = parser.parse_args()

    compact_policy = True
    if args.full_policy:
        compact_policy = False
    if args.compact_policy:
        compact_policy = True

    if args.verify:
        verify_dataset(args.output)
    else:
        download_lichess_evaluated(
            output_dir=args.output,
            num_positions=args.positions,
            batch_size=args.batch_size,
            compress=not args.no_compress,
            num_workers=args.workers,
            compact_policy=compact_policy,
        )
