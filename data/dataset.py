"""
Chess Dataset - PyTorch Dataset for training

Handles loading pre-processed chess positions with:
- Board state (18x8x8 tensor)
- Policy target (move probabilities)
- Value target (game outcome: -1, 0, 1)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List
import chess
import chess.pgn
from tqdm import tqdm
import io
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from .encoder import BoardEncoder


class ChessDataset(Dataset):
    """
    PyTorch Dataset for chess positions

    Loads pre-processed numpy arrays or processes PGN files on the fly
    """

    def __init__(
        self,
        data_path: str,
        encoder: Optional[BoardEncoder] = None,
        augment: bool = True,
        max_positions: Optional[int] = None,
    ):
        """
        Args:
            data_path: Path to .npz file with processed data or directory of .npz files
            encoder: BoardEncoder instance (created if not provided)
            augment: Whether to apply data augmentation (random flips)
            max_positions: Maximum positions to load (for debugging)
        """
        self.encoder = encoder or BoardEncoder()
        self.augment = augment

        if os.path.isfile(data_path) and data_path.endswith('.npz'):
            self._load_single_npz(data_path, max_positions)
        elif os.path.isdir(data_path):
            self._load_directory_fast(data_path, max_positions)
        else:
            raise ValueError(f"Invalid data path: {data_path}")

        print(f"Loaded {len(self)} positions")

    def _load_single_npz(self, path: str, max_positions: Optional[int] = None):
        """Load a single .npz file"""
        data = np.load(path)
        self.boards = data['boards']
        self.policies = data['policies']
        self.values = data['values']

        if max_positions and len(self.boards) > max_positions:
            self.boards = self.boards[:max_positions]
            self.policies = self.policies[:max_positions]
            self.values = self.values[:max_positions]

    def _load_directory_fast(self, path: str, max_positions: Optional[int] = None):
        """Load all .npz files from a directory - FAST version with pre-allocation"""
        files = sorted([f for f in os.listdir(path) if f.endswith('.npz')])

        if not files:
            raise ValueError(f"No .npz files found in {path}")

        # Phase 1: Scan files to get total size (fast - just reads headers)
        print("Scanning data files...")
        file_sizes = []
        total_positions = 0

        for f in files:
            data = np.load(os.path.join(path, f))
            n = len(data['boards'])
            file_sizes.append(n)
            total_positions += n
            data.close()

            if max_positions and total_positions >= max_positions:
                break

        if max_positions:
            total_positions = min(total_positions, max_positions)

        print(f"Found {total_positions:,} positions in {len(file_sizes)} files")

        # Phase 2: Pre-allocate arrays (single allocation - very fast)
        print("Allocating memory...")
        # Get shape from first file
        sample = np.load(os.path.join(path, files[0]))
        board_shape = sample['boards'].shape[1:]  # (18, 8, 8)
        policy_shape = sample['policies'].shape[1:]  # (1858,)
        sample.close()

        self.boards = np.zeros((total_positions, *board_shape), dtype=np.float32)
        self.policies = np.zeros((total_positions, *policy_shape), dtype=np.float32)
        self.values = np.zeros(total_positions, dtype=np.float32)

        # Phase 3: Load data directly into pre-allocated arrays (fast copy)
        print("Loading data...")
        offset = 0
        remaining = total_positions

        for i, f in enumerate(tqdm(files[:len(file_sizes)], desc="Loading data files")):
            if remaining <= 0:
                break

            data = np.load(os.path.join(path, f))
            n = min(file_sizes[i], remaining)

            # Direct copy into pre-allocated arrays (no Python list overhead)
            self.boards[offset:offset + n] = data['boards'][:n]
            self.policies[offset:offset + n] = data['policies'][:n]
            self.values[offset:offset + n] = data['values'][:n]

            data.close()
            offset += n
            remaining -= n

    def __len__(self) -> int:
        return len(self.boards)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single training example

        Returns:
            Tuple of (board_tensor, policy_tensor, value_tensor)
        """
        board = self.boards[idx].copy()
        policy = self.policies[idx].copy()
        value = self.values[idx]

        # Data augmentation: random horizontal flip
        if self.augment and np.random.random() > 0.5:
            board = self.encoder.flip_board(board)
            # Note: policy would also need to be flipped (complex, skipping for now)

        return (
            torch.from_numpy(board),
            torch.from_numpy(policy),
            torch.tensor(value, dtype=torch.float32),
        )


class ChessDatasetFromPGN(Dataset):
    """
    Dataset that loads directly from PGN files

    Useful for streaming large datasets without preprocessing
    """

    def __init__(
        self,
        pgn_path: str,
        encoder: Optional[BoardEncoder] = None,
        min_elo: int = 2300,
        max_games: Optional[int] = None,
        positions_per_game: int = 10,
    ):
        """
        Args:
            pgn_path: Path to PGN file
            encoder: BoardEncoder instance
            min_elo: Minimum player ELO to include
            max_games: Maximum games to process
            positions_per_game: Random positions to sample per game
        """
        self.encoder = encoder or BoardEncoder()
        self.min_elo = min_elo
        self.positions_per_game = positions_per_game

        # Process PGN and store positions
        self.positions = []
        self._process_pgn(pgn_path, max_games)

        print(f"Loaded {len(self)} positions from {pgn_path}")

    def _process_pgn(self, path: str, max_games: Optional[int] = None):
        """Process a PGN file and extract positions"""
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            game_count = 0

            while True:
                if max_games and game_count >= max_games:
                    break

                try:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                except Exception:
                    continue

                # Check ELO requirements
                white_elo = game.headers.get('WhiteElo', '0')
                black_elo = game.headers.get('BlackElo', '0')
                try:
                    if int(white_elo) < self.min_elo or int(black_elo) < self.min_elo:
                        continue
                except ValueError:
                    continue

                # Get game result
                result = game.headers.get('Result', '*')
                if result == '1-0':
                    outcome = 1.0
                elif result == '0-1':
                    outcome = -1.0
                elif result == '1/2-1/2':
                    outcome = 0.0
                else:
                    continue

                # Extract positions
                self._extract_positions(game, outcome)
                game_count += 1

                if game_count % 1000 == 0:
                    print(f"Processed {game_count} games, {len(self.positions)} positions")

    def _extract_positions(self, game, outcome: float):
        """Extract random positions from a game"""
        board = game.board()
        moves = list(game.mainline_moves())

        if len(moves) < 10:
            return

        # Sample random positions
        indices = np.random.choice(
            len(moves) - 1,
            min(self.positions_per_game, len(moves) - 1),
            replace=False
        )

        for i, move in enumerate(moves):
            if i in indices:
                # Encode current position
                board_tensor = self.encoder.encode_board(board)

                # Encode the move that was played
                policy = self.encoder.encode_policy(board, move)

                # Value from current player's perspective
                value = outcome if board.turn == chess.WHITE else -outcome

                self.positions.append((board_tensor, policy, value))

            board.push(move)

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        board, policy, value = self.positions[idx]
        return (
            torch.from_numpy(board),
            torch.from_numpy(policy),
            torch.tensor(value, dtype=torch.float32),
        )


def process_pgn_to_npz(
    pgn_path: str,
    output_path: str,
    encoder: Optional[BoardEncoder] = None,
    min_elo: int = 2300,
    max_games: Optional[int] = None,
    chunk_size: int = 100000,
):
    """
    Process a PGN file and save as .npz for fast loading

    Args:
        pgn_path: Input PGN file path
        output_path: Output directory for .npz files
        encoder: BoardEncoder instance
        min_elo: Minimum player ELO
        max_games: Maximum games to process
        chunk_size: Positions per output file
    """
    encoder = encoder or BoardEncoder()
    os.makedirs(output_path, exist_ok=True)

    boards = []
    policies = []
    values = []
    chunk_idx = 0
    game_count = 0

    with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as f:
        pbar = tqdm(desc="Processing games")

        while True:
            if max_games and game_count >= max_games:
                break

            try:
                game = chess.pgn.read_game(f)
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
                outcome = 1.0
            elif result == '0-1':
                outcome = -1.0
            elif result == '1/2-1/2':
                outcome = 0.0
            else:
                continue

            # Process all positions in game
            board = game.board()
            moves = list(game.mainline_moves())

            for i, move in enumerate(moves[:-1]):  # Skip last move
                board_tensor = encoder.encode_board(board)
                policy = encoder.encode_policy(board, move)
                value = outcome if board.turn == chess.WHITE else -outcome

                boards.append(board_tensor)
                policies.append(policy)
                values.append(value)

                board.push(move)

                # Save chunk if full
                if len(boards) >= chunk_size:
                    chunk_path = os.path.join(output_path, f"chunk_{chunk_idx:04d}.npz")
                    np.savez_compressed(
                        chunk_path,
                        boards=np.array(boards, dtype=np.float32),
                        policies=np.array(policies, dtype=np.float32),
                        values=np.array(values, dtype=np.float32),
                    )
                    print(f"\nSaved {chunk_path} with {len(boards)} positions")
                    boards, policies, values = [], [], []
                    chunk_idx += 1

            game_count += 1
            pbar.update(1)
            pbar.set_postfix(positions=len(boards) + chunk_idx * chunk_size)

        pbar.close()

    # Save remaining positions
    if boards:
        chunk_path = os.path.join(output_path, f"chunk_{chunk_idx:04d}.npz")
        np.savez_compressed(
            chunk_path,
            boards=np.array(boards, dtype=np.float32),
            policies=np.array(policies, dtype=np.float32),
            values=np.array(values, dtype=np.float32),
        )
        print(f"Saved {chunk_path} with {len(boards)} positions")

    print(f"\nProcessed {game_count} games into {chunk_idx + 1} chunks")


def create_dataloader(
    data_path: str,
    batch_size: int = 4096,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader for training

    Args:
        data_path: Path to data
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        PyTorch DataLoader
    """
    dataset = ChessDataset(data_path, **kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )


if __name__ == "__main__":
    # Test with a sample PGN
    import tempfile

    # Create a test PGN
    test_pgn = """[Event "Test"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]
[WhiteElo "2400"]
[BlackElo "2350"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O 1-0
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pgn', delete=False) as f:
        f.write(test_pgn)
        temp_path = f.name

    # Test loading from PGN
    dataset = ChessDatasetFromPGN(temp_path, min_elo=2300, positions_per_game=5)
    print(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        board, policy, value = dataset[0]
        print(f"Board shape: {board.shape}")
        print(f"Policy shape: {policy.shape}")
        print(f"Value: {value}")

    os.unlink(temp_path)
