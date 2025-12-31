"""
Data Download and Processing Script

Downloads the Lichess Elite Database and processes it into training-ready format.

Usage:
    python download_data.py --dataset lichess_elite
    python download_data.py --pgn my_games.pgn --output ./data/processed
"""

import os
import sys
import argparse
import requests
from pathlib import Path
from typing import List, Optional
import zipfile
import subprocess
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
import tempfile
import shutil

from data.encoder import BoardEncoder
from data.dataset import process_pgn_to_npz


# Lichess Elite Database URLs
LICHESS_ELITE_BASE = "https://database.nikonoel.fr/"

# Historical archive (2013-2020)
LICHESS_ARCHIVE_URL = "https://odysee.com/@nikonoel:4/lichess_elite_2020_05.7z"

# Monthly files (example URLs - these follow a pattern)
MONTHLY_FILES = [
    # 2020
    "lichess_elite_2020-06.zip",
    "lichess_elite_2020-07.zip",
    "lichess_elite_2020-08.zip",
    "lichess_elite_2020-09.zip",
    "lichess_elite_2020-10.zip",
    "lichess_elite_2020-11.zip",
    "lichess_elite_2020-12.zip",
    # 2021
    "lichess_elite_2021-01.zip",
    "lichess_elite_2021-02.zip",
    "lichess_elite_2021-03.zip",
    "lichess_elite_2021-04.zip",
    "lichess_elite_2021-05.zip",
    "lichess_elite_2021-06.zip",
    "lichess_elite_2021-07.zip",
    "lichess_elite_2021-08.zip",
    "lichess_elite_2021-09.zip",
    "lichess_elite_2021-10.zip",
    "lichess_elite_2021-11.zip",
    "lichess_elite_2021-12.zip",
    # 2022
    "lichess_elite_2022-01.zip",
    "lichess_elite_2022-02.zip",
    "lichess_elite_2022-03.zip",
    "lichess_elite_2022-04.zip",
    "lichess_elite_2022-05.zip",
    "lichess_elite_2022-06.zip",
    "lichess_elite_2022-07.zip",
    "lichess_elite_2022-08.zip",
    "lichess_elite_2022-09.zip",
    "lichess_elite_2022-10.zip",
    "lichess_elite_2022-11.zip",
    "lichess_elite_2022-12.zip",
    # 2023
    "lichess_elite_2023-01.zip",
    "lichess_elite_2023-02.zip",
    "lichess_elite_2023-03.zip",
    "lichess_elite_2023-04.zip",
    "lichess_elite_2023-05.zip",
    "lichess_elite_2023-06.zip",
    "lichess_elite_2023-07.zip",
    "lichess_elite_2023-08.zip",
    "lichess_elite_2023-09.zip",
    "lichess_elite_2023-10.zip",
    "lichess_elite_2023-11.zip",
    "lichess_elite_2023-12.zip",
    # 2024
    "lichess_elite_2024-01.zip",
    "lichess_elite_2024-02.zip",
    "lichess_elite_2024-03.zip",
    "lichess_elite_2024-04.zip",
    "lichess_elite_2024-05.zip",
    "lichess_elite_2024-06.zip",
]


def download_file(url: str, output_path: str, desc: str = None) -> bool:
    """
    Download a file with progress bar

    Args:
        url: URL to download
        output_path: Where to save the file
        desc: Description for progress bar

    Returns:
        True if successful
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f:
            with tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=desc or os.path.basename(output_path),
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def extract_zip(zip_path: str, output_dir: str) -> List[str]:
    """Extract a zip file and return list of extracted files"""
    extracted = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            if name.endswith('.pgn'):
                zf.extract(name, output_dir)
                extracted.append(os.path.join(output_dir, name))
    return extracted


def extract_7z(archive_path: str, output_dir: str) -> List[str]:
    """Extract a .7z file using 7z command"""
    try:
        subprocess.run(
            ['7z', 'x', archive_path, f'-o{output_dir}', '-y'],
            check=True,
            capture_output=True,
        )
        # Find extracted PGN files
        extracted = []
        for root, _, files in os.walk(output_dir):
            for f in files:
                if f.endswith('.pgn'):
                    extracted.append(os.path.join(root, f))
        return extracted
    except FileNotFoundError:
        print("7z not found. Please install p7zip or 7-zip.")
        print("On Ubuntu: sudo apt install p7zip-full")
        print("On Mac: brew install p7zip")
        return []
    except subprocess.CalledProcessError as e:
        print(f"Error extracting {archive_path}: {e}")
        return []


def download_lichess_elite(
    output_dir: str,
    max_files: Optional[int] = None,
    skip_archive: bool = False,
) -> List[str]:
    """
    Download Lichess Elite Database

    Args:
        output_dir: Output directory
        max_files: Maximum monthly files to download (None = all)
        skip_archive: Skip the large historical archive

    Returns:
        List of downloaded PGN file paths
    """
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    pgn_files = []

    # Download monthly files
    files_to_download = MONTHLY_FILES[:max_files] if max_files else MONTHLY_FILES

    print(f"Downloading {len(files_to_download)} monthly files...")

    for filename in tqdm(files_to_download, desc="Downloading"):
        url = f"{LICHESS_ELITE_BASE}{filename}"
        zip_path = os.path.join(raw_dir, filename)

        if os.path.exists(zip_path):
            print(f"  {filename} already exists, skipping download")
        else:
            success = download_file(url, zip_path, desc=filename)
            if not success:
                print(f"  Warning: Failed to download {filename}")
                continue

        # Extract
        try:
            extracted = extract_zip(zip_path, raw_dir)
            pgn_files.extend(extracted)
        except Exception as e:
            print(f"  Error extracting {filename}: {e}")

    print(f"\nDownloaded and extracted {len(pgn_files)} PGN files")
    return pgn_files


def process_all_pgns(
    pgn_files: List[str],
    output_dir: str,
    min_elo: int = 2300,
    max_games_per_file: Optional[int] = None,
    num_workers: int = 4,
) -> str:
    """
    Process all PGN files into training data

    Args:
        pgn_files: List of PGN file paths
        output_dir: Output directory for processed data
        min_elo: Minimum player ELO
        max_games_per_file: Max games to process per file
        num_workers: Number of parallel workers

    Returns:
        Path to processed data directory
    """
    processed_dir = os.path.join(output_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    encoder = BoardEncoder()

    print(f"\nProcessing {len(pgn_files)} PGN files...")

    for i, pgn_path in enumerate(pgn_files):
        print(f"\n[{i+1}/{len(pgn_files)}] Processing {os.path.basename(pgn_path)}")

        file_output = os.path.join(processed_dir, f"data_{i:04d}")
        os.makedirs(file_output, exist_ok=True)

        try:
            process_pgn_to_npz(
                pgn_path=pgn_path,
                output_path=file_output,
                encoder=encoder,
                min_elo=min_elo,
                max_games=max_games_per_file,
            )
        except Exception as e:
            print(f"  Error processing {pgn_path}: {e}")
            continue

    # Merge all chunks into single directory
    final_dir = os.path.join(output_dir, "train")
    os.makedirs(final_dir, exist_ok=True)

    chunk_idx = 0
    for subdir in sorted(os.listdir(processed_dir)):
        subdir_path = os.path.join(processed_dir, subdir)
        if os.path.isdir(subdir_path):
            for f in sorted(os.listdir(subdir_path)):
                if f.endswith('.npz'):
                    src = os.path.join(subdir_path, f)
                    dst = os.path.join(final_dir, f"chunk_{chunk_idx:06d}.npz")
                    shutil.move(src, dst)
                    chunk_idx += 1

    print(f"\nProcessed data saved to: {final_dir}")
    print(f"Total chunks: {chunk_idx}")

    return final_dir


def create_sample_data(output_dir: str, num_games: int = 100) -> str:
    """
    Create sample training data for testing

    Args:
        output_dir: Output directory
        num_games: Number of random games to generate

    Returns:
        Path to sample data
    """
    import chess
    import numpy as np

    sample_dir = os.path.join(output_dir, "sample")
    os.makedirs(sample_dir, exist_ok=True)

    encoder = BoardEncoder()

    boards = []
    policies = []
    values = []

    print(f"Generating {num_games} sample games...")

    for game_idx in tqdm(range(num_games)):
        board = chess.Board()

        # Random game outcome
        outcome = np.random.choice([-1.0, 0.0, 1.0])

        moves = 0
        while not board.is_game_over() and moves < 100:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break

            # Random move
            move = np.random.choice(legal_moves)

            # Store position
            board_tensor = encoder.encode_board(board)
            policy = encoder.encode_policy(board, move)
            value = outcome if board.turn == chess.WHITE else -outcome

            boards.append(board_tensor)
            policies.append(policy)
            values.append(value)

            board.push(move)
            moves += 1

    # Save
    output_path = os.path.join(sample_dir, "sample_data.npz")
    np.savez_compressed(
        output_path,
        boards=np.array(boards, dtype=np.float32),
        policies=np.array(policies, dtype=np.float32),
        values=np.array(values, dtype=np.float32),
    )

    print(f"Sample data saved to: {output_path}")
    print(f"Total positions: {len(boards)}")

    return sample_dir


def main():
    parser = argparse.ArgumentParser(description="Download and process chess training data")

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["lichess_elite", "sample"],
        default="sample",
        help="Dataset to download (default: sample)",
    )
    parser.add_argument(
        "--pgn",
        type=str,
        help="Process a specific PGN file instead",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data",
        help="Output directory (default: ./data)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum monthly files to download",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        help="Maximum games per PGN file",
    )
    parser.add_argument(
        "--min-elo",
        type=int,
        default=2300,
        help="Minimum player ELO (default: 2300)",
    )

    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    if args.pgn:
        # Process specific PGN file
        print(f"Processing {args.pgn}...")
        encoder = BoardEncoder()
        processed_dir = os.path.join(output_dir, "processed")
        process_pgn_to_npz(
            pgn_path=args.pgn,
            output_path=processed_dir,
            encoder=encoder,
            min_elo=args.min_elo,
            max_games=args.max_games,
        )

    elif args.dataset == "lichess_elite":
        # Download Lichess Elite
        print("Downloading Lichess Elite Database...")
        pgn_files = download_lichess_elite(
            output_dir=output_dir,
            max_files=args.max_files,
        )

        if pgn_files:
            print("\nProcessing downloaded files...")
            process_all_pgns(
                pgn_files=pgn_files,
                output_dir=output_dir,
                min_elo=args.min_elo,
                max_games_per_file=args.max_games,
            )

    elif args.dataset == "sample":
        # Generate sample data
        print("Generating sample data for testing...")
        create_sample_data(output_dir, num_games=1000)

    print("\nDone!")


if __name__ == "__main__":
    main()
