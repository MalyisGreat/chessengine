"""
Data Download and Processing Script

Downloads the Lichess Elite Database and processes it into training-ready format.
Uses 8 parallel threads for fast downloading (~2-3 min for all files).

Usage:
    python download_data.py --dataset lichess_elite
    python download_data.py --pgn my_games.pgn --output ./data/processed
"""

import os
import sys
import argparse
import requests
from pathlib import Path
from typing import List, Optional, Tuple
import zipfile
import subprocess
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
import tempfile
import shutil
import threading

from data.encoder import BoardEncoder
from data.dataset import process_pgn_to_npz


# Lichess Elite Database URLs
LICHESS_ELITE_BASE = "https://database.nikonoel.fr/"

# Historical archive (2013-2020)
LICHESS_ARCHIVE_URL = "https://odysee.com/@nikonoel:4/lichess_elite_2020_05.7z"

# All monthly files (2020-2024)
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
    "lichess_elite_2024-07.zip",
    "lichess_elite_2024-08.zip",
    "lichess_elite_2024-09.zip",
    "lichess_elite_2024-10.zip",
    "lichess_elite_2024-11.zip",
    "lichess_elite_2024-12.zip",
]

# Thread-safe progress tracking
progress_lock = threading.Lock()
download_progress = {}


def download_file_simple(url: str, output_path: str) -> Tuple[bool, str, int]:
    """
    Download a file without progress bar (for parallel downloads)

    Returns:
        Tuple of (success, filename, bytes_downloaded)
    """
    filename = os.path.basename(output_path)
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

        return True, filename, downloaded

    except Exception as e:
        return False, filename, 0


def download_and_extract(args: Tuple[str, str, str]) -> Tuple[bool, str, List[str]]:
    """
    Download and extract a single file (for parallel execution)

    Args:
        args: Tuple of (filename, url, raw_dir)

    Returns:
        Tuple of (success, filename, list of extracted pgn paths)
    """
    filename, url, raw_dir = args
    zip_path = os.path.join(raw_dir, filename)
    extracted_files = []

    # Skip if already exists
    if os.path.exists(zip_path):
        # Just extract
        try:
            extracted_files = extract_zip(zip_path, raw_dir)
            return True, filename, extracted_files
        except Exception as e:
            return False, filename, []

    # Download
    success, _, _ = download_file_simple(url, zip_path)

    if not success:
        return False, filename, []

    # Extract
    try:
        extracted_files = extract_zip(zip_path, raw_dir)
        return True, filename, extracted_files
    except Exception as e:
        return False, filename, []


def download_file(url: str, output_path: str, desc: str = None) -> bool:
    """
    Download a file with progress bar (for single file downloads)
    """
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f:
            with tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=desc or os.path.basename(output_path),
            ) as pbar:
                for chunk in response.iter_content(chunk_size=65536):
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


def download_lichess_elite_parallel(
    output_dir: str,
    max_files: Optional[int] = None,
    num_threads: int = 8,
) -> List[str]:
    """
    Download Lichess Elite Database using parallel threads

    Args:
        output_dir: Output directory
        max_files: Maximum monthly files to download (None = all)
        num_threads: Number of parallel download threads (default: 8)

    Returns:
        List of downloaded PGN file paths
    """
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # Prepare download tasks
    files_to_download = MONTHLY_FILES[:max_files] if max_files else MONTHLY_FILES
    tasks = [
        (filename, f"{LICHESS_ELITE_BASE}{filename}", raw_dir)
        for filename in files_to_download
    ]

    print(f"\n{'='*60}")
    print(f"PARALLEL DOWNLOAD - {len(tasks)} files with {num_threads} threads")
    print(f"{'='*60}\n")

    all_pgn_files = []
    successful = 0
    failed = 0

    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        futures = {
            executor.submit(download_and_extract, task): task[0]
            for task in tasks
        }

        # Track progress with tqdm
        with tqdm(total=len(tasks), desc="Downloading", unit="file") as pbar:
            for future in as_completed(futures):
                filename = futures[future]
                try:
                    success, name, extracted = future.result()
                    if success:
                        successful += 1
                        all_pgn_files.extend(extracted)
                        pbar.set_postfix({"last": name, "ok": successful, "fail": failed})
                    else:
                        failed += 1
                        pbar.set_postfix({"last": name, "ok": successful, "fail": failed})
                except Exception as e:
                    failed += 1
                    pbar.set_postfix({"error": str(e)[:20]})
                pbar.update(1)

    print(f"\n{'='*60}")
    print(f"DOWNLOAD COMPLETE")
    print(f"  Successful: {successful}/{len(tasks)}")
    print(f"  Failed: {failed}/{len(tasks)}")
    print(f"  PGN files extracted: {len(all_pgn_files)}")
    print(f"{'='*60}\n")

    return all_pgn_files


def process_all_pgns_parallel(
    pgn_files: List[str],
    output_dir: str,
    min_elo: int = 2300,
    max_games_per_file: Optional[int] = None,
    num_workers: int = 4,
) -> str:
    """
    Process all PGN files into training data (with parallel processing)
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
    """Create sample training data for testing"""
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
        outcome = np.random.choice([-1.0, 0.0, 1.0])

        moves = 0
        while not board.is_game_over() and moves < 100:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break

            move = np.random.choice(legal_moves)

            board_tensor = encoder.encode_board(board)
            policy = encoder.encode_policy(board, move)
            value = outcome if board.turn == chess.WHITE else -outcome

            boards.append(board_tensor)
            policies.append(policy)
            values.append(value)

            board.push(move)
            moves += 1

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
        help="Maximum monthly files to download (default: all)",
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
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of parallel download threads (default: 8)",
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
        # Download Lichess Elite with parallel threads
        print("Downloading Lichess Elite Database (parallel mode)...")
        print(f"Using {args.threads} threads for fast downloading\n")

        pgn_files = download_lichess_elite_parallel(
            output_dir=output_dir,
            max_files=args.max_files,
            num_threads=args.threads,
        )

        if pgn_files:
            print("\nProcessing downloaded files...")
            process_all_pgns_parallel(
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
