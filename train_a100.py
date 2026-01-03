#!/usr/bin/env python3
"""
Complete Training Pipeline for 1x A100 GPU

This script handles EVERYTHING:
1. Installs all dependencies
2. Downloads and installs Stockfish binary
3. Downloads Lichess evaluated positions (with REAL Stockfish evals)
4. Trains the chess engine
5. Runs benchmarks

Usage on RunPod:
    cd chess_engine
    python train_a100.py

Expected time: ~2-3 hours total
- Setup: ~5 min
- Download/process: ~30-60 min
- Training: ~1.5-2 hours
"""

import os
import sys
import subprocess
import platform
import tarfile
import zipfile
import urllib.request
import shutil
import argparse


def run_command(cmd, description, check=True):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")

    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        print(f"ERROR: {description} failed with code {result.returncode}")
        sys.exit(1)
    return result.returncode


def install_dependencies():
    """Install Python dependencies"""
    print("\n" + "="*60)
    print("Installing Python dependencies...")
    print("="*60 + "\n")

    # Core packages
    packages = [
        "torch",
        "numpy",
        "python-chess",
        "datasets",
        "tqdm",
        "tensorboard",
        "stockfish",
        "zstandard",
        "requests",
    ]

    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + packages)
    print("Dependencies installed!")


def download_stockfish():
    """Download and install Stockfish binary"""
    print("\n" + "="*60)
    print("Setting up Stockfish...")
    print("="*60 + "\n")

    stockfish_dir = os.path.expanduser("~/.stockfish")
    os.makedirs(stockfish_dir, exist_ok=True)

    system = platform.system().lower()
    machine = platform.machine().lower()

    # Determine download URL based on platform
    if system == "linux":
        if "x86_64" in machine or "amd64" in machine:
            # Use Stockfish 16.1 for Linux x86_64
            url = "https://github.com/official-stockfish/Stockfish/releases/download/sf_16.1/stockfish-ubuntu-x86-64-avx2.tar"
            archive_name = "stockfish-ubuntu-x86-64-avx2.tar"
            binary_name = "stockfish-ubuntu-x86-64-avx2"
        else:
            print(f"Unsupported Linux architecture: {machine}")
            print("Please install Stockfish manually: apt install stockfish")
            return None
    elif system == "darwin":
        url = "https://github.com/official-stockfish/Stockfish/releases/download/sf_16.1/stockfish-macos-x86-64-avx2.tar"
        archive_name = "stockfish-macos-x86-64-avx2.tar"
        binary_name = "stockfish-macos-x86-64-avx2"
    elif system == "windows":
        url = "https://github.com/official-stockfish/Stockfish/releases/download/sf_16.1/stockfish-windows-x86-64-avx2.zip"
        archive_name = "stockfish-windows-x86-64-avx2.zip"
        binary_name = "stockfish-windows-x86-64-avx2.exe"
    else:
        print(f"Unsupported platform: {system}")
        return None

    archive_path = os.path.join(stockfish_dir, archive_name)

    # Check if already installed
    if system == "windows":
        stockfish_path = os.path.join(stockfish_dir, "stockfish", binary_name)
    else:
        stockfish_path = os.path.join(stockfish_dir, "stockfish", binary_name)

    if os.path.exists(stockfish_path):
        print(f"Stockfish already installed at: {stockfish_path}")
        return stockfish_path

    # Download
    print(f"Downloading Stockfish from: {url}")
    try:
        urllib.request.urlretrieve(url, archive_path)
        print(f"Downloaded to: {archive_path}")
    except Exception as e:
        print(f"Failed to download Stockfish: {e}")
        print("Trying apt install as fallback...")
        subprocess.run(["apt", "update"], capture_output=True)
        result = subprocess.run(["apt", "install", "-y", "stockfish"], capture_output=True)
        if result.returncode == 0:
            return "/usr/games/stockfish"
        return None

    # Extract
    print("Extracting...")
    extract_dir = os.path.join(stockfish_dir, "stockfish")
    os.makedirs(extract_dir, exist_ok=True)

    if archive_path.endswith(".tar"):
        with tarfile.open(archive_path, "r") as tar:
            tar.extractall(extract_dir)
    elif archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

    # Find the binary
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            if "stockfish" in f.lower() and not f.endswith((".tar", ".zip")):
                found_path = os.path.join(root, f)
                # Make executable on Unix
                if system != "windows":
                    os.chmod(found_path, 0o755)
                print(f"Stockfish installed at: {found_path}")
                return found_path

    print("Could not find Stockfish binary after extraction")
    return None


def set_stockfish_env(stockfish_path):
    """Set environment variable for Stockfish"""
    if stockfish_path:
        os.environ["STOCKFISH_PATH"] = stockfish_path
        print(f"Set STOCKFISH_PATH={stockfish_path}")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Complete chess engine training pipeline for 1x A100",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full training run (50M positions, 3 epochs)
    python train_a100.py

    # Quick test (1M positions, 1 epoch)
    python train_a100.py --positions 1000000 --epochs 1

    # Resume with existing data
    python train_a100.py --skip-download

    # Include benchmark at the end
    python train_a100.py --benchmark
"""
    )
    parser.add_argument("--positions", type=int, default=50_000_000,
                        help="Number of training positions (default: 50M)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=8192,
                        help="Batch size (default: 8192 for A100 80GB)")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Learning rate (default: 0.1)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip data download (use existing data)")
    parser.add_argument("--skip-install", action="store_true",
                        help="Skip dependency installation")
    parser.add_argument("--data-dir", type=str, default="./data/lichess_eval",
                        help="Data directory")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark after training")

    args = parser.parse_args()

    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║         CHESS ENGINE TRAINING PIPELINE - 1x A100              ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  This script handles EVERYTHING:                              ║
    ║  1. Install dependencies                                      ║
    ║  2. Download Stockfish binary                                 ║
    ║  3. Download Lichess evaluated positions                      ║
    ║  4. Train the model                                           ║
    ║  5. Benchmark (optional)                                      ║
    ║                                                               ║
    ║  Dataset: 316M positions with REAL Stockfish evaluations      ║
    ║  This fixes the value head training problem!                  ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    print(f"Configuration:")
    print(f"  Positions:    {args.positions:,}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Data dir:     {args.data_dir}")
    print(f"  Benchmark:    {args.benchmark}")
    print()

    # Step 0: Install dependencies
    if not args.skip_install:
        install_dependencies()

    # Step 1: Download and setup Stockfish
    stockfish_path = download_stockfish()
    set_stockfish_env(stockfish_path)

    # Step 2: Download training data
    if not args.skip_download:
        run_command(
            f"{sys.executable} download_lichess_eval.py "
            f"--output {args.data_dir} "
            f"--positions {args.positions} "
            f"--batch-size 1000000 "  # Save every 1M positions (fewer files)
            f"--no-compress",  # Faster saves, larger files
            "Step 2/4: Downloading Lichess evaluated positions"
        )

        # Verify
        run_command(
            f"{sys.executable} download_lichess_eval.py --output {args.data_dir} --verify",
            "Verifying dataset"
        )
    else:
        print("\n[Skipping download - using existing data]")

    # Step 3: Train
    run_command(
        f"{sys.executable} train.py "
        f"--data {args.data_dir} "
        f"--batch-size {args.batch_size} "
        f"--epochs {args.epochs} "
        f"--lr {args.lr}",
        "Step 3/4: Training"
    )

    # Step 4: Benchmark (optional)
    if args.benchmark:
        model_path = "./outputs/chess_engine_v1/checkpoint_best.pt"
        if os.path.exists(model_path):
            # Set stockfish path for benchmark
            if stockfish_path:
                run_command(
                    f"STOCKFISH_PATH={stockfish_path} {sys.executable} benchmark.py "
                    f"--model {model_path} --quick",
                    "Step 4/4: Benchmarking"
                )
            else:
                print("\nWarning: Stockfish not available, skipping benchmark")
        else:
            print(f"\nWarning: Model not found at {model_path}")
            print("Skipping benchmark")

    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                     TRAINING COMPLETE!                        ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  Model saved to: ./outputs/chess_engine_v1/                   ║
    ║                                                               ║
    ║  Checkpoints:                                                 ║
    ║    - checkpoint_best.pt  (lowest validation loss)             ║
    ║    - checkpoint_latest.pt (most recent)                       ║
    ║                                                               ║
    ║  Next steps:                                                  ║
    ║    python benchmark.py --model ./outputs/.../checkpoint_best.pt ║
    ║    python -m engine.play --model ./outputs/.../checkpoint_best.pt ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
