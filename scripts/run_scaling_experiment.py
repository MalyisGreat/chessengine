#!/usr/bin/env python3
"""
Quick launcher for scaling experiments.

Usage (auto-download Stockfish):
    python scripts/run_scaling_experiment.py --auto-stockfish

Usage (local with existing Stockfish):
    python scripts/run_scaling_experiment.py --stockfish /path/to/stockfish

Usage (cloud with 64 vCPU):
    python scripts/run_scaling_experiment.py --auto-stockfish --cloud

Custom settings:
    python scripts/run_scaling_experiment.py --auto-stockfish --games 50 --workers 12
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

# Stockfish 16.1 URLs (compatible with our NNUE architecture)
# NOTE: Stockfish 17+ uses a different NNUE format and is NOT compatible
STOCKFISH_URLS = {
    "windows": "https://github.com/official-stockfish/Stockfish/releases/download/sf_16.1/stockfish-windows-x86-64-avx2.zip",
    "linux": "https://github.com/official-stockfish/Stockfish/releases/download/sf_16.1/stockfish-ubuntu-x86-64-avx2.tar",
}


def download_stockfish(repo_root: Path) -> Path:
    """Download Stockfish 16.1 and return path to executable."""
    system = platform.system().lower()
    if system == "windows":
        url = STOCKFISH_URLS["windows"]
        install_dir = repo_root / "bin" / "stockfish"
        exe_name = "stockfish-windows-x86-64-avx2.exe"
    else:
        url = STOCKFISH_URLS["linux"]
        install_dir = repo_root / "bin" / "stockfish"
        exe_name = "stockfish-ubuntu-x86-64-avx2"

    exe_path = install_dir / exe_name

    # Check if already downloaded
    if exe_path.exists():
        print(f"Stockfish already downloaded: {exe_path}")
        return exe_path

    print(f"Downloading Stockfish 16.1 from {url}...")
    install_dir.mkdir(parents=True, exist_ok=True)

    # Download to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip" if system == "windows" else ".tar") as tmp:
        tmp_path = Path(tmp.name)

    try:
        urllib.request.urlretrieve(url, tmp_path)
        print(f"Downloaded to {tmp_path}")

        # Extract
        if system == "windows":
            with zipfile.ZipFile(tmp_path, 'r') as zf:
                zf.extractall(install_dir)
            # Find the exe in extracted folder
            for f in install_dir.rglob("*.exe"):
                if "stockfish" in f.name.lower():
                    exe_path = f
                    break
        else:
            import tarfile
            with tarfile.open(tmp_path, 'r') as tf:
                tf.extractall(install_dir)
            # Find the binary
            for f in install_dir.rglob("stockfish*"):
                if f.is_file() and "stockfish" in f.name.lower() and not f.suffix:
                    exe_path = f
                    # Make executable
                    exe_path.chmod(exe_path.stat().st_mode | 0o755)
                    break

        print(f"Stockfish installed: {exe_path}")
        return exe_path

    finally:
        tmp_path.unlink(missing_ok=True)


def check_nnue_file(nnue_path: Path) -> bool:
    """Check if NNUE file exists and is valid (not a LFS pointer)."""
    if not nnue_path.exists():
        print(f"ERROR: NNUE file not found: {nnue_path}")
        return False

    size = nnue_path.stat().st_size
    if size < 1000:  # LFS pointer files are ~130 bytes
        print(f"ERROR: NNUE file appears to be a Git LFS pointer ({size} bytes)")
        print(f"       Expected ~58MB for a valid NNUE file.")
        print(f"       Run: git lfs pull")
        return False

    print(f"NNUE file OK: {nnue_path} ({size / 1024 / 1024:.1f} MB)")
    return True


def check_stockfish(stockfish_path: Path) -> bool:
    """Check if Stockfish exists and runs."""
    if not stockfish_path.exists():
        print(f"ERROR: Stockfish not found: {stockfish_path}")
        return False

    # Try to run stockfish
    try:
        result = subprocess.run(
            [str(stockfish_path)],
            input="uci\nquit\n",
            capture_output=True,
            text=True,
            timeout=10,
        )
        if "uciok" in result.stdout:
            print(f"Stockfish OK: {stockfish_path}")
            return True
        else:
            print(f"ERROR: Stockfish didn't respond correctly")
            print(f"stdout: {result.stdout[:200]}")
            print(f"stderr: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"ERROR: Stockfish timed out")
        return False
    except Exception as e:
        print(f"ERROR: Failed to run Stockfish: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run NNUE scaling experiment")
    parser.add_argument(
        "--stockfish",
        default=None,
        help="Path to Stockfish binary (or use --auto-stockfish)",
    )
    parser.add_argument(
        "--auto-stockfish",
        action="store_true",
        help="Auto-download Stockfish 16.1 (required for NNUE compatibility)",
    )
    parser.add_argument(
        "--nnue",
        default=None,
        help="Path to NNUE file (default: models/nn-epoch16-manual.nnue)",
    )
    parser.add_argument("--games", type=int, default=20, help="Games per test point")
    parser.add_argument("--threads", type=int, default=4, help="Threads per engine")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    parser.add_argument("--hash-mb", type=int, default=128, help="Hash MB per engine")
    parser.add_argument(
        "--times",
        default="0.05,0.1,0.2,0.4,0.8,1.5,2.0",
        help="Comma-separated time controls (seconds)",
    )
    parser.add_argument(
        "--base-elos",
        default="2800,2900,3000,3100",
        help="Comma-separated opponent Elo levels",
    )
    parser.add_argument(
        "--cloud",
        action="store_true",
        help="Use cloud-optimized settings (more workers)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Custom run ID (default: timestamp)",
    )
    args = parser.parse_args()

    # Validate arguments
    if not args.stockfish and not args.auto_stockfish:
        print("ERROR: Must specify either --stockfish or --auto-stockfish")
        print("       Use --auto-stockfish to automatically download Stockfish 16.1")
        sys.exit(1)

    # Find repo root
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    # Handle stockfish path
    if args.auto_stockfish:
        stockfish_path = download_stockfish(repo_root)
    elif args.stockfish:
        # Resolve stockfish path (handle relative paths)
        stockfish_path = Path(args.stockfish)
        if not stockfish_path.is_absolute():
            # Try relative to cwd first, then repo root
            if (Path.cwd() / stockfish_path).exists():
                stockfish_path = (Path.cwd() / stockfish_path).resolve()
            elif (repo_root / stockfish_path).exists():
                stockfish_path = (repo_root / stockfish_path).resolve()
            else:
                # Check if it's in PATH
                found = shutil.which(args.stockfish)
                if found:
                    stockfish_path = Path(found).resolve()
        else:
            stockfish_path = stockfish_path.resolve()

    # Default NNUE path
    if args.nnue is None:
        nnue_path = repo_root / "models" / "nn-epoch16-manual.nnue"
    else:
        nnue_path = Path(args.nnue)
        if not nnue_path.is_absolute():
            if (Path.cwd() / nnue_path).exists():
                nnue_path = (Path.cwd() / nnue_path).resolve()
            elif (repo_root / nnue_path).exists():
                nnue_path = (repo_root / nnue_path).resolve()
    nnue_path = nnue_path.resolve()

    # Validate files
    print("=" * 60)
    print("Checking prerequisites...")
    print("=" * 60)

    nnue_ok = check_nnue_file(nnue_path)
    stockfish_ok = check_stockfish(stockfish_path)

    if not nnue_ok or not stockfish_ok:
        print("\nPrerequisite check failed. Please fix the errors above.")
        sys.exit(1)

    # Cloud settings
    if args.cloud:
        args.workers = max(args.workers, 10)
        print(f"Cloud mode: using {args.workers} workers")

    # Generate run ID
    run_id = args.run_id or datetime.now().strftime("scaling_%Y%m%d_%H%M%S")

    # Build output paths
    out_dir = repo_root / "outputs" / "speed_demon" / "eval" / "scaling_runs"
    debug_dir = out_dir / run_id / "debug"

    # Calculate total games
    num_times = len(args.times.split(","))
    num_elos = len(args.base_elos.split(","))
    total_games = num_times * num_elos * args.games

    print()
    print("=" * 60)
    print("NNUE Scaling Experiment")
    print("=" * 60)
    print(f"NNUE:        {nnue_path}")
    print(f"Stockfish:   {stockfish_path}")
    print(f"Times:       {args.times}")
    print(f"Base Elos:   {args.base_elos}")
    print(f"Games/point: {args.games}")
    print(f"Threads:     {args.threads}")
    print(f"Workers:     {args.workers}")
    print(f"Hash MB:     {args.hash_mb}")
    print(f"Run ID:      {run_id}")
    print(f"Total games: {total_games}")
    print("=" * 60)

    # Build command with absolute paths
    cmd = [
        sys.executable,
        str(repo_root / "speed_demon" / "scaling_analysis.py"),
        "--nnue", str(nnue_path),
        "--stockfish", str(stockfish_path),
        "--games", str(args.games),
        "--threads", str(args.threads),
        "--workers", str(args.workers),
        "--hash-mb", str(args.hash_mb),
        "--times", args.times,
        "--base-elos", args.base_elos,
        "--out-dir", str(out_dir),
        "--run-id", run_id,
        "--debug-dir", str(debug_dir),
    ]

    print(f"\nRunning: {' '.join(cmd)}\n")

    # Run from repo root
    result = subprocess.run(cmd, cwd=str(repo_root))

    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("Experiment complete!")
        print(f"Results: {out_dir / run_id}")
        print("=" * 60)
    else:
        print(f"\nExperiment failed with code {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
