#!/usr/bin/env python3
"""
Quick launcher for scaling experiments.

Usage (local):
    python scripts/run_scaling_experiment.py --stockfish /path/to/stockfish

Usage (cloud with 64 vCPU):
    python scripts/run_scaling_experiment.py --stockfish /path/to/stockfish --cloud

Custom settings:
    python scripts/run_scaling_experiment.py --stockfish /path/to/stockfish --games 50 --workers 12
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run NNUE scaling experiment")
    parser.add_argument("--stockfish", required=True, help="Path to Stockfish binary")
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

    # Find repo root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent

    # Default NNUE path
    if args.nnue is None:
        args.nnue = str(repo_root / "models" / "nn-epoch16-manual.nnue")

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

    print("=" * 60)
    print("NNUE Scaling Experiment")
    print("=" * 60)
    print(f"NNUE:        {args.nnue}")
    print(f"Stockfish:   {args.stockfish}")
    print(f"Times:       {args.times}")
    print(f"Base Elos:   {args.base_elos}")
    print(f"Games/point: {args.games}")
    print(f"Threads:     {args.threads}")
    print(f"Workers:     {args.workers}")
    print(f"Hash MB:     {args.hash_mb}")
    print(f"Run ID:      {run_id}")
    print(f"Total games: {total_games}")
    print("=" * 60)

    # Build command
    cmd = [
        sys.executable,
        str(repo_root / "speed_demon" / "scaling_analysis.py"),
        "--nnue", args.nnue,
        "--stockfish", args.stockfish,
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

    # Run
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
