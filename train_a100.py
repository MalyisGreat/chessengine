#!/usr/bin/env python3
"""
Training script for 1x A100 GPU

This script:
1. Downloads Lichess evaluated positions (with REAL Stockfish centipawn evals)
2. Trains the chess engine
3. Runs benchmarks

Expected time: ~2-3 hours total
- Download/process: ~30-60 min
- Training: ~1.5-2 hours
"""

import os
import sys
import subprocess
import argparse


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")

    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"ERROR: {description} failed with code {result.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Train chess engine on 1x A100")
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
    parser.add_argument("--data-dir", type=str, default="./data/lichess_eval",
                        help="Data directory")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark after training")

    args = parser.parse_args()

    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║           CHESS ENGINE TRAINING - 1x A100                 ║
    ╠═══════════════════════════════════════════════════════════╣
    ║  Dataset: Lichess Evaluated Positions (Stockfish evals)   ║
    ║  This fixes the value head training problem!              ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    print(f"Settings:")
    print(f"  Positions: {args.positions:,}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Data directory: {args.data_dir}")

    # Step 1: Download data
    if not args.skip_download:
        run_command(
            f"python download_lichess_eval.py "
            f"--output {args.data_dir} "
            f"--positions {args.positions}",
            "Step 1/3: Downloading Lichess evaluated positions"
        )

        # Verify
        run_command(
            f"python download_lichess_eval.py --output {args.data_dir} --verify",
            "Verifying dataset"
        )
    else:
        print("\n[Skipping download - using existing data]")

    # Step 2: Train
    run_command(
        f"python train.py "
        f"--data {args.data_dir} "
        f"--batch-size {args.batch_size} "
        f"--epochs {args.epochs} "
        f"--lr {args.lr}",
        "Step 2/3: Training"
    )

    # Step 3: Benchmark (optional)
    if args.benchmark:
        model_path = "./outputs/chess_engine_v1/checkpoint_best.pt"
        if os.path.exists(model_path):
            run_command(
                f"python benchmark.py --model {model_path} --quick",
                "Step 3/3: Benchmarking"
            )
        else:
            print(f"\nWarning: Model not found at {model_path}")
            print("Skipping benchmark")

    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                    TRAINING COMPLETE!                     ║
    ╠═══════════════════════════════════════════════════════════╣
    ║  Model saved to: ./outputs/chess_engine_v1/               ║
    ║                                                           ║
    ║  Next steps:                                              ║
    ║  1. Benchmark: python benchmark.py --model <path> --quick ║
    ║  2. Play: python -m engine.play --model <path>            ║
    ╚═══════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
