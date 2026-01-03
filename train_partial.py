#!/usr/bin/env python3
"""
Train on partial data while download continues.

Usage:
    python train_partial.py --data ./data/lichess_eval --epochs 1
"""

import os
import sys
import subprocess

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train on partial downloaded data")
    parser.add_argument("--data", type=str, default="./data/lichess_eval")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=0.1)

    args = parser.parse_args()

    # Count available positions
    import glob
    import numpy as np

    files = sorted(glob.glob(os.path.join(args.data, "chunk_*.npz")))
    if not files:
        print(f"No data files found in {args.data}")
        sys.exit(1)

    total = 0
    for f in files:
        try:
            data = np.load(f)
            total += len(data['boards'])
        except:
            pass  # File might be being written

    print(f"\n{'='*60}")
    print(f"TRAINING ON PARTIAL DATA")
    print(f"{'='*60}")
    print(f"Found {len(files)} chunk files")
    print(f"Total positions: {total:,}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"{'='*60}\n")

    # Run training
    cmd = [
        sys.executable, "train.py",
        "--data", args.data,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
    ]

    print(f"Running: {' '.join(cmd)}\n")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
