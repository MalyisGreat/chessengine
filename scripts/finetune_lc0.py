#!/usr/bin/env python3
"""
Fine-tune the Epoch 16 NNUE on Lc0-derived data.

This script:
1. Converts the epoch 16 NNUE to PyTorch format (if needed)
2. Runs fine-tuning on the Lc0 binpack data
3. Exports the final model back to NNUE format

Usage:
    python scripts/finetune_lc0.py
    python scripts/finetune_lc0.py --epochs 10 --lr 0.0005
    python scripts/finetune_lc0.py --data path/to/custom.binpack
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
NNUE_PYTORCH_DIR = PROJECT_ROOT / "nnue-pytorch"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "finetune_lc0"

# Default files
DEFAULT_NNUE = MODELS_DIR / "nn-epoch16-manual.nnue"
DEFAULT_LC0_DATA = PROJECT_ROOT / "training_data" / "lc0" / "test80-2024-02-feb.binpack"

# Architecture (Stockfish 16.1 compatible)
FEATURES = "HalfKAv2_hm"
L1, L2, L3 = 2560, 15, 32


def convert_nnue_to_pt(nnue_path: Path, pt_path: Path):
    """Convert NNUE file to PyTorch format for resuming."""
    print(f"Converting {nnue_path.name} to PyTorch format...")

    cmd = [
        sys.executable, "serialize.py",
        str(nnue_path),
        str(pt_path),
        "--features", FEATURES,
        "--l1", str(L1)
    ]

    result = subprocess.run(cmd, cwd=NNUE_PYTORCH_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error converting NNUE: {result.stderr}")
        sys.exit(1)

    print(f"Converted to: {pt_path}")
    return pt_path


def convert_ckpt_to_nnue(ckpt_path: Path, nnue_path: Path):
    """Convert checkpoint to NNUE format."""
    print(f"Converting checkpoint to NNUE format...")

    cmd = [
        sys.executable, "serialize.py",
        str(ckpt_path),
        str(nnue_path),
        "--features", FEATURES,
        "--l1", str(L1)
    ]

    result = subprocess.run(cmd, cwd=NNUE_PYTORCH_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error converting to NNUE: {result.stderr}")
        return None

    print(f"Exported NNUE: {nnue_path}")
    return nnue_path


def run_finetuning(
    data_path: Path,
    resume_model: Path,
    output_dir: Path,
    epochs: int = 10,
    lr: float = 0.0005,
    batch_size: int = 16384,
    epoch_size: int = 50_000_000,
    num_workers: int = 4
):
    """Run the fine-tuning process."""

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("FINE-TUNING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Training data:    {data_path}")
    print(f"Resume from:      {resume_model}")
    print(f"Output dir:       {output_dir}")
    print(f"Epochs:           {epochs}")
    print(f"Learning rate:    {lr}")
    print(f"Batch size:       {batch_size}")
    print(f"Positions/epoch:  {epoch_size:,}")
    print(f"Architecture:     {FEATURES} ({L1}/{L2}/{L3})")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "train.py",
        str(data_path),
        "--default_root_dir", str(output_dir),
        "--resume-from-model", str(resume_model),
        "--features", FEATURES,
        "--l1", str(L1),
        "--l2", str(L2),
        "--l3", str(L3),
        "--max_epochs", str(epochs),
        "--epoch-size", str(epoch_size),
        "--batch-size", str(batch_size),
        "--num-workers", str(num_workers),
        "--lr", str(lr),
        "--gamma", "0.995",  # Slower LR decay for fine-tuning
        "--network-save-period", "1",
        "--save-last-network", "True",
        "--validation-size", str(min(1_000_000, epoch_size // 10)),
    ]

    print("Running training...")
    print(f"Command: {' '.join(cmd[:5])} ...")

    # Run training
    process = subprocess.Popen(
        cmd,
        cwd=NNUE_PYTORCH_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Stream output
    for line in process.stdout:
        print(line, end='')

    process.wait()

    if process.returncode != 0:
        print(f"\nTraining failed with code {process.returncode}")
        sys.exit(1)

    print("\nTraining complete!")
    return output_dir


def find_best_checkpoint(output_dir: Path) -> Path:
    """Find the best/last checkpoint in output directory."""
    ckpt_dirs = list(output_dir.glob("lightning_logs/version_*/checkpoints"))
    if not ckpt_dirs:
        return None

    latest_dir = max(ckpt_dirs, key=lambda p: p.stat().st_mtime)
    ckpts = list(latest_dir.glob("*.ckpt"))

    if not ckpts:
        return None

    # Prefer 'last' checkpoint, otherwise latest by epoch
    for ckpt in ckpts:
        if "last" in ckpt.name:
            return ckpt

    return max(ckpts, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Epoch 16 NNUE on Lc0 data")
    parser.add_argument("--data", type=Path, default=DEFAULT_LC0_DATA,
                        help="Path to training data binpack")
    parser.add_argument("--nnue", type=Path, default=DEFAULT_NNUE,
                        help="Path to source NNUE file")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR,
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=0.0005,
                        help="Learning rate (lower than initial training)")
    parser.add_argument("--batch-size", type=int, default=16384,
                        help="Batch size")
    parser.add_argument("--epoch-size", type=int, default=50_000_000,
                        help="Positions per epoch")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Data loader workers")
    parser.add_argument("--skip-convert", action="store_true",
                        help="Skip NNUE to PT conversion (if already done)")
    parser.add_argument("--export-only", type=Path, default=None,
                        help="Only export checkpoint to NNUE (skip training)")

    args = parser.parse_args()

    # Validate paths
    if not args.data.exists():
        print(f"Error: Training data not found: {args.data}")
        print(f"Run the download first or specify --data path")
        sys.exit(1)

    if not args.nnue.exists():
        print(f"Error: Source NNUE not found: {args.nnue}")
        sys.exit(1)

    # Export-only mode
    if args.export_only:
        ckpt = args.export_only
        if not ckpt.exists():
            print(f"Error: Checkpoint not found: {ckpt}")
            sys.exit(1)

        output_nnue = MODELS_DIR / "nn-finetuned-lc0.nnue"
        convert_ckpt_to_nnue(ckpt, output_nnue)
        return

    # Step 1: Convert NNUE to PT format
    pt_path = args.output / "epoch16_resume.pt"
    pt_path.parent.mkdir(parents=True, exist_ok=True)

    if args.skip_convert and pt_path.exists():
        print(f"Using existing PT file: {pt_path}")
    else:
        convert_nnue_to_pt(args.nnue, pt_path)

    # Step 2: Run fine-tuning
    run_finetuning(
        data_path=args.data,
        resume_model=pt_path,
        output_dir=args.output,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        epoch_size=args.epoch_size,
        num_workers=args.num_workers
    )

    # Step 3: Export best checkpoint to NNUE
    best_ckpt = find_best_checkpoint(args.output)
    if best_ckpt:
        output_nnue = MODELS_DIR / "nn-finetuned-lc0.nnue"
        convert_ckpt_to_nnue(best_ckpt, output_nnue)
        print(f"\nFinal model saved to: {output_nnue}")
    else:
        print("\nWarning: No checkpoint found to export")


if __name__ == "__main__":
    main()
