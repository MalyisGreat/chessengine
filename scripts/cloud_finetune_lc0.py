#!/usr/bin/env python3
"""
[EXPERIMENTAL] Train NNUE from scratch on Lc0 evaluation data.

This is a SEPARATE training pipeline that does NOT affect the main
speed_demon training. It uses:
  - Different data: Lc0 test80 rescored positions (~8GB)
  - Different output: outputs/finetune_lc0/
  - Same architecture: HalfKAv2_hm (2560/15/32) for Stockfish 16.1

The goal is to train on higher-quality Lc0 evaluations which may
produce better positional understanding than self-play data.

Usage:
    # Train from scratch on Lc0 data
    python scripts/cloud_finetune_lc0.py --force-fresh --epochs 15

    # With custom batch size for large GPU
    python scripts/cloud_finetune_lc0.py --force-fresh --epochs 15 --batch-size 32768

    # Resume from a checkpoint
    python scripts/cloud_finetune_lc0.py --checkpoint path/to/model.ckpt --epochs 10
"""

import argparse
import os
import sys
import subprocess
import urllib.request
import shutil
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Lc0 data URL (February 2024, ~7GB compressed, ~8GB uncompressed)
LC0_DATA_URL = "https://huggingface.co/datasets/linrock/test80-2024/resolve/main/test80-2024-02-feb-2tb7p.min-v2.v6.binpack.zst"

# Architecture (Stockfish 16.1 compatible)
FEATURES = "HalfKAv2_hm"
L1, L2, L3 = 2560, 15, 32

# ============================================================================
# PATHS
# ============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
NNUE_PYTORCH_DIR = PROJECT_ROOT / "nnue-pytorch"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "training_data" / "lc0"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "finetune_lc0"

SOURCE_NNUE = MODELS_DIR / "nn-epoch16-manual.nnue"
LC0_BINPACK = DATA_DIR / "test80-2024-02-feb.binpack"
LC0_BINPACK_ZST = DATA_DIR / "test80-2024-02-feb.binpack.zst"


def run_command(cmd, cwd=None, check=True):
    """Run a command and stream output."""
    print(f"\n>>> {' '.join(cmd[:6])}{'...' if len(cmd) > 6 else ''}")
    result = subprocess.run(cmd, cwd=cwd, check=check)
    return result.returncode == 0


def ensure_nnue_pytorch():
    """Ensure nnue-pytorch is cloned and patched."""
    if not NNUE_PYTORCH_DIR.exists():
        print("\nnnue-pytorch not found. Cloning...")
        run_command([
            "git", "clone",
            "https://github.com/official-stockfish/nnue-pytorch.git",
            str(NNUE_PYTORCH_DIR)
        ])

    # Apply L2/L3 patches if patch script exists
    patch_script = PROJECT_ROOT / "speed_demon" / "patch_nnue_pytorch.py"
    if patch_script.exists() and NNUE_PYTORCH_DIR.exists():
        print("\nPatching nnue-pytorch for L2/L3 support...")
        run_command([
            sys.executable, str(patch_script),
            "--repo", str(NNUE_PYTORCH_DIR)
        ], check=False)

    return NNUE_PYTORCH_DIR.exists()


def download_file(url, dest, desc="Downloading"):
    """Download a file with progress."""
    print(f"\n{desc}: {url}")
    print(f"Destination: {dest}")

    dest.parent.mkdir(parents=True, exist_ok=True)

    # Use curl for better progress display
    cmd = ["curl", "-L", "-o", str(dest), url, "--progress-bar"]
    return run_command(cmd, check=False)


def decompress_zst(src, dst):
    """Decompress a .zst file."""
    print(f"\nDecompressing: {src}")

    # Try zstd command first
    try:
        cmd = ["zstd", "-d", str(src), "-o", str(dst), "--force"]
        if run_command(cmd, check=False):
            return True
    except FileNotFoundError:
        pass

    # Fall back to Python zstandard
    try:
        import zstandard as zstd
        print("Using Python zstandard library...")

        with open(src, 'rb') as f_in:
            dctx = zstd.ZstdDecompressor()
            with open(dst, 'wb') as f_out:
                dctx.copy_stream(f_in, f_out)
        return True
    except ImportError:
        print("Installing zstandard...")
        subprocess.run([sys.executable, "-m", "pip", "install", "zstandard"], check=True)
        import zstandard as zstd

        with open(src, 'rb') as f_in:
            dctx = zstd.ZstdDecompressor()
            with open(dst, 'wb') as f_out:
                dctx.copy_stream(f_in, f_out)
        return True


def convert_nnue_to_pt(nnue_path, pt_path):
    """Convert NNUE to PyTorch format."""
    print(f"\nConverting NNUE to PyTorch format...")
    print(f"  Source: {nnue_path}")
    print(f"  Target: {pt_path}")

    pt_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "serialize.py",
        str(nnue_path),
        str(pt_path),
        "--features", FEATURES,
        "--l1", str(L1)
    ]

    return run_command(cmd, cwd=NNUE_PYTORCH_DIR)


def convert_ckpt_to_nnue(ckpt_path, nnue_path):
    """Convert checkpoint to NNUE format."""
    print(f"\nExporting to NNUE format...")
    print(f"  Source: {ckpt_path}")
    print(f"  Target: {nnue_path}")

    cmd = [
        sys.executable, "serialize.py",
        str(ckpt_path),
        str(nnue_path),
        "--features", FEATURES,
        "--l1", str(L1)
    ]

    return run_command(cmd, cwd=NNUE_PYTORCH_DIR, check=False)


def find_best_checkpoint(output_dir):
    """Find the best/last checkpoint."""
    ckpt_dirs = list(output_dir.glob("lightning_logs/version_*/checkpoints"))
    if not ckpt_dirs:
        return None

    latest_dir = max(ckpt_dirs, key=lambda p: p.stat().st_mtime)
    ckpts = list(latest_dir.glob("*.ckpt"))

    if not ckpts:
        return None

    # Prefer 'last' checkpoint
    for ckpt in ckpts:
        if "last" in ckpt.name:
            return ckpt

    return max(ckpts, key=lambda p: p.stat().st_mtime)


def run_training(data_path, resume_model, output_dir, args, from_scratch=False):
    """Run the training (fine-tuning or from scratch)."""

    output_dir.mkdir(parents=True, exist_ok=True)

    mode = "FRESH TRAINING" if from_scratch else "FINE-TUNING"
    print(f"\n{'='*60}")
    print(f"{mode} CONFIGURATION")
    print(f"{'='*60}")
    print(f"Training data:    {data_path} ({data_path.stat().st_size / 1e9:.2f} GB)")
    if not from_scratch:
        print(f"Resume from:      {resume_model}")
    print(f"Output dir:       {output_dir}")
    print(f"Epochs:           {args.epochs}")
    print(f"Learning rate:    {args.lr}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Positions/epoch:  {args.epoch_size:,}")
    print(f"Workers:          {args.num_workers}")
    print(f"Architecture:     {FEATURES} ({L1}/{L2}/{L3})")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "train.py",
        str(data_path),
        "--default_root_dir", str(output_dir),
        "--features", FEATURES,
        "--l1", str(L1),
        "--l2", str(L2),
        "--l3", str(L3),
        "--max_epochs", str(args.epochs),
        "--epoch-size", str(args.epoch_size),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--lr", str(args.lr),
        "--gamma", "0.995",  # Slower LR decay
        "--network-save-period", "1",
        "--save-last-network", "True",
        "--validation-size", str(min(1_000_000, args.epoch_size // 10)),
    ]

    # Add resume model only if not training from scratch
    if not from_scratch and resume_model:
        cmd.extend(["--resume-from-model", str(resume_model)])

    # Add threads if specified
    if args.threads:
        cmd.extend(["--threads", str(args.threads)])

    print("Starting training...\n")
    return run_command(cmd, cwd=NNUE_PYTORCH_DIR, check=False)


def main():
    parser = argparse.ArgumentParser(description="Cloud Fine-Tune Epoch 16 NNUE on Lc0 Data")

    # Data options
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading data (use existing)")
    parser.add_argument("--data", type=Path, default=None,
                        help="Custom data path (skip download)")

    # Training options
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of fine-tuning epochs (default: 10)")
    parser.add_argument("--lr", type=float, default=0.0005,
                        help="Learning rate (default: 0.0005)")
    parser.add_argument("--batch-size", type=int, default=16384,
                        help="Batch size (default: 16384, use 32768 for large GPU)")
    parser.add_argument("--epoch-size", type=int, default=50_000_000,
                        help="Positions per epoch (default: 50M)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Data loader workers (default: 4)")
    parser.add_argument("--threads", type=int, default=None,
                        help="Number of threads")

    # Resume options
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to .ckpt or .pt file to resume from")
    parser.add_argument("--force-fresh", action="store_true",
                        help="Train from scratch if NNUE conversion fails")

    # Output options
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR,
                        help="Output directory")
    parser.add_argument("--export-name", type=str, default="nn-finetuned-lc0.nnue",
                        help="Name for exported NNUE file")

    args = parser.parse_args()

    print("="*60)
    print("NNUE FINE-TUNING ON LC0 DATA")
    print("="*60)
    print(f"Project root: {PROJECT_ROOT}")

    # Ensure nnue-pytorch is available
    if not ensure_nnue_pytorch():
        print("ERROR: Could not set up nnue-pytorch")
        sys.exit(1)

    # ========================================================================
    # STEP 1: Download Lc0 data (if needed)
    # ========================================================================

    data_path = args.data if args.data else LC0_BINPACK

    if not args.skip_download and not args.data:
        if data_path.exists():
            print(f"\nData already exists: {data_path}")
            print(f"Size: {data_path.stat().st_size / 1e9:.2f} GB")
        else:
            print("\n" + "="*60)
            print("STEP 1: Download Lc0 Training Data")
            print("="*60)

            DATA_DIR.mkdir(parents=True, exist_ok=True)

            # Download compressed file
            if not LC0_BINPACK_ZST.exists():
                if not download_file(LC0_DATA_URL, LC0_BINPACK_ZST, "Downloading Lc0 data"):
                    print("ERROR: Download failed!")
                    sys.exit(1)

            # Decompress
            if not decompress_zst(LC0_BINPACK_ZST, LC0_BINPACK):
                print("ERROR: Decompression failed!")
                sys.exit(1)

            # Clean up compressed file
            print(f"\nCleaning up: {LC0_BINPACK_ZST}")
            LC0_BINPACK_ZST.unlink(missing_ok=True)

    if not data_path.exists():
        print(f"ERROR: Training data not found: {data_path}")
        sys.exit(1)

    # ========================================================================
    # STEP 2: Convert NNUE to PyTorch format (or use checkpoint)
    # ========================================================================

    print("\n" + "="*60)
    print("STEP 2: Convert NNUE to PyTorch Format")
    print("="*60)

    pt_path = args.output / "epoch16_resume.pt"
    train_from_scratch = False

    # Check if user provided a checkpoint file directly
    if args.checkpoint:
        if args.checkpoint.exists():
            pt_path = args.checkpoint
            print(f"Using provided checkpoint: {pt_path}")
        else:
            print(f"ERROR: Checkpoint not found: {args.checkpoint}")
            sys.exit(1)
    elif pt_path.exists():
        print(f"PyTorch model already exists: {pt_path}")
    else:
        if SOURCE_NNUE.exists():
            print(f"Attempting to convert: {SOURCE_NNUE}")
            if not convert_nnue_to_pt(SOURCE_NNUE, pt_path):
                print("\nWARNING: NNUE conversion failed (format incompatible)")
                print("The NNUE file was created with a different nnue-pytorch version.")
                if args.force_fresh:
                    print("--force-fresh specified, training from scratch...")
                    train_from_scratch = True
                else:
                    print("\nOptions:")
                    print("  1. Provide a checkpoint file with --checkpoint path/to/file.ckpt")
                    print("  2. Use --force-fresh to train from scratch on Lc0 data")
                    print("  3. Copy checkpoints from your training run")
                    sys.exit(1)
        else:
            print(f"Source NNUE not found: {SOURCE_NNUE}")
            if args.force_fresh:
                print("--force-fresh specified, training from scratch...")
                train_from_scratch = True
            else:
                print("Use --force-fresh to train from scratch on Lc0 data")
                sys.exit(1)

    # ========================================================================
    # STEP 3: Run Training
    # ========================================================================

    print("\n" + "="*60)
    print("STEP 3: Training")
    print("="*60)

    resume_model = None if train_from_scratch else pt_path
    if not run_training(data_path, resume_model, args.output, args, from_scratch=train_from_scratch):
        print("\nWARNING: Training may have encountered issues")

    # ========================================================================
    # STEP 4: Export Best Checkpoint
    # ========================================================================

    print("\n" + "="*60)
    print("STEP 4: Export Final Model")
    print("="*60)

    best_ckpt = find_best_checkpoint(args.output)
    if best_ckpt:
        output_nnue = MODELS_DIR / args.export_name
        if convert_ckpt_to_nnue(best_ckpt, output_nnue):
            print(f"\n{'='*60}")
            print("SUCCESS!")
            print(f"{'='*60}")
            print(f"Fine-tuned model: {output_nnue}")
            print(f"Size: {output_nnue.stat().st_size / 1e6:.1f} MB")
        else:
            print("\nWARNING: Could not export final NNUE")
    else:
        print("\nWARNING: No checkpoint found to export")

    print("\nDone!")


if __name__ == "__main__":
    main()
