#!/usr/bin/env python3
"""
Quick training throughput benchmark for small datasets.

Generates a tiny synthetic dataset (if needed) and runs a few
data loader/training variants to compare samples/sec.
"""

import argparse
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.encoder import BoardEncoder
from data.dataset import ChessDataset
from models.network import ChessNetwork, ChessLoss


def generate_dataset(output_dir: str, positions: int, chunk_size: int, seed: int) -> None:
    os.makedirs(output_dir, exist_ok=True)
    if any(name.endswith(".npz") for name in os.listdir(output_dir)):
        return

    rng = np.random.default_rng(seed)
    encoder = BoardEncoder()
    num_moves = encoder.num_moves
    remaining = positions
    chunk_idx = 0

    while remaining > 0:
        n = min(chunk_size, remaining)
        boards = rng.random((n, 18, 8, 8)).astype(np.float32)
        policies = np.zeros((n, num_moves), dtype=np.float32)
        move_idx = rng.integers(0, num_moves, size=n)
        policies[np.arange(n), move_idx] = 1.0
        values = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)

        chunk_path = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.npz")
        np.savez(chunk_path, boards=boards, policies=policies, values=values)
        chunk_idx += 1
        remaining -= n


def run_variant(name: str, args: argparse.Namespace, variant: dict) -> dict:
    dataset = ChessDataset(args.data_dir, augment=variant["augment"])

    loader_kwargs = {}
    if variant["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = variant["prefetch_factor"]
        loader_kwargs["persistent_workers"] = variant["persistent_workers"]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=variant["num_workers"],
        pin_memory=variant["pin_memory"],
        drop_last=True,
        **loader_kwargs,
    )

    device = torch.device(args.device)
    encoder = BoardEncoder()
    model = ChessNetwork(
        num_blocks=args.model_blocks,
        num_filters=args.model_filters,
        input_planes=18,
        num_moves=encoder.num_moves,
        se_ratio=8,
    ).to(device)

    if variant["channels_last"]:
        model = model.to(memory_format=torch.channels_last)

    loss_fn = ChessLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    use_amp = device.type == "cuda" and args.use_amp
    scaler = torch.amp.GradScaler(device.type) if use_amp else None
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    if use_amp:
        def autocast_ctx():
            return torch.amp.autocast(device.type, dtype=amp_dtype)
    else:
        def autocast_ctx():
            return nullcontext()

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    model.train()
    iterator = iter(loader)
    total_steps = args.warmup_steps + args.steps
    start_time = None

    for step in range(total_steps):
        try:
            boards, policies, values = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            boards, policies, values = next(iterator)

        if variant["channels_last"]:
            boards = boards.to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            boards = boards.to(device, non_blocking=True)
        policies = policies.to(device, non_blocking=True)
        values = values.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx():
            policy_logits, value_pred = model(boards)
            loss, _, _ = loss_fn(policy_logits, value_pred, policies, values)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if step == args.warmup_steps - 1:
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    samples = args.steps * args.batch_size

    return {
        "name": name,
        "samples_per_sec": samples / elapsed if elapsed > 0 else 0.0,
        "elapsed_sec": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny training throughput benchmark")
    parser.add_argument("--data-dir", type=str, default="./data/bench_small",
                        help="Directory for synthetic dataset")
    parser.add_argument("--positions", type=int, default=20000,
                        help="Total synthetic positions to generate")
    parser.add_argument("--chunk-size", type=int, default=5000,
                        help="Positions per chunk file")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for benchmark")
    parser.add_argument("--steps", type=int, default=50,
                        help="Measured steps per variant")
    parser.add_argument("--warmup-steps", type=int, default=5,
                        help="Warmup steps before timing")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on (cuda or cpu)")
    parser.add_argument("--use-amp", action="store_true",
                        help="Enable AMP if CUDA is available")
    parser.add_argument("--model-blocks", type=int, default=4,
                        help="Residual blocks for the benchmark model")
    parser.add_argument("--model-filters", type=int, default=64,
                        help="Filters per layer for the benchmark model")
    parser.add_argument("--variants", type=str, default="baseline,loader,loader_noaugment,channels_last",
                        help="Comma-separated list of variants to run")
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed for dataset generation")
    args = parser.parse_args()

    generate_dataset(args.data_dir, args.positions, args.chunk_size, args.seed)

    cpu_workers = max(1, (os.cpu_count() or 2) // 2)
    variants = {
        "baseline": {
            "num_workers": 0,
            "pin_memory": False,
            "prefetch_factor": 2,
            "persistent_workers": False,
            "augment": True,
            "channels_last": False,
        },
        "loader": {
            "num_workers": min(4, cpu_workers),
            "pin_memory": True,
            "prefetch_factor": 4,
            "persistent_workers": True,
            "augment": True,
            "channels_last": False,
        },
        "loader_noaugment": {
            "num_workers": min(4, cpu_workers),
            "pin_memory": True,
            "prefetch_factor": 4,
            "persistent_workers": True,
            "augment": False,
            "channels_last": False,
        },
        "channels_last": {
            "num_workers": min(4, cpu_workers),
            "pin_memory": True,
            "prefetch_factor": 4,
            "persistent_workers": True,
            "augment": True,
            "channels_last": True,
        },
    }

    requested = [v.strip() for v in args.variants.split(",") if v.strip()]
    results = []
    for name in requested:
        if name not in variants:
            print(f"Skipping unknown variant: {name}")
            continue
        if args.device == "cpu" and name == "channels_last":
            print("Skipping channels_last on CPU.")
            continue
        print(f"\nRunning variant: {name}")
        results.append(run_variant(name, args, variants[name]))

    print("\nResults (samples/sec):")
    for result in results:
        print(f"  {result['name']}: {result['samples_per_sec']:.1f} samples/s "
              f"({result['elapsed_sec']:.3f}s)")


if __name__ == "__main__":
    main()
