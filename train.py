"""
Chess Engine Training Script

Supports:
- Single GPU training
- Multi-GPU distributed training with DDP
- Mixed precision (BF16/FP16)
- torch.compile optimization
- TensorBoard logging
- Checkpoint saving/resuming

Usage:
    # Single GPU
    python train.py

    # Multi-GPU (5x H100)
    torchrun --nproc_per_node=5 train.py

    # Resume from checkpoint
    python train.py --resume ./outputs/checkpoint.pt
"""

import os
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import numpy as np

from config import Config, get_config
from models.network import ChessNetwork, ChessLoss
from data.dataset import ChessDataset, create_dataloader
from data.encoder import BoardEncoder


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])

        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """Check if this is the main process"""
    return rank == 0


class Trainer:
    """
    Chess engine trainer

    Handles training loop, logging, and checkpointing
    """

    def __init__(
        self,
        config: Config,
        rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1,
    ):
        self.config = config
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main = is_main_process(rank)

        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{local_rank}')
        else:
            self.device = torch.device('cpu')

        # Create output directory
        self.output_dir = Path(config.output_dir) / config.experiment_name
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._setup_model()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss()
        self._setup_logging()

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')

    def _setup_model(self):
        """Initialize model"""
        self.model = ChessNetwork(
            num_blocks=self.config.model.num_blocks,
            num_filters=self.config.model.num_filters,
            input_planes=self.config.model.input_planes,
            num_moves=self.config.model.policy_output_size,
            se_ratio=self.config.model.se_ratio,
        ).to(self.device)

        if self.is_main:
            print(f"Model parameters: {self.model.count_parameters():,}")

        # Compile model for faster training
        if self.config.hardware.compile_model and hasattr(torch, 'compile'):
            if self.is_main:
                print("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

        # Wrap with DDP for distributed training
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )

    def _setup_optimizer(self):
        """Initialize optimizer"""
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.training.lr,
            momentum=self.config.training.momentum,
            weight_decay=self.config.training.weight_decay,
        )

    def _setup_scheduler(self):
        """Initialize learning rate scheduler"""
        if self.config.training.lr_schedule == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs,
                eta_min=1e-6,
            )
        elif self.config.training.lr_schedule == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=1,
                gamma=0.1,
            )
        else:
            self.scheduler = None

    def _setup_loss(self):
        """Initialize loss function"""
        self.loss_fn = ChessLoss(
            policy_weight=self.config.training.policy_weight,
            value_weight=self.config.training.value_weight,
            label_smoothing=self.config.training.label_smoothing,
        )

    def _setup_logging(self):
        """Initialize logging"""
        if self.is_main:
            log_dir = self.output_dir / "logs"
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """
        Main training loop

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
        """
        if self.is_main:
            print(f"\nStarting training for {self.config.training.epochs} epochs")
            print(f"Training on {len(train_loader.dataset):,} positions")
            print(f"Batch size: {self.config.training.batch_size}")
            print(f"Device: {self.device}")
            print(f"World size: {self.world_size}")
            print()

        # Setup mixed precision
        scaler = torch.amp.GradScaler('cuda') if self.config.hardware.precision != "fp32" else None
        dtype = self.config.hardware.dtype

        for epoch in range(self.epoch, self.config.training.epochs):
            self.epoch = epoch

            # Set epoch for distributed sampler
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)

            # Training epoch
            train_metrics = self._train_epoch(train_loader, scaler, dtype)

            # Validation
            if val_loader is not None and self.is_main:
                val_metrics = self._validate(val_loader, dtype)
            else:
                val_metrics = None

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Logging
            if self.is_main:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"\nEpoch {epoch + 1}/{self.config.training.epochs}")
                print(f"  Train Loss: {train_metrics['loss']:.4f}")
                print(f"  Policy Loss: {train_metrics['policy_loss']:.4f}")
                print(f"  Value Loss: {train_metrics['value_loss']:.4f}")
                print(f"  LR: {lr:.6f}")

                if val_metrics:
                    print(f"  Val Loss: {val_metrics['loss']:.4f}")

                # TensorBoard
                self.writer.add_scalar('train/loss', train_metrics['loss'], epoch)
                self.writer.add_scalar('train/policy_loss', train_metrics['policy_loss'], epoch)
                self.writer.add_scalar('train/value_loss', train_metrics['value_loss'], epoch)
                self.writer.add_scalar('train/lr', lr, epoch)

                if val_metrics:
                    self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)

                # Save checkpoint
                is_best = train_metrics['loss'] < self.best_loss
                if is_best:
                    self.best_loss = train_metrics['loss']
                self.save_checkpoint(is_best=is_best)

        if self.is_main:
            print("\nTraining complete!")
            self.writer.close()

    def _train_epoch(
        self,
        train_loader: DataLoader,
        scaler: Optional[torch.amp.GradScaler],
        dtype: torch.dtype,
    ) -> dict:
        """Train for one epoch"""
        self.model.train()

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {self.epoch + 1}",
            disable=not self.is_main,
        )

        for batch_idx, (boards, policy_targets, value_targets) in enumerate(pbar):
            boards = boards.to(self.device, non_blocking=True)
            policy_targets = policy_targets.to(self.device, non_blocking=True)
            value_targets = value_targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', dtype=dtype):
                policy_logits, value_pred = self.model(boards)
                loss, p_loss, v_loss = self.loss_fn(
                    policy_logits, value_pred,
                    policy_targets, value_targets,
                )

            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            total_policy_loss += p_loss.item()
            total_value_loss += v_loss.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'p_loss': f'{p_loss.item():.4f}',
                'v_loss': f'{v_loss.item():.4f}',
            })

            # Periodic logging
            if self.is_main and self.global_step % self.config.training.log_every == 0:
                self.writer.add_scalar('step/loss', loss.item(), self.global_step)
                self.writer.add_scalar('step/policy_loss', p_loss.item(), self.global_step)
                self.writer.add_scalar('step/value_loss', v_loss.item(), self.global_step)

        return {
            'loss': total_loss / num_batches,
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches,
        }

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader, dtype: torch.dtype) -> dict:
        """Validate the model"""
        self.model.eval()

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        for boards, policy_targets, value_targets in val_loader:
            boards = boards.to(self.device, non_blocking=True)
            policy_targets = policy_targets.to(self.device, non_blocking=True)
            value_targets = value_targets.to(self.device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=dtype):
                policy_logits, value_pred = self.model(boards)
                loss, p_loss, v_loss = self.loss_fn(
                    policy_logits, value_pred,
                    policy_targets, value_targets,
                )

            total_loss += loss.item()
            total_policy_loss += p_loss.item()
            total_value_loss += v_loss.item()
            num_batches += 1

        return {
            'loss': total_loss / num_batches,
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches,
        }

    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_loss': self.best_loss,
            'config': self.config,
        }

        # Save latest checkpoint
        path = self.output_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, path)

        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
            print(f"  Saved best checkpoint (loss: {self.best_loss:.4f})")

        # Save epoch checkpoint
        if (self.epoch + 1) % 1 == 0:  # Every epoch
            epoch_path = self.output_dir / f'checkpoint_epoch_{self.epoch + 1}.pt'
            torch.save(checkpoint, epoch_path)

    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        # Load model
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load training state
        self.epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']

        print(f"Resumed from epoch {self.epoch}, step {self.global_step}")


def main():
    parser = argparse.ArgumentParser(description="Train chess engine")

    parser.add_argument(
        "--config",
        type=str,
        default="default",
        choices=["default", "fast", "accurate", "multi_gpu"],
        help="Configuration preset",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="./data/train",
        help="Path to training data",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Override learning rate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs",
        help="Output directory",
    )

    args = parser.parse_args()

    # Setup distributed
    rank, local_rank, world_size = setup_distributed()

    # Load config
    config = get_config(args.config)

    # Override config with CLI arguments
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.lr = args.lr
    if args.output:
        config.output_dir = args.output

    # Set seed
    torch.manual_seed(config.seed + rank)
    np.random.seed(config.seed + rank)

    # Enable TF32 for faster training on Ampere+ GPUs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    try:
        # Create dataset
        if is_main_process(rank):
            print(f"Loading data from {args.data}...")

        dataset = ChessDataset(args.data)

        # Split into train/val
        val_size = int(len(dataset) * config.training.val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        # Create data loaders
        if world_size > 1:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
        else:
            train_sampler = None

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size // world_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=config.hardware.num_workers,
            pin_memory=config.hardware.pin_memory,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size // world_size,
            shuffle=False,
            num_workers=config.hardware.num_workers,
            pin_memory=config.hardware.pin_memory,
        ) if val_size > 0 else None

        # Create trainer
        trainer = Trainer(
            config=config,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
        )

        # Resume if specified
        if args.resume:
            trainer.load_checkpoint(args.resume)

        # Train
        trainer.train(train_loader, val_loader)

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
