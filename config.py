"""
Chess Engine Training Configuration
Optimized for H100/A100 GPU training
"""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class ModelConfig:
    """Neural network architecture configuration"""
    num_blocks: int = 10          # Number of residual blocks
    num_filters: int = 256        # Filters per conv layer
    input_planes: int = 18        # 12 pieces + side to move + 4 castling + en passant
    policy_output_size: int = 1858  # All possible moves (from-to + promotions)
    se_ratio: int = 8             # Squeeze-excitation ratio


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # Data
    train_positions: int = 50_000_000   # Target training positions
    val_split: float = 0.01             # Validation set fraction

    # Batch size - H100 80GB can handle 8192 easily
    batch_size: int = 8192

    # Epochs
    epochs: int = 3

    # Optimizer
    lr: float = 0.1
    lr_schedule: str = "cosine"         # "cosine", "step", "constant"
    warmup_steps: int = 1000
    weight_decay: float = 1e-4
    momentum: float = 0.9

    # Loss weights
    policy_weight: float = 1.0
    value_weight: float = 1.0

    # Regularization
    dropout: float = 0.0
    label_smoothing: float = 0.0

    # Checkpointing and logging
    checkpoint_every: int = 1000
    eval_every: int = 500
    log_every: int = 100

    # Stockfish evaluation during training
    eval_every_n_epochs: int = 1      # Run Stockfish eval every N epochs (0 to disable)
    eval_num_games: int = 10          # Number of games per evaluation


@dataclass
class HardwareConfig:
    """Hardware and performance settings"""
    # GPU settings
    num_gpus: int = 1                   # For distributed training
    precision: str = "bf16"             # "fp32", "fp16", "bf16"
    compile_model: bool = True          # torch.compile for 40% speedup

    # DataLoader
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 4

    # Memory optimization
    gradient_checkpointing: bool = False
    gradient_accumulation_steps: int = 1

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        if self.precision == "bf16":
            return torch.bfloat16
        elif self.precision == "fp16":
            return torch.float16
        return torch.float32


@dataclass
class DataConfig:
    """Data paths and processing"""
    data_dir: str = "./data/processed"
    raw_dir: str = "./data/raw"
    cache_dir: str = "./data/cache"

    # Lichess Elite Database
    lichess_base_url: str = "https://database.nikonoel.fr/"

    # Processing
    max_game_length: int = 300          # Max moves per game
    min_elo: int = 2300                 # Minimum player ELO
    include_draws: bool = True

    # Augmentation
    random_flip: bool = True            # Mirror board horizontally


@dataclass
class BenchmarkConfig:
    """Benchmarking settings"""
    stockfish_path: Optional[str] = None  # Auto-detect if None
    stockfish_depth: int = 20

    # ELO testing
    num_games: int = 100
    time_control: float = 1.0           # Seconds per move

    # Tactical suite
    tactical_suite_path: str = "./data/puzzles.epd"
    tactical_time_limit: float = 10.0   # Seconds per puzzle


@dataclass
class Config:
    """Master configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    data: DataConfig = field(default_factory=DataConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    # Experiment
    experiment_name: str = "chess_engine_v1"
    output_dir: str = "./outputs"
    seed: int = 42

    def __post_init__(self):
        """Validate configuration"""
        assert self.model.num_blocks > 0
        assert self.training.batch_size > 0
        assert self.training.epochs > 0


# Default configuration
def get_config(preset: str = "default") -> Config:
    """Get configuration preset"""

    if preset == "default":
        return Config()

    elif preset == "fast":
        # Faster training, slightly less accuracy
        return Config(
            model=ModelConfig(num_blocks=6, num_filters=128),
            training=TrainingConfig(epochs=2, batch_size=4096),
        )

    elif preset == "accurate":
        # More accurate, slower training
        return Config(
            model=ModelConfig(num_blocks=20, num_filters=256),
            training=TrainingConfig(epochs=5, batch_size=8192),
        )

    elif preset == "multi_gpu":
        # For 5x H100 distributed training
        return Config(
            hardware=HardwareConfig(num_gpus=5, compile_model=True),
            training=TrainingConfig(batch_size=8192 * 5),  # Scale with GPUs
        )

    else:
        raise ValueError(f"Unknown preset: {preset}")


if __name__ == "__main__":
    # Print default config
    config = get_config()
    print(f"Model: {config.model.num_blocks} blocks, {config.model.num_filters} filters")
    print(f"Training: {config.training.epochs} epochs, batch size {config.training.batch_size}")
    print(f"Hardware: {config.hardware.device}, {config.hardware.precision} precision")
