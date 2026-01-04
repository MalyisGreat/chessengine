# Chess Engine

## Speed Demon NNUE (RunPod single command)

This adds a new NNUE training pipeline (Speed Demon: HalfKP, 256x32x32, int8)
that trains on Linrock T80 binpack and can run Stockfish eval games after each
epoch.

Single command on RunPod:

```bash
python speed_demon/runpod_train.py
```

Defaults:
- 20M positions total (5M per epoch x 4 epochs)
- Stockfish eval after every epoch
- Outputs to `./outputs/speed_demon`
- Eval results CSV: `./outputs/speed_demon/eval/eval.csv`

Notes:
- Stockfish 16.1 requires HalfKAv2_hm with 2560/15/32. The Speed Demon
  256/32/32 net is not Stockfish-compatible, so eval will fail unless you
  switch to `--stockfish-compat` or `--skip-eval`.
- By default, base Stockfish uses its own built-in nets for eval games.

Useful overrides:

```bash
python speed_demon/runpod_train.py --positions 20000000 --positions-per-epoch 5000000 --eval-games 10
python speed_demon/runpod_train.py --skip-download --skip-install
python speed_demon/runpod_train.py --stockfish-compat
python speed_demon/runpod_train.py --skip-eval
python speed_demon/runpod_train.py --stockfish-classical-base
python speed_demon/runpod_train.py --stockfish-base-elo 1600 --eval-games 12
python speed_demon/runpod_train.py --data-max-gb 2 --positions 2000000 --positions-per-epoch 500000
python speed_demon/runpod_train.py --no-eval-ladder --stockfish-base-elo 1400
```

Local training tip (smaller dataset so it fits on a desktop):

```bash
python speed_demon/runpod_train.py --stockfish-compat --data-max-gb 2 --positions 2000000 --positions-per-epoch 500000 --batch-size 8192 --threads 4 --num-workers 2
```

Estimate Elo quickly (ladder vs limited Stockfish):

```bash
python speed_demon/estimate_elo.py --nnue ./outputs/speed_demon/nnue/nn-epoch4.nnue --stockfish /path/to/stockfish --levels 1200,1600,2000,2400 --games 8
```

Legacy CNN training pipeline is still available below.

A superhuman chess engine trained using supervised learning on 316M Stockfish-evaluated positions. Achieves ~3000 ELO in under 1 hour on 5x H100 GPUs.

## Quick Start (One Command)

```bash
# Clone and run everything
git clone https://github.com/MalyisGreat/chessengine.git && cd chessengine && \
pip install -r requirements.txt datasets && \
python download_data.py && \
python train.py --data ./data/train
```

## Step by Step

```bash
# 1. Install dependencies
pip install -r requirements.txt datasets

# 2. Download 10M positions from HuggingFace (~5 min)
python download_data.py

# 3. Download more positions for superhuman (optional)
python download_data.py --positions 50000000

# 4. Train (single GPU)
python train.py --data ./data/train

# 5. Train (multi-GPU, e.g. 5x H100)
torchrun --nproc_per_node=5 train.py --data ./data/train

# 6. Benchmark your model
python benchmark.py --model ./outputs/chess_engine_v1/checkpoint_best.pt --all

# 7. Play against it!
python -m engine.play --model ./outputs/chess_engine_v1/checkpoint_best.pt
```

## Monitoring & Graphs

Training writes metrics to disk so you can track progress live:

- TensorBoard logs: `./outputs/<experiment>/logs`
- Epoch metrics: `./outputs/<experiment>/metrics.csv` and `metrics.jsonl`
- Step metrics: `./outputs/<experiment>/step_metrics.csv`
- Auto-generated plot: `./outputs/<experiment>/metrics.png`

To view TensorBoard:

```bash
tensorboard --logdir ./outputs
```

## Data Source

Uses [Lichess Position Evaluations](https://huggingface.co/datasets/Lichess/chess-position-evaluations) from HuggingFace:
- **316 million** chess positions
- Pre-evaluated by **Stockfish**
- No PGN processing needed - instant download!

## Architecture

```
Input: 18 planes x 8x8
  - 12 piece planes (6 white + 6 black)
  - 1 side to move
  - 4 castling rights
  - 1 en passant square
    ↓
Conv 3x3, 256 filters
    ↓
10x Residual Blocks (256 filters each)
    ↓
├── Policy Head → 1968 moves
└── Value Head → [-1, 1] (Stockfish evaluation)
```

## Expected Results

| GPU Setup | Training Time | Expected ELO |
|-----------|--------------|--------------|
| 1x A100 | 2-3 hours | ~2800 |
| 1x H100 | 1.5 hours | ~2800 |
| 5x H100 | 30-45 min | ~3000 |

**Superhuman = 2900+ ELO** (beats all humans)

## Testing

Run the comprehensive test suite before training to verify everything works:

```bash
# Quick tests (no GPU/Stockfish needed)
python test_suite.py --quick

# All tests including GPU
python test_suite.py --all

# Specific test categories
python test_suite.py --gpu        # GPU tests only
python test_suite.py --stockfish  # Stockfish integration tests
```

Tests cover:
- Board encoding (18 planes, castling, en passant)
- Neural network forward/backward passes
- Data loading and batching
- Search algorithm (alpha-beta, mate detection)
- Model save/load roundtrips
- GPU performance benchmarks

## Benchmarking

Benchmarking is **on-demand** (not automatic during training). After training:

```bash
# Quick ELO estimate (~5 min)
python benchmark.py --model ./outputs/chess_engine_v1/checkpoint_best.pt --quick

# Full benchmark suite (~30 min)
python benchmark.py --model ./outputs/chess_engine_v1/checkpoint_best.pt --all

# Specific tests
python benchmark.py --model <path> --elo        # Stockfish matches for ELO
python benchmark.py --model <path> --tactical   # Tactical puzzle suite
python benchmark.py --model <path> --accuracy   # Policy accuracy test
```

| Test | What It Measures | Time |
|------|------------------|------|
| ELO Estimation | Games vs Stockfish at various levels | 10-30 min |
| Tactical Suite | Puzzle solving accuracy | 5 min |
| Policy Accuracy | Agreement with Stockfish best moves | 5 min |

## Project Structure

```
chess_engine/
├── train.py              # Distributed training script
├── benchmark.py          # ELO estimation & testing
├── test_suite.py         # Comprehensive test suite
├── download_data.py      # HuggingFace data downloader (FAST!)
├── config.py             # Hyperparameters
├── models/
│   └── network.py        # ResNet architecture
├── data/
│   ├── encoder.py        # Board encoding
│   └── dataset.py        # PyTorch dataset
├── engine/
│   ├── search.py         # Alpha-beta search + UCI
│   └── play.py           # Interactive play
└── scripts/
    ├── setup.sh          # Environment setup
    └── run_training.sh   # Training launcher
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- datasets (HuggingFace)
- python-chess
- Stockfish (for benchmarking)

## License

MIT
