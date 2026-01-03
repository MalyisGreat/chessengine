#!/bin/bash
# Training script for 1x A100 GPU using Lc0 T80 data
# Expected training time: ~2-3 hours for 50M positions

set -e

echo "============================================"
echo "Chess Engine Training - 1x A100"
echo "Using Lc0 T80 Dataset (SOFT POLICY TARGETS)"
echo "============================================"

# Step 1: Install dependencies
echo ""
echo "[1/3] Installing dependencies..."
pip install -q torch numpy python-chess datasets tqdm tensorboard stockfish requests

# Step 2: Download T80 training data
echo ""
echo "[2/3] Downloading Lc0 T80 training data..."
echo "T80 = Training Run 80 from Leela Chess Zero"
echo "Contains SOFT POLICY TARGETS (MCTS visit distribution)"
echo "This trains the policy head much better than hard targets!"
echo ""

# Download ~20 T80 files (~50M positions total)
python download_t80.py \
    --output ./data/t80 \
    --num-files 20 \
    --positions 50000000

# Verify dataset
echo ""
echo "Verifying dataset..."
python download_t80.py --output ./data/t80 --verify

# Step 3: Train
echo ""
echo "[3/3] Starting training..."
echo "Training with 1x A100 optimized settings"
echo "Using 1858-move policy (Lc0 format)"
echo ""

python train.py \
    --data ./data/t80 \
    --batch-size 8192 \
    --epochs 3 \
    --lr 0.1

echo ""
echo "============================================"
echo "Training complete!"
echo "============================================"
echo ""
echo "To benchmark your model:"
echo "  python benchmark.py --model ./outputs/chess_engine_v1/checkpoint_best.pt --elo-test"
echo ""
echo "To play against it:"
echo "  python -m engine.play --model ./outputs/chess_engine_v1/checkpoint_best.pt"
