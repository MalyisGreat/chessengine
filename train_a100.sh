#!/bin/bash
# Training script for 1x A100 GPU
# Expected training time: ~2-3 hours for 50M positions

set -e

echo "============================================"
echo "Chess Engine Training - 1x A100"
echo "============================================"

# Step 1: Install dependencies
echo ""
echo "[1/3] Installing dependencies..."
pip install -q torch numpy python-chess datasets tqdm tensorboard stockfish

# Step 2: Download Lichess evaluated positions (has real Stockfish evals!)
echo ""
echo "[2/3] Downloading training data..."
echo "This dataset has REAL Stockfish centipawn evaluations"
echo "which will train the value head properly."
echo ""

# Download 50M positions (takes ~30-60 min due to processing)
python download_lichess_eval.py \
    --output ./data/lichess_eval \
    --positions 50000000 \
    --batch-size 100000

# Verify dataset
echo ""
echo "Verifying dataset..."
python download_lichess_eval.py --output ./data/lichess_eval --verify

# Step 3: Train
echo ""
echo "[3/3] Starting training..."
echo "Training with 1x A100 optimized settings"
echo ""

python train.py \
    --data ./data/lichess_eval \
    --batch-size 8192 \
    --epochs 3 \
    --lr 0.1

echo ""
echo "============================================"
echo "Training complete!"
echo "============================================"
echo ""
echo "To benchmark your model:"
echo "  python benchmark.py --model ./outputs/chess_engine_v1/checkpoint_best.pt --quick"
echo ""
echo "To play against it:"
echo "  python -m engine.play --model ./outputs/chess_engine_v1/checkpoint_best.pt"
