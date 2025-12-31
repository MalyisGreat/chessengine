#!/bin/bash
# Chess Engine Setup Script for RunPod/Cloud H100/A100 Instances
#
# Usage:
#   bash scripts/setup.sh
#
# This script:
# 1. Installs Python dependencies
# 2. Downloads Stockfish for benchmarking
# 3. Creates necessary directories
# 4. Verifies GPU availability

set -e  # Exit on error

echo "=========================================="
echo "Chess Engine Setup"
echo "=========================================="

# Check Python version
echo ""
echo "Checking Python version..."
python3 --version

# Check CUDA availability
echo ""
echo "Checking GPU availability..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]" || echo "PyTorch not installed yet"

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install Stockfish for benchmarking
echo ""
echo "Installing Stockfish..."
if command -v apt-get &> /dev/null; then
    # Debian/Ubuntu
    sudo apt-get update && sudo apt-get install -y stockfish p7zip-full
elif command -v yum &> /dev/null; then
    # CentOS/RHEL
    sudo yum install -y stockfish p7zip
elif command -v brew &> /dev/null; then
    # macOS
    brew install stockfish p7zip
else
    echo "Could not install Stockfish automatically. Please install manually."
fi

# Create directories
echo ""
echo "Creating directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/train
mkdir -p outputs
mkdir -p logs

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "
import torch
import chess
import numpy as np

print('Import check passed!')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Test model creation
echo ""
echo "Testing model creation..."
python3 -c "
from models.network import ChessNetwork
model = ChessNetwork(num_blocks=10, num_filters=256)
print(f'Model parameters: {model.count_parameters():,}')
print('Model test passed!')
"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Download training data:"
echo "     python download_data.py --dataset lichess_elite"
echo ""
echo "  2. Start training (single GPU):"
echo "     python train.py --data ./data/train"
echo ""
echo "  3. Start training (multi-GPU):"
echo "     torchrun --nproc_per_node=NUM_GPUS train.py --data ./data/train"
echo ""
echo "  4. Benchmark your model:"
echo "     python benchmark.py --model ./outputs/checkpoint_best.pt --all"
