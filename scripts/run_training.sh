#!/bin/bash
# Chess Engine Training Launch Script
#
# Usage:
#   bash scripts/run_training.sh                    # Single GPU
#   bash scripts/run_training.sh --gpus 5           # 5 GPUs
#   bash scripts/run_training.sh --gpus 5 --fast    # Fast training preset
#
# Options:
#   --gpus N      : Number of GPUs to use (default: 1)
#   --fast        : Use fast training preset (6 blocks, 2 epochs)
#   --accurate    : Use accurate training preset (20 blocks, 5 epochs)
#   --epochs N    : Override number of epochs
#   --batch-size N: Override batch size
#   --data PATH   : Path to training data (default: ./data/train)
#   --resume PATH : Resume from checkpoint

set -e

# Default values
NUM_GPUS=1
CONFIG="default"
DATA_PATH="./data/train"
EPOCHS=""
BATCH_SIZE=""
RESUME=""
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --fast)
            CONFIG="fast"
            shift
            ;;
        --accurate)
            CONFIG="accurate"
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --data)
            DATA_PATH="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Build command
CMD="python train.py --config $CONFIG --data $DATA_PATH"

if [[ -n "$EPOCHS" ]]; then
    CMD="$CMD --epochs $EPOCHS"
fi

if [[ -n "$BATCH_SIZE" ]]; then
    CMD="$CMD --batch-size $BATCH_SIZE"
fi

if [[ -n "$RESUME" ]]; then
    CMD="$CMD --resume $RESUME"
fi

CMD="$CMD $EXTRA_ARGS"

echo "=========================================="
echo "Chess Engine Training"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  GPUs: $NUM_GPUS"
echo "  Config: $CONFIG"
echo "  Data: $DATA_PATH"
echo ""

# Check GPU availability
echo "Checking GPUs..."
python3 -c "
import torch
available = torch.cuda.device_count()
print(f'Available GPUs: {available}')
if available < $NUM_GPUS:
    print(f'Warning: Requested {$NUM_GPUS} GPUs but only {available} available!')
"
echo ""

# Enable TF32 for Ampere+ GPUs
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# Launch training
if [[ $NUM_GPUS -gt 1 ]]; then
    echo "Launching distributed training with $NUM_GPUS GPUs..."
    echo "Command: torchrun --nproc_per_node=$NUM_GPUS $CMD"
    echo ""

    # Use torchrun for multi-GPU
    torchrun --nproc_per_node=$NUM_GPUS $CMD
else
    echo "Launching single GPU training..."
    echo "Command: $CMD"
    echo ""

    # Single GPU
    eval $CMD
fi

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Benchmark your model:"
echo "     python benchmark.py --model ./outputs/chess_engine_v1/checkpoint_best.pt --all"
echo ""
echo "  2. Play against your engine:"
echo "     python -m engine.search --model ./outputs/chess_engine_v1/checkpoint_best.pt"
