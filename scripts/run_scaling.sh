#!/bin/bash
# Quick scaling experiment launcher for Linux/Cloud
# Usage: ./run_scaling.sh /path/to/stockfish

if [ -z "$1" ]; then
    echo "Usage: ./run_scaling.sh <stockfish_path> [extra args]"
    echo "Example: ./run_scaling.sh /usr/bin/stockfish --cloud --games 50"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/run_scaling_experiment.py" --stockfish "$@"
