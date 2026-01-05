#!/bin/bash
# Quick scaling experiment launcher for Linux/Cloud
# Usage: ./run_scaling.sh [optional: stockfish_path] [extra args]
# Without arguments, auto-downloads Stockfish 16.1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "$1" ]; then
    echo "Auto-downloading Stockfish 16.1 and running scaling experiment..."
    python "$SCRIPT_DIR/run_scaling_experiment.py" --auto-stockfish
elif [ "$1" = "--auto-stockfish" ] || [ "$1" = "--cloud" ]; then
    python "$SCRIPT_DIR/run_scaling_experiment.py" "$@"
else
    python "$SCRIPT_DIR/run_scaling_experiment.py" --stockfish "$@"
fi
