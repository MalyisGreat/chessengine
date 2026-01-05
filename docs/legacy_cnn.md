# Legacy CNN Pipeline (Deprecated)

This repository originally included a CNN-based policy/value model trained on Lichess position evaluations.
The Speed Demon NNUE pipeline is now the preferred path.

## Quick Start

```bash
git clone https://github.com/MalyisGreat/chessengine.git
cd chessengine
pip install -r requirements.txt datasets
python download_data.py
python train.py --data ./data/train
```

## Larger Run

```bash
python download_data.py --positions 50000000
python train.py --data ./data/train
```

## Multi-GPU

```bash
torchrun --nproc_per_node=5 train.py --data ./data/train
```

## Data Source

Uses Lichess position evaluations from Hugging Face:
https://huggingface.co/datasets/Lichess/chess-position-evaluations

## Notes

- This pipeline is kept for reference and compatibility.
- New work should use the NNUE pipeline in `speed_demon/`.
