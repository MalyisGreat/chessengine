# Quick Start (Speed Demon NNUE)

This is the minimal path to run training and evaluation without touching the legacy CNN code.

## RunPod (recommended)

```bash
git clone https://github.com/MalyisGreat/chessengine.git
cd chessengine
python speed_demon/runpod_train.py --stockfish-compat
```

## Reproduce the 2023-2024 100GB run

```bash
python speed_demon/runpod_train.py \
  --stockfish-compat \
  --data-urls-file speed_demon/data_urls/test80_2023_2024_minv2_v6.txt \
  --data-max-gb 100 \
  --positions 3300000000 \
  --positions-per-epoch 100000000 \
  --batch-size 32768 \
  --num-workers 16 \
  --threads 16
```

## Reproduce the strongest snapshot (epoch 16)

This stops at epoch 16 (about 1.6B positions) using the same 100GB 2023-2024 data mix:

```bash
python speed_demon/runpod_train.py \
  --stockfish-compat \
  --data-urls-file speed_demon/data_urls/test80_2023_2024_minv2_v6.txt \
  --data-max-gb 100 \
  --positions-per-epoch 100000000 \
  --max-epochs 16 \
  --batch-size 32768 \
  --num-workers 16 \
  --threads 16
```

## Local (Windows)

```powershell
python -m pip install -r requirements.txt
python speed_demon\runpod_train.py --stockfish-compat
```

## Outputs

- Checkpoints: `outputs/speed_demon/lightning_logs/version_*/checkpoints/*.ckpt`
- NNUE nets: `outputs/speed_demon/nnue/*.nnue`
- Eval logs: `outputs/speed_demon/eval/`

## Eval a net vs Stockfish

```bash
python speed_demon/eval_vs_stockfish.py \
  --nnue /path/to/net.nnue \
  --stockfish /path/to/stockfish \
  --games 10 --time-per-move 0.1 --threads 6
```
