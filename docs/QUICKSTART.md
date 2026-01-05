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

## Search-Time Scaling Analysis

Measure how your NNUE's Elo scales with search time (simulates consumer PC performance).

### Quick run (local)

```bash
# Windows
scripts\run_scaling.bat C:\stockfish\stockfish\stockfish-windows-x86-64-avx2.exe

# Linux
./scripts/run_scaling.sh /usr/bin/stockfish
```

### Cloud run (64 vCPU for faster results)

```bash
python scripts/run_scaling_experiment.py \
  --stockfish /path/to/stockfish \
  --cloud \
  --games 20 \
  --workers 8
```

### Full custom run

```bash
python speed_demon/scaling_analysis.py \
  --nnue models/nn-epoch16-manual.nnue \
  --stockfish /path/to/stockfish \
  --games 20 \
  --threads 4 \
  --workers 8 \
  --hash-mb 128 \
  --times "0.05,0.1,0.2,0.4,0.8,1.5,2.0" \
  --base-elos "2800,2900,3000,3100" \
  --out-dir outputs/speed_demon/eval/scaling_runs \
  --run-id my_experiment \
  --debug-dir outputs/speed_demon/eval/scaling_runs/my_experiment/debug
```

### Scaling experiment outputs

- `raw_results.jsonl` - Every game result (wins/draws/losses, scores, timestamps)
- `summary.json` - Full stats, Elo estimates, scaling fit, parameters
- `summary.txt` - Human-readable summary
- `debug/` - Detailed debug logs per game

### Parameters explained

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--games` | 20 | Games per test point (more = tighter error bars) |
| `--threads` | 4 | Threads per engine (simulates consumer PC) |
| `--workers` | 8 | Parallel games (increase on cloud for speed) |
| `--times` | 0.05-2.0s | Time controls to test |
| `--base-elos` | 2800-3100 | Opponent Elo levels |
