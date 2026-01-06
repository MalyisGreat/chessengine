# Chess Engine

Speed Demon NNUE training pipeline focused on Stockfish‑compatible networks
(HalfKAv2_hm, 2560/15/32, int8). This repo is intentionally scoped to the
current NNUE approach only.

## Quick Start

RunPod (recommended):

```bash
git clone https://github.com/MalyisGreat/chessengine.git
cd chessengine
python speed_demon/runpod_train.py --stockfish-compat
```

Local (Windows):

```powershell
python -m pip install -r requirements.txt
python speed_demon\runpod_train.py --stockfish-compat
```

For full, reproducible commands (100 GB 2023–2024 run and epoch‑16 stop), see:
`docs/QUICKSTART.md`.

## Outputs

- Checkpoints: `outputs/speed_demon/lightning_logs/version_*/checkpoints/*.ckpt`
- NNUE nets: `outputs/speed_demon/nnue/*.nnue`
- Eval logs: `outputs/speed_demon/eval/`

## Results and Graphs

Full results and plots live in `docs/RESULTS.md`.

![Search-Time Scaling (Epoch 16 NNUE)](docs/scaling_run_20260105_220739.svg)
![Loss Curve (100GB Run, Best Epoch 16)](docs/loss_curve_latest.svg)

## Helpers

- Aggregate eval logs: `python scripts/aggregate_eval.py --eval-dir outputs/speed_demon/eval --out outputs/speed_demon/eval/combined_eval.jsonl`
- Plot scaling results: `python scripts/plot_scaling.py --csv docs/scaling_run_20260105_220739.csv --out docs/scaling_run_20260105_220739.svg`
