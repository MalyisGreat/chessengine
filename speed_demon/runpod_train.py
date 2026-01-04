import argparse
import csv
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
NNUE_REPO_URL = "https://github.com/official-stockfish/nnue-pytorch.git"
DEFAULT_DATA_URL = (
    "https://huggingface.co/datasets/linrock/test80-2024/resolve/main/"
    "test80-2024-01-jan-2tb7p.min-v2.v6.binpack.zst"
)
DEFAULT_DATA_URLS_FILE = (
    ROOT / "speed_demon" / "data_urls" / "test80_diverse_best.txt"
)
STOCKFISH_NET_URL = "https://tests.stockfishchess.org/api/nn/{name}"
STOCKFISH_COMPAT = {"features": "HalfKAv2_hm", "l1": 2560, "l2": 15, "l3": 32}


def run(cmd, cwd=None, env=None, check=True):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result.returncode


def ensure_system_packages(skip: bool) -> None:
    if skip:
        return
    if platform.system().lower() != "linux":
        print("System package install skipped (non-Linux).")
        return
    if shutil.which("apt-get") is None:
        print("apt-get not found; skipping system package install.")
        return

    env = os.environ.copy()
    env["DEBIAN_FRONTEND"] = "noninteractive"
    run(["apt-get", "update"], env=env)
    run(
        [
            "apt-get",
            "install",
            "-y",
            "git",
            "cmake",
            "build-essential",
            "zstd",
            "curl",
            "unzip",
        ],
        env=env,
    )


def ensure_python_packages(nnue_repo: Path, skip: bool) -> None:
    if skip:
        return
    req_path = nnue_repo / "requirements.txt"
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", "-r", str(req_path)])

    try:
        import torch  # noqa: F401

        torch_ok = True
    except Exception:
        torch_ok = False

    if not torch_ok:
        run([sys.executable, "-m", "pip", "install", "torch"])

    try:
        import cupy  # noqa: F401

        cupy_ok = True
    except Exception:
        cupy_ok = False

    if not cupy_ok:
        cuda_version = None
        try:
            import torch as _torch

            cuda_version = _torch.version.cuda
        except Exception:
            cuda_version = None

        if cuda_version and cuda_version.startswith("11"):
            cupy_pkg = "cupy-cuda11x"
        else:
            cupy_pkg = "cupy-cuda12x"
        run([sys.executable, "-m", "pip", "install", cupy_pkg])

    run([sys.executable, "-m", "pip", "install", "zstandard"])


def ensure_nnue_repo(repo_path: Path) -> None:
    if repo_path.exists():
        return
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", NNUE_REPO_URL, str(repo_path)])


def patch_nnue(repo_path: Path) -> None:
    patch_script = ROOT / "speed_demon" / "patch_nnue_pytorch.py"
    run([sys.executable, str(patch_script), "--repo", str(repo_path)])


def ensure_data_loader(repo_path: Path, skip: bool) -> None:
    if skip:
        return
    for ext in (".so", ".dylib", ".dll"):
        if (repo_path / f"training_data_loader{ext}").exists():
            return

    build_dir = repo_path / "build"
    run(
        [
            "cmake",
            "-S",
            ".",
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_INSTALL_PREFIX=.",
        ],
        cwd=repo_path,
    )
    run(
        ["cmake", "--build", str(build_dir), "--config", "Release", "--target", "install"],
        cwd=repo_path,
    )


def ensure_stockfish(explicit_path: Optional[str]) -> str:
    if explicit_path:
        return explicit_path

    env_path = os.environ.get("STOCKFISH_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    for path in ("/usr/games/stockfish", "/usr/bin/stockfish"):
        if os.path.exists(path):
            return path

    system = platform.system().lower()
    machine = platform.machine().lower()
    if system != "linux" or ("x86_64" not in machine and "amd64" not in machine):
        raise RuntimeError("Stockfish auto-install only supported on Linux x86_64.")

    stockfish_dir = Path.home() / ".stockfish"
    stockfish_dir.mkdir(parents=True, exist_ok=True)
    url = (
        "https://github.com/official-stockfish/Stockfish/releases/download/"
        "sf_16.1/stockfish-ubuntu-x86-64-avx2.tar"
    )
    archive_path = stockfish_dir / "stockfish-ubuntu-x86-64-avx2.tar"

    if not archive_path.exists():
        print(f"Downloading Stockfish: {url}")
        urlretrieve(url, archive_path)

    extract_dir = stockfish_dir / "stockfish"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r") as tar:
        tar.extractall(extract_dir)

    for root, _, files in os.walk(extract_dir):
        for name in files:
            if "stockfish" in name and not name.endswith((".tar", ".zip")):
                path = os.path.join(root, name)
                os.chmod(path, 0o755)
                os.environ["STOCKFISH_PATH"] = path
                return path

    raise RuntimeError("Stockfish binary not found after extraction.")


def _parse_stockfish_default_nets(stockfish_path: str) -> list[str]:
    result = subprocess.run(
        [stockfish_path],
        input="uci\nquit\n",
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return []

    nets: list[str] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("option name EvalFile") or line.startswith(
            "option name EvalFileSmall"
        ):
            parts = line.split("default", 1)
            if len(parts) != 2:
                continue
            name = parts[1].strip()
            if name and name != "<empty>":
                nets.append(name)
    return nets


def ensure_stockfish_nets(stockfish_path: str) -> None:
    root = Path(stockfish_path).resolve().parent
    candidates = list(root.glob("nn-*.nnue")) + list(root.glob("*.nnue"))
    if candidates:
        return

    nets = _parse_stockfish_default_nets(stockfish_path)
    for name in nets:
        target = root / name
        if target.exists():
            continue
        url = STOCKFISH_NET_URL.format(name=name)
        print(f"Downloading Stockfish net: {url}")
        urlretrieve(url, target)


def find_baseline_nnue(stockfish_path: str) -> Optional[str]:
    root = Path(stockfish_path).resolve().parent
    candidates = list(root.glob("nn-*.nnue")) + list(root.glob("*.nnue"))
    if not candidates:
        parent = root.parent
        candidates = list(parent.glob("nn-*.nnue")) + list(parent.glob("*.nnue"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def download_dataset(
    output_path: Path,
    urls: Optional[list[str]],
    urls_file: Optional[str],
    skip: bool,
    max_gb: Optional[float],
    resume: bool,
    retries: int,
    retry_backoff: float,
    connect_timeout: float,
    read_timeout: float,
) -> None:
    if skip:
        return
    download_script = ROOT / "speed_demon" / "download_binpack.py"
    cmd = [
        sys.executable,
        str(download_script),
        "--output",
        str(output_path),
    ]
    if urls:
        for url in urls:
            cmd += ["--url", url]
    if urls_file:
        cmd += ["--urls-file", urls_file]
    if not urls and not urls_file:
        cmd += ["--url", DEFAULT_DATA_URL]
    if max_gb is not None:
        cmd += ["--max-gb", str(max_gb)]
    if resume:
        cmd += ["--resume"]
    cmd += [
        "--retries",
        str(retries),
        "--retry-backoff",
        str(retry_backoff),
        "--connect-timeout",
        str(connect_timeout),
        "--read-timeout",
        str(read_timeout),
    ]
    run(cmd)


def _parse_epoch(path: Path) -> Optional[int]:
    match = re.search(r"epoch=([0-9]+)", path.name)
    if not match:
        return None
    return int(match.group(1))


def _tail_log(path: Path, max_bytes: int = 20000) -> str:
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read().decode(errors="replace")
        return "\n".join(data.splitlines()[-50:])
    except Exception:
        return ""


def _run_serialization(
    repo_path: Path,
    output_dir: Path,
    ckpt: Path,
    nnue_path: Path,
    features: str,
    l1: int,
    l2: int,
    l3: int,
    eval_ft_compression: str,
    verbose: bool,
) -> bool:
    cmd = [
        sys.executable,
        "serialize.py",
        str(ckpt),
        str(nnue_path),
        f"--features={features}",
        "--l1",
        str(l1),
        "--l2",
        str(l2),
        "--l3",
        str(l3),
        "--ft_compression",
        eval_ft_compression,
    ]
    if verbose:
        result = subprocess.run(cmd, cwd=repo_path)
    else:
        log_dir = output_dir / "serialize_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{nnue_path.stem}.log"
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(cmd, cwd=repo_path, stdout=log, stderr=log)
    if result.returncode != 0:
        print(f"Serialize failed for {ckpt}.")
        if not verbose:
            tail = _tail_log(log_path)
            if tail:
                print(tail)
        return False
    return True


def eval_watcher(
    repo_path: Path,
    output_dir: Path,
    features: str,
    l1: int,
    l2: int,
    l3: int,
    stockfish_path: str,
    base_nnue: Optional[str],
    base_classical: bool,
    base_elo: Optional[int],
    base_skill: Optional[int],
    eval_games: int,
    eval_time_per_move: float,
    eval_max_moves: int,
    eval_every_epochs: int,
    eval_ft_compression: str,
    serialize_verbose: bool,
    eval_ladder: bool,
    ladder_start_elo: int,
    ladder_step: int,
    ladder_games: int,
    ladder_win_rate: float,
    ladder_max_elo: int,
    min_ckpt_mtime: float,
    stop_event: threading.Event,
) -> None:
    seen = set()
    nnue_dir = output_dir / "nnue"
    eval_csv = output_dir / "eval" / "eval.csv"
    eval_json = output_dir / "eval" / "eval.jsonl"
    eval_json = output_dir / "eval" / "eval.jsonl"
    eval_script = ROOT / "speed_demon" / "eval_vs_stockfish.py"

    while not stop_event.is_set():
        ckpts = sorted(output_dir.rglob("epoch=*.ckpt"))
        for ckpt in ckpts:
            if ckpt in seen:
                continue
            try:
                if ckpt.stat().st_mtime < min_ckpt_mtime:
                    seen.add(ckpt)
                    continue
            except FileNotFoundError:
                continue
            epoch = _parse_epoch(ckpt)
            seen.add(ckpt)
            if epoch is None or (epoch + 1) % eval_every_epochs != 0:
                continue

            nnue_dir.mkdir(parents=True, exist_ok=True)
            nnue_path = nnue_dir / f"nn-epoch{epoch + 1}.nnue"
            if not _run_serialization(
                repo_path=repo_path,
                output_dir=output_dir,
                ckpt=ckpt,
                nnue_path=nnue_path,
                features=features,
                l1=l1,
                l2=l2,
                l3=l3,
                eval_ft_compression=eval_ft_compression,
                verbose=serialize_verbose,
            ):
                continue

            if eval_ladder:
                current_elo = max(1400, ladder_start_elo)
                while current_elo <= ladder_max_elo:
                    print(f"Ladder eval vs base Elo {current_elo}...")
                    metrics = _run_eval_once(
                        eval_script=eval_script,
                        nnue_path=nnue_path,
                        stockfish_path=stockfish_path,
                        eval_csv=eval_csv,
                        eval_json=eval_json,
                        epoch_label=epoch + 1,
                        eval_games=ladder_games,
                        eval_time_per_move=eval_time_per_move,
                        eval_max_moves=eval_max_moves,
                        base_nnue=base_nnue,
                        base_classical=base_classical,
                        base_elo=current_elo,
                        base_skill=base_skill,
                        debug_dir=output_dir / "eval" / "debug",
                    )
                    if metrics is None:
                        break
                    win_rate = _parse_win_rate(metrics)
                    if win_rate is None or win_rate < ladder_win_rate:
                        break
                    current_elo += ladder_step
            else:
                _run_eval_once(
                    eval_script=eval_script,
                    nnue_path=nnue_path,
                    stockfish_path=stockfish_path,
                    eval_csv=eval_csv,
                    eval_json=eval_json,
                    epoch_label=epoch + 1,
                    eval_games=eval_games,
                    eval_time_per_move=eval_time_per_move,
                    eval_max_moves=eval_max_moves,
                    base_nnue=base_nnue,
                    base_classical=base_classical,
                    base_elo=base_elo,
                    base_skill=base_skill,
                    debug_dir=output_dir / "eval" / "debug",
                )

        if (output_dir / "training_finished").exists():
            return

        time.sleep(15)


def _latest_checkpoint(output_dir: Path) -> Optional[Path]:
    ckpts = list(output_dir.rglob("epoch=*.ckpt"))
    if not ckpts:
        return None
    ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0]


def _read_logged_epochs(csv_path: Path) -> set[int]:
    if not csv_path.exists():
        return set()
    epochs = set()
    try:
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "epoch" in row and row["epoch"]:
                    epochs.add(int(row["epoch"]))
    except Exception:
        return set()
    return epochs


def _read_last_metrics(csv_path: Path) -> Optional[dict]:
    if not csv_path.exists():
        return None
    last_row = None
    try:
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                last_row = row
    except Exception:
        return None
    return last_row


def _latest_metrics_csv(output_dir: Path) -> Optional[Path]:
    csvs = list(output_dir.rglob("metrics.csv"))
    if not csvs:
        return None
    csvs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return csvs[0]


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _write_loss_history(output_dir: Path) -> None:
    metrics_csv = _latest_metrics_csv(output_dir)
    if not metrics_csv:
        print("No metrics.csv found for loss history.")
        return

    by_epoch: dict[int, dict] = {}
    try:
        with metrics_csv.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "epoch" not in row or not row["epoch"]:
                    continue
                try:
                    epoch = int(float(row["epoch"]))
                except ValueError:
                    continue
                entry = by_epoch.get(epoch, {"epoch": epoch})
                step = _parse_float(row.get("step"))
                if step is not None:
                    entry["step"] = int(step)
                train_loss = _parse_float(
                    row.get("train_loss_epoch") or row.get("train_loss")
                )
                if train_loss is not None:
                    entry["train_loss"] = train_loss
                val_loss = _parse_float(row.get("val_loss_epoch") or row.get("val_loss"))
                if val_loss is not None:
                    entry["val_loss"] = val_loss
                by_epoch[epoch] = entry
    except Exception:
        print(f"Failed to read metrics from {metrics_csv}")
        return

    if not by_epoch:
        print("No epoch loss metrics found in metrics.csv.")
        return

    history = [by_epoch[key] for key in sorted(by_epoch)]
    loss_dir = output_dir / "loss"
    loss_dir.mkdir(parents=True, exist_ok=True)
    output_path = loss_dir / "loss_history.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Loss history saved to {output_path}")


def _parse_win_rate(metrics: dict) -> Optional[float]:
    try:
        if "win_rate" in metrics and metrics["win_rate"] not in (None, ""):
            return float(metrics["win_rate"])
        wins = int(metrics.get("wins", 0))
        games = int(metrics.get("games", 0))
        if games > 0:
            return wins / games
    except Exception:
        return None
    return None


def _run_eval_once(
    eval_script: Path,
    nnue_path: Path,
    stockfish_path: str,
    eval_csv: Path,
    eval_json: Path,
    epoch_label: int,
    eval_games: int,
    eval_time_per_move: float,
    eval_max_moves: int,
    base_nnue: Optional[str],
    base_classical: bool,
    base_elo: Optional[int],
    base_skill: Optional[int],
    debug_dir: Path,
) -> Optional[dict]:
    cmd = [
        sys.executable,
        str(eval_script),
        "--nnue",
        str(nnue_path),
        "--stockfish",
        stockfish_path,
        "--games",
        str(eval_games),
        "--time-per-move",
        str(eval_time_per_move),
        "--max-moves",
        str(eval_max_moves),
        "--csv",
        str(eval_csv),
        "--json",
        str(eval_json),
        "--epoch",
        str(epoch_label),
        "--debug-dir",
        str(debug_dir),
    ]
    if base_nnue:
        cmd += ["--stockfish-base-nnue", base_nnue]
    if base_classical:
        cmd += ["--stockfish-classical-base"]
    if base_elo is not None:
        cmd += ["--stockfish-base-elo", str(base_elo)]
    if base_skill is not None:
        cmd += ["--stockfish-base-skill", str(base_skill)]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Eval command failed (code {result.returncode}).")
        return None
    return _read_last_metrics(eval_csv)


def _run_final_eval(
    repo_path: Path,
    output_dir: Path,
    features: str,
    l1: int,
    l2: int,
    l3: int,
    stockfish_path: str,
    base_nnue: Optional[str],
    base_classical: bool,
    base_elo: Optional[int],
    base_skill: Optional[int],
    eval_games: int,
    eval_time_per_move: float,
    eval_max_moves: int,
    eval_ft_compression: str,
    serialize_verbose: bool,
    eval_ladder: bool,
    ladder_start_elo: int,
    ladder_step: int,
    ladder_games: int,
    ladder_win_rate: float,
    ladder_max_elo: int,
) -> None:
    ckpt = _latest_checkpoint(output_dir)
    if not ckpt:
        print("No checkpoints found for final eval.")
        return

    epoch = _parse_epoch(ckpt)
    if epoch is None:
        print(f"Could not parse epoch from {ckpt}. Skipping final eval.")
        return

    eval_csv = output_dir / "eval" / "eval.csv"
    logged_epochs = _read_logged_epochs(eval_csv)
    epoch_label = epoch + 1
    if epoch_label in logged_epochs:
        return

    nnue_dir = output_dir / "nnue"
    nnue_dir.mkdir(parents=True, exist_ok=True)
    nnue_path = nnue_dir / f"nn-epoch{epoch_label}.nnue"
    if not nnue_path.exists():
        if not _run_serialization(
            repo_path=repo_path,
            output_dir=output_dir,
            ckpt=ckpt,
            nnue_path=nnue_path,
            features=features,
            l1=l1,
            l2=l2,
            l3=l3,
            eval_ft_compression=eval_ft_compression,
            verbose=serialize_verbose,
        ):
            return

    eval_script = ROOT / "speed_demon" / "eval_vs_stockfish.py"
    if eval_ladder:
        current_elo = max(1400, ladder_start_elo)
        while current_elo <= ladder_max_elo:
            print(f"Ladder eval vs base Elo {current_elo}...")
            metrics = _run_eval_once(
                eval_script=eval_script,
                nnue_path=nnue_path,
                stockfish_path=stockfish_path,
                eval_csv=eval_csv,
                eval_json=eval_json,
                epoch_label=epoch_label,
                eval_games=ladder_games,
                eval_time_per_move=eval_time_per_move,
                eval_max_moves=eval_max_moves,
                base_nnue=base_nnue,
                base_classical=base_classical,
                base_elo=current_elo,
                base_skill=base_skill,
                debug_dir=output_dir / "eval" / "debug",
            )
            if metrics is None:
                break
            win_rate = _parse_win_rate(metrics)
            if win_rate is None or win_rate < ladder_win_rate:
                break
            current_elo += ladder_step
    else:
        result = subprocess.run(
            [
                sys.executable,
                str(eval_script),
                "--nnue",
                str(nnue_path),
                "--stockfish",
                stockfish_path,
                "--games",
                str(eval_games),
                "--time-per-move",
                str(eval_time_per_move),
                "--max-moves",
                str(eval_max_moves),
                "--csv",
                str(eval_csv),
                "--json",
                str(eval_json),
                "--epoch",
                str(epoch_label),
                "--debug-dir",
                str(output_dir / "eval" / "debug"),
            ]
            + (["--stockfish-base-nnue", base_nnue] if base_nnue else [])
            + (["--stockfish-classical-base"] if base_classical else [])
            + (["--stockfish-base-elo", str(base_elo)] if base_elo is not None else [])
            + (["--stockfish-base-skill", str(base_skill)] if base_skill is not None else []),
        )
        if result.returncode != 0:
            print(f"Final eval failed (code {result.returncode}).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Speed Demon NNUE training")
    parser.add_argument("--skip-system", action="store_true", help="Skip apt installs")
    parser.add_argument("--skip-install", action="store_true", help="Skip pip installs")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download")
    parser.add_argument("--skip-compile", action="store_true", help="Skip data loader build")
    parser.add_argument("--skip-eval", action="store_true", help="Skip eval watcher")
    parser.add_argument(
        "--data-url",
        action="append",
        default=None,
        help="Dataset URL (repeatable or comma-separated).",
    )
    parser.add_argument(
        "--data-urls-file",
        type=str,
        default=None,
        help="Path to a newline-delimited list of dataset URLs.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(ROOT / "data" / "binpack" / "training_data.binpack"),
    )
    parser.add_argument(
        "--data-max-gb",
        type=float,
        default=None,
        help="Download only the first N GB of the binpack dataset (chunk-aligned).",
    )
    parser.add_argument(
        "--download-resume",
        action="store_true",
        help="Resume multi-URL dataset downloads using a .resume.json state file.",
    )
    parser.add_argument(
        "--download-retries",
        type=int,
        default=5,
        help="Retries per URL during dataset download.",
    )
    parser.add_argument(
        "--download-retry-backoff",
        type=float,
        default=5.0,
        help="Base seconds for download retry backoff.",
    )
    parser.add_argument(
        "--download-connect-timeout",
        type=float,
        default=30.0,
        help="Download connect timeout in seconds.",
    )
    parser.add_argument(
        "--download-read-timeout",
        type=float,
        default=300.0,
        help="Download read timeout in seconds.",
    )
    parser.add_argument("--positions", type=int, default=20_000_000)
    parser.add_argument("--positions-per-epoch", type=int, default=5_000_000)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--features", type=str, default="HalfKP")
    parser.add_argument("--l1", type=int, default=256)
    parser.add_argument("--l2", type=int, default=32)
    parser.add_argument("--l3", type=int, default=32)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=1.0)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Initial learning rate override for nnue-pytorch.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Learning rate decay per epoch override for nnue-pytorch.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument(
        "--matmul-precision",
        type=str,
        choices=["high", "medium"],
        default=None,
        help="Set TORCH_MATMUL_PRECISION for training (enables Tensor Cores).",
    )
    parser.add_argument(
        "--enable-tf32",
        action="store_true",
        help="Enable TF32 matmul in torch for training.",
    )
    parser.add_argument("--validation-size", type=int, default=100_000)
    parser.add_argument("--eval-games", type=int, default=8)
    parser.add_argument("--eval-time-per-move", type=float, default=0.05)
    parser.add_argument("--eval-max-moves", type=int, default=200)
    parser.add_argument("--eval-every-epochs", type=int, default=1)
    parser.add_argument(
        "--eval-ladder",
        action="store_true",
        default=True,
        help="Run a Stockfish Elo ladder per epoch (default on).",
    )
    parser.add_argument(
        "--no-eval-ladder",
        action="store_false",
        dest="eval_ladder",
        help="Disable the ladder and run a single eval per epoch.",
    )
    parser.add_argument("--eval-ladder-start-elo", type=int, default=1400)
    parser.add_argument("--eval-ladder-step", type=int, default=200)
    parser.add_argument("--eval-ladder-games", type=int, default=10)
    parser.add_argument("--eval-ladder-win-rate", type=float, default=0.9)
    parser.add_argument("--eval-ladder-max-elo", type=int, default=3190)
    parser.add_argument(
        "--eval-ft-compression",
        type=str,
        default="none",
        choices=["none", "leb128"],
        help="Feature transformer compression for eval nets",
    )
    parser.add_argument("--stockfish-path", type=str, default=None)
    parser.add_argument("--stockfish-base-nnue", type=str, default=None)
    parser.add_argument(
        "--stockfish-base-elo",
        type=int,
        default=None,
        help="Approximate Elo for the base Stockfish engine (uses UCI_LimitStrength).",
    )
    parser.add_argument(
        "--stockfish-base-skill",
        type=int,
        default=None,
        help="Skill Level for the base Stockfish engine (0-20).",
    )
    parser.add_argument(
        "--stockfish-classical-base",
        action="store_true",
        help="Disable NNUE for the base Stockfish engine during eval.",
    )
    parser.add_argument(
        "--stockfish-compat",
        action="store_true",
        help="Use Stockfish 16.1 compatible NNUE settings (HalfKAv2_hm, 2560/15/32).",
    )
    parser.add_argument(
        "--serialize-verbose",
        action="store_true",
        help="Print serializer output (weight histograms).",
    )
    parser.add_argument("--repo-path", type=str, default=str(ROOT / "third_party" / "nnue-pytorch"))
    args = parser.parse_args()

    if args.data_url is None and args.data_urls_file is None:
        if DEFAULT_DATA_URLS_FILE.exists():
            args.data_urls_file = str(DEFAULT_DATA_URLS_FILE)

    if args.stockfish_compat:
        args.features = STOCKFISH_COMPAT["features"]
        args.l1 = STOCKFISH_COMPAT["l1"]
        args.l2 = STOCKFISH_COMPAT["l2"]
        args.l3 = STOCKFISH_COMPAT["l3"]
        if args.eval_ft_compression == "none":
            args.eval_ft_compression = "leb128"

    positions_per_epoch = min(args.positions, args.positions_per_epoch)
    epochs = args.epochs
    if epochs is None:
        epochs = max(1, (args.positions + positions_per_epoch - 1) // positions_per_epoch)

    print("Speed Demon NNUE config:")
    print(f"  Positions:         {args.positions}")
    print(f"  Positions/epoch:   {positions_per_epoch}")
    print(f"  Epochs:            {epochs}")
    print(f"  Batch size:        {args.batch_size}")
    print(f"  Features:          {args.features}")
    print(f"  Layers:            {args.l1}/{args.l2}/{args.l3}")
    print(f"  Lambda:            {args.lambda_}")
    if not args.skip_eval:
        if args.eval_ladder:
            print(
                "  Eval ladder:      start {} step {} games {} win_rate {} max {}".format(
                    args.eval_ladder_start_elo,
                    args.eval_ladder_step,
                    args.eval_ladder_games,
                    args.eval_ladder_win_rate,
                    args.eval_ladder_max_elo,
                )
            )
        else:
            print(f"  Eval games:        {args.eval_games}")
    if not args.skip_eval and not (
        args.features == STOCKFISH_COMPAT["features"]
        and args.l1 == STOCKFISH_COMPAT["l1"]
        and args.l2 == STOCKFISH_COMPAT["l2"]
        and args.l3 == STOCKFISH_COMPAT["l3"]
    ):
        print(
            "Warning: Stockfish 16.1 expects HalfKAv2_hm with 2560/15/32. "
            "Eval may fail unless you use --stockfish-compat or --skip-eval."
        )

    if args.eval_ladder and args.eval_ladder_start_elo < 1400:
        print("Eval ladder start Elo < 1400; forcing to 1400.")
        args.eval_ladder_start_elo = 1400
    if args.eval_ladder and args.eval_ladder_max_elo < args.eval_ladder_start_elo:
        args.eval_ladder_max_elo = args.eval_ladder_start_elo

    ensure_system_packages(args.skip_system)

    repo_path = Path(args.repo_path)
    ensure_nnue_repo(repo_path)
    patch_nnue(repo_path)
    ensure_python_packages(repo_path, args.skip_install)

    data_path = Path(args.data_path)
    download_dataset(
        data_path,
        args.data_url,
        args.data_urls_file,
        args.skip_download,
        args.data_max_gb,
        args.download_resume,
        args.download_retries,
        args.download_retry_backoff,
        args.download_connect_timeout,
        args.download_read_timeout,
    )
    ensure_data_loader(repo_path, args.skip_compile)

    stockfish_path = ensure_stockfish(args.stockfish_path)
    ensure_stockfish_nets(stockfish_path)
    base_nnue = args.stockfish_base_nnue
    print(f"Stockfish: {stockfish_path}")
    if base_nnue:
        print(f"Baseline NNUE: {base_nnue}")
    elif args.stockfish_classical_base:
        print("Baseline NNUE disabled, base engine will use classical eval.")
    else:
        print("Baseline NNUE: using Stockfish defaults.")

    output_dir = ROOT / "outputs" / "speed_demon"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        sys.executable,
        "train.py",
        str(data_path),
        "--default_root_dir",
        str(output_dir),
        "--max_epochs",
        str(epochs),
        "--epoch-size",
        str(positions_per_epoch),
        "--batch-size",
        str(args.batch_size),
        "--lambda",
        str(args.lambda_),
        "--features",
        args.features,
        "--l1",
        str(args.l1),
        "--l2",
        str(args.l2),
        "--l3",
        str(args.l3),
        "--num-workers",
        str(args.num_workers),
        "--threads",
        str(args.threads),
        "--network-save-period",
        "1",
        "--save-last-network",
        "True",
        "--validation-size",
        str(args.validation_size),
    ]
    if args.lr is not None:
        train_cmd += ["--lr", str(args.lr)]
    if args.gamma is not None:
        train_cmd += ["--gamma", str(args.gamma)]
    if args.gpus:
        train_cmd += ["--gpus", args.gpus]

    stop_event = threading.Event()
    watcher_thread = None
    if not args.skip_eval:
        min_ckpt_mtime = time.time() - 1.0
        watcher_thread = threading.Thread(
            target=eval_watcher,
            args=(
                repo_path,
                output_dir,
                args.features,
                args.l1,
                args.l2,
                args.l3,
                stockfish_path,
                base_nnue,
                args.stockfish_classical_base,
                args.stockfish_base_elo,
                args.stockfish_base_skill,
                args.eval_games,
                args.eval_time_per_move,
                args.eval_max_moves,
                args.eval_every_epochs,
                args.eval_ft_compression,
                args.serialize_verbose,
                args.eval_ladder,
                args.eval_ladder_start_elo,
                args.eval_ladder_step,
                args.eval_ladder_games,
                args.eval_ladder_win_rate,
                args.eval_ladder_max_elo,
                min_ckpt_mtime,
                stop_event,
            ),
            daemon=True,
        )
        watcher_thread.start()

    train_env = None
    if args.matmul_precision or args.enable_tf32:
        train_env = os.environ.copy()
        if args.matmul_precision:
            train_env["TORCH_MATMUL_PRECISION"] = args.matmul_precision
        if args.enable_tf32:
            train_env["TORCH_TF32"] = "1"

    run(train_cmd, cwd=repo_path, env=train_env)
    stop_event.set()
    if watcher_thread:
        watcher_thread.join()

    if not args.skip_eval:
        _run_final_eval(
            repo_path=repo_path,
            output_dir=output_dir,
            features=args.features,
            l1=args.l1,
            l2=args.l2,
            l3=args.l3,
            stockfish_path=stockfish_path,
            base_nnue=base_nnue,
            base_classical=args.stockfish_classical_base,
            base_elo=args.stockfish_base_elo,
            base_skill=args.stockfish_base_skill,
            eval_games=args.eval_games,
            eval_time_per_move=args.eval_time_per_move,
            eval_max_moves=args.eval_max_moves,
            eval_ft_compression=args.eval_ft_compression,
            serialize_verbose=args.serialize_verbose,
            eval_ladder=args.eval_ladder,
            ladder_start_elo=args.eval_ladder_start_elo,
            ladder_step=args.eval_ladder_step,
            ladder_games=args.eval_ladder_games,
            ladder_win_rate=args.eval_ladder_win_rate,
            ladder_max_elo=args.eval_ladder_max_elo,
        )

    _write_loss_history(output_dir)

    print("Training complete.")


if __name__ == "__main__":
    main()
