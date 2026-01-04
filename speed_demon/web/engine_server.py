import argparse
import atexit
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Optional

import chess
import chess.engine


ENGINE: Optional[chess.engine.SimpleEngine] = None
ENGINE_LOCK = Lock()
ENGINE_INFO = {}


def _find_stockfish() -> Optional[str]:
    env_path = os.environ.get("STOCKFISH_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    candidates = [
        "/root/.stockfish/stockfish/stockfish/stockfish-ubuntu-x86-64-avx2",
        "/usr/games/stockfish",
        "/usr/bin/stockfish",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _find_latest_nnue(repo_root: Path) -> Optional[Path]:
    nnue_dir = repo_root / "outputs" / "speed_demon" / "nnue"
    candidates: list[Path] = []
    if nnue_dir.exists():
        candidates = list(nnue_dir.glob("*.nnue"))
    if not candidates:
        fallback = repo_root / "speed_demon.nnue"
        if fallback.exists():
            return fallback
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_engine(stockfish_path: str, nnue_path: str, threads: int, hash_mb: int) -> None:
    global ENGINE, ENGINE_INFO
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    options = {}
    if "Threads" in engine.options:
        options["Threads"] = threads
    if "Hash" in engine.options:
        options["Hash"] = hash_mb
    if "EvalFile" in engine.options:
        options["EvalFile"] = nnue_path
    if "Use NNUE" in engine.options:
        options["Use NNUE"] = True
    if options:
        engine.configure(options)

    ENGINE = engine
    ENGINE_INFO = {
        "stockfish": stockfish_path,
        "nnue": nnue_path,
        "threads": threads,
        "hash_mb": hash_mb,
    }


def _shutdown_engine() -> None:
    global ENGINE
    if ENGINE is not None:
        try:
            ENGINE.quit()
        except Exception:
            pass
        ENGINE = None


class Handler(BaseHTTPRequestHandler):
    server_version = "SpeedDemonEngine/1.0"

    def _send_json(self, data: dict, status: int = 200) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_file(self, path: Path) -> None:
        if not path.exists():
            self.send_error(404, "File not found")
            return
        data = path.read_bytes()
        self.send_response(200)
        if path.suffix == ".html":
            content_type = "text/html"
        elif path.suffix == ".js":
            content_type = "text/javascript"
        elif path.suffix == ".css":
            content_type = "text/css"
        else:
            content_type = "application/octet-stream"
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send_file(Path(__file__).with_name("index.html"))
            return
        if self.path == "/health":
            self._send_json({"ok": True, **ENGINE_INFO})
            return
        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        if self.path != "/move":
            self.send_error(404, "Not found")
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(length).decode("utf-8")
        try:
            data = json.loads(payload)
            fen = data.get("fen")
            time_limit = float(data.get("time", 0.05))
        except Exception:
            self._send_json({"error": "Invalid JSON"}, status=400)
            return

        if not fen:
            self._send_json({"error": "Missing fen"}, status=400)
            return

        if ENGINE is None:
            self._send_json({"error": "Engine not ready"}, status=500)
            return

        board = chess.Board(fen)
        with ENGINE_LOCK:
            result = ENGINE.play(board, chess.engine.Limit(time=time_limit))
        if result.move is None:
            self._send_json({"error": "No move returned"}, status=500)
            return

        move = result.move.uci()
        self._send_json({"move": move})


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a local NNUE engine over HTTP.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--stockfish", type=str, default=None)
    parser.add_argument("--nnue", type=str, default=None)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--hash-mb", type=int, default=128)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    stockfish_path = os.path.abspath(args.stockfish) if args.stockfish else _find_stockfish()
    if not stockfish_path:
        print("Stockfish not found. Pass --stockfish /path/to/stockfish.")
        sys.exit(2)

    nnue_path = None
    if args.nnue:
        nnue_path = os.path.abspath(args.nnue)
    else:
        latest_nnue = _find_latest_nnue(repo_root)
        if latest_nnue is not None:
            nnue_path = latest_nnue.as_posix()
    if not nnue_path or not os.path.exists(nnue_path):
        print("NNUE not found. Pass --nnue /path/to/nnue.")
        sys.exit(2)

    print(f"Using Stockfish: {stockfish_path}")
    print(f"Using NNUE:      {nnue_path}")

    _load_engine(stockfish_path, nnue_path, args.threads, args.hash_mb)
    atexit.register(_shutdown_engine)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving on http://{args.host}:{args.port}")
    print(f"Local URL:    http://127.0.0.1:{args.port}")
    print(f"Localhost:    http://localhost:{args.port}")
    if args.host == "0.0.0.0":
        print("Bind address: 0.0.0.0 (use a real host/IP in your browser)")
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if pod_id:
        print(f"RunPod proxy: https://{pod_id}-{args.port}.proxy.runpod.net")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
