"""
Stockfish helper utilities.
"""

from __future__ import annotations

import os
import shutil
from typing import Optional

import chess.engine


def find_stockfish_binary(explicit_path: Optional[str] = None) -> Optional[str]:
    """Find a Stockfish binary path from explicit path, env, or common locations."""
    candidates = []

    if explicit_path:
        candidates.append(explicit_path)

    env_path = os.environ.get("STOCKFISH_PATH")
    if env_path:
        candidates.append(env_path)

    candidates.extend([
        "stockfish",
        "/usr/local/bin/stockfish",
        "/usr/bin/stockfish",
        "/usr/games/stockfish",
        os.path.expanduser("~/.stockfish/stockfish/stockfish-ubuntu-x86-64-avx2"),
        "C:\\Program Files\\Stockfish\\stockfish.exe",
        "stockfish.exe",
    ])

    for path in candidates:
        if not path:
            continue
        if os.path.isdir(path):
            for name in ("stockfish", "stockfish.exe"):
                candidate = os.path.join(path, name)
                if os.path.exists(candidate):
                    return candidate
            continue
        if os.path.exists(path):
            return path
        resolved = shutil.which(path)
        if resolved:
            return resolved

    return None


def open_stockfish_engine(path: str) -> chess.engine.SimpleEngine:
    """Open Stockfish via UCI."""
    return chess.engine.SimpleEngine.popen_uci(path)
