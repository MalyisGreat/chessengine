"""
Chess Engine - Neural Network + Search

Combines the trained neural network with alpha-beta search for playing games.
Supports both pure policy network play and search-enhanced play.
"""

import chess
import torch
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
import time

import sys
sys.path.append('..')
from models.network import ChessNetwork
from data.encoder import BoardEncoder


@dataclass
class SearchResult:
    """Result of a search"""
    best_move: chess.Move
    score: float  # Evaluation in centipawns (or win probability)
    nodes: int
    depth: int
    pv: List[chess.Move]  # Principal variation
    time_ms: float


class ChessEngine:
    """
    Chess engine using neural network evaluation

    Supports:
    - Pure policy network play (fast)
    - Alpha-beta search with NN evaluation (stronger)
    - MCTS (optional, for strongest play)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        search_depth: int = 4,
        use_search: bool = True,
    ):
        """
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run on
            search_depth: Depth for alpha-beta search
            use_search: Whether to use search (False = pure policy)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.search_depth = search_depth
        self.use_search = use_search
        self.nodes_searched = 0

        # Initialize encoder
        self.encoder = BoardEncoder()

        # Load model
        self.model = self._load_model(model_path)

        # Transposition table for search
        self.tt = {}
        self.tt_hits = 0

    def _load_model(self, model_path: Optional[str]) -> ChessNetwork:
        """Load trained model"""
        model = ChessNetwork(
            num_blocks=10,
            num_filters=256,
            num_moves=self.encoder.num_moves,  # Use encoder's actual move count
        ).to(self.device)

        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded model from {model_path}")

        model.eval()
        return model

    @torch.no_grad()
    def evaluate(self, board: chess.Board) -> Tuple[np.ndarray, float]:
        """
        Evaluate a position with the neural network

        Args:
            board: Chess board position

        Returns:
            Tuple of (policy_probs, value)
        """
        # Encode board
        board_tensor = self.encoder.encode_board(board)
        board_tensor = torch.from_numpy(board_tensor).unsqueeze(0).to(self.device)

        # Forward pass
        policy_logits, value = self.model(board_tensor)

        # Convert to probabilities
        policy_probs = torch.softmax(policy_logits, dim=-1).cpu().numpy()[0]
        value = value.cpu().item()

        return policy_probs, value

    def get_move_probs(self, board: chess.Board) -> List[Tuple[chess.Move, float]]:
        """
        Get move probabilities for legal moves

        Args:
            board: Chess board position

        Returns:
            List of (move, probability) tuples, sorted by probability
        """
        policy_probs, _ = self.evaluate(board)

        move_probs = []
        for move in board.legal_moves:
            move_idx = self.encoder.encode_move(move)
            if move_idx >= 0:
                prob = policy_probs[move_idx]
                move_probs.append((move, prob))

        # Sort by probability
        move_probs.sort(key=lambda x: x[1], reverse=True)
        return move_probs

    def search(
        self,
        board: chess.Board,
        depth: Optional[int] = None,
        time_limit: Optional[float] = None,
    ) -> SearchResult:
        """
        Search for the best move

        Args:
            board: Chess board position
            depth: Search depth (overrides default)
            time_limit: Time limit in seconds

        Returns:
            SearchResult with best move and evaluation
        """
        start_time = time.time()
        self.nodes_searched = 0
        self.tt.clear()
        self.tt_hits = 0

        depth = depth or self.search_depth

        if not self.use_search:
            # Pure policy network play
            move_probs = self.get_move_probs(board)
            if not move_probs:
                return None

            best_move = move_probs[0][0]
            _, value = self.evaluate(board)

            return SearchResult(
                best_move=best_move,
                score=value * 100,  # Convert to centipawns-like scale
                nodes=1,
                depth=0,
                pv=[best_move],
                time_ms=(time.time() - start_time) * 1000,
            )

        # Alpha-beta search
        best_move = None
        best_score = -float('inf')
        pv = []

        # Iterative deepening
        for current_depth in range(1, depth + 1):
            score, move, line = self._alpha_beta(
                board,
                current_depth,
                -float('inf'),
                float('inf'),
                True,  # Maximizing
            )

            if move is not None:
                best_move = move
                best_score = score
                pv = line

            # Check time limit
            if time_limit and (time.time() - start_time) > time_limit:
                break

        elapsed = (time.time() - start_time) * 1000

        return SearchResult(
            best_move=best_move,
            score=best_score * 100,
            nodes=self.nodes_searched,
            depth=current_depth,
            pv=pv,
            time_ms=elapsed,
        )

    def _alpha_beta(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
    ) -> Tuple[float, Optional[chess.Move], List[chess.Move]]:
        """
        Alpha-beta search with neural network evaluation

        Returns:
            Tuple of (score, best_move, principal_variation)
        """
        self.nodes_searched += 1

        # Check for game over
        if board.is_game_over():
            result = board.result()
            if result == "1-0":
                return (1.0 if maximizing else -1.0, None, [])
            elif result == "0-1":
                return (-1.0 if maximizing else 1.0, None, [])
            else:
                return (0.0, None, [])

        # Leaf node - use neural network evaluation
        if depth == 0:
            _, value = self.evaluate(board)
            # Value is from current player's perspective
            if not board.turn:  # Black to move
                value = -value
            return (value, None, [])

        # Transposition table lookup
        board_hash = board.fen()
        if board_hash in self.tt:
            tt_entry = self.tt[board_hash]
            if tt_entry['depth'] >= depth:
                self.tt_hits += 1
                return (tt_entry['score'], tt_entry['move'], tt_entry['pv'])

        # Get move ordering from policy network
        move_probs = self.get_move_probs(board)

        best_move = None
        best_pv = []

        if maximizing:
            max_score = -float('inf')

            for move, _ in move_probs:
                board.push(move)
                score, _, child_pv = self._alpha_beta(board, depth - 1, alpha, beta, False)
                board.pop()

                if score > max_score:
                    max_score = score
                    best_move = move
                    best_pv = [move] + child_pv

                alpha = max(alpha, score)
                if beta <= alpha:
                    break  # Beta cutoff

            # Store in transposition table
            self.tt[board_hash] = {
                'score': max_score,
                'move': best_move,
                'pv': best_pv,
                'depth': depth,
            }

            return (max_score, best_move, best_pv)

        else:
            min_score = float('inf')

            for move, _ in move_probs:
                board.push(move)
                score, _, child_pv = self._alpha_beta(board, depth - 1, alpha, beta, True)
                board.pop()

                if score < min_score:
                    min_score = score
                    best_move = move
                    best_pv = [move] + child_pv

                beta = min(beta, score)
                if beta <= alpha:
                    break  # Alpha cutoff

            # Store in transposition table
            self.tt[board_hash] = {
                'score': min_score,
                'move': best_move,
                'pv': best_pv,
                'depth': depth,
            }

            return (min_score, best_move, best_pv)

    def play_game(
        self,
        opponent=None,
        color: chess.Color = chess.WHITE,
        time_per_move: float = 1.0,
        verbose: bool = True,
    ) -> Tuple[chess.Board, str]:
        """
        Play a game

        Args:
            opponent: Opponent engine (None = self-play)
            color: Color to play as
            time_per_move: Time limit per move
            verbose: Print moves

        Returns:
            Tuple of (final_board, result)
        """
        board = chess.Board()

        while not board.is_game_over():
            is_our_turn = (board.turn == color)

            if is_our_turn or opponent is None:
                result = self.search(board, time_limit=time_per_move)
                move = result.best_move
            else:
                result = opponent.search(board, time_limit=time_per_move)
                move = result.best_move

            if verbose:
                side = "White" if board.turn else "Black"
                print(f"{side}: {move.uci()} (score: {result.score:.1f}, nodes: {result.nodes})")

            board.push(move)

        result = board.result()
        if verbose:
            print(f"\nGame over: {result}")

        return board, result


class UCIEngine:
    """
    UCI protocol interface for the chess engine

    Allows the engine to be used with chess GUIs and tournaments.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.engine = ChessEngine(model_path)
        self.board = chess.Board()

    def run(self):
        """Run the UCI loop"""
        while True:
            try:
                line = input().strip()
            except EOFError:
                break

            if not line:
                continue

            tokens = line.split()
            cmd = tokens[0]

            if cmd == "uci":
                print("id name ChessNN")
                print("id author Claude")
                print("option name Depth type spin default 4 min 1 max 20")
                print("uciok")

            elif cmd == "isready":
                print("readyok")

            elif cmd == "ucinewgame":
                self.board = chess.Board()
                self.engine.tt.clear()

            elif cmd == "position":
                self._handle_position(tokens[1:])

            elif cmd == "go":
                self._handle_go(tokens[1:])

            elif cmd == "quit":
                break

    def _handle_position(self, tokens: List[str]):
        """Handle position command"""
        if tokens[0] == "startpos":
            self.board = chess.Board()
            tokens = tokens[1:]
        elif tokens[0] == "fen":
            fen = " ".join(tokens[1:7])
            self.board = chess.Board(fen)
            tokens = tokens[7:]

        if tokens and tokens[0] == "moves":
            for move_uci in tokens[1:]:
                move = chess.Move.from_uci(move_uci)
                self.board.push(move)

    def _handle_go(self, tokens: List[str]):
        """Handle go command"""
        depth = self.engine.search_depth
        time_limit = None

        i = 0
        while i < len(tokens):
            if tokens[i] == "depth" and i + 1 < len(tokens):
                depth = int(tokens[i + 1])
                i += 2
            elif tokens[i] == "movetime" and i + 1 < len(tokens):
                time_limit = int(tokens[i + 1]) / 1000.0
                i += 2
            elif tokens[i] in ("wtime", "btime", "winc", "binc", "movestogo"):
                i += 2  # Skip time control for now
            else:
                i += 1

        result = self.engine.search(self.board, depth=depth, time_limit=time_limit)

        # Print info
        print(f"info depth {result.depth} nodes {result.nodes} score cp {int(result.score)} pv {' '.join(m.uci() for m in result.pv)}")

        # Print best move
        print(f"bestmove {result.best_move.uci()}")


# Test the engine
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--uci", action="store_true", help="Run in UCI mode")
    parser.add_argument("--depth", type=int, default=4, help="Search depth")
    args = parser.parse_args()

    if args.uci:
        uci = UCIEngine(args.model)
        uci.run()
    else:
        # Demo game
        engine = ChessEngine(args.model, search_depth=args.depth)

        board = chess.Board()
        print("Starting position:")
        print(board)
        print()

        # Play a few moves
        for _ in range(10):
            result = engine.search(board, depth=args.depth)
            print(f"Best move: {result.best_move.uci()}")
            print(f"Score: {result.score:.2f}")
            print(f"Nodes: {result.nodes}")
            print(f"Depth: {result.depth}")
            print(f"PV: {' '.join(m.uci() for m in result.pv)}")
            print()

            board.push(result.best_move)
            print(board)
            print()

            if board.is_game_over():
                break
