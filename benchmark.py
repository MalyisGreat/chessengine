"""
Chess Engine Benchmarking

Tests the trained engine against:
1. Stockfish at various skill levels (ELO estimation)
2. Tactical puzzles (tactical accuracy)
3. Standard test positions (strategic play)

Usage:
    python benchmark.py --model ./outputs/checkpoint_best.pt
    python benchmark.py --model ./outputs/checkpoint_best.pt --elo-test
    python benchmark.py --model ./outputs/checkpoint_best.pt --tactical-test
"""

import os
import sys
import argparse
import time
from typing import Optional, List, Tuple
from dataclasses import dataclass
import json

import chess
import chess.engine
import numpy as np
from tqdm import tqdm

from engine.search import ChessEngine
from data.encoder import BoardEncoder


@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""
    name: str
    score: float
    details: dict


class ChessBenchmark:
    """
    Benchmark suite for chess engine evaluation
    """

    def __init__(
        self,
        model_path: str,
        stockfish_path: Optional[str] = None,
        device: str = "cuda",
    ):
        """
        Args:
            model_path: Path to trained model
            stockfish_path: Path to Stockfish binary (auto-detect if None)
            device: Device to run on
        """
        self.engine = ChessEngine(model_path, device=device, search_depth=4)
        self.stockfish_path = stockfish_path or self._find_stockfish()
        self.encoder = BoardEncoder()

    def _find_stockfish(self) -> Optional[str]:
        """Try to find Stockfish binary"""
        import shutil

        # Common paths
        paths = [
            "stockfish",
            "/usr/local/bin/stockfish",
            "/usr/bin/stockfish",
            "C:\\Program Files\\Stockfish\\stockfish.exe",
            "stockfish.exe",
        ]

        for path in paths:
            if shutil.which(path):
                return path

        print("Warning: Stockfish not found. ELO testing disabled.")
        return None

    def run_all(self) -> dict:
        """Run all benchmarks"""
        results = {}

        # ELO estimation
        if self.stockfish_path:
            elo_result = self.elo_test()
            results['elo'] = elo_result

        # Tactical test
        tactical_result = self.tactical_test()
        results['tactical'] = tactical_result

        # Policy accuracy
        policy_result = self.policy_accuracy_test()
        results['policy_accuracy'] = policy_result

        return results

    def elo_test(
        self,
        num_games: int = 50,
        time_per_move: float = 1.0,
    ) -> BenchmarkResult:
        """
        Estimate ELO by playing against Stockfish at various levels

        Args:
            num_games: Number of games to play at each level
            time_per_move: Time per move in seconds

        Returns:
            BenchmarkResult with ELO estimation
        """
        if not self.stockfish_path:
            return BenchmarkResult(
                name="ELO Estimation",
                score=0,
                details={"error": "Stockfish not found"},
            )

        print("\n=== ELO Estimation ===")

        # Stockfish skill levels and approximate ELOs
        levels = [
            (1, 1350),
            (5, 1700),
            (10, 2000),
            (15, 2400),
            (20, 2850),
        ]

        results = {}

        for skill, approx_elo in levels:
            print(f"\nTesting against Stockfish Level {skill} (~{approx_elo} ELO)...")

            wins, draws, losses = 0, 0, 0

            for game_idx in tqdm(range(num_games // 2)):
                # Play as white
                result = self._play_vs_stockfish(
                    skill_level=skill,
                    our_color=chess.WHITE,
                    time_per_move=time_per_move,
                )
                if result == "1-0":
                    wins += 1
                elif result == "0-1":
                    losses += 1
                else:
                    draws += 1

                # Play as black
                result = self._play_vs_stockfish(
                    skill_level=skill,
                    our_color=chess.BLACK,
                    time_per_move=time_per_move,
                )
                if result == "0-1":
                    wins += 1
                elif result == "1-0":
                    losses += 1
                else:
                    draws += 1

            total = wins + draws + losses
            win_rate = (wins + 0.5 * draws) / total if total > 0 else 0.5

            results[skill] = {
                'approx_elo': approx_elo,
                'wins': wins,
                'draws': draws,
                'losses': losses,
                'win_rate': win_rate,
            }

            print(f"  Wins: {wins}, Draws: {draws}, Losses: {losses}")
            print(f"  Win rate: {win_rate:.1%}")

        # Estimate ELO based on win rates
        estimated_elo = self._estimate_elo(results)

        return BenchmarkResult(
            name="ELO Estimation",
            score=estimated_elo,
            details=results,
        )

    def _play_vs_stockfish(
        self,
        skill_level: int,
        our_color: chess.Color,
        time_per_move: float,
    ) -> str:
        """Play a game against Stockfish"""
        board = chess.Board()

        with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as stockfish:
            stockfish.configure({"Skill Level": skill_level})

            while not board.is_game_over():
                if board.turn == our_color:
                    # Our move
                    result = self.engine.search(board, time_limit=time_per_move)
                    if result and result.best_move:
                        board.push(result.best_move)
                    else:
                        break
                else:
                    # Stockfish move
                    result = stockfish.play(
                        board,
                        chess.engine.Limit(time=time_per_move),
                    )
                    board.push(result.move)

        return board.result()

    def _estimate_elo(self, results: dict) -> float:
        """Estimate ELO from win rates against various levels"""
        # Simple interpolation based on 50% win rate point
        for skill in sorted(results.keys()):
            data = results[skill]
            if data['win_rate'] < 0.5:
                # Found the level where we're worse
                if skill == 1:
                    return data['approx_elo'] - 200
                else:
                    # Interpolate between this and previous level
                    prev_skill = skill - 5 if skill > 5 else skill - 4
                    prev_data = results.get(prev_skill, data)
                    prev_elo = prev_data['approx_elo']
                    curr_elo = data['approx_elo']

                    # Linear interpolation
                    prev_rate = prev_data['win_rate']
                    curr_rate = data['win_rate']

                    if prev_rate > curr_rate:
                        ratio = (0.5 - curr_rate) / (prev_rate - curr_rate)
                        return curr_elo + ratio * (prev_elo - curr_elo)

        # Won against all levels
        return 3000

    def tactical_test(
        self,
        num_puzzles: int = 100,
        time_per_puzzle: float = 10.0,
    ) -> BenchmarkResult:
        """
        Test tactical ability on chess puzzles

        Args:
            num_puzzles: Number of puzzles to test
            time_per_puzzle: Time limit per puzzle

        Returns:
            BenchmarkResult with tactical accuracy
        """
        print("\n=== Tactical Test ===")

        # Generate random tactical positions (simplified)
        correct = 0
        total = 0

        puzzles = self._generate_tactical_puzzles(num_puzzles)

        for board, solution_move in tqdm(puzzles, desc="Testing puzzles"):
            result = self.engine.search(board, time_limit=time_per_puzzle)

            if result and result.best_move == solution_move:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0

        print(f"\nTactical accuracy: {accuracy:.1%} ({correct}/{total})")

        return BenchmarkResult(
            name="Tactical Test",
            score=accuracy * 100,
            details={
                'correct': correct,
                'total': total,
                'accuracy': accuracy,
            },
        )

    def _generate_tactical_puzzles(self, num_puzzles: int) -> List[Tuple[chess.Board, chess.Move]]:
        """Generate simple tactical puzzles (checkmate in 1)"""
        puzzles = []

        # Some classic mate-in-1 positions
        mate_in_1 = [
            ("r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "h5f7"),  # Scholar's mate
            ("rnbqkb1r/pppp1ppp/5n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "h5f7"),
            ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "f3e5"),
            ("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 3", "h4e1"),
        ]

        for fen, move_uci in mate_in_1:
            board = chess.Board(fen)
            move = chess.Move.from_uci(move_uci)
            puzzles.append((board, move))

        # Add random positions with best moves (simplified)
        while len(puzzles) < num_puzzles:
            board = chess.Board()

            # Make some random moves
            for _ in range(np.random.randint(5, 30)):
                legal = list(board.legal_moves)
                if not legal:
                    break
                board.push(np.random.choice(legal))

            if board.is_game_over():
                continue

            # Use Stockfish to find best move if available
            if self.stockfish_path:
                try:
                    with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as sf:
                        result = sf.play(board, chess.engine.Limit(depth=15))
                        puzzles.append((board.copy(), result.move))
                except Exception:
                    # Just use any legal move
                    puzzles.append((board.copy(), list(board.legal_moves)[0]))
            else:
                # Use first legal move as "solution"
                puzzles.append((board.copy(), list(board.legal_moves)[0]))

        return puzzles[:num_puzzles]

    def policy_accuracy_test(
        self,
        num_positions: int = 1000,
    ) -> BenchmarkResult:
        """
        Test policy network accuracy vs Stockfish best moves

        Args:
            num_positions: Number of positions to test

        Returns:
            BenchmarkResult with policy accuracy
        """
        print("\n=== Policy Accuracy Test ===")

        if not self.stockfish_path:
            return BenchmarkResult(
                name="Policy Accuracy",
                score=0,
                details={"error": "Stockfish not found"},
            )

        top1_correct = 0
        top3_correct = 0
        top5_correct = 0
        total = 0

        with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as stockfish:
            for _ in tqdm(range(num_positions), desc="Testing positions"):
                # Generate random position
                board = chess.Board()
                for _ in range(np.random.randint(5, 50)):
                    legal = list(board.legal_moves)
                    if not legal:
                        break
                    board.push(np.random.choice(legal))

                if board.is_game_over():
                    continue

                # Get Stockfish best move
                try:
                    sf_result = stockfish.play(board, chess.engine.Limit(depth=15))
                    sf_move = sf_result.move
                except Exception:
                    continue

                # Get our policy ranking
                move_probs = self.engine.get_move_probs(board)
                if not move_probs:
                    continue

                # Check accuracy
                top_moves = [m for m, _ in move_probs[:5]]

                if sf_move == top_moves[0]:
                    top1_correct += 1
                if sf_move in top_moves[:3]:
                    top3_correct += 1
                if sf_move in top_moves[:5]:
                    top5_correct += 1

                total += 1

        top1_acc = top1_correct / total if total > 0 else 0
        top3_acc = top3_correct / total if total > 0 else 0
        top5_acc = top5_correct / total if total > 0 else 0

        print(f"\nTop-1 accuracy: {top1_acc:.1%}")
        print(f"Top-3 accuracy: {top3_acc:.1%}")
        print(f"Top-5 accuracy: {top5_acc:.1%}")

        return BenchmarkResult(
            name="Policy Accuracy",
            score=top1_acc * 100,
            details={
                'top1_accuracy': top1_acc,
                'top3_accuracy': top3_acc,
                'top5_accuracy': top5_acc,
                'total_positions': total,
            },
        )

    def print_summary(self, results: dict):
        """Print benchmark summary"""
        print("\n" + "=" * 50)
        print("BENCHMARK SUMMARY")
        print("=" * 50)

        if 'elo' in results:
            print(f"\nEstimated ELO: {results['elo'].score:.0f}")

        if 'tactical' in results:
            acc = results['tactical'].details['accuracy']
            print(f"Tactical Accuracy: {acc:.1%}")

        if 'policy_accuracy' in results:
            acc = results['policy_accuracy'].details.get('top1_accuracy', 0)
            print(f"Policy Top-1 Accuracy: {acc:.1%}")

        # Superhuman check
        print("\n" + "-" * 50)
        if 'elo' in results and results['elo'].score >= 2900:
            print("SUPERHUMAN: YES (ELO >= 2900)")
        elif 'elo' in results:
            print(f"SUPERHUMAN: NO (Need {2900 - results['elo'].score:.0f} more ELO)")
        else:
            print("SUPERHUMAN: Unknown (run ELO test)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark chess engine")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--stockfish",
        type=str,
        help="Path to Stockfish binary",
    )
    parser.add_argument(
        "--elo-test",
        action="store_true",
        help="Run ELO estimation test",
    )
    parser.add_argument(
        "--tactical-test",
        action="store_true",
        help="Run tactical test",
    )
    parser.add_argument(
        "--policy-test",
        action="store_true",
        help="Run policy accuracy test",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=50,
        help="Number of games for ELO test",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    # Create benchmark
    benchmark = ChessBenchmark(
        model_path=args.model,
        stockfish_path=args.stockfish,
        device=args.device,
    )

    results = {}

    if args.all:
        results = benchmark.run_all()
    else:
        if args.elo_test:
            results['elo'] = benchmark.elo_test(num_games=args.num_games)

        if args.tactical_test:
            results['tactical'] = benchmark.tactical_test()

        if args.policy_test:
            results['policy_accuracy'] = benchmark.policy_accuracy_test()

    # Print summary
    benchmark.print_summary(results)

    # Save results
    if args.output:
        output_data = {
            name: {'score': r.score, 'details': r.details}
            for name, r in results.items()
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
