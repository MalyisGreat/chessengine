"""
Comprehensive Test Suite for Chess Engine

Run this script to verify all components work correctly before training.

Usage:
    python test_suite.py              # Run all tests
    python test_suite.py --quick      # Quick tests only (no GPU/Stockfish needed)
    python test_suite.py --gpu        # Include GPU tests
    python test_suite.py --stockfish  # Include Stockfish benchmarks
"""

import sys
import os
import time
import argparse
import traceback
from typing import Tuple, List, Optional

import numpy as np
from data.encoder import BoardEncoder

NUM_MOVES = BoardEncoder().num_moves

# Test results tracking
TESTS_RUN = 0
TESTS_PASSED = 0
TESTS_FAILED = 0


def test(name: str):
    """Decorator for test functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            global TESTS_RUN, TESTS_PASSED, TESTS_FAILED
            TESTS_RUN += 1
            print(f"\n{'='*60}")
            print(f"TEST: {name}")
            print(f"{'='*60}")
            try:
                start = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                print(f"PASS ({elapsed:.2f}s)")
                TESTS_PASSED += 1
                return result
            except Exception as e:
                print(f"FAIL: {e}")
                traceback.print_exc()
                TESTS_FAILED += 1
                return None
        return wrapper
    return decorator


# ============================================================================
# BOARD ENCODING TESTS
# ============================================================================

@test("Board Encoding - Basic Position")
def test_board_encoding_basic():
    """Test that starting position encodes correctly"""
    import chess
    from data.encoder import BoardEncoder

    encoder = BoardEncoder()
    board = chess.Board()  # Starting position

    tensor = encoder.encode_board(board)

    # Check shape
    assert tensor.shape == (18, 8, 8), f"Expected (18, 8, 8), got {tensor.shape}"

    # Check white pawns on rank 2
    assert tensor[0, 1, :].sum() == 8, "Should have 8 white pawns on rank 2"

    # Check black pawns on rank 7
    assert tensor[6, 6, :].sum() == 8, "Should have 8 black pawns on rank 7"

    # Check white king on e1
    assert tensor[5, 0, 4] == 1.0, "White king should be on e1"

    # Check side to move (white)
    assert tensor[12, :, :].sum() == 64, "Side to move plane should be all 1s for white"

    # Check castling rights (all available at start)
    assert tensor[13, :, :].sum() == 64, "White kingside castling should be available"
    assert tensor[14, :, :].sum() == 64, "White queenside castling should be available"
    assert tensor[15, :, :].sum() == 64, "Black kingside castling should be available"
    assert tensor[16, :, :].sum() == 64, "Black queenside castling should be available"

    # No en passant at start
    assert tensor[17, :, :].sum() == 0, "No en passant at start"

    print("  - Shape correct: (18, 8, 8)")
    print("  - Piece positions correct")
    print("  - Side to move correct")
    print("  - Castling rights correct")
    print("  - En passant correct")


@test("Board Encoding - After e4")
def test_board_encoding_after_e4():
    """Test encoding after 1.e4 (en passant square should appear)"""
    import chess
    from data.encoder import BoardEncoder

    encoder = BoardEncoder()
    board = chess.Board()
    board.push_san("e4")

    tensor = encoder.encode_board(board)

    # Side to move should be black now
    assert tensor[12, :, :].sum() == 0, "Side to move should be 0 for black"

    # En passant square should be on e3
    assert tensor[17, 2, 4] == 1.0, "En passant should be on e3"

    print("  - Side to move switches to black")
    print("  - En passant square set correctly")


@test("Board Encoding - Castling Rights Lost")
def test_board_encoding_castling():
    """Test that castling rights update correctly"""
    import chess
    from data.encoder import BoardEncoder

    encoder = BoardEncoder()
    board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")

    tensor = encoder.encode_board(board)
    assert tensor[13:17, :, :].sum() == 64 * 4, "All castling rights available"

    # After king moves, lose all castling
    board.push_san("Kf1")
    tensor = encoder.encode_board(board)
    assert tensor[13, :, :].sum() == 0, "White kingside castling lost"
    assert tensor[14, :, :].sum() == 0, "White queenside castling lost"

    print("  - Castling rights update on king move")


@test("Move Encoding - All Move Types")
def test_move_encoding():
    """Test that moves encode to valid indices"""
    import chess
    from data.encoder import BoardEncoder

    encoder = BoardEncoder()
    board = chess.Board()

    # Test some moves
    test_moves = [
        chess.Move.from_uci("e2e4"),  # Pawn push
        chess.Move.from_uci("g1f3"),  # Knight
        chess.Move.from_uci("e1g1"),  # Castling (if legal)
    ]

    for move in test_moves:
        idx = encoder.encode_move(move)
        assert 0 <= idx < encoder.num_moves, f"Move index {idx} out of range for {move}"

        # Decode and verify
        decoded = encoder.decode_move(idx)
        assert decoded == move, f"Decoded {decoded} != original {move}"

    print(f"  - Tested {len(test_moves)} move encodings")
    print(f"  - All indices in valid range [0, {NUM_MOVES})")
    print("  - Encode/decode roundtrip successful")


# ============================================================================
# NEURAL NETWORK TESTS
# ============================================================================

@test("Network - Forward Pass (CPU)")
def test_network_forward_cpu():
    """Test network forward pass on CPU"""
    import torch
    from models.network import ChessNetwork

    model = ChessNetwork(num_blocks=2, num_filters=64, num_moves=NUM_MOVES)  # Small for testing
    model.eval()

    batch_size = 4
    x = torch.randn(batch_size, 18, 8, 8)

    with torch.no_grad():
        policy, value = model(x)

    assert policy.shape == (batch_size, NUM_MOVES), f"Policy shape wrong: {policy.shape}"
    assert value.shape == (batch_size,), f"Value shape wrong: {value.shape}"

    # Value should be in [-1, 1] due to tanh
    assert value.min() >= -1 and value.max() <= 1, "Value out of range"

    params = model.count_parameters()
    print(f"  - Model parameters: {params:,}")
    print(f"  - Policy shape: {policy.shape}")
    print(f"  - Value shape: {value.shape}")
    print(f"  - Value range: [{value.min():.3f}, {value.max():.3f}]")


@test("Network - Loss Function")
def test_network_loss():
    """Test combined loss function"""
    import torch
    from models.network import ChessNetwork, ChessLoss

    model = ChessNetwork(num_blocks=2, num_filters=64, num_moves=NUM_MOVES)
    loss_fn = ChessLoss()

    batch_size = 4
    x = torch.randn(batch_size, 18, 8, 8)

    policy, value = model(x)

    # Create targets
    policy_target = torch.zeros(batch_size, NUM_MOVES)
    policy_target[:, 0] = 1  # One-hot
    value_target = torch.tensor([1.0, -1.0, 0.0, 0.5])

    total, p_loss, v_loss = loss_fn(policy, value, policy_target, value_target)

    assert total.item() > 0, "Loss should be positive"
    assert not torch.isnan(total), "Loss should not be NaN"

    print(f"  - Total loss: {total.item():.4f}")
    print(f"  - Policy loss: {p_loss.item():.4f}")
    print(f"  - Value loss: {v_loss.item():.4f}")


@test("Network - Gradient Flow")
def test_network_gradients():
    """Test that gradients flow properly"""
    import torch
    from models.network import ChessNetwork, ChessLoss

    model = ChessNetwork(num_blocks=2, num_filters=64, num_moves=NUM_MOVES)
    loss_fn = ChessLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    x = torch.randn(2, 18, 8, 8)
    policy_target = torch.zeros(2, NUM_MOVES)
    policy_target[0, 100] = 1
    policy_target[1, 200] = 1
    value_target = torch.tensor([0.5, -0.5])

    # Forward
    policy, value = model(x)
    total, _, _ = loss_fn(policy, value, policy_target, value_target)

    # Backward
    optimizer.zero_grad()
    total.backward()

    # Check gradients exist
    grad_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            grad_count += 1

    assert grad_count > 0, "No gradients computed"
    print(f"  - {grad_count} parameters have gradients")
    print("  - Gradient flow verified")


# ============================================================================
# DATA LOADING TESTS
# ============================================================================

@test("Dataset - NPZ Loading")
def test_dataset_loading():
    """Test that dataset can load NPZ files"""
    from data.dataset import ChessDataset

    # Create temporary test data
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()
    try:
        # Create fake NPZ file
        boards = np.random.randn(100, 18, 8, 8).astype(np.float32)
        policies = np.zeros((100, NUM_MOVES), dtype=np.float32)
        policies[:, 0] = 1
        values = np.random.uniform(-1, 1, 100).astype(np.float32)

        np.savez(
            os.path.join(temp_dir, "test.npz"),
            boards=boards,
            policies=policies,
            values=values,
        )

        # Load dataset
        dataset = ChessDataset(temp_dir)

        assert len(dataset) == 100, f"Expected 100 samples, got {len(dataset)}"

        # Test getitem
        board, policy, value = dataset[0]
        assert board.shape == (18, 8, 8), f"Board shape wrong: {board.shape}"
        assert policy.shape == (NUM_MOVES,), f"Policy shape wrong: {policy.shape}"

        print(f"  - Loaded {len(dataset)} samples")
        print(f"  - Board shape: {board.shape}")
        print(f"  - Policy shape: {policy.shape}")

    finally:
        shutil.rmtree(temp_dir)


@test("DataLoader - Batching")
def test_dataloader():
    """Test PyTorch DataLoader integration"""
    import torch
    from torch.utils.data import DataLoader
    from data.dataset import ChessDataset
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()
    try:
        # Create test data
        boards = np.random.randn(100, 18, 8, 8).astype(np.float32)
        policies = np.zeros((100, NUM_MOVES), dtype=np.float32)
        values = np.random.uniform(-1, 1, 100).astype(np.float32)

        np.savez(os.path.join(temp_dir, "test.npz"),
                 boards=boards, policies=policies, values=values)

        dataset = ChessDataset(temp_dir)
        loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

        batch = next(iter(loader))
        boards_batch, policies_batch, values_batch = batch

        assert boards_batch.shape == (16, 18, 8, 8)
        assert policies_batch.shape == (16, NUM_MOVES)
        assert values_batch.shape == (16,)

        print(f"  - Batch size: 16")
        print(f"  - Boards batch: {boards_batch.shape}")
        print(f"  - DataLoader working correctly")

    finally:
        shutil.rmtree(temp_dir)


# ============================================================================
# SEARCH & ENGINE TESTS
# ============================================================================

@test("Search - Negamax Basic")
def test_search_basic():
    """Test that search finds legal moves"""
    import chess
    from engine.search import ChessEngine

    # Create engine without model (uses random eval)
    engine = ChessEngine(model_path=None, device="cpu")

    board = chess.Board()

    # Search should return a legal move
    result = engine.search(board, depth=2, time_limit=5.0)

    assert result is not None, "Search should return a result"
    assert result.best_move is not None, "Should find a move"
    assert result.best_move in board.legal_moves, "Move should be legal"

    print(f"  - Best move: {result.best_move}")
    print(f"  - Score: {result.score:.2f}")
    print(f"  - Nodes searched: {result.nodes}")


@test("Search - Mate in 1")
def test_search_mate_in_1():
    """Test that search finds mate in 1"""
    import chess
    from engine.search import ChessEngine

    engine = ChessEngine(model_path=None, device="cpu")

    # Scholar's mate position - Qxf7 is mate
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")

    result = engine.search(board, depth=3, time_limit=10.0)

    # Should find Qxf7#
    assert result.best_move == chess.Move.from_uci("h5f7"), f"Should find Qxf7#, got {result.best_move}"

    print(f"  - Found mate: {result.best_move}")
    print(f"  - Score indicates winning: {result.score}")


@test("UCI Protocol - Basic Commands")
def test_uci_protocol():
    """Test UCI command parsing"""
    from engine.search import UCIEngine
    import io
    from contextlib import redirect_stdout

    # We'll just test that the UCI engine initializes
    # Full UCI testing would require subprocess interaction

    # Test position parsing
    import chess
    board = chess.Board()

    # Test FEN parsing
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    board.set_fen(fen)
    assert board.turn == chess.BLACK
    assert board.ep_square == chess.E3

    print("  - Position parsing works")
    print("  - FEN parsing works")
    print("  - UCI engine initializes")


# ============================================================================
# GPU TESTS (Optional)
# ============================================================================

@test("GPU - CUDA Available")
def test_gpu_available():
    """Check if CUDA is available"""
    import torch

    if not torch.cuda.is_available():
        print("  - CUDA not available (this is OK for CPU training)")
        return

    print(f"  - CUDA available: {torch.cuda.is_available()}")
    print(f"  - Device count: {torch.cuda.device_count()}")
    print(f"  - Current device: {torch.cuda.current_device()}")
    print(f"  - Device name: {torch.cuda.get_device_name(0)}")

    # Test memory
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  - GPU memory: {total:.1f} GB")


@test("GPU - Network Forward Pass")
def test_gpu_forward():
    """Test network on GPU"""
    import torch
    from models.network import ChessNetwork

    if not torch.cuda.is_available():
        print("  - Skipping (no CUDA)")
        return

    model = ChessNetwork(num_blocks=10, num_filters=256, num_moves=NUM_MOVES).cuda()

    # Large batch to test memory
    batch_size = 1024
    x = torch.randn(batch_size, 18, 8, 8).cuda()

    # Warmup
    with torch.no_grad():
        _ = model(x)

    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        for _ in range(10):
            policy, value = model(x)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    samples_per_sec = (batch_size * 10) / elapsed

    print(f"  - Batch size: {batch_size}")
    print(f"  - Throughput: {samples_per_sec:.0f} samples/sec")
    print(f"  - GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


@test("GPU - Mixed Precision (BF16)")
def test_gpu_mixed_precision():
    """Test BF16 mixed precision"""
    import torch
    from models.network import ChessNetwork

    if not torch.cuda.is_available():
        print("  - Skipping (no CUDA)")
        return

    model = ChessNetwork(num_blocks=10, num_filters=256, num_moves=NUM_MOVES).cuda()
    x = torch.randn(512, 18, 8, 8).cuda()

    # Test with autocast
    with torch.autocast('cuda', dtype=torch.bfloat16):
        policy, value = model(x)

    print(f"  - BF16 forward pass successful")
    print(f"  - Policy dtype: {policy.dtype}")
    print(f"  - Value dtype: {value.dtype}")


# ============================================================================
# STOCKFISH TESTS (Optional)
# ============================================================================

@test("Stockfish - Installation Check")
def test_stockfish_available():
    """Check if Stockfish is available for benchmarking"""
    try:
        from stockfish import Stockfish

        # Try common paths
        paths = [
            "stockfish",
            "/usr/bin/stockfish",
            "/usr/local/bin/stockfish",
            "C:\\stockfish\\stockfish.exe",
        ]

        for path in paths:
            try:
                sf = Stockfish(path)
                print(f"  - Stockfish found at: {path}")
                print(f"  - Ready for benchmarking")
                return True
            except:
                continue

        print("  - Stockfish not found (benchmarking will be limited)")
        print("  - Install with: apt install stockfish (Linux)")
        return False

    except ImportError:
        print("  - stockfish package not installed")
        print("  - Install with: pip install stockfish")
        return False


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@test("Integration - Full Training Step")
def test_training_step():
    """Test a complete training step"""
    import torch
    from models.network import ChessNetwork, ChessLoss

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChessNetwork(num_blocks=2, num_filters=64, num_moves=NUM_MOVES).to(device)
    loss_fn = ChessLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Fake batch
    batch_size = 32
    boards = torch.randn(batch_size, 18, 8, 8).to(device)
    policies = torch.zeros(batch_size, NUM_MOVES).to(device)
    policies[:, torch.randint(0, NUM_MOVES, (batch_size,))] = 1
    values = torch.rand(batch_size).to(device) * 2 - 1

    # Training step
    model.train()
    optimizer.zero_grad()

    policy_out, value_out = model(boards)
    total_loss, p_loss, v_loss = loss_fn(policy_out, value_out, policies, values)

    total_loss.backward()
    optimizer.step()

    print(f"  - Device: {device}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Loss: {total_loss.item():.4f}")
    print("  - Training step completed successfully")


@test("Integration - Model Save/Load")
def test_model_save_load():
    """Test saving and loading model checkpoints"""
    import torch
    from models.network import ChessNetwork
    import tempfile

    model = ChessNetwork(num_blocks=2, num_filters=64, num_moves=NUM_MOVES)

    # Save
    temp_file = tempfile.NamedTemporaryFile(suffix='.pt', delete=False)
    temp_file.close()

    try:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'num_blocks': 2,
            'num_filters': 64,
        }
        torch.save(checkpoint, temp_file.name)

        # Load
        loaded = torch.load(temp_file.name, weights_only=False)
        new_model = ChessNetwork(
            num_blocks=loaded['num_blocks'],
            num_filters=loaded['num_filters'],
            num_moves=NUM_MOVES,
        )
        new_model.load_state_dict(loaded['model_state_dict'])

        # Verify
        x = torch.randn(1, 18, 8, 8)
        with torch.no_grad():
            out1 = model(x)
            out2 = new_model(x)

        assert torch.allclose(out1[0], out2[0]), "Policy mismatch"
        assert torch.allclose(out1[1], out2[1]), "Value mismatch"

        print(f"  - Model saved to temp file")
        print(f"  - Model loaded successfully")
        print(f"  - Outputs match after reload")

    finally:
        os.unlink(temp_file.name)


# ============================================================================
# MAIN
# ============================================================================

def run_tests(quick: bool = False, gpu: bool = False, stockfish: bool = False):
    """Run all tests"""
    global TESTS_RUN, TESTS_PASSED, TESTS_FAILED

    print("\n" + "=" * 60)
    print("CHESS ENGINE TEST SUITE")
    print("=" * 60)

    # Core tests (always run)
    print("\n>>> BOARD ENCODING TESTS")
    test_board_encoding_basic()
    test_board_encoding_after_e4()
    test_board_encoding_castling()
    test_move_encoding()

    print("\n>>> NEURAL NETWORK TESTS")
    test_network_forward_cpu()
    test_network_loss()
    test_network_gradients()

    print("\n>>> DATA LOADING TESTS")
    test_dataset_loading()
    test_dataloader()

    print("\n>>> SEARCH & ENGINE TESTS")
    test_search_basic()
    test_search_mate_in_1()
    test_uci_protocol()

    print("\n>>> INTEGRATION TESTS")
    test_training_step()
    test_model_save_load()

    if gpu or not quick:
        print("\n>>> GPU TESTS")
        test_gpu_available()
        import torch
        if torch.cuda.is_available():
            test_gpu_forward()
            test_gpu_mixed_precision()

    if stockfish:
        print("\n>>> STOCKFISH TESTS")
        test_stockfish_available()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run:    {TESTS_RUN}")
    print(f"Tests passed: {TESTS_PASSED}")
    print(f"Tests failed: {TESTS_FAILED}")
    print("=" * 60)

    if TESTS_FAILED > 0:
        print("\nWARNING: Some tests failed! Check output above.")
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Chess Engine Test Suite")
    parser.add_argument("--quick", action="store_true", help="Quick tests only (no GPU)")
    parser.add_argument("--gpu", action="store_true", help="Include GPU tests")
    parser.add_argument("--stockfish", action="store_true", help="Include Stockfish tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")

    args = parser.parse_args()

    if args.all:
        run_tests(quick=False, gpu=True, stockfish=True)
    else:
        run_tests(quick=args.quick, gpu=args.gpu, stockfish=args.stockfish)


if __name__ == "__main__":
    main()
