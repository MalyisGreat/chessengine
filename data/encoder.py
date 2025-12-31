"""
Board Encoder - Converts chess positions to neural network input tensors

Input representation (18 planes of 8x8):
- 6 planes: White pieces (P, N, B, R, Q, K)
- 6 planes: Black pieces (p, n, b, r, q, k)
- 1 plane: Side to move (all 1s if white, all 0s if black)
- 1 plane: Castling rights (white kingside)
- 1 plane: Castling rights (white queenside)
- 1 plane: Castling rights (black kingside)
- 1 plane: Castling rights (black queenside)
- 1 plane: En passant square
"""

import chess
import numpy as np
import torch
from typing import Tuple, Optional


class BoardEncoder:
    """Encode chess board positions into tensor format for neural network"""

    # Piece type to plane index mapping
    PIECE_PLANES = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    NUM_PLANES = 18  # Total input planes
    BOARD_SIZE = 8

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._init_move_tables()

    def _init_move_tables(self):
        """Initialize move encoding/decoding tables"""
        # All possible moves: 64 from squares * 73 possible move types
        # Move types: 56 queen moves (8 directions * 7 distances) +
        #             8 knight moves + 9 underpromotions
        self.move_to_idx = {}
        self.idx_to_move = {}
        idx = 0

        # Generate all legal move patterns
        for from_sq in range(64):
            from_rank = from_sq // 8
            from_file = from_sq % 8

            # Queen-like moves (includes rook and bishop)
            directions = [
                (0, 1), (0, -1), (1, 0), (-1, 0),  # Orthogonal
                (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonal
            ]
            for dr, df in directions:
                for dist in range(1, 8):
                    to_rank = from_rank + dr * dist
                    to_file = from_file + df * dist
                    if 0 <= to_rank < 8 and 0 <= to_file < 8:
                        to_sq = to_rank * 8 + to_file
                        move_key = (from_sq, to_sq, None)
                        if move_key not in self.move_to_idx:
                            self.move_to_idx[move_key] = idx
                            self.idx_to_move[idx] = move_key
                            idx += 1

            # Knight moves
            knight_moves = [
                (2, 1), (2, -1), (-2, 1), (-2, -1),
                (1, 2), (1, -2), (-1, 2), (-1, -2)
            ]
            for dr, df in knight_moves:
                to_rank = from_rank + dr
                to_file = from_file + df
                if 0 <= to_rank < 8 and 0 <= to_file < 8:
                    to_sq = to_rank * 8 + to_file
                    move_key = (from_sq, to_sq, None)
                    if move_key not in self.move_to_idx:
                        self.move_to_idx[move_key] = idx
                        self.idx_to_move[idx] = move_key
                        idx += 1

            # Pawn promotions (only from ranks 1 and 6)
            if from_rank in [1, 6]:
                promo_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
                promo_dirs = [(1, 0), (1, 1), (1, -1)] if from_rank == 6 else [(-1, 0), (-1, 1), (-1, -1)]
                for dr, df in promo_dirs:
                    to_rank = from_rank + dr
                    to_file = from_file + df
                    if 0 <= to_file < 8:
                        to_sq = to_rank * 8 + to_file
                        for promo in promo_pieces:
                            move_key = (from_sq, to_sq, promo)
                            if move_key not in self.move_to_idx:
                                self.move_to_idx[move_key] = idx
                                self.idx_to_move[idx] = move_key
                                idx += 1

        self.num_moves = idx

    def encode_board(self, board: chess.Board) -> np.ndarray:
        """
        Encode a chess board position into an 18x8x8 tensor

        Args:
            board: python-chess Board object

        Returns:
            numpy array of shape (18, 8, 8)
        """
        planes = np.zeros((self.NUM_PLANES, 8, 8), dtype=np.float32)

        # Encode pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                rank = square // 8
                file = square % 8

                plane_idx = self.PIECE_PLANES[piece.piece_type]
                if piece.color == chess.BLACK:
                    plane_idx += 6  # Black pieces are planes 6-11

                planes[plane_idx, rank, file] = 1.0

        # Side to move (plane 12)
        if board.turn == chess.WHITE:
            planes[12, :, :] = 1.0

        # Castling rights (planes 13-16)
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[13, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[14, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[15, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[16, :, :] = 1.0

        # En passant square (plane 17)
        if board.ep_square is not None:
            ep_rank = board.ep_square // 8
            ep_file = board.ep_square % 8
            planes[17, ep_rank, ep_file] = 1.0

        return planes

    def encode_board_batch(self, boards: list) -> torch.Tensor:
        """Encode multiple boards into a batch tensor"""
        batch = np.stack([self.encode_board(b) for b in boards])
        return torch.from_numpy(batch).to(self.device)

    def encode_move(self, move: chess.Move) -> int:
        """
        Encode a move as an integer index

        Args:
            move: python-chess Move object

        Returns:
            Integer index for the move
        """
        from_sq = move.from_square
        to_sq = move.to_square
        promotion = move.promotion

        move_key = (from_sq, to_sq, promotion)
        return self.move_to_idx.get(move_key, -1)

    def decode_move(self, idx: int) -> Optional[chess.Move]:
        """
        Decode an integer index back to a move

        Args:
            idx: Move index

        Returns:
            python-chess Move object
        """
        if idx not in self.idx_to_move:
            return None

        from_sq, to_sq, promotion = self.idx_to_move[idx]
        return chess.Move(from_sq, to_sq, promotion=promotion)

    def encode_policy(self, board: chess.Board, move: chess.Move) -> np.ndarray:
        """
        Create a one-hot policy vector for a move

        Args:
            board: Current board position
            move: Move to encode

        Returns:
            numpy array of shape (num_moves,) with 1.0 at the move index
        """
        policy = np.zeros(self.num_moves, dtype=np.float32)
        move_idx = self.encode_move(move)
        if move_idx >= 0:
            policy[move_idx] = 1.0
        return policy

    def get_legal_move_mask(self, board: chess.Board) -> np.ndarray:
        """
        Create a mask of legal moves for the current position

        Args:
            board: Current board position

        Returns:
            numpy array of shape (num_moves,) with 1.0 for legal moves
        """
        mask = np.zeros(self.num_moves, dtype=np.float32)
        for move in board.legal_moves:
            move_idx = self.encode_move(move)
            if move_idx >= 0:
                mask[move_idx] = 1.0
        return mask

    def flip_board(self, planes: np.ndarray) -> np.ndarray:
        """
        Flip board horizontally for data augmentation

        Args:
            planes: Board tensor of shape (18, 8, 8)

        Returns:
            Flipped board tensor
        """
        return np.flip(planes, axis=2).copy()

    def mirror_board(self, planes: np.ndarray) -> np.ndarray:
        """
        Mirror board vertically and swap colors (view from black's perspective)

        Args:
            planes: Board tensor of shape (18, 8, 8)

        Returns:
            Mirrored board tensor
        """
        mirrored = np.zeros_like(planes)

        # Swap white and black pieces and flip ranks
        mirrored[0:6] = np.flip(planes[6:12], axis=1)  # Black pieces become white
        mirrored[6:12] = np.flip(planes[0:6], axis=1)  # White pieces become black

        # Flip side to move
        mirrored[12] = 1.0 - planes[12]

        # Swap castling rights
        mirrored[13] = np.flip(planes[15], axis=1)  # Black kingside -> White
        mirrored[14] = np.flip(planes[16], axis=1)  # Black queenside -> White
        mirrored[15] = np.flip(planes[13], axis=1)  # White kingside -> Black
        mirrored[16] = np.flip(planes[14], axis=1)  # White queenside -> Black

        # Flip en passant
        mirrored[17] = np.flip(planes[17], axis=1)

        return mirrored


# Test the encoder
if __name__ == "__main__":
    encoder = BoardEncoder()

    # Test with starting position
    board = chess.Board()
    encoded = encoder.encode_board(board)
    print(f"Encoded shape: {encoded.shape}")
    print(f"Number of possible moves: {encoder.num_moves}")

    # Test move encoding
    move = chess.Move.from_uci("e2e4")
    move_idx = encoder.encode_move(move)
    print(f"Move e2e4 encoded as: {move_idx}")

    # Decode back
    decoded = encoder.decode_move(move_idx)
    print(f"Decoded back: {decoded.uci()}")

    # Legal move mask
    mask = encoder.get_legal_move_mask(board)
    print(f"Legal moves in starting position: {np.sum(mask)}")
