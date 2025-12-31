"""
Chess Neural Network - ResNet architecture with policy and value heads

Based on AlphaZero architecture:
- Input: 18 planes of 8x8 (board representation)
- Body: Stack of residual blocks
- Policy head: Move probabilities
- Value head: Position evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.size()

        # Global average pooling
        y = x.view(batch, channels, -1).mean(dim=2)

        # Excitation
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))

        # Scale
        return x * y.view(batch, channels, 1, 1)


class ResidualBlock(nn.Module):
    """
    Residual block with optional squeeze-excitation

    Structure: Conv -> BN -> ReLU -> Conv -> BN -> SE -> Add -> ReLU
    """

    def __init__(
        self,
        channels: int,
        se_ratio: int = 0,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.se = SEBlock(channels, se_ratio) if se_ratio > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.se is not None:
            out = self.se(out)

        out = out + identity
        out = F.relu(out)

        return out


class PolicyHead(nn.Module):
    """
    Policy head - predicts move probabilities

    Structure: Conv 1x1 -> BN -> ReLU -> Flatten -> FC
    """

    def __init__(
        self,
        in_channels: int,
        num_moves: int = 1858,
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, 32, 1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32 * 8 * 8, num_moves)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x  # Raw logits (use with CrossEntropyLoss)


class ValueHead(nn.Module):
    """
    Value head - predicts position evaluation

    Structure: Conv 1x1 -> BN -> ReLU -> Flatten -> FC -> ReLU -> FC -> Tanh
    """

    def __init__(
        self,
        in_channels: int,
        hidden_size: int = 256,
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(8 * 8, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x.squeeze(-1)  # Shape: (batch,)


class ChessNetwork(nn.Module):
    """
    Main chess neural network

    Architecture:
    - Input convolution (18 -> num_filters)
    - N residual blocks
    - Policy head (move probabilities)
    - Value head (position evaluation)
    """

    def __init__(
        self,
        num_blocks: int = 10,
        num_filters: int = 256,
        input_planes: int = 18,
        num_moves: int = 1858,
        se_ratio: int = 8,
    ):
        """
        Args:
            num_blocks: Number of residual blocks
            num_filters: Filters per convolution layer
            input_planes: Input feature planes (18 for standard encoding)
            num_moves: Number of possible moves
            se_ratio: Squeeze-excitation ratio (0 to disable)
        """
        super().__init__()

        self.num_blocks = num_blocks
        self.num_filters = num_filters

        # Input convolution
        self.input_conv = nn.Conv2d(input_planes, num_filters, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(num_filters)

        # Residual tower
        self.residual_tower = nn.Sequential(*[
            ResidualBlock(num_filters, se_ratio)
            for _ in range(num_blocks)
        ])

        # Output heads
        self.policy_head = PolicyHead(num_filters, num_moves)
        self.value_head = ValueHead(num_filters)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch, 18, 8, 8)

        Returns:
            Tuple of:
            - policy: Move logits of shape (batch, num_moves)
            - value: Position evaluation of shape (batch,)
        """
        # Input block
        x = F.relu(self.input_bn(self.input_conv(x)))

        # Residual tower
        x = self.residual_tower(x)

        # Output heads
        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value

    def predict(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with softmax policy

        Args:
            x: Input tensor
            temperature: Softmax temperature (lower = more greedy)

        Returns:
            Tuple of (policy_probs, value)
        """
        policy_logits, value = self.forward(x)

        # Apply temperature and softmax
        policy_probs = F.softmax(policy_logits / temperature, dim=-1)

        return policy_probs, value

    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def from_config(config) -> 'ChessNetwork':
        """Create network from config"""
        return ChessNetwork(
            num_blocks=config.model.num_blocks,
            num_filters=config.model.num_filters,
            input_planes=config.model.input_planes,
            num_moves=config.model.policy_output_size,
            se_ratio=config.model.se_ratio,
        )


class ChessLoss(nn.Module):
    """
    Combined loss function for chess network

    Total loss = policy_weight * cross_entropy(policy) + value_weight * mse(value)
    """

    def __init__(
        self,
        policy_weight: float = 1.0,
        value_weight: float = 1.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight

        self.policy_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.value_loss = nn.MSELoss()

    def forward(
        self,
        policy_logits: torch.Tensor,
        value_pred: torch.Tensor,
        policy_target: torch.Tensor,
        value_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute losses

        Args:
            policy_logits: Predicted move logits (batch, num_moves)
            value_pred: Predicted value (batch,)
            policy_target: Target move probabilities (batch, num_moves)
            value_target: Target value (batch,)

        Returns:
            Tuple of (total_loss, policy_loss, value_loss)
        """
        # Policy loss - use cross entropy with soft targets
        # Convert soft targets to hard targets for standard cross entropy
        policy_target_idx = policy_target.argmax(dim=-1)
        p_loss = self.policy_loss(policy_logits, policy_target_idx)

        # Value loss
        v_loss = self.value_loss(value_pred, value_target)

        # Total loss
        total = self.policy_weight * p_loss + self.value_weight * v_loss

        return total, p_loss, v_loss


def create_model(
    num_blocks: int = 10,
    num_filters: int = 256,
    compile_model: bool = True,
    device: str = "cuda",
) -> ChessNetwork:
    """
    Create and optionally compile a chess network

    Args:
        num_blocks: Number of residual blocks
        num_filters: Filters per layer
        compile_model: Whether to use torch.compile
        device: Device to place model on

    Returns:
        ChessNetwork instance
    """
    model = ChessNetwork(
        num_blocks=num_blocks,
        num_filters=num_filters,
    ).to(device)

    if compile_model and hasattr(torch, 'compile'):
        model = torch.compile(model)

    return model


# Test the network
if __name__ == "__main__":
    # Create model
    model = ChessNetwork(num_blocks=10, num_filters=256)
    print(f"Model parameters: {model.count_parameters():,}")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 18, 8, 8)

    policy, value = model(x)
    print(f"Policy shape: {policy.shape}")  # (4, 1858)
    print(f"Value shape: {value.shape}")    # (4,)

    # Test loss
    loss_fn = ChessLoss()
    policy_target = torch.zeros(batch_size, 1858)
    policy_target[:, 0] = 1  # One-hot
    value_target = torch.tensor([1.0, -1.0, 0.0, 0.5])

    total, p_loss, v_loss = loss_fn(policy, value, policy_target, value_target)
    print(f"Total loss: {total.item():.4f}")
    print(f"Policy loss: {p_loss.item():.4f}")
    print(f"Value loss: {v_loss.item():.4f}")

    # Test with CUDA if available
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
        policy, value = model(x)
        print(f"\nCUDA forward pass successful!")
        print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
