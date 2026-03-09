import flax.linen as nn
import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple
class ResBlock(nn.Module):
    """Residual block with LayerNorm (AlphaZero-style order)"""
    filters: int

    @nn.compact
    def __call__(self, x):
        # First conv block
        y = nn.Conv(self.filters, (3, 3), padding='SAME')(x)
        y = nn.LayerNorm()(y)
        y = nn.relu(y)

        # Second conv block
        y = nn.Conv(self.filters, (3, 3), padding='SAME')(y)
        y = nn.LayerNorm()(y)

        # Skip connection
        y = y + x

        # ReLU AFTER addition (correct order!)
        y = nn.relu(y)
        return y
class AbaloneModel(nn.Module):
    num_actions: int = 1734
    num_filters: int = 128
    num_blocks: int = 8

    @nn.compact
    def __call__(self, board, marbles_out, history=None, moves_count=None):
        # Normalize and reshape inputs
        marbles_out = marbles_out.reshape(-1, 2) / 6.0  # Normalize to [0,1]
        board = board / 1.0  # Normalize board values (-1, 0, 1)

        # Process current board
        x = board[..., None]  # (batch, 9, 9, 1)

        # Process history if provided
        if history is not None:
            # history shape: (batch, 8, 9, 9) - 8 previous positions
            history = history / 1.0  # Normalize values
            history = history[..., None]  # (batch, 8, 9, 9, 1)

            # Concatenate current board with history
            # x: (batch, 9, 9, 1), history: (batch, 8, 9, 9, 1)
            # We want: (batch, 9, 9, 9) with 9 channels (1 current + 8 history)
            history_channels = jnp.transpose(history, (0, 2, 3, 1, 4)).squeeze(-1)  # (batch, 9, 9, 8)
            x = jnp.concatenate([x, history_channels], axis=-1)  # (batch, 9, 9, 9)

        # Common trunk with LayerNorm
        x = nn.Conv(self.num_filters, (3, 3), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        # Residual tower
        for _ in range(self.num_blocks):
            x = ResBlock(self.num_filters)(x)

        # Prepare auxiliary features (marbles_out + moves_count)
        # Always include moves_count for consistent network shape
        if moves_count is not None:
            moves_count_norm = moves_count.reshape(-1, 1) / 300.0  # Normalize to [0,1]
        else:
            # Default to 0 if not provided (backward compatibility)
            moves_count_norm = jnp.zeros((marbles_out.shape[0], 1), dtype=marbles_out.dtype)

        auxiliary_features = jnp.concatenate([marbles_out, moves_count_norm], axis=1)  # (batch, 3)

        # === POLICY HEAD (AlphaZero-style with Conv 1x1) ===
        # Spatial processing with 1x1 conv to reduce dimensionality
        policy_conv = nn.Conv(features=32, kernel_size=(1, 1))(x)
        policy_conv = nn.LayerNorm()(policy_conv)
        policy_conv = nn.relu(policy_conv)
        policy_flat = policy_conv.reshape((policy_conv.shape[0], -1))  # (batch, 9*9*32 = 2592)

        # Combine spatial features with auxiliary information
        policy_combined = jnp.concatenate([policy_flat, auxiliary_features], axis=1)

        # Dense layers for policy
        policy = nn.Dense(1024)(policy_combined)
        policy = nn.relu(policy)
        policy = nn.Dense(512)(policy)  # Additional layer for expressiveness
        policy = nn.relu(policy)
        prior_logits = nn.Dense(self.num_actions)(policy)

        # === VALUE HEAD (keep simple, works well) ===
        # Flatten spatial features for value head
        x_flat = x.reshape((x.shape[0], -1))
        value_combined = jnp.concatenate([x_flat, auxiliary_features], axis=1)

        value = nn.Dense(256)(value_combined)
        value = nn.relu(value)
        value = nn.Dense(1)(value)
        value = nn.tanh(value)
        value = value.squeeze(-1)

        return prior_logits, value
