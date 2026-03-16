import flax.linen as nn
import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple
class ResBlock(nn.Module):
    """Residual block with LayerNorm (AlphaZero-style order)"""
    filters: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        # First conv block
        y = nn.Conv(self.filters, (3, 3), padding='SAME', dtype=self.dtype)(x)
        y = nn.LayerNorm(dtype=self.dtype)(y)
        y = nn.relu(y)

        # Second conv block
        y = nn.Conv(self.filters, (3, 3), padding='SAME', dtype=self.dtype)(y)
        y = nn.LayerNorm(dtype=self.dtype)(y)

        # Skip connection
        y = y + x

        # ReLU AFTER addition (correct order!)
        y = nn.relu(y)
        return y
class AbaloneModel(nn.Module):
    num_actions: int = 1734
    num_filters: int = 128
    num_blocks: int = 8
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, board, marbles_out, history=None, moves_count=None):
        # board: (batch, 9, 9) canonical int8 with values {-2, -1, 0, 1}
        # history: (batch, 8, 9, 9) canonical int8
        # marbles_out: (batch, 2) raw counts
        # moves_count: (batch,) raw count

        batch_size = board.shape[0]
        board_f = board.astype(self.dtype)

        # === Binary plane encoding (AlphaZero-style) ===
        # Current board: 2 planes
        our_pieces = (board_f == 1).astype(self.dtype)    # (batch, 9, 9)
        opp_pieces = (board_f == -1).astype(self.dtype)   # (batch, 9, 9)
        planes = [our_pieces, opp_pieces]

        # History: 8 steps × 2 planes = 16 planes
        if history is not None:
            history_f = history.astype(self.dtype)
            for t in range(8):
                h = history_f[:, t]  # (batch, 9, 9)
                planes.append((h == 1).astype(self.dtype))   # our pieces at t
                planes.append((h == -1).astype(self.dtype))  # opp pieces at t

        # Valid positions mask (hexagonal board mask)
        valid_mask = (board_f != -2).astype(self.dtype)  # (batch, 9, 9)
        planes.append(valid_mask)

        # Scalar features broadcast to spatial planes
        marbles_norm = marbles_out.reshape(-1, 2) / 6.0
        our_marbles_plane = jnp.broadcast_to(
            marbles_norm[:, 0:1, None], (batch_size, 9, 9))  # (batch, 9, 9)
        opp_marbles_plane = jnp.broadcast_to(
            marbles_norm[:, 1:2, None], (batch_size, 9, 9))
        planes.append(our_marbles_plane.astype(self.dtype))
        planes.append(opp_marbles_plane.astype(self.dtype))

        if moves_count is not None:
            mc_plane = jnp.broadcast_to(
                (moves_count.reshape(-1, 1, 1) / 300.0), (batch_size, 9, 9))
            planes.append(mc_plane.astype(self.dtype))
        else:
            planes.append(jnp.zeros((batch_size, 9, 9), dtype=self.dtype))

        # Stack all planes: (batch, 9, 9, C) where C = 2 + 16 + 1 + 2 + 1 = 22
        x = jnp.stack(planes, axis=-1)  # (batch, 9, 9, 22)

        # Common trunk with LayerNorm (bfloat16)
        x = nn.Conv(self.num_filters, (3, 3), padding='SAME', dtype=self.dtype)(x)
        x = nn.LayerNorm(dtype=self.dtype)(x)
        x = nn.relu(x)

        # Residual tower (bfloat16)
        for _ in range(self.num_blocks):
            x = ResBlock(self.num_filters, dtype=self.dtype)(x)

        # === POLICY HEAD (bfloat16) ===
        policy_conv = nn.Conv(features=32, kernel_size=(1, 1), dtype=self.dtype)(x)
        policy_conv = nn.LayerNorm(dtype=self.dtype)(policy_conv)
        policy_conv = nn.relu(policy_conv)
        policy_flat = policy_conv.reshape((batch_size, -1))  # (batch, 9*9*32 = 2592)

        policy = nn.Dense(1024, dtype=self.dtype)(policy_flat)
        policy = nn.relu(policy)
        policy = nn.Dense(512, dtype=self.dtype)(policy)
        policy = nn.relu(policy)
        prior_logits = nn.Dense(self.num_actions, dtype=self.dtype)(policy)
        # Cast logits back to float32 for numerical stability in softmax/loss
        prior_logits = prior_logits.astype(jnp.float32)

        # === VALUE HEAD (float32 for stability) ===
        # Conv 1x1 to reduce spatial features (like policy head)
        value_conv = nn.Conv(features=1, kernel_size=(1, 1), dtype=jnp.float32)(x.astype(jnp.float32))
        value_conv = nn.LayerNorm(dtype=jnp.float32)(value_conv)
        value_conv = nn.relu(value_conv)
        value_flat = value_conv.reshape((batch_size, -1))  # (batch, 9*9*1 = 81)

        value = nn.Dense(256, dtype=jnp.float32)(value_flat)
        value = nn.relu(value)
        value = nn.Dense(1, dtype=jnp.float32)(value)
        value = nn.tanh(value)
        value = value.squeeze(-1)

        return prior_logits, value
