import flax.linen as nn
import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple


# Board cells carry -2 when they are outside the hexagon (see cube_to_2d).
OFF_BOARD = -2


def _norm(num_features: int):
    """
    GroupNorm rather than BatchNorm.

    BatchNorm would need its running statistics synchronised across devices and
    hosts under pmap, and would add a mutable collection to thread through every
    apply() call. GroupNorm is batch-independent, so training and self-play (which
    runs at a completely different batch size) see exactly the same function.
    """
    return nn.GroupNorm(num_groups=min(32, num_features))


class ResBlock(nn.Module):
    """Residual block: conv -> norm -> relu -> conv -> norm -> (+x) -> relu"""
    filters: int

    @nn.compact
    def __call__(self, x):
        y = nn.Conv(self.filters, (3, 3), padding='SAME', use_bias=False)(x)
        y = _norm(self.filters)(y)
        y = nn.relu(y)
        y = nn.Conv(self.filters, (3, 3), padding='SAME', use_bias=False)(y)
        y = _norm(self.filters)(y)
        return nn.relu(x + y)


class AbaloneModel(nn.Module):
    """
    AlphaZero-style network for Abalone.

    Input encoding: one signed plane per position was replaced by binary planes
    (my marbles / opponent marbles), plus an explicit on-board mask, so the
    network no longer has to learn that "-2" means "this cell does not exist"
    and that the sign of a cell encodes ownership.

    Head sizing follows AlphaZero: a 1x1 convolution collapses the channel
    dimension before the dense layers. Flattening the full 9x9x128 trunk straight
    into Dense(1024) put ~15M of 18M parameters in the heads.
    """
    num_actions: int = 1734
    num_filters: int = 128
    num_blocks: int = 8
    policy_channels: int = 32
    value_channels: int = 1
    value_hidden: int = 256

    def _planes(self, board, history, marbles_out):
        """Build the (batch, 9, 9, C) input stack."""
        board = board.astype(jnp.float32)
        planes = [
            (board == 1).astype(jnp.float32),          # my marbles
            (board == -1).astype(jnp.float32),         # opponent marbles
            (board != OFF_BOARD).astype(jnp.float32),  # playable cells
        ]

        if history is not None:
            history = history.astype(jnp.float32)              # (batch, 8, 9, 9)
            history = jnp.transpose(history, (0, 2, 3, 1))     # (batch, 9, 9, 8)
            planes.append((history == 1).astype(jnp.float32))
            planes.append((history == -1).astype(jnp.float32))

        x = jnp.concatenate([p if p.ndim == 4 else p[..., None] for p in planes], axis=-1)

        # Marbles out, broadcast as constant planes so the trunk can use them.
        scalars = marbles_out.reshape(-1, 2).astype(jnp.float32) / 6.0
        scalars = jnp.broadcast_to(scalars[:, None, None, :], x.shape[:3] + (2,))
        return jnp.concatenate([x, scalars], axis=-1)

    @nn.compact
    def __call__(self, board, marbles_out, history=None):
        x = self._planes(board, history, marbles_out)

        # Common trunk
        x = nn.Conv(self.num_filters, (3, 3), padding='SAME', use_bias=False)(x)
        x = _norm(self.num_filters)(x)
        x = nn.relu(x)

        for _ in range(self.num_blocks):
            x = ResBlock(self.num_filters)(x)

        # Policy head: 1x1 conv -> flatten -> dense
        p = nn.Conv(self.policy_channels, (1, 1), use_bias=False)(x)
        p = _norm(self.policy_channels)(p)
        p = nn.relu(p)
        prior_logits = nn.Dense(self.num_actions)(p.reshape((p.shape[0], -1)))

        # Value head: 1x1 conv -> flatten -> dense -> tanh
        v = nn.Conv(self.value_channels, (1, 1), use_bias=False)(x)
        v = _norm(self.value_channels)(v)
        v = nn.relu(v)
        v = nn.Dense(self.value_hidden)(v.reshape((v.shape[0], -1)))
        v = nn.relu(v)
        value = nn.tanh(nn.Dense(1)(v)).squeeze(-1)

        return prior_logits, value
