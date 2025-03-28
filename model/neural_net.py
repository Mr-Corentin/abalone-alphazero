import flax.linen as nn
import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple
class ResBlock(nn.Module):
    """Bloc résiduel"""
    filters: int

    @nn.compact
    def __call__(self, x):
        y = nn.Conv(self.filters, (3, 3), padding='SAME')(x)
        y = nn.relu(y)
        y = nn.Conv(self.filters, (3, 3), padding='SAME')(y)
        return nn.relu(x + y)
class AbaloneModel(nn.Module):
    num_actions: int = 1734
    num_filters: int = 128
    num_blocks: int = 8

    @nn.compact
    def __call__(self, board, marbles_out):
        # Normalisation et reshape des entrées
        marbles_out = marbles_out.reshape(-1, 2) / 6.0  # Normalise à [0,1]
        board = board / 1.0  # Normalise les valeurs du plateau (-1, 0, 1)

        x = board[..., None]  # (batch, 9, 9, 1)

        # Tronc commun
        x = nn.Conv(self.num_filters, (3, 3), padding='SAME')(x)
        x = nn.relu(x)

        for _ in range(self.num_blocks):
            x = ResBlock(self.num_filters)(x)

        # Aplatir les features spatiales
        x_flat = x.reshape((x.shape[0], -1))

        # Concaténer avec l'information des billes sorties
        combined = jnp.concatenate([x_flat, marbles_out], axis=1)

        # Tête de politique
        policy = nn.Dense(1024)(combined)
        policy = nn.relu(policy)
        prior_logits = nn.Dense(self.num_actions)(policy)

        # Tête de valeur
        value = nn.Dense(256)(combined)
        value = nn.relu(value)
        value = nn.Dense(1)(value)
        value = nn.tanh(value)
        value = value.squeeze(-1)

        return prior_logits, value
