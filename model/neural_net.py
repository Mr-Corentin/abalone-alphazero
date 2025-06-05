import flax.linen as nn
import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, Any 

class ResBlock(nn.Module):
    """Bloc résiduel avec Batch Normalization"""
    filters: int
    train: bool # Ajouter l'argument train pour contrôler la BatchNorm

    @nn.compact
    def __call__(self, x):
        # Séquence : Conv -> BatchNorm -> ReLU
        y = nn.Conv(self.filters, (3, 3), padding='SAME', use_bias=False)(x)
        y = nn.BatchNorm(use_running_average=not self.train, momentum=0.9)(y)
        y = nn.relu(y)

        y = nn.Conv(self.filters, (3, 3), padding='SAME', use_bias=False)(y)
        y = nn.BatchNorm(use_running_average=not self.train, momentum=0.9)(y)
        output = nn.relu(x + y)
        return output

class AbaloneModel(nn.Module):
    num_actions: int = 1734
    num_filters: int = 128
    num_blocks: int = 8

    @nn.compact
    def __call__(self, board, marbles_out, train: bool):
        """
        Forward pass du réseau avec support de l'historique
        
        Args:
            board: Plateau avec historique, shape (batch, 9, 9, 9)
                   Canal 0: Position actuelle
                   Canaux 1-8: Historique (du plus récent au plus ancien)
            marbles_out: Billes sorties, shape (batch, 2)
            train: Mode entraînement pour BatchNorm
        """
        # Normalisation et reshape des entrées
        marbles_out = marbles_out.reshape(-1, 2) / 6.0  # Normalise à [0,1]
        
        # Board contient déjà l'historique avec 9 canaux : (batch, 9, 9, 9)
        # Plus besoin d'ajouter une dimension avec [..., None]
        board = board.astype(jnp.float32)
        x = board  # Shape: (batch, 9, 9, 9)

        # Tronc commun - la première convolution prend maintenant 9 canaux
        # Séquence : Conv -> BatchNorm -> ReLU
        x = nn.Conv(self.num_filters, (3, 3), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.9)(x)
        x = nn.relu(x)

        # Blocs résiduels - inchangés
        for _ in range(self.num_blocks):
            x = ResBlock(self.num_filters, train=train)(x)

        # Aplatir les features spatiales
        x_flat = x.reshape((x.shape[0], -1))

        # Concaténer avec l'information des billes sorties
        combined = jnp.concatenate([x_flat, marbles_out], axis=1)

        # Tête de politique - inchangée
        policy = nn.Dense(1024)(combined)
        policy = nn.relu(policy)
        prior_logits = nn.Dense(self.num_actions)(policy)

        # Tête de valeur - inchangée
        value = nn.Dense(256)(combined)
        value = nn.relu(value)
        value = nn.Dense(1)(value)
        value = nn.tanh(value)
        value = value.squeeze(-1)

        return prior_logits, value