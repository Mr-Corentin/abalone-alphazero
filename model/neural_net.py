# import flax.linen as nn
# import jax
# import jax.numpy as jnp
# from functools import partial
# from typing import Tuple
# class ResBlock(nn.Module):
#     """Bloc résiduel"""
#     filters: int

#     @nn.compact
#     def __call__(self, x):
#         y = nn.Conv(self.filters, (3, 3), padding='SAME')(x)
#         y = nn.relu(y)
#         y = nn.Conv(self.filters, (3, 3), padding='SAME')(y)
#         return nn.relu(x + y)
# class AbaloneModel(nn.Module):
#     num_actions: int = 1734
#     num_filters: int = 128
#     num_blocks: int = 8

#     @nn.compact
#     def __call__(self, board, marbles_out):
#         # Normalisation et reshape des entrées
#         marbles_out = marbles_out.reshape(-1, 2) / 6.0  # Normalise à [0,1]
#         board = board / 1.0  # Normalise les valeurs du plateau (-1, 0, 1)

#         x = board[..., None]  # (batch, 9, 9, 1)

#         # Tronc commun
#         x = nn.Conv(self.num_filters, (3, 3), padding='SAME')(x)
#         x = nn.relu(x)

#         for _ in range(self.num_blocks):
#             x = ResBlock(self.num_filters)(x)

#         # Aplatir les features spatiales
#         x_flat = x.reshape((x.shape[0], -1))

#         # Concaténer avec l'information des billes sorties
#         combined = jnp.concatenate([x_flat, marbles_out], axis=1)

#         # Tête de politique
#         policy = nn.Dense(1024)(combined)
#         policy = nn.relu(policy)
#         prior_logits = nn.Dense(self.num_actions)(policy)

#         # Tête de valeur
#         value = nn.Dense(256)(combined)
#         value = nn.relu(value)
#         value = nn.Dense(1)(value)
#         value = nn.tanh(value)
#         value = value.squeeze(-1)

#         return prior_logits, value


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
    # train: bool # Pas besoin ici si on le passe en argument de __call__

    @nn.compact
    def __call__(self, board, marbles_out, train: bool): # Ajouter l'argument train
        # Normalisation et reshape des entrées
        marbles_out = marbles_out.reshape(-1, 2) / 6.0  # Normalise à [0,1]
        # board = board / 1.0  # Si board est déjà -1, 0, 1, cette ligne n'est pas strictement nécessaire
                                # mais ne pose pas de problème.
        # Assurons-nous que board est float pour la suite
        board = board.astype(jnp.float32)


        x = board[..., None]  # (batch, 9, 9, 1)

        # Tronc commun
        # Séquence : Conv -> BatchNorm -> ReLU
        x = nn.Conv(self.num_filters, (3, 3), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train, momentum=0.9)(x)
        x = nn.relu(x)

        for _ in range(self.num_blocks):
            x = ResBlock(self.num_filters, train=train)(x) # Passer l'argument train au ResBlock

        # Aplatir les features spatiales
        x_flat = x.reshape((x.shape[0], -1))

        # Concaténer avec l'information des billes sorties
        combined = jnp.concatenate([x_flat, marbles_out], axis=1)

        # Tête de politique
        # On peut aussi ajouter BN ici, mais c'est moins courant/critique que dans le tronc.
        # Pour la simplicité, on peut commencer sans.
        policy = nn.Dense(1024)(combined)
        # policy = nn.BatchNorm(use_running_average=not train, momentum=0.9)(policy) # Optionnel
        policy = nn.relu(policy)
        prior_logits = nn.Dense(self.num_actions)(policy)

        # Tête de valeur
        value = nn.Dense(256)(combined)
        # value = nn.BatchNorm(use_running_average=not train, momentum=0.9)(value) # Optionnel
        value = nn.relu(value)
        value = nn.Dense(1)(value)
        value = nn.tanh(value) # tanh est déjà une sorte de normalisation de sortie à [-1, 1]
        value = value.squeeze(-1)

        return prior_logits, value