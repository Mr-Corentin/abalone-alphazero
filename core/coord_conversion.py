import jax
import jax.numpy as jnp
import chex
from typing import Tuple
from functools import partial

@partial(jax.jit, static_argnames=['radius'])
def get_valid_positions(radius: int = 4):
    """
    Retourne une liste des positions valides dans le plateau hexagonal
    """
    return [
        # Ligne 0 (z = -4)
        (0,4,-4), (1,3,-4), (2,2,-4), (3,1,-4), (4,0,-4),
        # Ligne 1 (z = -3)
        (-1,4,-3), (0,3,-3), (1,2,-3), (2,1,-3), (3,0,-3), (4,-1,-3),
        # Ligne 2
        (-2,4,-2), (-1,3,-2), (0,2,-2), (1,1,-2), (2,0,-2), (3,-1,-2), (4,-2,-2),
        # Ligne 3
        (-3,4,-1), (-2,3,-1), (-1,2,-1), (0,1,-1), (1,0,-1), (2,-1,-1), (3,-2,-1), (4,-3,-1),
        #Ligne 4
        (-4,4,0), (-3,3,0), (-2,2,0), (-1,1,0), (0,0,0), (1,-1,0), (2,-2,0), (3,-3,0), (4,-4,0),
        # Ligne 5
        (-4,3,1), (-3,2,1), (-2,1,1), (-1,0,1), (0,-1,1), (1,-2,1), (2,-3,1), (3,-4,1),
        # Ligne 6
        (-4,2,2), (-3,1,2), (-2,0,2), (-1,-1,2), (0,-2,2), (1,-3,2), (2,-4,2),
        # Ligne 7
        (-4,1,3), (-3,0,3), (-2,-1,3), (-1,-2,3), (0,-3,3), (1,-4,3),
        # Ligne 8
        (-4,0,4), (-3,-1,4), (-2,-2,4), (-1,-3,4), (0,-4,4)
        # etc...
        # (ajouter toutes les positions valides)
    ]

@partial(jax.jit, static_argnames=['radius'])
def cube_to_2d(board_3d: chex.Array, radius: int = 4) -> chex.Array:
    """
    Convertit le plateau de la représentation cubique (3D) vers une grille 2D 9x9
    Version vectorisée pour vmap
    """
    # Utiliser -2 pour les cases invalides (au lieu de NaN)
    board_2d = jnp.full((9, 9), -2, dtype=board_3d.dtype)
    
    # Positions valides pré-calculées comme array JAX
    valid_positions = jnp.array([
        # Ligne 0 (z = -4)
        [0,4,-4], [1,3,-4], [2,2,-4], [3,1,-4], [4,0,-4],
        # Ligne 1 (z = -3)
        [-1,4,-3], [0,3,-3], [1,2,-3], [2,1,-3], [3,0,-3], [4,-1,-3],
        # Ligne 2
        [-2,4,-2], [-1,3,-2], [0,2,-2], [1,1,-2], [2,0,-2], [3,-1,-2], [4,-2,-2],
        # Ligne 3
        [-3,4,-1], [-2,3,-1], [-1,2,-1], [0,1,-1], [1,0,-1], [2,-1,-1], [3,-2,-1], [4,-3,-1],
        #Ligne 4
        [-4,4,0], [-3,3,0], [-2,2,0], [-1,1,0], [0,0,0], [1,-1,0], [2,-2,0], [3,-3,0], [4,-4,0],
        # Ligne 5
        [-4,3,1], [-3,2,1], [-2,1,1], [-1,0,1], [0,-1,1], [1,-2,1], [2,-3,1], [3,-4,1],
        # Ligne 6
        [-4,2,2], [-3,1,2], [-2,0,2], [-1,-1,2], [0,-2,2], [1,-3,2], [2,-4,2],
        # Ligne 7
        [-4,1,3], [-3,0,3], [-2,-1,3], [-1,-2,3], [0,-3,3], [1,-4,3],
        # Ligne 8
        [-4,0,4], [-3,-1,4], [-2,-2,4], [-1,-3,4], [0,-4,4]
    ])
    
    def convert_single_position(carry, position):
        board = carry
        x, y, z = position[0], position[1], position[2]
        
        # Convertir en indices de tableau 3D
        array_x = x + radius
        array_y = y + radius
        array_z = z + radius
        
        # Obtenir la valeur
        value = board_3d[array_x, array_y, array_z]
        
        # Calculer les coordonnées 2D
        row = z + radius
        col = x + 4
        
        # Mettre à jour le tableau
        new_board = board.at[row, col].set(value)
        return new_board, None
    
    # Utiliser scan pour appliquer la conversion séquentiellement
    final_board, _ = jax.lax.scan(convert_single_position, board_2d, valid_positions)
    
    return final_board

@partial(jax.jit, static_argnames=['radius'])
def compute_coord_map(radius: int = 4):
    """
    Pré-calcule la correspondance entre coordonnées 3D et 2D
    Returns:
        Dict avec :
        - indices_3d : positions dans le tableau 3D (shape (61, 3))
        - indices_2d : positions correspondantes dans le tableau 2D (shape (61, 2))
    """
    valid_positions = get_valid_positions(radius)
    n_positions = len(valid_positions)
    
    # Pré-calculer les indices 3D et 2D
    indices_3d = jnp.array([(x + radius, y + radius, z + radius) 
                           for x, y, z in valid_positions])
    
    indices_2d = jnp.array([(z + radius, x + 4) 
                           for x, y, z in valid_positions])
    
    return {'indices_3d': indices_3d, 'indices_2d': indices_2d}


@partial(jax.jit, static_argnames=['radius'])
def prepare_input(board_3d: jnp.ndarray, 
                  history_3d: jnp.ndarray,
                  actual_player: jnp.ndarray,
                  our_marbles_out: jnp.ndarray, 
                  opponent_marbles_out: jnp.ndarray, 
                  radius: int = 4):
    """
    Prépare les entrées pour le réseau avec support du batching et historique canonique.

    Args:
        board_3d: Shape (batch_size, x, y, z) en batch ou (x, y, z) sans batch
        history_3d: Shape (batch_size, 8, x, y, z) en batch ou (8, x, y, z) sans batch
        actual_player: Shape (batch_size,) ou scalaire - joueur actuel (1 ou -1)
        our_marbles_out: Shape (batch_size,) ou scalaire
        opponent_marbles_out: Shape (batch_size,) ou scalaire
        radius: Rayon du plateau
        
    Returns:
        board_2d: Shape (batch_size, 9, 9, 9) - board actuel + 8 positions historiques (canoniques)
        marbles_out: Shape (batch_size, 2) - billes sorties [nous, adversaire]
    """
    # Détecter si on a un batch ou un seul exemple
    is_batched = board_3d.ndim > 3  # car board_3d est déjà en 3D

    if not is_batched:
        # Si single example, ajouter dimension de batch
        board_3d = board_3d[None, ...]  # (1, x, y, z)
        history_3d = history_3d[None, ...]  # (1, 8, x, y, z)
        actual_player = jnp.array([actual_player])  # (1,)
        our_marbles_out = jnp.array([our_marbles_out])
        opponent_marbles_out = jnp.array([opponent_marbles_out])

    # Convertir le board actuel : (batch_size, x, y, z) -> (batch_size, 9, 9)
    # Le board actuel est déjà canonique
    current_board_2d = jax.vmap(lambda b: cube_to_2d(b, radius))(board_3d)
    
    # Appliquer la transformation canonique à l'historique
    # L'historique contient les positions "réelles", il faut les adapter au joueur actuel
    def canonicalize_history_for_player(history, player):
        """Transforme l'historique pour que le joueur actuel voie ses pièces comme 1"""
        return jnp.where(player == 1, history, -history)
    
    # Appliquer la canonicalisation à chaque élément du batch
    canonical_history = jax.vmap(canonicalize_history_for_player)(history_3d, actual_player)
    
    # Convertir l'historique canonique : (batch_size, 8, x, y, z) -> (batch_size, 8, 9, 9)
    history_2d = jax.vmap(jax.vmap(lambda h: cube_to_2d(h, radius)))(canonical_history)
    
    # Empiler board actuel + historique en canaux
    # current_board_2d: (batch_size, 9, 9) -> (batch_size, 9, 9, 1)
    current_board_2d = current_board_2d[..., None]
    
    # history_2d: (batch_size, 8, 9, 9) -> (batch_size, 9, 9, 8)
    history_2d = jnp.transpose(history_2d, (0, 2, 3, 1))
    
    # Concaténer : (batch_size, 9, 9, 1) + (batch_size, 9, 9, 8) = (batch_size, 9, 9, 9)
    board_with_history = jnp.concatenate([current_board_2d, history_2d], axis=-1)

    # Remplacer les NaN par -2 et convertir en int8
    board_with_history = jnp.nan_to_num(board_with_history, -2.0)
    board_with_history = board_with_history.astype(jnp.int8)

    # Créer le vecteur des billes sorties (batch_size, 2)
    marbles_out = jnp.stack([our_marbles_out, opponent_marbles_out], axis=-1)

    return board_with_history, marbles_out


# Fonction de compatibilité pour l'ancien code
@partial(jax.jit, static_argnames=['radius'])
def prepare_input_legacy(board_3d: jnp.ndarray, our_marbles_out: jnp.ndarray, opponent_marbles_out: jnp.ndarray, radius: int = 4):
    """
    Version legacy de prepare_input sans historique pour compatibilité.
    """
    # Détecter si on a un batch ou un seul exemple
    is_batched = board_3d.ndim > 3  # car board_3d est déjà en 3D

    if not is_batched:
        # Si single example, ajouter dimension de batch
        board_3d = board_3d[None, ...]  # (1, x, y, z)
        our_marbles_out = jnp.array([our_marbles_out])
        opponent_marbles_out = jnp.array([opponent_marbles_out])

    # Utiliser vmap pour appliquer cube_to_2d sur chaque élément du batch
    board_2d = jax.vmap(lambda b: cube_to_2d(b, radius))(board_3d)

    # Remplacer les NaN par -2 et convertir en int8
    board_2d = jnp.nan_to_num(board_2d, -2.0)
    board_2d = board_2d.astype(jnp.int8)

    # Créer le vecteur des billes sorties (batch_size, 2)
    marbles_out = jnp.stack([our_marbles_out, opponent_marbles_out], axis=-1)

    return board_2d, marbles_out


def display_2d_board(board_2d: chex.Array):
    """
    Affiche le plateau 2D
    """
    print("\nPlateau 2D:")
    for row in range(9):
        indent = abs(4 - row) if row <= 4 else 0
        print(" " * indent, end="")
        
        for col in range(9):
            value = board_2d[row, col]
            if jnp.isnan(value):
                print(" ", end=" ")
            elif value == -2:
                print(" ", end=" ")
            elif value == 1:
                print("●", end=" ")
            elif value == -1:
                print("○", end=" ")
            else:
                print("·", end=" ")
        print()

