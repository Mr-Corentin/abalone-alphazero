import jax
import jax.numpy as jnp
import chex
from functools import partial
from typing import Dict, Tuple
from core.moves import move_single_marble , move_group_inline, move_group_parallel
import numpy as np

@partial(jax.jit, static_argnames=['radius'])
def create_player_positions_mask(board: chex.Array, radius: int = 4) -> chex.Array:
    """
    Crée un masque booléen des positions où se trouvent les billes du joueur courant (toujours 1)
    """
    return board == 1

@partial(jax.jit, static_argnames=['radius'])
def filter_moves_by_positions(player_mask: chex.Array, 
                            moves_index: Dict[str, chex.Array],
                            radius: int = 4) -> chex.Array:
    """
    Crée un masque des mouvements dont toutes les positions de départ sont des billes du joueur
    """
    def check_move_positions(move_idx):
        positions = moves_index['positions'][move_idx]
        group_size = moves_index['group_sizes'][move_idx]
        
        board_positions = positions + radius
        
        has_pieces = player_mask[board_positions[:, 0],
                               board_positions[:, 1],
                               board_positions[:, 2]]
        
        # Créer masque pour nombre correct de positions
        positions_mask = jnp.arange(3) < group_size

       # True si toutes les positions requises ont nos pièces
        return jnp.all(jnp.where(positions_mask, has_pieces, True))
    
    return jax.vmap(check_move_positions)(jnp.arange(len(moves_index['directions'])))


@partial(jax.jit, static_argnames=['radius'])
def check_moves_validity(board: chex.Array,
                        moves_index: Dict[str, chex.Array],
                        filtered_moves: chex.Array,
                        radius: int = 4) -> chex.Array:
    """
    Vérifie quels mouvements filtrés sont légaux selon les règles du jeu
    """
    def check_move(move_idx):

        # Si le mouvement n'a pas passé le premier filtre, retourner False
        is_filtered = filtered_moves[move_idx]
        
        positions = moves_index['positions'][move_idx]
        direction = moves_index['directions'][move_idx]
        move_type = moves_index['move_types'][move_idx]
        group_size = moves_index['group_sizes'][move_idx]  
        
        # Vérifier les différents types de mouvements
        _, success_single = move_single_marble(board, positions[0], direction, radius)
        
        _, success_parallel = move_group_parallel(board, positions, direction, group_size, radius)
        
        _, success_inline, _ = move_group_inline(board, positions, direction, group_size, radius)
        
        # Sélectionner le bon résultat selon le type
        is_valid = jnp.where(
            move_type == 0, success_single,
            jnp.where(move_type == 1, success_parallel, success_inline)
        )

        return jnp.where(is_filtered, is_valid, False)
    
    return jax.vmap(check_move)(jnp.arange(len(moves_index['directions'])))


@partial(jax.jit, static_argnames=['radius'])
def get_legal_moves(board: chex.Array,
                   moves_index: Dict[str, chex.Array],
                   radius: int = 4) -> chex.Array:
    """
    Détermine tous les mouvements légaux pour le joueur courant (toujours 1)
    """
    position_filtered = filter_moves_by_positions(
        create_player_positions_mask(board),  
        moves_index,
        radius
    )
    
    return check_moves_validity(board, moves_index, position_filtered, radius)


def filter_moves_by_positions_for_eval(player_mask: chex.Array, 
                                     moves_index: Dict[str, chex.Array],
                                     radius: int = 4) -> chex.Array:
    """
    Version sans vmap de filter_moves_by_positions pour l'évaluation
    """
    num_moves = len(moves_index['directions'])
    results = np.zeros(num_moves, dtype=bool)
    
    for move_idx in range(num_moves):
        positions = moves_index['positions'][move_idx]
        group_size = moves_index['group_sizes'][move_idx]
        
        board_positions = positions + radius
        
        # Vérifier chaque position requise
        valid = True
        for i in range(group_size):
            pos = board_positions[i]
            if not player_mask[pos[0], pos[1], pos[2]]:
                valid = False
                break
        
        results[move_idx] = valid
    
    return results

def check_moves_validity_for_eval(board: chex.Array,
                                moves_index: Dict[str, chex.Array],
                                filtered_moves: np.ndarray,
                                radius: int = 4) -> np.ndarray:
    """
    Version sans vmap de check_moves_validity pour l'évaluation
    """
    num_moves = len(moves_index['directions'])
    results = np.zeros(num_moves, dtype=bool)
    
    for move_idx in range(num_moves):
        # Si le mouvement n'a pas passé le premier filtre, passer au suivant
        if not filtered_moves[move_idx]:
            continue
        
        positions = moves_index['positions'][move_idx]
        direction = moves_index['directions'][move_idx]
        move_type = moves_index['move_types'][move_idx]
        group_size = moves_index['group_sizes'][move_idx]
        
        # Vérifier selon le type de mouvement
        if move_type == 0:
            _, success = move_single_marble(board, positions[0], direction, radius)
            results[move_idx] = success
        elif move_type == 1:
            _, success = move_group_parallel(board, positions, direction, group_size, radius)
            results[move_idx] = success
        else:  # move_type == 2
            _, success, _ = move_group_inline(board, positions, direction, group_size, radius)
            results[move_idx] = success
    
    return results

def get_legal_moves_for_eval(board: chex.Array,
                           moves_index: Dict[str, chex.Array],
                           radius: int = 4) -> np.ndarray:
    """
    Version sans vmap de get_legal_moves pour l'évaluation
    """
    player_mask = (board == 1)
    position_filtered = filter_moves_by_positions_for_eval(player_mask, moves_index, radius)
    return check_moves_validity_for_eval(board, moves_index, position_filtered, radius)



@partial(jax.jit, static_argnames=['radius'])
def get_legal_moves_for_single(board, moves_index, radius=4):
    """
    Version jit mais non-vectorisée pour un seul état.
    Réutilise les fonctions internes de get_legal_moves mais sans vmap.
    """
    # Créer le masque du joueur
    player_mask = board == 1
    
    # Utiliser la fonction interne de filter_moves_by_positions mais l'appliquer 
    # à chaque mouvement un par un sans vmap
    def check_move_positions(move_idx):
        positions = moves_index['positions'][move_idx]
        group_size = moves_index['group_sizes'][move_idx]
        
        board_positions = positions + radius
        
        has_pieces = player_mask[board_positions[:, 0],
                               board_positions[:, 1],
                               board_positions[:, 2]]
        
        # Créer masque pour nombre correct de positions
        positions_mask = jnp.arange(3) < group_size

        # True si toutes les positions requises ont nos pièces
        return jnp.all(jnp.where(positions_mask, has_pieces, True))
    
    # Appliquer à chaque mouvement avec un scan ou cumulativement
    num_moves = len(moves_index['directions'])
    position_filtered = jnp.zeros(num_moves, dtype=jnp.bool_)
    
    # Utiliser jax.lax.fori_loop au lieu d'une boucle Python
    def body_fn(i, filtered):
        filtered = filtered.at[i].set(check_move_positions(i))
        return filtered
        
    position_filtered = jax.lax.fori_loop(
        0, num_moves, body_fn, position_filtered
    )
    
    # Fonction interne check_move de check_moves_validity
    def check_move(move_idx):
        # Si le mouvement n'a pas passé le premier filtre, retourner False
        is_filtered = position_filtered[move_idx]
        
        positions = moves_index['positions'][move_idx]
        direction = moves_index['directions'][move_idx]
        move_type = moves_index['move_types'][move_idx]
        group_size = moves_index['group_sizes'][move_idx]  
        
        # Vérifier les différents types de mouvements
        _, success_single = move_single_marble(board, positions[0], direction, radius)
        
        _, success_parallel = move_group_parallel(board, positions, direction, group_size, radius)
        
        _, success_inline, _ = move_group_inline(board, positions, direction, group_size, radius)
        
        # Sélectionner le bon résultat selon le type
        is_valid = jnp.where(
            move_type == 0, success_single,
            jnp.where(move_type == 1, success_parallel, success_inline)
        )

        return jnp.where(is_filtered, is_valid, False)
    
    # Appliquer à chaque mouvement
    legal_moves = jnp.zeros(num_moves, dtype=jnp.bool_)
    
    def body_fn2(i, legal):
        legal = legal.at[i].set(check_move(i))
        return legal
        
    legal_moves = jax.lax.fori_loop(
        0, num_moves, body_fn2, legal_moves
    )
    
    return legal_moves