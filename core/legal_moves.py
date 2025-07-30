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
    Create boolean mask of positions where current player's marbles are located (always 1)
    """
    return board == 1

@partial(jax.jit, static_argnames=['radius'])
def filter_moves_by_positions(player_mask: chex.Array, 
                            moves_index: Dict[str, chex.Array],
                            radius: int = 4) -> chex.Array:
    """
    Create mask of moves where all start positions have player marbles
    """
    def check_move_positions(move_idx):
        positions = moves_index['positions'][move_idx]
        group_size = moves_index['group_sizes'][move_idx]
        
        board_positions = positions + radius
        
        has_pieces = player_mask[board_positions[:, 0],
                               board_positions[:, 1],
                               board_positions[:, 2]]
        
        # Create mask for correct number of positions
        positions_mask = jnp.arange(3) < group_size

        # True if all required positions have our pieces
        return jnp.all(jnp.where(positions_mask, has_pieces, True))
    
    return jax.vmap(check_move_positions)(jnp.arange(len(moves_index['directions'])))


@partial(jax.jit, static_argnames=['radius'])
def check_moves_validity(board: chex.Array,
                        moves_index: Dict[str, chex.Array],
                        filtered_moves: chex.Array,
                        radius: int = 4) -> chex.Array:
    """
    Check which filtered moves are legal according to game rules
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
    Determine all legal moves for current player (always 1)
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
    Non-vmap version of filter_moves_by_positions for evaluation
    """
    num_moves = len(moves_index['directions'])
    results = np.zeros(num_moves, dtype=bool)
    
    for move_idx in range(num_moves):
        positions = moves_index['positions'][move_idx]
        group_size = moves_index['group_sizes'][move_idx]
        
        board_positions = positions + radius
        
        # Check each required position
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
    Non-vmap version of check_moves_validity for evaluation
    """
    num_moves = len(moves_index['directions'])
    results = np.zeros(num_moves, dtype=bool)
    
    for move_idx in range(num_moves):
        # If move didn't pass first filter, skip to next
        if not filtered_moves[move_idx]:
            continue
        
        positions = moves_index['positions'][move_idx]
        direction = moves_index['directions'][move_idx]
        move_type = moves_index['move_types'][move_idx]
        group_size = moves_index['group_sizes'][move_idx]
        
        # Check based on move type
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
    Non-vmap version of get_legal_moves for evaluation
    """
    player_mask = (board == 1)
    position_filtered = filter_moves_by_positions_for_eval(player_mask, moves_index, radius)
    return check_moves_validity_for_eval(board, moves_index, position_filtered, radius)



@partial(jax.jit, static_argnames=['radius'])
def get_legal_moves_for_single(board, moves_index, radius=4):
    """
    JIT version but non-vectorized for single state.
    Reuses internal functions from get_legal_moves but without vmap.
    """
    # Create player mask
    player_mask = board == 1
    
    # Use internal function from filter_moves_by_positions but apply
    # to each move one by one without vmap
    def check_move_positions(move_idx):
        positions = moves_index['positions'][move_idx]
        group_size = moves_index['group_sizes'][move_idx]
        
        board_positions = positions + radius
        
        has_pieces = player_mask[board_positions[:, 0],
                               board_positions[:, 1],
                               board_positions[:, 2]]
        
        # Create mask for correct number of positions
        positions_mask = jnp.arange(3) < group_size

        # True if all required positions have our pieces
        return jnp.all(jnp.where(positions_mask, has_pieces, True))
    
    # Apply to each move with scan or cumulatively
    num_moves = len(moves_index['directions'])
    position_filtered = jnp.zeros(num_moves, dtype=jnp.bool_)
    
    # Use jax.lax.fori_loop instead of Python loop
    def body_fn(i, filtered):
        filtered = filtered.at[i].set(check_move_positions(i))
        return filtered
        
    position_filtered = jax.lax.fori_loop(
        0, num_moves, body_fn, position_filtered
    )
    
    # Internal check_move function from check_moves_validity
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
    
    # Apply to each move
    legal_moves = jnp.zeros(num_moves, dtype=jnp.bool_)
    
    def body_fn2(i, legal):
        legal = legal.at[i].set(check_move(i))
        return legal
        
    legal_moves = jax.lax.fori_loop(
        0, num_moves, body_fn2, legal_moves
    )
    
    return legal_moves