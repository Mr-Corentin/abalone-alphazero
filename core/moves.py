import jax
import jax.numpy as jnp
import chex
from functools import partial
from core.core import Direction, DIRECTIONS
from core.positions import get_valid_neighbors, analyze_group

# Dans moves.py
#@partial(jax.jit, static_argnames=['radius', 'direction_idx'])
@partial(jax.jit)
def move_single_marble(board: chex.Array, 
                               position: chex.Array, 
                               direction_idx: int,
                               radius: int = 4) -> tuple[chex.Array, bool]:
    """
    Version optimisée du mouvement d'une seule bille
    Le joueur courant est toujours représenté par 1
    """
    # 1. Calcul des positions en une fois
    start_pos_idx = position + radius
    dir_vec = DIRECTIONS[direction_idx]
    new_pos_idx = start_pos_idx + dir_vec

    # 2. Vérification des conditions en une seule lecture du board
    start_value = board[start_pos_idx[0], start_pos_idx[1], start_pos_idx[2]]
    has_marble = start_value == 1
    
    # 3. Vérification de validité combinée
    valid_mask, _ = get_valid_neighbors(position, board, radius)
    dest_value = board[new_pos_idx[0], new_pos_idx[1], new_pos_idx[2]]
    
    # 4. Validation finale
    success = has_marble & valid_mask[direction_idx] & (dest_value == 0)
    
    # 5. Mise à jour du plateau en une seule opération
    new_board = jnp.where(success,
                         board.at[start_pos_idx[0], start_pos_idx[1], start_pos_idx[2]].set(0)
                              .at[new_pos_idx[0], new_pos_idx[1], new_pos_idx[2]].set(1),
                         board)
    
    return new_board, success


#@partial(jax.jit, static_argnames=['radius'])
@partial(jax.jit)
def move_group_parallel(board: chex.Array, 
                       positions: chex.Array,
                       direction: int,
                       group_size: int,
                       radius: int = 4) -> tuple[chex.Array, bool]:
    """
    Version optimisée du mouvement parallèle
    Le joueur courant est toujours représenté par 1
    """
    # 1. Préparation des positions et masques
    positions_mask = jnp.arange(positions.shape[0]) < group_size
    valid_positions = jnp.where(positions_mask[:, None], positions, 0)
    
    # 2. Analyse du groupe et validation
    is_valid, _, parallel_dirs = analyze_group(valid_positions, board, group_size, radius)
    is_valid_direction = parallel_dirs[direction]

    # 3. Calcul des positions
    dir_vec = DIRECTIONS[direction]
    start_positions = valid_positions + radius
    new_positions = jnp.where(positions_mask[:, None], positions + dir_vec, 0)
    board_positions = new_positions + radius

    # 4. Vérification des limites et validité
    in_bounds = jnp.all((board_positions >= 0) & (board_positions < board.shape[0]))
    actual_board_state = jnp.where(
        positions_mask,
        board[board_positions[:, 0], board_positions[:, 1], board_positions[:, 2]],
        0
    )
    destinations_empty = jnp.all(actual_board_state == 0)
    
    # 5. Validation finale
    success = is_valid & is_valid_direction & in_bounds & destinations_empty

    # 6. Préparation des indices pour la mise à jour
    start_indices = (
        jnp.where(positions_mask, start_positions[:, 0], 0),
        jnp.where(positions_mask, start_positions[:, 1], 0),
        jnp.where(positions_mask, start_positions[:, 2], 0)
    )
    board_indices = (
        jnp.where(positions_mask, board_positions[:, 0], 0),
        jnp.where(positions_mask, board_positions[:, 1], 0),
        jnp.where(positions_mask, board_positions[:, 2], 0)
    )
    
    # 7. Mise à jour du plateau
    updated_board = board.at[start_indices].set(
        jnp.where(positions_mask, 0., board[start_indices])
    )
    updated_board = updated_board.at[board_indices].set(
        jnp.where(positions_mask, 1., updated_board[board_indices])
    )

    # 8. Finalisation
    new_board = jnp.where(success, updated_board, board)
    
    return new_board, success

#@partial(jax.jit, static_argnames=['radius'])
@partial(jax.jit)
def move_group_inline(board: chex.Array, 
                              positions: chex.Array,
                              direction: int,
                              group_size: int,
                              radius: int = 4) -> tuple[chex.Array, bool, int]:
    """
    Version optimisée du mouvement inline
    Gère à la fois les déplacements simples et les poussées
    Le joueur courant est toujours représenté par 1, l'adversaire par -1
    """
    # 1. Préparation optimisée des positions et masques
    positions_mask = jnp.arange(positions.shape[0]) < group_size
    valid_positions = jnp.where(positions_mask[:, None], positions, 0)
    dir_vec = DIRECTIONS[direction]
    
    # 2. Analyse du groupe et validation de base
    is_valid, inline_dirs, _ = analyze_group(valid_positions, board, group_size, radius)
    is_valid_direction = inline_dirs[direction]

    # 3. Calcul de la tête et des positions
    scores = jnp.sum(valid_positions * dir_vec, axis=1)
    scores = jnp.where(positions_mask, scores, -jnp.inf)
    head_index = jnp.argmax(scores)
    head_position = valid_positions[head_index]
    
    # 4. Vérification du contenu devant la tête du groupe
    position_ahead = head_position + dir_vec
    position_ahead_board = position_ahead + radius
    content_ahead = board[position_ahead_board[0], 
                        position_ahead_board[1], 
                        position_ahead_board[2]]
    is_blocking_friend_ahead = content_ahead == 1
    
    # 5. Calcul des nouvelles positions
    start_positions = valid_positions + radius
    new_positions = jnp.where(positions_mask[:, None], positions + dir_vec, 0)
    board_positions = new_positions + radius

    # 6. Vérification des limites et validité
    in_bounds = jnp.all((board_positions >= 0) & (board_positions < board.shape[0]))
    actual_board_state = jnp.where(
        positions_mask,
        board[board_positions[:, 0], board_positions[:, 1], board_positions[:, 2]],
        0
    )
    is_empty = actual_board_state == 0
    is_moving_piece = positions_mask  # Gardé simple pour permettre les poussées
    is_valid_position = is_empty | is_moving_piece

    # 7. Early check pour la présence de bille adverse
    push_positions = head_position + jnp.array([dir_vec * (i + 1) for i in range(3)])
    push_board_positions = push_positions + radius
    push_contents = board[push_board_positions[:, 0],
                        push_board_positions[:, 1],
                        push_board_positions[:, 2]]
    has_opposing = push_contents[0] == -1

    # 8. Préparation du mouvement simple
    start_indices = (
        jnp.where(positions_mask, start_positions[:, 0], 0),
        jnp.where(positions_mask, start_positions[:, 1], 0),
        jnp.where(positions_mask, start_positions[:, 2], 0)
    )
    board_indices = (
        jnp.where(positions_mask, board_positions[:, 0], 0),
        jnp.where(positions_mask, board_positions[:, 1], 0),
        jnp.where(positions_mask, board_positions[:, 2], 0)
    )
    
    # Mise à jour simple du plateau
    updated_board = board.at[start_indices].set(
        jnp.where(positions_mask, 0., board[start_indices])
    )
    updated_board = updated_board.at[board_indices].set(
        jnp.where(positions_mask, 1., updated_board[board_indices])
    )

    # 9. Calcul de la validité de la poussée
    push_in_bounds = jnp.all((push_board_positions >= 0) & 
                            (push_board_positions < board.shape[0]), axis=1)
    is_opposing = push_contents == -1
    n_opposing = jnp.where(has_opposing, 
                          jnp.argmin(is_opposing),
                          0)
    no_friendly_behind = (n_opposing == 0) | (push_contents[n_opposing] != 1)
    can_push = jnp.where(has_opposing,
                        (n_opposing > 0) & (n_opposing < group_size) & no_friendly_behind,
                        False)
    
    # 10. Validation finale avec la condition combinée
    destinations_valid = jnp.all(is_valid_position) & ~is_blocking_friend_ahead & ((content_ahead == 0) | (has_opposing & can_push))
    success = is_valid & is_valid_direction & in_bounds & destinations_valid

    # Reste du code identique pour la mise à jour avec poussée...
    push_dest_positions = push_positions + dir_vec
    push_dest_board_positions = push_dest_positions + radius
    dest_indices = (push_dest_board_positions[:, 0], 
                   push_dest_board_positions[:, 1], 
                   push_dest_board_positions[:, 2])
    valid_push = jnp.array([True, True, False]) 
    valid_push = valid_push & (jnp.arange(3) < n_opposing) & push_in_bounds

    push_updated_board = jnp.where(
        has_opposing & success,
        updated_board.at[dest_indices].set(
            jnp.where(valid_push, -1, updated_board[dest_indices])
        ),
        updated_board
    )

    # Calcul des billes sorties
    push_possibility = jnp.array([False, True, True])
    out_of_bounds = push_possibility & ~push_in_bounds
    opposing_mask = jnp.roll(jnp.arange(3) < n_opposing, shift=1)
    potential_exits = jnp.where(opposing_mask, out_of_bounds, False)
    billes_sorties = jnp.where(has_opposing & success, jnp.sum(potential_exits), 0)

    # Finalisation
    new_board = jnp.where(success, push_updated_board, board)

    return new_board, success, billes_sorties
