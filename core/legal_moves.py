import jax
import jax.numpy as jnp
import chex
from functools import partial
from typing import Dict, Tuple
from core.moves import move_single_marble , move_group_inline, move_group_parallel
from core.core import DIRECTIONS
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
    OPTIMIZED: Uses jax.lax.cond to skip validation for filtered-out moves
    """
    def check_move(move_idx):
        # Si le mouvement n'a pas passé le premier filtre, retourner False
        is_filtered = filtered_moves[move_idx]

        def compute_validity(_):
            """Only called if is_filtered == True"""
            positions = moves_index['positions'][move_idx]
            direction = moves_index['directions'][move_idx]
            move_type = moves_index['move_types'][move_idx]
            group_size = moves_index['group_sizes'][move_idx]

            # Définir les branches comme des fonctions
            # Seulement la branche sélectionnée sera évaluée (économie de calcul)
            def branch_single(_):
                _, success = move_single_marble(board, positions[0], direction, radius)
                return success

            def branch_parallel(_):
                _, success = move_group_parallel(board, positions, direction, group_size, radius)
                return success

            def branch_inline(_):
                _, success, _ = move_group_inline(board, positions, direction, group_size, radius)
                return success

            # Switch n'évalue QUE la branche sélectionnée (vs jnp.where qui évalue tout)
            return jax.lax.switch(
                move_type,
                [branch_single, branch_parallel, branch_inline],
                None
            )

        def return_false(_):
            """Only called if is_filtered == False"""
            return False

        # ✓ VRAIE OPTIMISATION: compute_validity n'est évaluée que si is_filtered == True
        return jax.lax.cond(is_filtered, compute_validity, return_false, None)

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


# ============================================================================
# OPTIMIZED VERSION - Precalculated Indices
# ============================================================================

def precalculate_legal_moves_indices(moves_index: Dict[str, chex.Array], radius: int = 4) -> Dict[str, chex.Array]:
    """
    Precalculate all indices needed for optimized legal move checking.
    This should be called ONCE during environment initialization.

    Returns:
        Dict with precalculated arrays for fast legal move checking
    """
    num_moves = len(moves_index['directions'])

    # Precalculate board indices for all source positions
    source_indices = []
    for move_idx in range(num_moves):
        positions = moves_index['positions'][move_idx]
        # Convert to board indices
        board_pos = positions + radius  # (3, 3)
        source_indices.append(board_pos)

    source_indices = jnp.array(source_indices)  # (1734, 3, 3)

    return {
        'source_indices': source_indices,
        'group_sizes': moves_index['group_sizes'],
        'move_types': moves_index['move_types'],
        'directions': moves_index['directions'],
        'positions': moves_index['positions']
    }


@partial(jax.jit, static_argnames=['radius'])
def get_legal_moves_optimized(board: chex.Array,
                               precalc_indices: Dict[str, chex.Array],
                               radius: int = 4) -> chex.Array:
    """
    OPTIMIZED version of get_legal_moves using precalculated indices.

    Improvements:
    - Pure vectorization instead of vmap
    - Precalculated board indices
    - Single pass for position checking

    NOTE: This is a FAST APPROXIMATION that only checks if source positions
    have the player's pieces. It does NOT validate full move legality
    (destination checks, push rules, etc.).

    For full validation, still use check_moves_validity.
    This function is meant for MCTS tree pruning where speed >> accuracy.

    Args:
        board: Game board (canonical, player to move is always 1)
        precalc_indices: Dict from precalculate_legal_moves_indices()
        radius: Board radius

    Returns:
        Boolean array (1734,) indicating probable legal moves
    """
    # Get precalculated source indices (1734, 3, 3) where last dim is [x, y, z]
    source_idx = precalc_indices['source_indices']

    # Vectorized check: does each position have our piece?
    has_pieces = board[source_idx[:, :, 0],
                       source_idx[:, :, 1],
                       source_idx[:, :, 2]]  # (1734, 3)

    has_pieces = (has_pieces == 1)

    # Create mask for group sizes
    group_sizes = precalc_indices['group_sizes']  # (1734,)
    positions_mask = jnp.arange(3)[None, :] < group_sizes[:, None]  # (1734, 3)

    # Check if all required positions have our pieces
    position_valid = jnp.all(jnp.where(positions_mask, has_pieces, True), axis=1)  # (1734,)

    return position_valid


@partial(jax.jit, static_argnames=['radius'])
def get_legal_moves_full_optimized(board: chex.Array,
                                    moves_index: Dict[str, chex.Array],
                                    precalc_indices: Dict[str, chex.Array],
                                    radius: int = 4) -> chex.Array:
    """
    FULLY OPTIMIZED version that combines fast position checking with full validation.

    This is the recommended version for production use:
    1. Fast position check (vectorized, precalculated indices)
    2. Full validation only for position-valid moves (reduced work)

    Args:
        board: Game board
        moves_index: Full moves index (for validation)
        precalc_indices: Precalculated indices
        radius: Board radius

    Returns:
        Boolean array (1734,) indicating truly legal moves
    """
    # Step 1: Fast position filtering (vectorized)
    position_filtered = get_legal_moves_optimized(board, precalc_indices, radius)

    # Step 2: Full validation only for position-valid moves
    # This uses the existing check_moves_validity function
    return check_moves_validity(board, moves_index, position_filtered, radius)


# ============================================================================
# VECTORIZED VERSION - No branches, pure tensor ops
# ============================================================================

def precalculate_vectorized_legal_moves_data(moves_index: Dict[str, chex.Array], radius: int = 4) -> Dict[str, jnp.ndarray]:
    """
    Precalculate ALL data needed for fully vectorized legal move checking.
    Called ONCE at environment initialization.

    Precalculates:
    - Source board indices for position filtering
    - Destination board indices for single/parallel moves
    - Head position and push chain positions for inline moves
    - In-bounds masks for all positions

    Returns:
        Dict with all precalculated arrays
    """
    positions = np.array(moves_index['positions'])    # (1734, 3, 3)
    directions = np.array(moves_index['directions'])  # (1734,)
    move_types = np.array(moves_index['move_types'])  # (1734,)
    group_sizes = np.array(moves_index['group_sizes'])  # (1734,)
    num_moves = len(directions)
    board_size = 2 * radius + 1  # 9

    # Direction vectors for each move
    DIRS_NP = np.array(DIRECTIONS)  # (6, 3)
    dir_vecs = DIRS_NP[directions]  # (1734, 3)

    # ── Source positions (for position filter) ──
    source_board_idx = positions + radius  # (1734, 3, 3)
    positions_mask = np.arange(3)[None, :] < group_sizes[:, None]  # (1734, 3)

    # ── Destinations for single/parallel: each marble + direction ──
    dest_positions = positions + dir_vecs[:, None, :]  # (1734, 3, 3)
    dest_board_idx = dest_positions + radius  # (1734, 3, 3)
    dest_in_bounds = np.all((dest_board_idx >= 0) & (dest_board_idx < board_size), axis=2)  # (1734, 3)
    dest_board_idx_clamped = np.clip(dest_board_idx, 0, board_size - 1)

    # ── Head position for inline moves ──
    # Head = marble furthest in the direction of movement
    scores = np.sum(positions * dir_vecs[:, None, :], axis=2)  # (1734, 3)
    scores_masked = np.where(positions_mask, scores, -1e9)
    head_indices = np.argmax(scores_masked, axis=1)  # (1734,)
    head_positions = positions[np.arange(num_moves), head_indices]  # (1734, 3)

    # ── Push chain: head+dir, head+2*dir, head+3*dir ──
    push_pos = np.stack([
        head_positions + 1 * dir_vecs,
        head_positions + 2 * dir_vecs,
        head_positions + 3 * dir_vecs,
    ], axis=1)  # (1734, 3, 3)
    push_board_idx = push_pos + radius  # (1734, 3, 3)
    push_in_bounds = np.all((push_board_idx >= 0) & (push_board_idx < board_size), axis=2)  # (1734, 3)
    push_board_idx_clamped = np.clip(push_board_idx, 0, board_size - 1)

    return {
        # Source position check
        'source_board_idx': jnp.array(source_board_idx),         # (1734, 3, 3)
        'positions_mask': jnp.array(positions_mask),             # (1734, 3)
        'group_sizes': jnp.array(group_sizes),                   # (1734,)
        'move_types': jnp.array(move_types),                     # (1734,)
        # Destination check (single/parallel)
        'dest_board_idx_clamped': jnp.array(dest_board_idx_clamped),  # (1734, 3, 3)
        'dest_in_bounds': jnp.array(dest_in_bounds),             # (1734, 3)
        # Push chain (inline)
        'push_board_idx_clamped': jnp.array(push_board_idx_clamped),  # (1734, 3, 3)
        'push_in_bounds': jnp.array(push_in_bounds),             # (1734, 3)
    }


@partial(jax.jit, static_argnames=['radius'])
def get_legal_moves_vectorized(board: chex.Array,
                                precalc: Dict[str, chex.Array],
                                radius: int = 4) -> chex.Array:
    """
    Fully vectorized legal move checking. No vmap, no cond, no switch.

    All move types (single, parallel, inline) are evaluated simultaneously
    using pure tensor operations, then the correct result is selected via
    jnp.where based on move_type.

    ~10-20x faster than the vmap+cond+switch version on TPU/GPU.

    Args:
        board: Game board (canonical, player to move is always 1)
        precalc: Dict from precalculate_vectorized_legal_moves_data()
        radius: Board radius

    Returns:
        Boolean array (1734,) indicating legal moves
    """
    source_idx = precalc['source_board_idx']      # (1734, 3, 3)
    pos_mask = precalc['positions_mask']           # (1734, 3)
    group_sizes = precalc['group_sizes']           # (1734,)
    move_types = precalc['move_types']             # (1734,)
    dest_idx = precalc['dest_board_idx_clamped']   # (1734, 3, 3)
    dest_in_bounds = precalc['dest_in_bounds']     # (1734, 3)
    push_idx = precalc['push_board_idx_clamped']   # (1734, 3, 3)
    push_in_bounds = precalc['push_in_bounds']     # (1734, 3)

    # ── 1. Position filter: do source positions have our marbles? ──
    source_vals = board[source_idx[:, :, 0],
                        source_idx[:, :, 1],
                        source_idx[:, :, 2]]  # (1734, 3)
    has_pieces = (source_vals == 1)
    position_valid = jnp.all(jnp.where(pos_mask, has_pieces, True), axis=1)  # (1734,)

    # ── 2. Destination values (used by single + parallel) ──
    dest_vals = board[dest_idx[:, :, 0],
                      dest_idx[:, :, 1],
                      dest_idx[:, :, 2]]  # (1734, 3)
    dest_empty = (dest_vals == 0) & dest_in_bounds  # (1734, 3)

    # Single: only first marble's destination matters
    single_valid = dest_empty[:, 0]  # (1734,)

    # Parallel: all group members' destinations must be empty
    parallel_valid = jnp.all(jnp.where(pos_mask, dest_empty, True), axis=1)  # (1734,)

    # ── 3. Inline push logic ──
    push_vals = board[push_idx[:, :, 0],
                      push_idx[:, :, 1],
                      push_idx[:, :, 2]]  # (1734, 3)

    # What's immediately ahead of the head?
    ahead_in_bounds = push_in_bounds[:, 0]            # (1734,)
    content_ahead = push_vals[:, 0]                    # (1734,)
    is_empty_ahead = (content_ahead == 0) & ahead_in_bounds
    is_opponent_ahead = (content_ahead == -1) & ahead_in_bounds

    # Count consecutive opponents (max 2 pushable)
    opp_1 = is_opponent_ahead                                                      # (1734,)
    opp_2 = opp_1 & (push_vals[:, 1] == -1) & push_in_bounds[:, 1]               # (1734,)
    n_opponents = opp_1.astype(jnp.int32) + opp_2.astype(jnp.int32)              # (1734,)

    # What's after the last opponent?
    # n_opp=1 → check position 1 (head+2*dir), n_opp=2 → check position 2 (head+3*dir)
    after_opp_val = jnp.where(n_opponents <= 1, push_vals[:, 1], push_vals[:, 2])
    after_opp_in_bounds = jnp.where(n_opponents <= 1, push_in_bounds[:, 1], push_in_bounds[:, 2])

    # Push valid if: opponent(s) can be pushed somewhere (empty or off-board)
    after_is_ok = (~after_opp_in_bounds) | (after_opp_val == 0)
    can_push = is_opponent_ahead & (n_opponents < group_sizes) & after_is_ok

    # Inline valid: destination ahead is empty OR valid push
    inline_valid = is_empty_ahead | can_push

    # ── 4. Select result by move type ──
    validity = jnp.where(
        move_types == 0, single_valid,
        jnp.where(move_types == 1, parallel_valid, inline_valid)
    )

    return position_valid & validity