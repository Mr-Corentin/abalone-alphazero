import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Tuple, List, Dict, Any

from environment.env import AbaloneStateNonCanonical, AbaloneEnvNonCanonical
from core.core import DIRECTIONS
from core.board import display_board, create_custom_board


@partial(jax.jit, static_argnames=['radius', 'board_center_weight'])
def geometric_score(state: AbaloneStateNonCanonical, player: int, radius: int = 4, board_center_weight: float = 0.3) -> float:
    """
    Implementation of the geometric evaluation function for non-canonical environment.
    
    Args:
        state: Current game state (non-canonical)
        player: The player (1 for black, -1 for white)
        radius: Board radius
        board_center_weight: Weight of the board center (between 0 and 1)
    
    Returns:
        Geometric score (higher value is better for the specified player)
    """
    # 1. Calculate centers of mass for black and white marbles
    size = 2 * radius + 1
    x_grid, y_grid, z_grid = jnp.mgrid[0:size, 0:size, 0:size]
    x_grid, y_grid, z_grid = x_grid - radius, y_grid - radius, z_grid - radius
    
    # Masks for each player's marbles (black=1, white=-1 always)
    black_mask = (state.board == 1)
    white_mask = (state.board == -1)
    
    # Count marbles
    black_count = jnp.sum(black_mask)
    white_count = jnp.sum(white_mask)
    
    # Centers of mass (with protection against division by zero)
    black_center_x = jnp.sum(x_grid * black_mask) / jnp.maximum(black_count, 1)
    black_center_y = jnp.sum(y_grid * black_mask) / jnp.maximum(black_count, 1)
    black_center_z = jnp.sum(z_grid * black_mask) / jnp.maximum(black_count, 1)
    
    white_center_x = jnp.sum(x_grid * white_mask) / jnp.maximum(white_count, 1)
    white_center_y = jnp.sum(y_grid * white_mask) / jnp.maximum(white_count, 1)
    white_center_z = jnp.sum(z_grid * white_mask) / jnp.maximum(white_count, 1)
    
    # 2. Create reference point R (weighted average of centers)
    board_center = jnp.array([0., 0., 0.])
    black_center = jnp.array([black_center_x, black_center_y, black_center_z])
    white_center = jnp.array([white_center_x, white_center_y, white_center_z])
    R = ((black_center + white_center) / 2) * (1 - board_center_weight) + board_center * board_center_weight
    
    # 3. Calculate distances
    coords = jnp.stack([x_grid, y_grid, z_grid], axis=-1)
    
    def hex_manhattan_distance(pos, ref):
        return jnp.sum(jnp.abs(pos - ref)) / 2
    
    distances_flat = jax.vmap(lambda pos: hex_manhattan_distance(pos, R))(coords.reshape(-1, 3))
    distances = distances_flat.reshape(size, size, size)
    
    # Distances for each player
    out_distance = float(radius * 2)
    black_distances = jnp.sum(distances * black_mask)
    white_distances = jnp.sum(distances * white_mask)
    
    # Add distances for eliminated marbles
    black_distances += state.black_out * out_distance
    white_distances += state.white_out * out_distance
    
    # 4. Calculate difference based on player
    score = jnp.where(player == 1, 
                      white_distances - black_distances,  # For black (1)
                      black_distances - white_distances)  # For white (-1)
    
    return score


def order_moves(state: AbaloneStateNonCanonical, legal_indices, env: AbaloneEnvNonCanonical, radius=4):
    """
    Order legal moves by their estimated quality to improve alpha-beta pruning.
    
    Args:
        state: Current game state
        legal_indices: Indices of legal moves
        env: Game environment
        radius: Board radius
        
    Returns:
        List of move indices ordered by estimated quality (best first)
    """
    scores = []
    
    for idx in legal_indices:
        # Get move information
        move_type = env.moves_index['move_types'][idx]
        group_size = env.moves_index['group_sizes'][idx]
        
        # Base score by move type
        if move_type == 0:  # Single marble
            base_score = 1.0
        elif move_type == 1:  # Parallel group
            base_score = 3.0
        else:  # Inline group (potentially a push)
            base_score = 5.0
            
        # Bonus for group size
        size_bonus = float(group_size - 1) * 2.0
        
        # Simulate the move
        next_state = env.step(state, idx)
        
        # Calculate geometric score from current player's perspective
        geo_score = geometric_score(next_state, state.current_player, radius)
        
        # Check for marbles pushed out
        if state.current_player == 1:  # Black playing
            marbles_out = next_state.white_out - state.white_out
        else:  # White playing
            marbles_out = next_state.black_out - state.black_out
        
        # Give maximum priority to moves that push marbles out
        if marbles_out > 0:
            if state.current_player == 1 and state.white_out >= 5:
                final_score = 500  # Very high score for imminent victory
            elif state.current_player == -1 and state.black_out >= 5:
                final_score = 500
            else:
                final_score = 30
        else:
            # Bonus for inline moves that push opponent marbles
            push_bonus = 0.0
            if move_type == 2:
                # With non-canonical representation, it's simpler:
                # Enemy marbles are always of the opposite color
                enemy_value = -1 if state.current_player == 1 else 1
                before_enemy_count = (state.board == enemy_value).sum()
                after_enemy_count = (next_state.board == enemy_value).sum()
                
                # Check if board has changed for enemy marbles
                if before_enemy_count != after_enemy_count or \
                   ((state.board == enemy_value) != (next_state.board == enemy_value)).any():
                    push_bonus = 15
            
            final_score = base_score + size_bonus + geo_score + push_bonus
        
        scores.append((idx, final_score))
    
    # Sort by decreasing score
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in sorted_scores]


def evaluate_position(state: AbaloneStateNonCanonical, env: AbaloneEnvNonCanonical, 
                    player: int, radius: int = 4) -> float:
    """
    Position evaluation function from the specified player's perspective.
    
    Args:
        state: Current game state
        env: Game environment
        player: Player to evaluate for (1 for black, -1 for white)
        radius: Board radius
        
    Returns:
        Evaluation score (higher is better for the specified player)
    """
    # 1. Geometric score from the specified player's perspective
    geo_score = geometric_score(state, player, radius)
    
    # 2. Marbles out with non-linear bonus
    if player == 1:  # Evaluating for black player
        marbles_out = state.white_out  # White marbles pushed out
    else:  # Evaluating for white player
        marbles_out = state.black_out  # Black marbles pushed out
    
    # Non-linear bonus
    if marbles_out > 0:
        marbles_out_score = 30
    else:
        marbles_out_score = 0
    
    # 3. Final score
    final_score = geo_score + marbles_out_score
    
    return final_score


def alphabeta_pruning(state: AbaloneStateNonCanonical, depth: int, alpha: float, beta: float, 
                     env: AbaloneEnvNonCanonical, radius: int = 4, original_player: int = None):
    """
    Alpha-beta pruning algorithm for Abalone.
    
    Args:
        state: Current game state
        depth: Search depth
        alpha: Alpha value for pruning
        beta: Beta value for pruning
        env: Game environment
        radius: Board radius
        original_player: The player whose perspective we're optimizing for
        
    Returns:
        Tuple of (score, best_move_index)
    """
    # Initialize original_player if not provided
    if original_player is None:
        original_player = state.current_player
        
    # Check if the current state is terminal
    if env.is_terminal(state):
        winner = env.get_winner(state)
        if winner == original_player:
            return float('inf'), -1  # Victory
        elif winner == -original_player:
            return float('-inf'), -1  # Defeat
        else:
            return 0.0, -1  # Draw
    
    # If depth 0, direct evaluation
    if depth == 0:
        score = evaluate_position(state, env, original_player, radius)
        return score, -1
    
    # Get legal actions
    legal_moves = env.get_legal_moves(state)
    legal_indices = jnp.where(legal_moves)[0]
    
    if legal_indices.size == 0:
        return evaluate_position(state, env, original_player, radius), -1

    # Order moves for better pruning
    ordered_indices = order_moves(state, legal_indices, env, radius)
    best_action = ordered_indices[0] if len(ordered_indices) > 0 else int(legal_indices[0])
    
    if state.current_player == original_player:  # Maximizing player
        best_score = float('-inf')
        
        for action_idx in ordered_indices:
            next_state = env.step(state, int(action_idx))
            
            # Check if this action leads directly to victory or defeat
            if next_state.white_out >= 6 and original_player == 1:  
                return float('inf'), int(action_idx)  # Immediate victory for black
            elif next_state.black_out >= 6 and original_player == -1:
                return float('-inf'), int(action_idx)  # Immediate defeat for black
                
            score, _ = alphabeta_pruning(next_state, depth - 1, alpha, beta, env, radius, original_player)
            
            if score > best_score:
                best_score = score
                best_action = int(action_idx)
            
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break  # Beta cutoff
        
        return best_score, best_action
    
    else:  # Minimizing player
        best_score = float('inf')
        
        for action_idx in ordered_indices:
            next_state = env.step(state, int(action_idx))
            
            # Check if this action leads directly to victory or defeat
            if next_state.black_out >= 6 and original_player == -1:  
                return float('inf'), int(action_idx)  # Immediate victory for white
            elif next_state.white_out >= 6 and original_player == 1:
                return float('-inf'), int(action_idx)  # Immediate defeat for white
            
            score, _ = alphabeta_pruning(next_state, depth - 1, alpha, beta, env, radius, original_player)
            
            if score < best_score:
                best_score = score
                best_action = int(action_idx)
            
            beta = min(beta, best_score)
            if alpha >= beta:
                break  # Alpha cutoff
        
        return best_score, best_action
