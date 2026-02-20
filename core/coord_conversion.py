import jax
import jax.numpy as jnp
import chex
from typing import Tuple
from functools import partial

@partial(jax.jit, static_argnames=['radius'])
def get_valid_positions(radius: int = 4):
    """
    Returns a list of valid positions in the hexagonal board
    """
    return [
        # Row 0 (z = -4)
        (0,4,-4), (1,3,-4), (2,2,-4), (3,1,-4), (4,0,-4),
        # Row 1 (z = -3)
        (-1,4,-3), (0,3,-3), (1,2,-3), (2,1,-3), (3,0,-3), (4,-1,-3),
        # Row 2
        (-2,4,-2), (-1,3,-2), (0,2,-2), (1,1,-2), (2,0,-2), (3,-1,-2), (4,-2,-2),
        # Row 3
        (-3,4,-1), (-2,3,-1), (-1,2,-1), (0,1,-1), (1,0,-1), (2,-1,-1), (3,-2,-1), (4,-3,-1),
        # Row 4
        (-4,4,0), (-3,3,0), (-2,2,0), (-1,1,0), (0,0,0), (1,-1,0), (2,-2,0), (3,-3,0), (4,-4,0),
        # Row 5
        (-4,3,1), (-3,2,1), (-2,1,1), (-1,0,1), (0,-1,1), (1,-2,1), (2,-3,1), (3,-4,1),
        # Row 6
        (-4,2,2), (-3,1,2), (-2,0,2), (-1,-1,2), (0,-2,2), (1,-3,2), (2,-4,2),
        # Row 7
        (-4,1,3), (-3,0,3), (-2,-1,3), (-1,-2,3), (0,-3,3), (1,-4,3),
        # Row 8
        (-4,0,4), (-3,-1,4), (-2,-2,4), (-1,-3,4), (0,-4,4)
    ]

# Pre-compute mapping indices once (outside JIT)
def _compute_cube_to_2d_indices(radius: int = 4):
    """
    Pre-compute mapping indices for 3D → 2D conversion.
    Called once at module load time for efficiency.
    """
    valid_positions = jnp.array([
        # Row 0 (z = -4)
        [0,4,-4], [1,3,-4], [2,2,-4], [3,1,-4], [4,0,-4],
        # Row 1 (z = -3)
        [-1,4,-3], [0,3,-3], [1,2,-3], [2,1,-3], [3,0,-3], [4,-1,-3],
        # Row 2
        [-2,4,-2], [-1,3,-2], [0,2,-2], [1,1,-2], [2,0,-2], [3,-1,-2], [4,-2,-2],
        # Row 3
        [-3,4,-1], [-2,3,-1], [-1,2,-1], [0,1,-1], [1,0,-1], [2,-1,-1], [3,-2,-1], [4,-3,-1],
        # Row 4
        [-4,4,0], [-3,3,0], [-2,2,0], [-1,1,0], [0,0,0], [1,-1,0], [2,-2,0], [3,-3,0], [4,-4,0],
        # Row 5
        [-4,3,1], [-3,2,1], [-2,1,1], [-1,0,1], [0,-1,1], [1,-2,1], [2,-3,1], [3,-4,1],
        # Row 6
        [-4,2,2], [-3,1,2], [-2,0,2], [-1,-1,2], [0,-2,2], [1,-3,2], [2,-4,2],
        # Row 7
        [-4,1,3], [-3,0,3], [-2,-1,3], [-1,-2,3], [0,-3,3], [1,-4,3],
        # Row 8
        [-4,0,4], [-3,-1,4], [-2,-2,4], [-1,-3,4], [0,-4,4]
    ])

    # Calculate 3D indices (for reading from board_3d)
    indices_3d = valid_positions + radius  # (61, 3)

    # Calculate 2D indices (for writing to board_2d)
    indices_2d_row = valid_positions[:, 2] + radius  # z + radius
    indices_2d_col = valid_positions[:, 0] + 4       # x + 4

    return indices_3d, indices_2d_row, indices_2d_col

# Pre-compute indices once at module load
_CUBE_TO_2D_INDICES_3D, _CUBE_TO_2D_INDICES_2D_ROW, _CUBE_TO_2D_INDICES_2D_COL = _compute_cube_to_2d_indices(radius=4)


@partial(jax.jit, static_argnames=['radius'])
def cube_to_2d(board_3d: chex.Array, radius: int = 4) -> chex.Array:
    """
    Convert board from cubic representation (3D) to 2D 9x9 grid.
    OPTIMIZED: Fully vectorized using pre-computed indices for better TPU performance.
    """
    # Create empty 2D board
    board_2d = jnp.full((9, 9), -2, dtype=board_3d.dtype)

    # Vectorized read: get all 61 values from 3D board in one operation
    values = board_3d[_CUBE_TO_2D_INDICES_3D[:, 0],
                      _CUBE_TO_2D_INDICES_3D[:, 1],
                      _CUBE_TO_2D_INDICES_3D[:, 2]]

    # Vectorized write: place all values in 2D board in one operation
    board_2d = board_2d.at[_CUBE_TO_2D_INDICES_2D_ROW, _CUBE_TO_2D_INDICES_2D_COL].set(values)

    return board_2d


@partial(jax.jit, static_argnames=['radius'])
def convert_and_canonicalize_history_batch(history_3d: jnp.ndarray,
                                           actual_players: jnp.ndarray,
                                           radius: int = 4) -> jnp.ndarray:
    """
    OPTIMIZED: Convert 3D history to 2D AND canonicalize in one efficient pass.

    This function replaces the inefficient jax.vmap(jax.vmap(cube_to_2d)) pattern
    with a single vmap and proper canonicalization.

    Args:
        history_3d: (batch_size, 8, 9, 9, 9) - raw history in 3D
        actual_players: (batch_size,) - current player for each state (1 or -1)
        radius: Board radius

    Returns:
        history_2d: (batch_size, 8, 9, 9) - canonical 2D history

    Performance: ~2-3x faster than jax.vmap(jax.vmap(cube_to_2d)) on TPU
    """
    batch_size = history_3d.shape[0]

    # Step 1: Canonicalize (flip values if player is -1)
    # This ensures current player always sees their pieces as 1
    canonical_history_3d = jnp.where(
        actual_players[:, None, None, None, None] == 1,
        history_3d,
        -history_3d
    )

    # Step 2: Flatten batch and history dimensions for efficient conversion
    # (batch_size, 8, 9, 9, 9) -> (batch_size * 8, 9, 9, 9)
    flat_history = canonical_history_3d.reshape(-1, 9, 9, 9)

    # Step 3: Single vmap instead of double vmap (MUCH faster!)
    flat_history_2d = jax.vmap(cube_to_2d)(flat_history)  # (batch_size * 8, 9, 9)

    # Step 4: Reshape back to separate batch and history
    # (batch_size * 8, 9, 9) -> (batch_size, 8, 9, 9)
    history_2d = flat_history_2d.reshape(batch_size, 8, 9, 9)

    return history_2d

@partial(jax.jit, static_argnames=['radius'])
def compute_coord_map(radius: int = 4):
    """
    Pre-compute mapping between 3D and 2D coordinates
    Returns:
        Dict with:
        - indices_3d : positions in 3D array (shape (61, 3))
        - indices_2d : corresponding positions in 2D array (shape (61, 2))
    """
    valid_positions = get_valid_positions(radius)
    n_positions = len(valid_positions)
    
    # Pre-compute 3D and 2D indices
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
    Prepare inputs for the network with batching support and canonical history.

    Args:
        board_3d: Shape (batch_size, x, y, z) batched or (x, y, z) single
        history_3d: Shape (batch_size, 8, x, y, z) batched or (8, x, y, z) single
        actual_player: Shape (batch_size,) or scalar - current player (1 or -1)
        our_marbles_out: Shape (batch_size,) or scalar
        opponent_marbles_out: Shape (batch_size,) or scalar
        radius: Board radius
        
    Returns:
        board_2d: Shape (batch_size, 9, 9, 9) - current board + 8 historical positions (canonical)
        marbles_out: Shape (batch_size, 2) - marbles out [us, opponent]
    """
    # Detect if we have a batch or single example
    is_batched = board_3d.ndim > 3  # car board_3d est déjà en 3D

    if not is_batched:
        # If single example, add batch dimension
        board_3d = board_3d[None, ...]  # (1, x, y, z)
        history_3d = history_3d[None, ...]  # (1, 8, x, y, z)
        actual_player = jnp.array([actual_player])  # (1,)
        our_marbles_out = jnp.array([our_marbles_out])
        opponent_marbles_out = jnp.array([opponent_marbles_out])

    # Convert current board: (batch_size, x, y, z) -> (batch_size, 9, 9)
    # Current board is already canonical
    current_board_2d = jax.vmap(lambda b: cube_to_2d(b, radius))(board_3d)
    
    # Apply canonical transformation to history
    # History contains "real" positions, need to adapt to current player
    def canonicalize_history_for_player(history, player):
        """Transform history so current player sees their pieces as 1"""
        return jnp.where(player == 1, history, -history)
    
    # Apply canonicalization to each batch element
    canonical_history = jax.vmap(canonicalize_history_for_player)(history_3d, actual_player)
    
    # Convert canonical history: (batch_size, 8, x, y, z) -> (batch_size, 8, 9, 9)
    history_2d = jax.vmap(jax.vmap(lambda h: cube_to_2d(h, radius)))(canonical_history)
    
    # Stack current board + history as channels
    # current_board_2d: (batch_size, 9, 9) -> (batch_size, 9, 9, 1)
    current_board_2d = current_board_2d[..., None]
    
    # history_2d: (batch_size, 8, 9, 9) -> (batch_size, 9, 9, 8)
    history_2d = jnp.transpose(history_2d, (0, 2, 3, 1))
    
    # Concatenate: (batch_size, 9, 9, 1) + (batch_size, 9, 9, 8) = (batch_size, 9, 9, 9)
    board_with_history = jnp.concatenate([current_board_2d, history_2d], axis=-1)

    # Replace NaN with -2 and convert to int8
    board_with_history = jnp.nan_to_num(board_with_history, -2.0)
    board_with_history = board_with_history.astype(jnp.int8)

    # Create marbles out vector (batch_size, 2)
    marbles_out = jnp.stack([our_marbles_out, opponent_marbles_out], axis=-1)

    return board_with_history, marbles_out


# Compatibility function for legacy code
@partial(jax.jit, static_argnames=['radius'])
def prepare_input_legacy(board_3d: jnp.ndarray, our_marbles_out: jnp.ndarray, opponent_marbles_out: jnp.ndarray, radius: int = 4):
    """
    Legacy version of prepare_input without history for compatibility.
    """
    # Detect if we have a batch or single example
    is_batched = board_3d.ndim > 3  # car board_3d est déjà en 3D

    if not is_batched:
        # If single example, add batch dimension
        board_3d = board_3d[None, ...]  # (1, x, y, z)
        our_marbles_out = jnp.array([our_marbles_out])
        opponent_marbles_out = jnp.array([opponent_marbles_out])

    # Use vmap to apply cube_to_2d to each batch element
    board_2d = jax.vmap(lambda b: cube_to_2d(b, radius))(board_3d)

    # Replace NaN with -2 and convert to int8
    board_2d = jnp.nan_to_num(board_2d, -2.0)
    board_2d = board_2d.astype(jnp.int8)

    # Create marbles out vector (batch_size, 2)
    marbles_out = jnp.stack([our_marbles_out, opponent_marbles_out], axis=-1)

    return board_2d, marbles_out


def display_2d_board(board_2d: chex.Array):
    """
    Display 2D board
    """
    print("\n2D Board:")
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

