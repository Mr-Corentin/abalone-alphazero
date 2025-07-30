from functools import partial
import jax
import jax.numpy as jnp
import chex
from core.core import Direction, DIRECTIONS, DIR_TO_IDX

@partial(jax.jit)
def get_valid_neighbors(pos: chex.Array, board: chex.Array, radius: int = 4) -> tuple[chex.Array, chex.Array]:
    """
    Find valid neighbors of a given position in cubic coordinates.
    
    Args:
        pos: Current position [x, y, z]
        board: Board state
        radius: Board radius
    
    Returns:
        tuple[chex.Array, chex.Array]: 
            - valid directions mask (6,)
            - neighbor positions (6, 3)
    """
    # Calculate all possible neighbors
    neighbors = pos[None, :] + DIRECTIONS  # (6, 3)
    
    # Convert to array indices
    board_pos = neighbors + radius  # Add radius to all coordinates
    
    # Create mask for valid positions
    within_bounds = ((board_pos >= 0) & (board_pos < board.shape[0])).all(axis=1)
    
    # Check if positions are on board (non-nan)
    valid_pos = ~jnp.isnan(board[board_pos[:, 0], board_pos[:, 1], board_pos[:, 2]])
    
    # Combine masks
    valid_mask = within_bounds & valid_pos
    
    return valid_mask, neighbors

@partial(jax.jit, static_argnames=['radius'])
def is_valid_group(positions: chex.Array, board: chex.Array, radius: int = 4) -> tuple[chex.Array, jnp.int32]:
    """
    Check if a group of positions forms a valid group
    Args:
        positions: coordinate array (N, 3) where N is 2 or 3
        board: board state
        radius: board radius
    Returns:
        tuple[chex.Array, jnp.int32]: (is_valid, direction_index)
            direction_index corresponds to index in DIRECTIONS
    """
    # Check that all positions have same color
    board_positions = positions + radius
    values = board[board_positions[:, 0], board_positions[:, 1], board_positions[:, 2]]
    same_color = (values[0] != 0) & jnp.all(values == values[0])
    
    # For 2 marbles
    if len(positions) == 2:
        # Calculate vector between two positions
        diff = positions[1] - positions[0]
        
        # Check if this vector corresponds to valid direction
        is_valid_direction = jnp.any(jnp.all(diff == DIRECTIONS, axis=1))
        direction_idx = jnp.argmax(jnp.all(diff == DIRECTIONS, axis=1))
        
        return same_color & is_valid_direction, direction_idx
    
    # For 3 marbles
    elif len(positions) == 3:
        # Calculate vectors between consecutive positions
        diff1 = positions[1] - positions[0]
        diff2 = positions[2] - positions[1]
        
        # Differences must be identical for alignment
        aligned = jnp.all(diff1 == diff2)
        # Check if first difference corresponds to valid direction
        is_valid_direction = jnp.any(jnp.all(diff1 == DIRECTIONS, axis=1))
        direction_idx = jnp.argmax(jnp.all(diff1 == DIRECTIONS, axis=1))
        
        return same_color & aligned & is_valid_direction, direction_idx
    
    # If number of positions is not 2 or 3
    return jnp.array(False), jnp.array(-1)

def test_group(positions: list[tuple[int, int, int]], board: chex.Array):
    """
    Test function to display group validation results
    Args:
        positions: list of tuples (x, y, z)
        board: board state
    """
    pos_array = jnp.array(positions)
    is_valid, direction_idx = is_valid_group(pos_array, board)
    
    print(f"\nGroup test:")
    for x, y, z in positions:
        print(f"({x}, {y}, {z})")
    
    if is_valid:
        direction = list(Direction)[int(direction_idx)]
        print(f"Valid group, aligned in direction: {direction.name}")
    else:
        print("Invalid group")
    
    # Display position values
    print("\nPosition values:")
    for x, y, z in positions:
        value = board[x + 4, y + 4, z + 4]
        content = "●" if value == 1 else "○" if value == -1 else "·"
        print(f"Position ({x}, {y}, {z}): {content}")


@partial(jax.jit)
def analyze_group(positions: chex.Array, board: chex.Array, group_size: int, radius: int = 4) -> tuple[chex.Array, chex.Array, chex.Array]:
    """
    Analyze a group of positions to determine its validity and possible movements.
    
    Args:
        positions: Position array (Nx3) where N is 2 or 3
        board: Board state
        group_size: Actual number of marbles in group
        radius: Board radius
    
    Returns:
        tuple:
            - is_valid: if group is valid
            - inline_dirs: mask of possible directions for inline movement (6,)
            - parallel_dirs: mask of possible directions for parallel movement (6,)
    """
    # Check that all marbles are same color
    board_positions = positions + radius
    values = board[board_positions[:, 0], board_positions[:, 1], board_positions[:, 2]]
    values_mask = jnp.arange(values.shape[0]) < group_size
    same_color = (values[0] != 0) & jnp.all(jnp.where(values_mask, values == values[0], True))
    
    # Calculate difference vector between adjacent marbles
    diff = positions[1] - positions[0]
    
    # Identify alignment direction
    is_ew = jnp.all(diff == jnp.array([1, -1, 0])) | jnp.all(diff == jnp.array([-1, 1, 0]))
    is_ne_sw = jnp.all(diff == jnp.array([1, 0, -1])) | jnp.all(diff == jnp.array([-1, 0, 1]))
    is_se_nw = jnp.all(diff == jnp.array([0, -1, 1])) | jnp.all(diff == jnp.array([0, 1, -1]))
    
    # Check adjacency
    is_adjacent = is_ew | is_ne_sw | is_se_nw
    
    # Handle differently based on group size
    def handle_size_3(dummy):
        diff2 = positions[2] - positions[1]
        is_aligned = jnp.all(diff == diff2)
        return same_color & is_adjacent & is_aligned
    def handle_size_2(dummy):
        return same_color & is_adjacent
    
    is_valid = jax.lax.switch(group_size - 2,
                             [handle_size_2, handle_size_3],
                             None)
    
    # Initialize direction masks
    inline_dirs = jnp.zeros(6, dtype=jnp.bool_)
    parallel_dirs = jnp.zeros(6, dtype=jnp.bool_)
    
    # Create masks for each alignment type
    ew_inline = jnp.zeros(6, dtype=jnp.bool_).at[
        jnp.array([DIR_TO_IDX[Direction.E], DIR_TO_IDX[Direction.W]])
    ].set(True)
    
    ew_parallel = jnp.zeros(6, dtype=jnp.bool_).at[
        jnp.array([DIR_TO_IDX[Direction.NE], DIR_TO_IDX[Direction.SE], 
                  DIR_TO_IDX[Direction.NW], DIR_TO_IDX[Direction.SW]])
    ].set(True)
    
    ne_sw_inline = jnp.zeros(6, dtype=jnp.bool_).at[
        jnp.array([DIR_TO_IDX[Direction.NE], DIR_TO_IDX[Direction.SW]])
    ].set(True)
    
    ne_sw_parallel = jnp.zeros(6, dtype=jnp.bool_).at[
        jnp.array([DIR_TO_IDX[Direction.E], DIR_TO_IDX[Direction.W],
                  DIR_TO_IDX[Direction.SE], DIR_TO_IDX[Direction.NW]])
    ].set(True)
    
    se_nw_inline = jnp.zeros(6, dtype=jnp.bool_).at[
        jnp.array([DIR_TO_IDX[Direction.SE], DIR_TO_IDX[Direction.NW]])
    ].set(True)
    
    se_nw_parallel = jnp.zeros(6, dtype=jnp.bool_).at[
        jnp.array([DIR_TO_IDX[Direction.E], DIR_TO_IDX[Direction.W],
                  DIR_TO_IDX[Direction.NE], DIR_TO_IDX[Direction.SW]])
    ].set(True)
    
    # Assign directions based on alignment
    inline_dirs = jnp.where(is_ew, ew_inline, 
                  jnp.where(is_ne_sw, ne_sw_inline,
                  jnp.where(is_se_nw, se_nw_inline, inline_dirs)))
    
    parallel_dirs = jnp.where(is_ew, ew_parallel,
                    jnp.where(is_ne_sw, ne_sw_parallel,
                    jnp.where(is_se_nw, se_nw_parallel, parallel_dirs)))
    
    # Mask directions if group is not valid
    inline_dirs = jnp.where(is_valid, inline_dirs, jnp.zeros_like(inline_dirs))
    parallel_dirs = jnp.where(is_valid, parallel_dirs, jnp.zeros_like(parallel_dirs))
    
    return is_valid, inline_dirs, parallel_dirs

def print_group_analysis(positions: list[tuple[int, int, int]], board: chex.Array):
    """
    Display group analysis for debugging
    """
    pos_array = jnp.array(positions)
    is_valid, axis_idx, inline_dirs, parallel_dirs = analyze_group(pos_array, board)
    
    print(f"\nGroup analysis:")
    print(f"Positions: {positions}")
    print(f"Valid: {bool(is_valid)}")
    
    if bool(is_valid):
        print("\nPossible inline movements:")
        inline_dirs = jnp.array(inline_dirs)
        for dir_enum in Direction:
            idx = DIR_TO_IDX[dir_enum]
            if inline_dirs[idx]:
                print(f"- {dir_enum.name}")
        
        print("\nPossible parallel movements:")
        parallel_dirs = jnp.array(parallel_dirs)
        for dir_enum in Direction:
            idx = DIR_TO_IDX[dir_enum]
            if parallel_dirs[idx]:
                print(f"- {dir_enum.name}")


