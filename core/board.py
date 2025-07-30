import jax
import jax.numpy as jnp
import chex
from functools import partial
from typing import Tuple

from core.core import CubeCoord, Direction, DIRECTIONS

def create_board_mask(radius: int = 4) -> chex.Array:
    """
    Creates a 3D mask of valid positions on the board
    Returns:
        chex.Array: Boolean mask (2*radius+1, 2*radius+1, 2*radius+1)
    """
    size = 2 * radius + 1
    mask = jnp.full((size, size, size), False)
    
    # List of all valid coordinates
    valid_coords = [
        # Row 1 (z = -4)
        (0,4,-4), (1,3,-4), (2,2,-4), (3,1,-4), (4,0,-4),
        # Row 2 (z = -3)
        (-1,4,-3), (0,3,-3), (1,2,-3), (2,1,-3), (3,0,-3), (4,-1,-3),
        # Row 3 (z = -2)
        (-2,4,-2), (-1,3,-2), (0,2,-2), (1,1,-2), (2,0,-2), (3,-1,-2), (4,-2,-2),
        # Row 4 (z = -1)
        (-3,4,-1), (-2,3,-1), (-1,2,-1), (0,1,-1), (1,0,-1), (2,-1,-1), (3,-2,-1), (4,-3,-1),
        # Row 5 (z = 0)
        (-4,4,0), (-3,3,0), (-2,2,0), (-1,1,0), (0,0,0), (1,-1,0), (2,-2,0), (3,-3,0), (4,-4,0),
        # Row 6 (z = 1)
        (-4,3,1), (-3,2,1), (-2,1,1), (-1,0,1), (0,-1,1), (1,-2,1), (2,-3,1), (3,-4,1),
        # Row 7 (z = 2)
        (-4,2,2), (-3,1,2), (-2,0,2), (-1,-1,2), (0,-2,2), (1,-3,2), (2,-4,2),
        # Row 8 (z = 3)
        (-4,1,3), (-3,0,3), (-2,-1,3), (-1,-2,3), (0,-3,3), (1,-4,3),
        # Row 9 (z = 4)
        (-4,0,4), (-3,-1,4), (-2,-2,4), (-1,-3,4), (0,-4,4)
    ]
    
    # Mark all valid positions in the mask
    for x, y, z in valid_coords:
        array_x = x + radius
        array_y = y + radius
        array_z = z + radius
        mask = mask.at[array_x, array_y, array_z].set(True)
    
    return mask
def initialize_board(radius: int = 4) -> chex.Array:
    """
    Initialize the board with starting positions
    Returns:
        chex.Array: Board with initial positions
    """
    size = 2 * radius + 1
    board = jnp.full((size, size, size), jnp.nan)
    
    # Create valid positions mask and initialize valid cells to 0
    valid_mask = create_board_mask(radius)
    board = jnp.where(valid_mask, 0., board)
    

    # Belgian Daisy config
    black_coords = [
        (3,1,-4), (4,0,-4),
        (2,1,-3), (3,0,-3), (4,-1,-3),
        (2,0,-2), (3,-1,-2),
        (-3,1,2), (-2,0,2),
        (-4,1,3), (-3,0,3), (-2,-1,3),
        (-4,0,4), (-3,-1,4)
    ]
    
    white_coords = [
        (0,4,-4), (1,3,-4),
        (-1,4,-3), (0,3,-3), (1,2,-3),
        (-1,3,-2), (0,2,-2),
        (0,-2,2), (1,-3,2),
        (-1,-2,3), (0,-3,3), (1,-4,3),
        (-1,-3,4), (0,-4,4)
    ]
    
    # Place black marbles (1) and white marbles (-1)
    for x, y, z in black_coords:
        array_x, array_y, array_z = x + radius, y + radius, z + radius
        board = board.at[array_x, array_y, array_z].set(1.)
    
    for x, y, z in white_coords:
        array_x, array_y, array_z = x + radius, y + radius, z + radius
        board = board.at[array_x, array_y, array_z].set(-1.)
    
    return board

    
def display_board(board: chex.Array, radius: int = 4):
    """
    Display the Abalone board in 2D following cubic coordinates
    """
    print("\nAbalone Board:")
    print("Legend: ● (black), ○ (white), · (empty)\n")
    
    # Starting coordinates for each row
    line_starts = [
        [(0,4,-4), 5],
        [(-1,4,-3), 6],
        [(-2,4,-2), 7],
        [(-3,4,-1), 8],
        [(-4,4,0), 9],
        [(-4,3,1), 8],
        [(-4,2,2), 7],
        [(-4,1,3), 6],
        [(-4,0,4), 5]
    ]
    
    # For each row
    for row_idx, ((start_x, start_y, start_z), num_cells) in enumerate(line_starts):
        indent = abs(4 - row_idx)
        print(" " * indent, end="")
        
        for i in range(num_cells):
            x = start_x + i
            y = start_y - i
            z = start_z
            
            array_x = x + radius
            array_y = y + radius
            array_z = z + radius
            
            value = board[array_x, array_y, array_z]
            if value == 1:
                print("● ", end="")
            elif value == -1:
                print("○ ", end="")
            else:
                print("· ", end="")
        print() 
                

def create_custom_board(marbles: list[tuple[tuple[int, int, int], int]], radius: int = 4) -> chex.Array:
    """
    Create a board with marbles placed at specific positions
    
    Args:
        marbles: List of tuples ((x,y,z), color) where color is 1 for black, -1 for white
        radius: Board radius
    
    Returns:
        chex.Array: Custom board
    """
    # Create empty board
    size = 2 * radius + 1
    board = jnp.full((size, size, size), jnp.nan)
    
    # Fill all valid positions with 0 (empty)
    valid_mask = create_board_mask(radius)
    board = jnp.where(valid_mask, 0., board)
    
    # Place marbles at specified positions
    for (x, y, z), color in marbles:
        board_pos = jnp.array([x, y, z]) + radius
        board = board.at[board_pos[0], board_pos[1], board_pos[2]].set(float(color))
    
    return board



@jax.jit
def get_neighbors_array(pos_array: chex.Array) -> chex.Array:
    """
    Returns neighboring positions of a given coordinate in array format
    Args:
        pos_array: Origin position (array [x, y, z])
    Returns:
        chex.Array: Array (6, 3) of neighboring coordinates
    """
    return pos_array[None, :] + DIRECTIONS

def get_neighbors(pos: CubeCoord) -> chex.Array:
    """
    Wrapper version that accepts a CubeCoord
    """
    return get_neighbors_array(pos.to_array())

@partial(jax.jit, static_argnums=(2,))
def is_valid_position(pos: chex.Array, board: chex.Array, radius: int = 4) -> chex.Array:
    """
    Check if a position is valid on the board
    Args:
        pos: Position to check (x, y, z)
        board: Board state
        radius: Board radius
    Returns:
        chex.Array: True if position is valid
    """
    x, y, z = pos
    size = 2 * radius + 1
    
    # Use jnp.where instead of Python boolean operations
    within_bounds = ((x + radius >= 0) & 
                    (x + radius < size) & 
                    (y + radius >= 0) & 
                    (y + radius < size) & 
                    (z + radius >= 0) & 
                    (z + radius < size))
    
    # Now use jnp.where to combine conditions
    return jnp.where(
        within_bounds,
        ~jnp.isnan(board[x + radius, y + radius, z + radius]),
        False
    )
