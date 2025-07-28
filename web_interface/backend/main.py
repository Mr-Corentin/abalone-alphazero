#!/usr/bin/env python3
"""
FastAPI backend for Abalone web interface
Integrates with existing AbaloneEnv
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple, Optional, Union
import sys
import os
import jax
import jax.numpy as jnp
import numpy as np

# Add project root directory to path to import our modules
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from environment.env import AbaloneEnv, AbaloneState
from core.board import display_board, create_custom_board

app = FastAPI(title="Abalone API", description="Backend for Abalone AlphaZero web interface")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global game state
env = AbaloneEnv(radius=4)
current_state: Optional[AbaloneState] = None

class MoveRequest(BaseModel):
    marbles: List[Tuple[int, int, int]]  # Selected marble coordinates
    direction: str  # Direction name (NE, E, SE, SW, W, NW)

class GameStateResponse(BaseModel):
    board: List[List[List[Optional[float]]]]  # 3D board representation (None for invalid positions)
    current_player: int  # 1 for black, -1 for white
    actual_player: int  # The actual player (for canonical representation)
    black_out: int
    white_out: int
    moves_count: int
    is_terminal: bool
    winner: Optional[int] = None

class LegalMovesResponse(BaseModel):
    legal_moves: List[int]  # Move indices
    move_count: int

def state_to_response(state: AbaloneState) -> GameStateResponse:
    """Convert AbaloneState to API response format"""
    
    # IMPORTANT: Convert canonical representation to consistent visual representation
    # In canonical: current player's pieces are always 1, opponent's are -1
    # For frontend: Black pieces should always be 1, White pieces should always be -1
    
    board_array = state.board
    actual_player = state.actual_player
    
    # If it's White's turn (actual_player = -1), the canonical board is flipped
    # We need to flip it back for consistent visual representation
    if actual_player == -1:
        # Flip the canonical board back: 1 -> -1, -1 -> 1, 0 stays 0
        visual_board = -board_array
        # But keep empty spaces as 0
        visual_board = jnp.where(board_array == 0, 0, visual_board)
    else:
        # Black's turn: canonical board is already correct for visual
        visual_board = board_array
    
    board_list = []
    for i in range(visual_board.shape[0]):
        layer = []
        for j in range(visual_board.shape[1]):
            row = []
            for k in range(visual_board.shape[2]):
                value = float(visual_board[i, j, k])
                # Replace NaN with None for JSON compatibility
                if jnp.isnan(value):
                    row.append(None)
                else:
                    row.append(value)
            layer.append(row)
        board_list.append(layer)
    
    # Check if game is terminal
    is_terminal = env.is_terminal(state)
    winner = None
    if is_terminal:
        winner = env.get_winner(state)
    
    return GameStateResponse(
        board=board_list,
        current_player=state.actual_player,
        actual_player=state.actual_player,
        black_out=int(state.black_out),
        white_out=int(state.white_out), 
        moves_count=int(state.moves_count),
        is_terminal=is_terminal,
        winner=winner
    )

def find_move_index(marbles: List[Tuple[int, int, int]], direction: str) -> Optional[int]:
    """Find the move index that matches selected marbles and direction"""
    
    # Direction name to index mapping
    direction_map = {'NE': 0, 'E': 1, 'SE': 2, 'SW': 3, 'W': 4, 'NW': 5}
    direction_idx = direction_map.get(direction)
    if direction_idx is None:
        return None
    
    # IMPORTANT: The frontend sends coordinates based on visual representation
    # We need to check if these marbles belong to the current player
    
    actual_player = current_state.actual_player
    canonical_positions = set()
    
    # Check each selected marble
    for x, y, z in marbles:
        array_x, array_y, array_z = x + 4, y + 4, z + 4
        canonical_value = current_state.board[array_x, array_y, array_z]
        
        # In canonical representation, current player's pieces are always 1
        # If the marble belongs to the current player, add it to canonical positions
        if canonical_value == 1:
            canonical_positions.add((x, y, z))
        else:
            # This marble doesn't belong to the current player
            print(f"Warning: Marble at ({x},{y},{z}) has canonical value {canonical_value}, not 1 (current player)")
    
    # If no valid positions found, the selected marbles don't belong to current player
    if not canonical_positions:
        print(f"Error: None of the selected marbles {marbles} belong to current player {actual_player}")
        return None
    
    # Get legal moves
    legal_moves = env.get_legal_moves(current_state)
    legal_indices = np.where(np.array(legal_moves))[0]
    
    # Debug: Print what we're looking for
    print(f"Frontend marbles: {marbles}, direction: {direction}")
    print(f"Canonical positions: {canonical_positions}")
    print(f"Current player: {actual_player}")
    print(f"Legal moves count: {len(legal_indices)}")
    
    for move_idx in legal_indices:
        # Get move data
        move_positions = env.moves_index['positions'][move_idx]
        move_direction = env.moves_index['directions'][move_idx]
        move_group_size = env.moves_index['group_sizes'][move_idx]
        
        # Check if direction matches
        if move_direction != direction_idx:
            continue
        
        # Get positions from move (only first group_size positions are used)
        # Convert np.int8 to int for proper comparison
        move_coords = set()
        for i in range(move_group_size):
            pos = tuple(int(x) for x in move_positions[i])
            move_coords.add(pos)
        
        # Check if canonical positions match move positions
        if canonical_positions == move_coords:
            print(f"Found matching move: {move_idx}")
            return int(move_idx)
    
    print(f"No matching move found for canonical positions {canonical_positions}")
    return None

@app.get("/")
async def root():
    return {"message": "Abalone API is running!"}

@app.post("/game/new")
async def new_game():
    """Start a new game"""
    global current_state
    
    # Create new RNG key
    rng = jax.random.PRNGKey(np.random.randint(0, 1000000))
    
    # Initialize new game state
    current_state = env.reset(rng)
    
    return state_to_response(current_state)

@app.post("/game/custom")
async def new_custom_game():
    """Start a game with the custom test board configuration"""
    global current_state
    
    # Create the custom board with exact positions from debug output
    black_positions = [
        (-4, 0, 4), (-4, 1, 3), (-3, -1, 4), (-2, -2, 4), (-1, -1, 2),
        (0, -2, 2), (1, -3, 2), (2, 0, -2), (3, -1, -2), (4, 0, -4)
    ]
    
    white_positions = [
        (-1, -3, 4), (-1, -2, 3), (0, -4, 4), (0, -3, 3), (1, -4, 3),
        (1, 2, -3), (1, 3, -4), (2, -4, 2), (2, 2, -4), (3, 0, -3),
        (3, 1, -4), (4, -1, -3)
    ]
    
    # Create marble list for custom board function
    marbles = []
    
    # Add black marbles (value 1)
    for pos in black_positions:
        marbles.append((pos, 1))
    
    # Add white marbles (value -1)  
    for pos in white_positions:
        marbles.append((pos, -1))
    
    # Create custom board
    custom_board = create_custom_board(marbles, radius=4)
    
    # Create initial empty history
    history_shape = (8,) + custom_board.shape
    history = jnp.zeros(history_shape)
    
    # Create state with custom board (black to move, player 1)
    current_state = AbaloneState(
        board=custom_board,
        history=history,
        actual_player=1,  # Black to move
        black_out=0,
        white_out=0,
        moves_count=0
    )
    
    return state_to_response(current_state)

@app.get("/game/state")
async def get_game_state():
    """Get current game state"""
    if current_state is None:
        raise HTTPException(status_code=400, detail="No active game. Start a new game first.")
    
    return state_to_response(current_state)

@app.get("/game/legal-moves")
async def get_legal_moves():
    """Get legal moves for current state"""
    if current_state is None:
        raise HTTPException(status_code=400, detail="No active game. Start a new game first.")
    
    legal_moves = env.get_legal_moves(current_state)
    legal_indices = np.where(np.array(legal_moves))[0].tolist()
    
    return LegalMovesResponse(
        legal_moves=legal_indices,
        move_count=len(legal_indices)
    )

@app.post("/game/move")
async def make_move(move_request: MoveRequest):
    """Execute a move"""
    global current_state
    
    if current_state is None:
        raise HTTPException(status_code=400, detail="No active game. Start a new game first.")
    
    if env.is_terminal(current_state):
        raise HTTPException(status_code=400, detail="Game is already finished.")
    
    # Find the move index for the selected marbles and direction
    move_idx = find_move_index(move_request.marbles, move_request.direction)
    
    if move_idx is None:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid move: marbles {move_request.marbles} in direction {move_request.direction}"
        )
    
    try:
        # Execute the move using the environment
        current_state = env.step(current_state, move_idx)
        
        return state_to_response(current_state)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing move: {str(e)}")

@app.get("/debug/board")
async def debug_board():
    """Debug endpoint to print board to console"""
    if current_state is None:
        raise HTTPException(status_code=400, detail="No active game. Start a new game first.")
    
    print("\n" + "="*50)
    print("CURRENT BOARD STATE DEBUG")
    print("="*50)
    print(f"Actual player: {current_state.actual_player}")
    print(f"Black out: {current_state.black_out}, White out: {current_state.white_out}")
    print(f"Moves count: {current_state.moves_count}")
    print("\nBoard display:")
    display_board(current_state.board, env.radius)
    
    # Show ALL positions and their values (organized)
    print("\nALL positions and values:")
    black_positions = []
    white_positions = []
    empty_positions = []
    
    for x in range(-4, 5):
        for y in range(-4, 5):
            for z in range(-4, 5):
                if x + y + z == 0:  # Valid cubic coordinate
                    array_x, array_y, array_z = x + 4, y + 4, z + 4
                    value = current_state.board[array_x, array_y, array_z]
                    if not jnp.isnan(value):
                        if value == 1:
                            black_positions.append((x, y, z))
                        elif value == -1:
                            white_positions.append((x, y, z))
                        elif value == 0:
                            empty_positions.append((x, y, z))
    
    print(f"Black marbles (value 1): {len(black_positions)}")
    for pos in sorted(black_positions):
        print(f"  {pos}")
    print(f"White marbles (value -1): {len(white_positions)}")  
    for pos in sorted(white_positions):
        print(f"  {pos}")
    print(f"Empty positions: {len(empty_positions)}")
    
    # Show legal moves with the specific move we're looking for
    legal_moves = env.get_legal_moves(current_state)
    legal_indices = np.where(np.array(legal_moves))[0]
    print(f"\nLegal moves: {len(legal_indices)} total")
    
    # Look specifically for the move with marbles (1,-3,2), (0,-2,2) in direction E (1)
    target_marbles = {(1, -3, 2), (0, -2, 2)}
    target_direction = 1  # E
    
    found_target = False
    for move_idx in legal_indices:
        positions = env.moves_index['positions'][move_idx]
        direction = env.moves_index['directions'][move_idx]
        group_size = env.moves_index['group_sizes'][move_idx]
        
        # Get move coordinates
        move_coords = set()
        for i in range(group_size):
            pos = tuple(positions[i])
            move_coords.add(pos)
        
        if move_coords == target_marbles and direction == target_direction:
            found_target = True
            print(f"🎯 FOUND TARGET MOVE {move_idx}: {move_coords} dir={direction}")
        
        # Show moves around 890 for context
        if 880 <= move_idx <= 900:
            print(f"  Move {move_idx}: {move_coords} dir={direction} size={group_size}")
    
    if not found_target:
        print(f"❌ TARGET MOVE NOT FOUND: {target_marbles} direction E({target_direction})")
    
    print("="*50)
    
    return {"message": "Board printed to console"}

if __name__ == "__main__":
    import uvicorn
    print("Starting Abalone API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)