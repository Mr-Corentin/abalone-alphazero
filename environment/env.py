import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Dict, NamedTuple
from core.board import initialize_board, create_custom_board, create_board_mask
from core.legal_moves import get_legal_moves
from core.moves import move_group_inline, move_group_parallel, move_single_marble
import numpy as np
from core.coord_conversion import compute_coord_map
from functools import partial

class AbaloneState(NamedTuple):
    """Abalone game state (canonical version)"""
    board: chex.Array  # Board where current player is always 1
    history: chex.Array  # Last 8 positions (8, 9, 9, 9) - also canonical
    actual_player: int  # Real player (1=black, -1=white)
    black_out: int  # Number of black marbles out
    white_out: int  # Number of white marbles out
    moves_count: int

class AbaloneEnv:
    def __init__(self, radius: int = 4):
      self.radius = radius
      self.moves_index = self._load_moves_index()
      self.coord_map = compute_coord_map(radius)

    # RNG arg to parallelize
    def reset(self, rng: chex.PRNGKey) -> AbaloneState:
        """Reset with RNG key for batch compatibility."""
        board = initialize_board()  # Can be randomized with rng later
        
        # Initialize history with same structure as board (including NaN)
        # but with all valid positions at 0
        history_shape = (8,) + board.shape
        valid_mask = create_board_mask(self.radius)
        
        # Create history: NaN for invalid positions, 0 for valid positions
        single_history_layer = jnp.where(valid_mask, 0.0, jnp.nan)
        history = jnp.repeat(single_history_layer[None, ...], 8, axis=0)
        
        return AbaloneState(
            board=board,
            history=history,
            actual_player=1,
            black_out=0,
            white_out=0,
            moves_count=0
        )

    @partial(jax.jit, static_argnames=['self', 'batch_size'])
    def reset_batch(self, rng: chex.PRNGKey, batch_size: int = 1) -> AbaloneState:
        """Reset for a batch of states"""
        # Initialize single board
        single_board = initialize_board(self.radius)  # shape: (size, size, size)

        # Create batch_size copies of board
        # We want: (batch_size, size, size, size)
        boards = jnp.repeat(single_board[None, ...], batch_size, axis=0)
        
        # Initialize history for batch with same logic
        valid_mask = create_board_mask(self.radius)
        single_history_layer = jnp.where(valid_mask, 0.0, jnp.nan)
        single_history = jnp.repeat(single_history_layer[None, ...], 8, axis=0)
        histories = jnp.repeat(single_history[None, ...], batch_size, axis=0)

        return AbaloneState(
            board=boards,  # shape: (batch_size, size, size, size)
            history=histories,  # shape: (batch_size, 8, size, size, size)
            actual_player=jnp.ones(batch_size, dtype=jnp.int32),
            black_out=jnp.zeros(batch_size, dtype=jnp.int32),
            white_out=jnp.zeros(batch_size, dtype=jnp.int32),
            moves_count=jnp.zeros(batch_size, dtype=jnp.int32)
        )

    @partial(jax.jit, static_argnames=['self'])
    def step(self, state: AbaloneState, move_idx: int) -> AbaloneState:
        """Execute a move and return new state"""
        # Convert move_idx to scalar integer
        move_idx = move_idx.astype(jnp.int32).reshape(())

        # Access data (already in JAX arrays, no conversion needed)
        direction = self.moves_index['directions'][move_idx]
        move_type = self.moves_index['move_types'][move_idx]
        group_size = self.moves_index['group_sizes'][move_idx]
        position = self.moves_index['positions'][move_idx]

        # Define move type branches (cleaner than nested lambdas)
        def single_marble_branch(_):
            new_board, _ = move_single_marble(state.board, position[0], direction, self.radius)
            return new_board, 0

        def group_parallel_branch(_):
            new_board, _ = move_group_parallel(state.board, position, direction, group_size, self.radius)
            return new_board, 0

        def group_inline_branch(_):
            new_board, _, billes_sorties = move_group_inline(state.board, position, direction, group_size, self.radius)
            return new_board, billes_sorties

        # Use switch for move type
        new_board, billes_sorties = jax.lax.switch(
            move_type,
            [single_marble_branch, group_parallel_branch, group_inline_branch],
            None  # Operand not used by branches
        )

        # Ensure actual_player is scalar
        actual_player = state.actual_player.reshape(())

        # Update history BEFORE canonical transformation
        # Shift history and add old current position (REAL, not canonical)
        new_history = jnp.roll(state.history, shift=1, axis=0)  # Shift everything right
        new_history = new_history.at[0].set(state.board)  # Save old REAL position

        # Update marbles out
        black_out = state.black_out + billes_sorties * (actual_player == -1)
        white_out = state.white_out + billes_sorties * (actual_player == 1)

        return AbaloneState(
            board=-new_board,  # New canonical board for new player
            history=new_history, 
            actual_player=-actual_player,
            black_out=black_out,
            white_out=white_out,
            moves_count=state.moves_count + 1
        )

    @partial(jax.jit, static_argnames=['self'])
    def step_batch(self, states: AbaloneState, move_idxs: chex.Array) -> AbaloneState:
        return jax.vmap(self.step)(states, move_idxs)

    
    def _load_moves_index(self):
        """Load moves index from npz file and convert to JAX arrays once"""
        import os

        # Build path to file in data/ folder
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'move_map.npz')

        # Load data
        moves_data = np.load(data_path)

        # Convert to JAX arrays ONCE (not at every step)
        return {
            'positions': jnp.array(moves_data['positions']),
            'directions': jnp.array(moves_data['directions']),
            'move_types': jnp.array(moves_data['move_types']),
            'group_sizes': jnp.array(moves_data['group_sizes'])
        }

    @partial(jax.jit, static_argnames=['self'])
    def get_legal_moves(self, state: AbaloneState) -> chex.Array:
        """Return legal moves mask"""
        return get_legal_moves(state.board, self.moves_index, self.radius)

    @partial(jax.jit, static_argnames=['self'])
    def get_legal_moves_batch(self, states: AbaloneState) -> chex.Array:
        """Return legal moves mask for batch of states"""
        # FIXED: Use lambda to avoid vmapping over moves_index and radius
        return jax.vmap(lambda board: get_legal_moves(board, self.moves_index, self.radius))(states.board)

    def is_terminal(self, state: AbaloneState) -> bool:
      """Check if state is terminal"""
      # Replace or with jnp.logical_or
      return jnp.logical_or(
          jnp.logical_or(
              state.black_out >= 6,
              state.white_out >= 6
          ),
          state.moves_count >= 300
      )
    @partial(jax.jit, static_argnames=['self'])
    def is_terminal_batch(self, states: AbaloneState) -> chex.Array:
        return jax.vmap(lambda s: jnp.logical_or(
            jnp.logical_or(
                s.black_out >= 6,
                s.white_out >= 6
            ),
            s.moves_count >= 300
        ))(states)
    def get_winner(self, state: AbaloneState) -> int:
        """
        Determine winner

        Returns:
            1 if black wins, -1 if white wins, 0 if draw
        """
        if state.white_out >= 6:
            return 1  # Black wins
        elif state.black_out >= 6:
            return -1  # White wins
        elif state.moves_count >= 300:
            return 0  # Draw
        return 0  # Game in progress
    @partial(jax.jit, static_argnames=['self'])
    def get_winner_batch(self, states: AbaloneState) -> chex.Array:
        return jax.vmap(lambda s: jnp.where(
            s.white_out >= 6,
            1,  # Black wins
            jnp.where(
                s.black_out >= 6,
                -1,  # White wins
                0  # Draw or in progress
            )
        ))(states)

    def is_legal_move(self, state: AbaloneState, move_idx: int) -> bool:
        """Check if specific move is legal"""
        legal_moves = self.get_legal_moves(state)
        return legal_moves[move_idx]

    def get_score(self, state: AbaloneState) -> dict:
        """Return current score as dictionary"""
        return {
            'black_out': state.black_out,
            'white_out': state.white_out,
            'moves': state.moves_count
        }

    def get_canonical_state(self, board: chex.Array, actual_player: int) -> chex.Array:
        """
        Convert board to canonical representation where player to move is always 1

        Args:
            board: Board state
            actual_player: Player to move (1 or -1)

        Returns:
            board_canonical: Board in canonical representation
        """
        return jnp.where(actual_player == 1, board, -board)