import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Dict, NamedTuple
from core.board import initialize_board, create_custom_board, create_board_mask
from core.legal_moves import (get_legal_moves, build_fast_tables,
                              get_legal_moves_fast, apply_move_fast)
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
    HISTORY_LENGTH_DEFAULT = 0

    def __init__(self, radius: int = 4, max_moves: int = 200,
                 history_length: int = HISTORY_LENGTH_DEFAULT):
      """
      Args:
          radius: board radius
          max_moves: draw / truncation limit. NOT a rule of Abalone -- it is an
              artificial cutoff so self-play games terminate.
          history_length: number of past positions fed to the network.
              Defaults to 0: Abalone is fully observable and has no repetition
              rule here, so past positions carry no information the current board
              lacks, while costing ~23 KB per MCTS tree node and 8x the replay
              buffer volume. Set it back to 8 to restore the previous behaviour --
              every code path is still in place, the arrays are simply empty.
      """
      self.radius = radius
      self.max_moves = max_moves
      self.history_length = history_length
      self.moves_index = self._load_moves_index()
      self.legality_tables = build_fast_tables(self.moves_index, radius)
      self.coord_map = compute_coord_map(radius)

    # RNG arg to parallelize
    def reset(self, rng: chex.PRNGKey) -> AbaloneState:
        """Reset with RNG key for batch compatibility."""
        board = initialize_board()  # Can be randomized with rng later
        
        # Initialize history with same structure as board (including NaN)
        # but with all valid positions at 0
        valid_mask = create_board_mask(self.radius)
        
        # Create history: NaN for invalid positions, 0 for valid positions
        single_history_layer = jnp.where(valid_mask, 0.0, jnp.nan)
        history = jnp.repeat(single_history_layer[None, ...], self.history_length, axis=0)
        
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
        single_history = jnp.repeat(single_history_layer[None, ...], self.history_length, axis=0)
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

        # Vectorised move application: no lax.switch, so vmap does not have to
        # evaluate all three move functions for every state in the batch.
        # It also rejects moves whose group is made of OPPONENT marbles, which
        # the old move_group_* path accepted (analyze_group only checked that the
        # group was single-coloured, not that it was ours).
        new_board, _success, billes_sorties = apply_move_fast(
            state.board, move_idx, self.legality_tables)

        # Ensure actual_player is scalar
        actual_player = state.actual_player.reshape(())

        # Update history BEFORE canonical transformation.
        # state.board is CANONICAL (current player is 1); multiplying by
        # actual_player converts it back to the absolute frame (black = 1).
        # prepare_input / convert_and_canonicalize_history_batch re-canonicalise
        # for whoever is to move, so the history must be stored absolute --
        # storing it canonical made every other plane come out sign-flipped.
        if self.history_length > 0:
            new_history = jnp.roll(state.history, shift=1, axis=0)
            new_history = new_history.at[0].set(state.board * actual_player)
        else:
            new_history = state.history  # empty, nothing to shift

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
        return get_legal_moves_fast(state.board, self.legality_tables)

    @partial(jax.jit, static_argnames=['self'])
    def get_legal_moves_batch(self, states: AbaloneState) -> chex.Array:
        """Return legal moves mask for batch of states"""
        return jax.vmap(lambda board: get_legal_moves_fast(board, self.legality_tables))(states.board)

    @partial(jax.jit, static_argnames=['self'])
    def get_legal_moves_from_board(self, board: chex.Array) -> chex.Array:
        """Legal moves for a raw canonical board (no AbaloneState needed)"""
        return get_legal_moves_fast(board, self.legality_tables)

    def is_terminal(self, state: AbaloneState) -> bool:
      """Check if state is terminal"""
      # Replace or with jnp.logical_or
      return jnp.logical_or(
          jnp.logical_or(
              state.black_out >= 6,
              state.white_out >= 6
          ),
          state.moves_count >= self.max_moves
      )
    @partial(jax.jit, static_argnames=['self'])
    def is_terminal_batch(self, states: AbaloneState) -> chex.Array:
        return jax.vmap(self.is_terminal)(states)
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
        elif state.moves_count >= self.max_moves:
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