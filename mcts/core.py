import mctx
import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Dict, Any
from functools import partial

from environment.env import AbaloneEnv, AbaloneState
from model.neural_net import AbaloneModel
from core.coord_conversion import prepare_input_legacy, cube_to_2d, convert_and_canonicalize_history_batch

@partial(jax.jit, static_argnames=['win_threshold'])
def calculate_reward_terminal_only(current_state: AbaloneState, next_state: AbaloneState,
                                   win_threshold: int = 6) -> float:
    """
    TERMINAL REWARDS ONLY (AlphaZero approach)
    Calculate transition reward - rewards only at game end
    - +1.0 for winning the game (from current player perspective)
    - -1.0 for losing the game (from current player perspective)
    - 0.0 for all other transitions

    `win_threshold` is the curriculum threshold (AbaloneEnv.win_threshold), not
    necessarily Abalone's 6. It is static: it is a Python int at trace time.
    """
    game_over = (next_state.black_out >= win_threshold) | (next_state.white_out >= win_threshold)
    
    reward = jnp.where(~game_over, 0.0,
        jnp.where(
            next_state.white_out >= win_threshold,
            1.0 * current_state.actual_player,
            jnp.where(
                next_state.black_out >= win_threshold, 
                -1.0 * current_state.actual_player,
                0.0
            )
        )
    )
    
    return reward

@partial(jax.jit, static_argnames=['win_threshold'])
def calculate_reward_with_intermediate(current_state: AbaloneState, next_state: AbaloneState,
                                       weight: float = 0.1, win_threshold: int = 6) -> float:
    """
    INTERMEDIATE REWARDS VERSION - FOR TESTING
    Calculate reward with intermediate rewards for pushing marbles
    - +1.0 for winning the game
    - +weight for each opponent marble pushed
    - -1.0 for losing the game
    
    Canonical version: always from current player perspective
    """
    black_diff = next_state.black_out - current_state.black_out
    white_diff = next_state.white_out - current_state.white_out
    
    opponent_marbles_pushed = jnp.where(
        current_state.actual_player == 1,
        white_diff,
        black_diff
    )
    
    intermediate_reward = weight * opponent_marbles_pushed
    
    game_over = (next_state.black_out >= win_threshold) | (next_state.white_out >= win_threshold)
    
    terminal_reward = jnp.where(~game_over, 0.0,
        jnp.where(
            next_state.white_out >= win_threshold,
            1.0 * current_state.actual_player,
            jnp.where(
                next_state.black_out >= win_threshold, 
                -1.0 * current_state.actual_player,
                0.0
            )
        )
    )
    
    return intermediate_reward + terminal_reward

@partial(jax.jit, static_argnames=['win_threshold'])
def calculate_reward_curriculum(current_state: AbaloneState, next_state: AbaloneState, iteration: int,
                                win_threshold: int = 6) -> float:
    """
    CURRICULUM REWARD VERSION
    Switches between intermediate and terminal rewards based on iteration
    """

    weight = jnp.where(
        iteration < 10,
        0.1,
        jnp.where(
            iteration < 15,
            0.05,
            jnp.where(
                iteration < 30,
                0.01,
                0.0
            )
        )
    )

    # Calculate only the intermediate part of the reward
    black_diff = next_state.black_out - current_state.black_out
    white_diff = next_state.white_out - current_state.white_out
    
    opponent_marbles_pushed = jnp.where(
        current_state.actual_player == 1,
        white_diff,
        black_diff
    )
    
    intermediate_reward_part = weight * opponent_marbles_pushed

    # Always add the terminal reward
    terminal_reward = calculate_reward_terminal_only(current_state, next_state, win_threshold)
    
    return intermediate_reward_part + terminal_reward

# Default function (can be switched for testing)
@partial(jax.jit, static_argnames=['win_threshold'])
def calculate_reward(current_state: AbaloneState, next_state: AbaloneState, iteration: int,
                     win_threshold: int = 6) -> float:
    """
    Reward used by the search. Terminal outcomes only (standard AlphaZero).

    The intermediate variants below are kept for experiments but are NOT used:
    the value head is trained on the final game outcome alone, so shaping only the
    search reward makes the two disagree. Concretely, +0.1 per marble pushed made a
    won game worth 0.6 + 1.0 = 1.6, which a tanh-bounded value head can never
    predict, and qtransform_completed_by_mix_value then blended that Q with a value
    living on a different scale.

    If you want to reward pushing marbles, do it on the TRAINING TARGET instead
    (e.g. score the move limit by the marble differential), not here.
    """
    return calculate_reward_terminal_only(current_state, next_state, win_threshold)
    # return calculate_reward_curriculum(current_state, next_state, iteration)
    # return calculate_reward_with_intermediate(current_state, next_state)
    
@partial(jax.jit, static_argnames=['max_moves', 'win_threshold'])
def calculate_discount(state: AbaloneState, max_moves: int = 300,
                       win_threshold: int = 6) -> float:
    """
    Discount for a two-player zero-sum game under mctx.

    mctx backs values up with `leaf_value = reward + discount * leaf_value`, with
    NO per-level sign flip of its own. `reward` is expressed from the point of view
    of the player who moved (see calculate_reward) and the child's value from the
    point of view of the player to move at the child -- the opposite side. The
    discount is what converts between the two, hence -1.0.

    Terminal states return 0.0 so nothing below them propagates: the terminal
    outcome is already carried by the reward of the transition into them.
    """
    is_terminal = ((state.black_out >= win_threshold) | (state.white_out >= win_threshold)
                   | (state.moves_count >= max_moves))
    return jnp.where(is_terminal, 0.0, -1.0)


# Logit used to mask illegal actions. Large and negative, but finite: -inf would
# produce NaNs in mctx's softmax when a state has no legal move at all.
MASKED_LOGIT = -1e9


class AbaloneMCTSRecurrentFn:
    """Recurrent function class for MCTS using mctx"""
    def __init__(self, env: AbaloneEnv, network: AbaloneModel):
        self.env = env
        self.network = network

    @partial(jax.jit, static_argnums=(0,))
    def recurrent_fn(self, params, rng_key, action, embedding):
        """
        Recurrent function for MCTS handling a batch of states

        Args:
            params: Network parameters
            rng_key: JAX RNG key
            action: Actions to apply (shape: (batch_size,))
            embedding: Dict containing batch state
        """
        current_states = AbaloneState(
            board=embedding['board_3d'],
            history=embedding['history_3d'],
            actual_player=embedding['actual_player'],
            black_out=embedding['black_out'],
            white_out=embedding['white_out'],
            moves_count=embedding['moves_count']
        )

        next_states = jax.vmap(self.env.step)(current_states, action)

        iteration = embedding['iteration']
        # win_threshold est capture depuis l'env : c'est un int Python, donc un
        # argument statique valide meme a l'interieur du vmap.
        win_threshold = self.env.win_threshold
        reward = jax.vmap(
            lambda c, n, it: calculate_reward(c, n, it, win_threshold)
        )(current_states, next_states, iteration)
        discount = jax.vmap(
            lambda s: calculate_discount(s, self.env.max_moves, win_threshold)
        )(next_states)
        our_marbles = jnp.where(next_states.actual_player == 1,
                               next_states.black_out,
                               next_states.white_out)
        opp_marbles = jnp.where(next_states.actual_player == 1,
                               next_states.white_out,
                               next_states.black_out)

        board_2d, marbles_out = prepare_input_legacy(next_states.board, our_marbles, opp_marbles)

        # OPTIMIZED: Use efficient single-vmap conversion with canonicalization
        history_2d = convert_and_canonicalize_history_batch(next_states.history, next_states.actual_player)

        prior_logits, value = self.network.apply(params, board_2d, marbles_out, history_2d)

        # mctx only masks invalid actions at the ROOT. Without this, interior nodes
        # expand illegal moves, which env.step turns into null moves (board unchanged,
        # turn passed) -- with ~52 legal moves out of 1734, that is ~97% of the tree.
        legal_moves = self.env.get_legal_moves_batch(next_states)
        prior_logits = jnp.where(legal_moves, prior_logits, MASKED_LOGIT)

        next_embedding = {
            'board_3d': next_states.board,
            'history_3d': next_states.history,
            'actual_player': next_states.actual_player,
            'black_out': next_states.black_out,
            'white_out': next_states.white_out,
            'moves_count': next_states.moves_count,
            'iteration': iteration
        }

        return mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=prior_logits,
            # Value stays in the perspective of the player to move at the child.
            # The sign flip between levels is handled by discount = -1.
            value=value
        ), next_embedding


@partial(jax.jit, static_argnames=['network', 'env'])
def get_root_output_batch(states: AbaloneState, network: AbaloneModel, params, env: AbaloneEnv, iteration: int = 0):
    """
    Vectorized version of get_root_output for processing a batch of states

    Args:
        states: Batch of AbaloneState (with batch_size states)
        network: Neural network
        params: Network parameters
        env: Abalone environment
    """
    our_marbles = jnp.where(states.actual_player == 1,
                           states.black_out,
                           states.white_out)
    opp_marbles = jnp.where(states.actual_player == 1,
                           states.white_out,
                           states.black_out)

    board_2d, marbles_out = prepare_input_legacy(states.board, our_marbles, opp_marbles)

    # OPTIMIZED: Use efficient single-vmap conversion with canonicalization
    history_2d = convert_and_canonicalize_history_batch(states.history, states.actual_player)

    prior_logits, value = network.apply(params, board_2d, marbles_out, history_2d)

    embedding = {
        'board_3d': states.board,
        'history_3d': states.history,
        'actual_player': states.actual_player,
        'black_out': states.black_out,
        'white_out': states.white_out,
        'moves_count': states.moves_count,
        'iteration': jnp.full_like(states.actual_player, iteration)
    }

    return mctx.RootFnOutput(
        prior_logits=prior_logits,
        value=value,
        embedding=embedding
    )