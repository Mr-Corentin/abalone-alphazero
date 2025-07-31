import mctx
import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Dict, Any
from functools import partial

from environment.env import AbaloneEnv, AbaloneState
from model.neural_net import AbaloneModel
from core.coord_conversion import prepare_input_legacy, cube_to_2d

@partial(jax.jit)
def calculate_reward_terminal_only(current_state: AbaloneState, next_state: AbaloneState) -> float:
    """
    TERMINAL REWARDS ONLY (AlphaZero approach)
    Calculate transition reward - rewards only at game end
    - +1.0 for winning the game (from current player perspective)
    - -1.0 for losing the game (from current player perspective)
    - 0.0 for all other transitions
    """
    game_over = (next_state.black_out >= 6) | (next_state.white_out >= 6)
    
    reward = jnp.where(~game_over, 0.0,
        jnp.where(
            next_state.white_out >= 6,
            1.0 * current_state.actual_player,
            jnp.where(
                next_state.black_out >= 6, 
                -1.0 * current_state.actual_player,
                0.0
            )
        )
    )
    
    return reward

@partial(jax.jit)
def calculate_reward_with_intermediate(current_state: AbaloneState, next_state: AbaloneState, weight: float = 0.1) -> float:
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
    
    game_over = (next_state.black_out >= 6) | (next_state.white_out >= 6)
    
    terminal_reward = jnp.where(~game_over, 0.0,
        jnp.where(
            next_state.white_out >= 6,
            1.0 * current_state.actual_player,
            jnp.where(
                next_state.black_out >= 6, 
                -1.0 * current_state.actual_player,
                0.0
            )
        )
    )
    
    return intermediate_reward + terminal_reward

@partial(jax.jit)
def calculate_reward_curriculum(current_state: AbaloneState, next_state: AbaloneState, iteration: int) -> float:
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

    
    return jnp.where(
        weight > 0.0,
        calculate_reward_with_intermediate(current_state, next_state, weight),
        calculate_reward_terminal_only(current_state, next_state)
    )

# Default function (can be switched for testing)
@partial(jax.jit)
def calculate_reward(current_state: AbaloneState, next_state: AbaloneState, iteration: int) -> float:
    """Current reward function - can be switched between terminal-only and intermediate"""
    #return calculate_reward_terminal_only(current_state, next_state)
    return calculate_reward_curriculum(current_state,next_state, iteration)
    
@partial(jax.jit)
def calculate_discount(state: AbaloneState) -> float:
    """Return discount factor using jnp.where"""
    is_terminal = (state.black_out >= 6) | (state.white_out >= 6) | (state.moves_count >= 300)
    return jnp.where(is_terminal, 0.0, 1.0)


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
        batch_size = action.shape[0]
        current_states = jax.vmap(lambda b: AbaloneState(
            board=embedding['board_3d'][b],
            history=embedding.get('history_3d', jnp.zeros((8,) + embedding['board_3d'][b].shape))[b],
            actual_player=embedding['actual_player'][b],
            black_out=embedding['black_out'][b],
            white_out=embedding['white_out'][b],
            moves_count=embedding['moves_count'][b]
        ))(jnp.arange(batch_size))

        next_states = jax.vmap(self.env.step)(current_states, action)

        iteration = embedding.get('iteration', jnp.zeros_like(current_states.actual_player))
        reward = jax.vmap(calculate_reward)(current_states, next_states, iteration)
        discount = jax.vmap(calculate_discount)(next_states)
        our_marbles = jnp.where(next_states.actual_player == 1,
                               next_states.black_out,
                               next_states.white_out)
        opp_marbles = jnp.where(next_states.actual_player == 1,
                               next_states.white_out,
                               next_states.black_out)

        board_2d, marbles_out = prepare_input_legacy(next_states.board, our_marbles, opp_marbles)
        history_2d = jax.vmap(jax.vmap(cube_to_2d))(next_states.history)

        prior_logits, value = self.network.apply(params, board_2d, marbles_out, history_2d)
        next_embedding = {
            'board_3d': next_states.board,
            'history_3d': next_states.history,
            'actual_player': next_states.actual_player,
            'black_out': next_states.black_out,
            'white_out': next_states.white_out,
            'moves_count': next_states.moves_count,
            'iteration': embedding.get('iteration', jnp.zeros_like(next_states.actual_player))
        }

        return mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=prior_logits,
            value=-value
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
    history_2d = jax.vmap(jax.vmap(cube_to_2d))(states.history)

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