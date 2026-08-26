import mctx
import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, Dict, Any

# Local imports
from environment.env import AbaloneEnv, AbaloneState
from model.neural_net import AbaloneModel
from core.coord_conversion import cube_to_2d, convert_and_canonicalize_history_batch
from mcts.core import AbaloneMCTSRecurrentFn, get_root_output_batch


@partial(jax.jit, static_argnames=['recurrent_fn', 'network', 'env', 'num_simulations', 'max_num_considered_actions'])
def run_search_batch(states: AbaloneState,
                    recurrent_fn: AbaloneMCTSRecurrentFn,
                    network: AbaloneModel,
                    params,
                    rng_key,
                    env: AbaloneEnv,
                    *,
                    num_simulations: int,
                    max_num_considered_actions: int,
                    iteration: int = 0):
    """
    Batched version of run_search

    `num_simulations` and `max_num_considered_actions` are keyword-only and have
    no default on purpose: a positional call once passed the simulation count in
    the `iteration` slot, and the default silently pinned every search to 600
    simulations no matter what the config said. Forgetting them is now a
    TypeError instead of a silent wrong number.

    Args:
        states: Batch of AbaloneState states
        recurrent_fn: Recurrent function for MCTS
        network: Network model
        params: Model parameters
        rng_key: JAX random key
        env: Game environment
        iteration: Training iteration (used by curriculum rewards)
        num_simulations: Number of MCTS simulations
        max_num_considered_actions: Maximum number of actions to consider

    Returns:
        policy_output for each state in batch
    """
    # Get root for batch
    root = get_root_output_batch(states, network, params, env, iteration)

    # Get legal moves masks for batch
    legal_moves = env.get_legal_moves_batch(states)  # shape: (batch_size, num_actions)
    invalid_actions = ~legal_moves  # shape: (batch_size, num_actions)

    # Launch MCTS search
    policy_output = mctx.gumbel_muzero_policy(
        params=params,
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn.recurrent_fn,
        num_simulations=num_simulations,
        max_num_considered_actions=max_num_considered_actions,
        invalid_actions=invalid_actions,
        gumbel_scale=1.0
    )

    # Create simplified output structure without search tree
    lightweight_output = type(policy_output)(
        action=policy_output.action,
        action_weights=policy_output.action_weights,
        search_tree=None
    )

    return lightweight_output


@partial(jax.jit, static_argnames=['env', 'network', 'num_simulations',
                                   'max_num_considered_actions', 'batch_size'])
def generate_game_mcts_batch(rng_key, params, network, env, batch_size,
                             *, num_simulations, max_num_considered_actions,
                             iteration=0):
    """
    Generate a batch of self-play games using MCTS for action selection.

    Games that reach a terminal state are frozen: their state stops being stepped,
    nothing more is written for them, and the loop exits as soon as every game in
    the batch is done. This matters -- an earlier version kept stepping finished
    games for the rest of the fixed 300-step horizon, so `final_black_out` /
    `final_white_out` both climbed past 6 and the recorded winner was wrong for
    every decisive game.

    Args:
        rng_key: JAX random key
        params: Model parameters
        network: Network model
        env: Game environment
        batch_size: Number of games to generate in parallel
        iteration: Training iteration (traced, so changing it does not recompile)
        num_simulations: Number of MCTS simulations per move
        max_num_considered_actions: Root actions expanded by Gumbel MuZero

    Returns:
        Generated game data
    """
    max_moves = env.max_moves
    num_actions = env.moves_index['positions'].shape[0]

    # Initial reset for batch
    init_states = env.reset_batch(rng_key, batch_size)

    # Initialize recurrent_fn for MCTS
    recurrent_fn = AbaloneMCTSRecurrentFn(env, network)

    # Pre-allocate buffers (+1 slot to also record the terminal state)
    boards_2d = jnp.zeros((batch_size, max_moves + 1, 9, 9), dtype=jnp.int8)
    history_2d = jnp.zeros((batch_size, max_moves + 1, env.history_length, 9, 9), dtype=jnp.int8)
    actual_players = jnp.zeros((batch_size, max_moves + 1), dtype=jnp.int8)
    black_outs = jnp.zeros((batch_size, max_moves + 1), dtype=jnp.int8)
    white_outs = jnp.zeros((batch_size, max_moves + 1), dtype=jnp.int8)
    policies = jnp.zeros((batch_size, max_moves + 1, num_actions), dtype=jnp.float32)
    is_terminal_states = jnp.zeros((batch_size, max_moves + 1), dtype=jnp.bool_)

    moves_per_game = jnp.zeros(batch_size, dtype=jnp.int32)
    batch_indices = jnp.arange(batch_size)

    def game_step(carry):
        states, rng, arrays, moves_per_game, active = carry
        (boards_2d, history_2d, actual_players, black_outs,
         white_outs, policies, is_terminal_states) = arrays

        terminal_states = env.is_terminal_batch(states)
        active_games = active & ~terminal_states

        rng, search_rng = jax.random.split(rng)
        search_outputs = run_search_batch(
            states, recurrent_fn, network, params, search_rng, env,
            iteration=iteration,
            num_simulations=num_simulations,
            max_num_considered_actions=max_num_considered_actions,
        )

        next_states = jax.vmap(env.step)(states, search_outputs.action)

        slot = (batch_indices, moves_per_game)

        def write(buffer, value, mask):
            """Write `value` at the current slot of every still-running game."""
            extra_dims = value.ndim - mask.ndim
            mask = mask.reshape(mask.shape + (1,) * extra_dims)
            return buffer.at[slot].set(jnp.where(mask, value, buffer[slot]))

        # `active` (not `active_games`): the terminal position is recorded too,
        # it is a valid training input even though no move follows it.
        boards_2d = write(boards_2d, jax.vmap(cube_to_2d)(states.board).astype(jnp.int8), active)
        history_2d = write(
            history_2d,
            convert_and_canonicalize_history_batch(states.history, states.actual_player).astype(jnp.int8),
            active)
        actual_players = write(actual_players, states.actual_player.astype(jnp.int8), active)
        black_outs = write(black_outs, states.black_out.astype(jnp.int8), active)
        white_outs = write(white_outs, states.white_out.astype(jnp.int8), active)
        is_terminal_states = write(is_terminal_states, terminal_states, active)

        # mctx's action_weights is ALREADY a probability distribution (it is a
        # softmax over completed search logits, zero on illegal actions). Running
        # softmax on it again flattened the target to near-uniform over all 1734
        # actions, i.e. ~97% of the target mass on illegal moves.
        policies = write(policies, search_outputs.action_weights,
                         active_games)

        # Freeze finished games: their state must stop evolving.
        final_states = jax.tree.map(
            lambda nxt, cur: jnp.where(
                active_games.reshape(active_games.shape + (1,) * (nxt.ndim - 1)), nxt, cur),
            next_states, states)

        new_moves_per_game = jnp.where(active_games, moves_per_game + 1, moves_per_game)

        new_arrays = (boards_2d, history_2d, actual_players, black_outs,
                      white_outs, policies, is_terminal_states)

        return (final_states, rng, new_arrays, new_moves_per_game, active_games)

    def cond_fn(carry):
        return jnp.any(carry[4])

    arrays = (boards_2d, history_2d, actual_players, black_outs,
              white_outs, policies, is_terminal_states)
    initial_active = jnp.ones(batch_size, dtype=jnp.bool_)

    final_states, _, final_arrays, final_moves_per_game, _ = jax.lax.while_loop(
        cond_fn,
        game_step,
        (init_states, rng_key, arrays, moves_per_game, initial_active)
    )

    (final_boards_2d, final_history_2d, final_actual_players, final_black_outs,
     final_white_outs, final_policies, final_terminal_states) = final_arrays

    return {
        'boards_2d': final_boards_2d,            # 2D boards for each move
        'history_2d': final_history_2d,          # 2D history for each move
        'policies': final_policies,              # MCTS policies for each move
        'moves_per_game': final_moves_per_game,  # Actual length of each game
        'actual_players': final_actual_players,  # Active player at each move
        'black_outs': final_black_outs,          # Black marbles out at each move
        'white_outs': final_white_outs,          # White marbles out at each move
        'is_terminal': final_terminal_states,    # Indicator if state is terminal
        'final_black_out': final_states.black_out,   # Black marbles out at end
        'final_white_out': final_states.white_out,   # White marbles out at end
        'final_player': final_states.actual_player,  # Last active player
    }


def create_optimized_game_generator(num_simulations: int, max_num_considered_actions: int = 16):
    """
    Create the pmapped self-play generator.

    `iteration` is deliberately NOT a static argument: making it static forced a
    full recompilation of the whole search loop on every training iteration.

    Args:
        num_simulations: Number of MCTS simulations per move
        max_num_considered_actions: Root actions expanded by Gumbel MuZero

    Returns:
        Pmapped function (rng_keys, params, network, env, batch_size, iterations)
    """

    @partial(jax.pmap, axis_name='device', static_broadcasted_argnums=(2, 3, 4))
    def generate_games_pmap(rng_key, params, network, env, batch_size_per_device, iteration):
        return generate_game_mcts_batch(
            rng_key, params, network, env, batch_size_per_device,
            iteration=iteration,
            num_simulations=num_simulations,
            max_num_considered_actions=max_num_considered_actions,
        )

    return generate_games_pmap
