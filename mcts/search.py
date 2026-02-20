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
                    iteration: int = 0,
                    num_simulations: int = 600,
                    max_num_considered_actions: int = 16):
    """
    Batched version of run_search
    
    Args:
        states: Batch of AbaloneState states
        recurrent_fn: Recurrent function for MCTS
        network: Network model
        params: Model parameters
        rng_key: JAX random key
        env: Game environment
        num_simulations: Number of MCTS simulations
        max_num_considered_actions: Maximum number of actions to consider
        
    Returns:
        policy_output for each state in batch
    """
    # Get root for batch
    root = get_root_output_batch(states, network, params, env, iteration)

    # Get legal moves masks for batch
    legal_moves = jax.vmap(env.get_legal_moves)(states)  # shape: (batch_size, num_actions)
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


@partial(jax.jit, static_argnames=['env', 'network', 'num_simulations', 'batch_size'])
def generate_game_mcts_batch(rng_key, params, network, env, batch_size, iteration=0, num_simulations=500):
    """
    Generate batch of games using MCTS for action selection
    
    Args:
        rng_key: JAX random key
        params: Model parameters
        network: Network model
        env: Game environment
        batch_size: Number of games to generate in parallel
        num_simulations: Number of MCTS simulations per action
        
    Returns:
        Generated game data
    """
    max_moves = 300

    # Initial reset for batch
    init_states = env.reset_batch(rng_key, batch_size)

    # Initialize recurrent_fn for MCTS
    recurrent_fn = AbaloneMCTSRecurrentFn(env, network)

    # Pre-allocate buffers - now with 2D boards and history
    # Assume 3D->2D conversion gives 9x9 board
    boards_3d = jnp.zeros((batch_size, max_moves + 1) + init_states.board.shape[1:], dtype=jnp.int8)  # +1 for terminal state
    boards_2d = jnp.zeros((batch_size, max_moves + 1, 9, 9), dtype=jnp.int8)  # Boards converted to 2D, +1 for terminal state
    history_2d = jnp.zeros((batch_size, max_moves + 1, 8, 9, 9), dtype=jnp.int8)  # 2D history with 8 positions
    actual_players = jnp.zeros((batch_size, max_moves + 1), dtype=jnp.int8)
    black_outs = jnp.zeros((batch_size, max_moves + 1), dtype=jnp.int8)
    white_outs = jnp.zeros((batch_size, max_moves + 1), dtype=jnp.int8)
    moves_counts = jnp.zeros((batch_size, max_moves + 1), dtype=jnp.int32)
    policies = jnp.zeros((batch_size, max_moves + 1, env.moves_index['positions'].shape[0]), dtype=jnp.float32)
    is_terminal_states = jnp.zeros((batch_size, max_moves + 1), dtype=jnp.bool_)

    # Array to track number of moves per game
    moves_per_game = jnp.zeros(batch_size, dtype=jnp.int32)

    # Vectorize cube_to_2d conversion function
    batch_cube_to_2d = jax.vmap(cube_to_2d)

    def game_step(carry):
        states, rng, arrays, moves_per_game, active = carry
        boards_3d, boards_2d, history_2d, actual_players, black_outs, white_outs, moves_counts, policies, is_terminal_states = arrays

        # Check terminations
        terminal_states = jax.vmap(env.is_terminal)(states)
        
        # Use active to mask, and also stop if terminal
        active_games = active & ~terminal_states

        # Store terminal state before making new move
        # (This records final state of game when it ends)
        for_terminal = terminal_states & active  # only games that just ended
        
        # Store terminal info for these games
        is_terminal_states = is_terminal_states.at[jnp.arange(batch_size), moves_per_game].set(
            jnp.where(for_terminal, True, is_terminal_states[jnp.arange(batch_size), moves_per_game])
        )

        # MCTS and moves only for active non-terminal games
        rng, search_rng = jax.random.split(rng)
        search_outputs = run_search_batch(states, recurrent_fn, network, params, search_rng, env, iteration, num_simulations)

        # Update states
        next_states = jax.vmap(env.step)(states, search_outputs.action)

        # Store 3D states
        new_boards_3d = boards_3d.at[jnp.arange(batch_size), moves_per_game].set(
            jnp.where(active[:, None, None, None], states.board, boards_3d[jnp.arange(batch_size), moves_per_game])
        )
        
        # Convert and store 2D boards
        current_boards_2d = batch_cube_to_2d(states.board)
        new_boards_2d = boards_2d.at[jnp.arange(batch_size), moves_per_game].set(
            jnp.where(active[:, None, None], current_boards_2d, boards_2d[jnp.arange(batch_size), moves_per_game])
        )
        
        # Convert and store 2D history
        # OPTIMIZED: Use efficient conversion with canonicalization
        # states.history shape: (batch_size, 8, x, y, z) -> (batch_size, 8, 9, 9)
        current_history_2d = convert_and_canonicalize_history_batch(states.history, states.actual_player)
        new_history_2d = history_2d.at[jnp.arange(batch_size), moves_per_game].set(
            jnp.where(active[:, None, None, None], current_history_2d, history_2d[jnp.arange(batch_size), moves_per_game])
        )
        
        # Rest of code identical
        new_actual_players = actual_players.at[jnp.arange(batch_size), moves_per_game].set(
            jnp.where(active, states.actual_player, actual_players[jnp.arange(batch_size), moves_per_game])
        )
        new_black_outs = black_outs.at[jnp.arange(batch_size), moves_per_game].set(
            jnp.where(active, states.black_out, black_outs[jnp.arange(batch_size), moves_per_game])
        )
        new_white_outs = white_outs.at[jnp.arange(batch_size), moves_per_game].set(
            jnp.where(active, states.white_out, white_outs[jnp.arange(batch_size), moves_per_game])
        )
        new_moves_counts = moves_counts.at[jnp.arange(batch_size), moves_per_game].set(
            jnp.where(active, states.moves_count, moves_counts[jnp.arange(batch_size), moves_per_game])
        )
        new_policies = policies.at[jnp.arange(batch_size), moves_per_game].set(
            jnp.where(active_games[:, None], jax.nn.softmax(search_outputs.action_weights),
                     jnp.where(for_terminal[:, None], jnp.zeros_like(search_outputs.action_weights),
                              policies[jnp.arange(batch_size), moves_per_game]))
        )

        # Update state for next turn (only for active non-terminal games)
        final_states = AbaloneState(
            board=jnp.where(active_games[:, None, None, None], next_states.board, states.board),
            history=jnp.where(active_games[:, None, None, None, None], next_states.history, states.history),
            actual_player=jnp.where(active_games, next_states.actual_player, states.actual_player),
            black_out=jnp.where(active_games, next_states.black_out, states.black_out),
            white_out=jnp.where(active_games, next_states.white_out, states.white_out),
            moves_count=jnp.where(active_games, next_states.moves_count, states.moves_count)
        )

        # Increment moves_per_game only for active non-terminal games
        new_moves_per_game = jnp.where(active_games, moves_per_game + 1, moves_per_game)

        new_arrays = (new_boards_3d, new_boards_2d, new_history_2d, new_actual_players, new_black_outs,
                    new_white_outs, new_moves_counts, new_policies, is_terminal_states)

        return (final_states, rng, new_arrays, new_moves_per_game, active_games)

    def cond_fn(carry):
        _, _, _, _, active = carry
        return jnp.any(active)

    arrays = (boards_3d, boards_2d, history_2d, actual_players, black_outs, white_outs, moves_counts, policies, is_terminal_states)
    initial_active = jnp.ones(batch_size, dtype=jnp.bool_)

    final_states, _, final_arrays, final_moves_per_game, _ = jax.lax.while_loop(
        cond_fn,
        game_step,
        (init_states, rng_key, arrays, moves_per_game, initial_active)
    )
    
    # Extract final arrays
    _, final_boards_2d, final_history_2d, final_actual_players, final_black_outs, final_white_outs, final_moves_counts, final_policies, final_terminal_states = final_arrays
    
    # Create pytree dictionary to optimize transfer
    essential_data = {
        'boards_2d': final_boards_2d,          # 2D boards for each move
        'history_2d': final_history_2d,        # 2D history for each move
        'policies': final_policies,            # MCTS policies for each move
        'moves_per_game': final_moves_per_game,  # Actual length of each game
        'actual_players': final_actual_players,  # Active player at each move
        'black_outs': final_black_outs,        # Black marbles out at each move
        'white_outs': final_white_outs,        # White marbles out at each move
        'is_terminal': final_terminal_states,  # Indicator if state is terminal
        'final_black_out': final_states.black_out,  # Black marbles out at end
        'final_white_out': final_states.white_out,  # White marbles out at end
        'final_player': final_states.actual_player  # Last active player
    }


    return essential_data


@partial(jax.pmap, axis_name='device', static_broadcasted_argnums=(2, 3, 4, 5))
def generate_parallel_games_pmap(rngs, params, network, env, batch_size_per_device, iteration=0):
    """
    Pmapped version of generate_game_mcts_batch to parallelize across multiple devices
    """
    return generate_game_mcts_batch(rngs, params, network, env, batch_size_per_device, iteration)



def create_optimized_game_generator(num_simulations=500):
    """
    Create pmapped version of optimized game generation
    
    Args:
        num_simulations: Number of MCTS simulations per move
        
    Returns:
        Pmapped function to generate games
    """
    
    @partial(jax.pmap, axis_name='device', static_broadcasted_argnums=(2, 3, 4, 5))
    def generate_optimized_games_pmap(rng_key, params, network, env, batch_size_per_device, iteration=0):
        """Optimized version with lax.scan and filtering of ended games"""

        def game_step(carry, _):
            states, rng, moves_per_game, active, game_data = carry

            # Ignore already terminated states
            terminal_states = jax.vmap(env.is_terminal)(states)
            active_games = active & ~terminal_states

            # Generate new RNG key
            rng, search_rng = jax.random.split(rng)

            # Execute MCTS only on active states
            search_outputs = run_search_batch(states, recurrent_fn, network, params, search_rng, env, num_simulations)

            # Apply actions to get new states
            next_states = jax.vmap(env.step)(states, search_outputs.action)

            # Calculate data for this turn
            current_boards_2d = jax.vmap(cube_to_2d)(states.board)
            # OPTIMIZED: Add history conversion with canonicalization
            current_history_2d = convert_and_canonicalize_history_batch(states.history, states.actual_player)

            # Update arrays - OPTIMIZED: No Python loop, explicit JAX updates
            batch_indices = jnp.arange(batch_size_per_device)
            move_indices = moves_per_game

            # Update boards_2d
            game_data['boards_2d'] = game_data['boards_2d'].at[batch_indices, move_indices].set(
                jnp.where(active_games[:, None, None], current_boards_2d, game_data['boards_2d'][batch_indices, move_indices])
            )

            # Update history_2d (NEW)
            game_data['history_2d'] = game_data['history_2d'].at[batch_indices, move_indices].set(
                jnp.where(active_games[:, None, None, None], current_history_2d, game_data['history_2d'][batch_indices, move_indices])
            )

            # Update policies
            game_data['policies'] = game_data['policies'].at[batch_indices, move_indices].set(
                jnp.where(active_games[:, None], search_outputs.action_weights, game_data['policies'][batch_indices, move_indices])
            )

            # Update actual_players
            game_data['actual_players'] = game_data['actual_players'].at[batch_indices, move_indices].set(
                jnp.where(active_games, states.actual_player, game_data['actual_players'][batch_indices, move_indices])
            )

            # Update black_outs
            game_data['black_outs'] = game_data['black_outs'].at[batch_indices, move_indices].set(
                jnp.where(active_games, states.black_out, game_data['black_outs'][batch_indices, move_indices])
            )

            # Update white_outs
            game_data['white_outs'] = game_data['white_outs'].at[batch_indices, move_indices].set(
                jnp.where(active_games, states.white_out, game_data['white_outs'][batch_indices, move_indices])
            )

            # Update is_terminal
            game_data['is_terminal'] = game_data['is_terminal'].at[batch_indices, move_indices].set(
                jnp.where(active_games, terminal_states, game_data['is_terminal'][batch_indices, move_indices])
            )

            # Increment move count for active games
            new_moves_per_game = jnp.where(active_games, moves_per_game + 1, moves_per_game)

            return (next_states, rng, new_moves_per_game, active_games, game_data), None

        # Initialize recurrent state for MCTS
        recurrent_fn = AbaloneMCTSRecurrentFn(env, network)
        
        # Initialize states
        init_states = env.reset_batch(rng_key, batch_size_per_device)

        # Pre-allocate data arrays
        max_moves = 300  # Limit maximum number of moves
        game_data = {
            'boards_2d': jnp.zeros((batch_size_per_device, max_moves + 1, 9, 9), dtype=jnp.int8),
            'history_2d': jnp.zeros((batch_size_per_device, max_moves + 1, 8, 9, 9), dtype=jnp.int8),  # ADDED: History storage
            'policies': jnp.zeros((batch_size_per_device, max_moves + 1, 1734), dtype=jnp.float32),
            'actual_players': jnp.zeros((batch_size_per_device, max_moves + 1), dtype=jnp.int32),
            'black_outs': jnp.zeros((batch_size_per_device, max_moves + 1), dtype=jnp.int32),
            'white_outs': jnp.zeros((batch_size_per_device, max_moves + 1), dtype=jnp.int32),
            'is_terminal': jnp.zeros((batch_size_per_device, max_moves + 1), dtype=jnp.bool_)
        }

        # Initialize games
        active = jnp.ones(batch_size_per_device, dtype=jnp.bool_)
        moves_per_game = jnp.zeros(batch_size_per_device, dtype=jnp.int32)

        # Run game simulation with lax.scan (more efficient than while_loop)
        (final_states, _, final_moves_per_game, _, final_data), _ = jax.lax.scan(
            game_step,
            (init_states, rng_key, moves_per_game, active, game_data),
            None,
            length=max_moves
        )

        # Add final states to return dictionary
        essential_data = {
            **final_data,
            'moves_per_game': final_moves_per_game,
            'final_black_out': final_states.black_out,
            'final_white_out': final_states.white_out,
            'final_player': final_states.actual_player
        }

        return essential_data
    
    return generate_optimized_games_pmap