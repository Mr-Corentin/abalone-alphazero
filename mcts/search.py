import mctx
import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, Dict, Any

# Importations locales
from environment.env import AbaloneEnv, AbaloneState
from model.neural_net import AbaloneModel
from core.coord_conversion import cube_to_2d
from mcts.core import AbaloneMCTSRecurrentFn, get_root_output_batch


@partial(jax.jit, static_argnames=['recurrent_fn_instance', 'network', 'env', 'num_simulations', 'max_num_considered_actions'])
def run_search_batch(states: AbaloneState,
                     recurrent_fn_instance: AbaloneMCTSRecurrentFn,
                     network: AbaloneModel,
                     model_variables: Dict[str, Any],
                     rng_key,
                     env: AbaloneEnv,
                     num_simulations: int = 600,
                     max_num_considered_actions: int = 16):
    """
    Version batchée de run_search
    
    Args:
        states: Batch d'états AbaloneState
        recurrent_fn_instance: Instance de la classe AbaloneMCTSRecurrentFn
        network: Modèle réseau
        model_variables: Dictionnaire contenant les 'params' et 'batch_stats' du modèle
        rng_key: Clé aléatoire JAX
        env: Environnement du jeu
        num_simulations: Nombre de simulations MCTS
        max_num_considered_actions: Nombre maximum d'actions à considérer
        
    Returns:
        policy_output pour chaque état du batch
    """
    # Obtenir root pour le batch
    root = get_root_output_batch(states, network, model_variables, env)

    # Obtenir les masques de mouvements légaux pour le batch
    legal_moves = jax.vmap(env.get_legal_moves)(states)
    invalid_actions = ~legal_moves

    # Lancer la recherche MCTS
    policy_output = mctx.gumbel_muzero_policy(
        params=model_variables,
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn_instance.recurrent_fn,
        num_simulations=num_simulations,
        max_num_considered_actions=max_num_considered_actions,
        invalid_actions=invalid_actions,
        gumbel_scale=1.0
    )

    # Créer une structure de sortie simplifiée qui ne contient pas l'arbre de recherche
    lightweight_output = type(policy_output)(
        action=policy_output.action,
        action_weights=policy_output.action_weights,
        search_tree=None
    )

    return lightweight_output


@partial(jax.jit, static_argnames=['env', 'network', 'num_simulations', 'batch_size'])
def generate_game_mcts_batch(rng_key,
                             model_variables: Dict[str, Any],
                             network: AbaloneModel,
                             env: AbaloneEnv,
                             batch_size: int,
                             num_simulations: int = 500,
                             shaping_factor: float = 0.1):  # NOUVEAU PARAMÈTRE
    """
    Génère un batch de parties en utilisant MCTS pour la sélection des actions
    
    Args:
        rng_key: Clé aléatoire JAX
        model_variables: Dictionnaire contenant les 'params' et 'batch_stats' du modèle
        network: Modèle réseau (instance nn.Module)
        env: Environnement du jeu
        batch_size: Nombre de parties à générer en parallèle
        num_simulations: Nombre de simulations MCTS par action
        shaping_factor: Facteur de shaping pour les récompenses intermédiaires  # NOUVEAU
        
    Returns:
        Données des parties générées
    """
    max_moves = 200

    init_states = env.reset_batch(rng_key, batch_size)
    # MODIFIÉ: Créer l'instance avec le facteur de shaping
    recurrent_fn_instance = AbaloneMCTSRecurrentFn(env, network, shaping_factor)

    # ... (pré-allocation des buffers reste identique) ...
    boards_3d = jnp.zeros((batch_size, max_moves + 1) + init_states.board.shape[1:], dtype=jnp.int8)
    boards_2d = jnp.zeros((batch_size, max_moves + 1, 9, 9), dtype=jnp.int8)
    actual_players = jnp.zeros((batch_size, max_moves + 1), dtype=jnp.int8)
    black_outs = jnp.zeros((batch_size, max_moves + 1), dtype=jnp.int8)
    white_outs = jnp.zeros((batch_size, max_moves + 1), dtype=jnp.int8)
    moves_counts = jnp.zeros((batch_size, max_moves + 1), dtype=jnp.int32)
    policies = jnp.zeros((batch_size, max_moves + 1, env.moves_index['positions'].shape[0]), dtype=jnp.float32)
    is_terminal_states = jnp.zeros((batch_size, max_moves + 1), dtype=jnp.bool_)
    moves_per_game = jnp.zeros(batch_size, dtype=jnp.int32)
    batch_cube_to_2d = jax.vmap(cube_to_2d)

    def game_step(carry):
        states, rng, arrays, current_moves_per_game, active = carry
        boards_3d_arr, boards_2d_arr, actual_players_arr, black_outs_arr, white_outs_arr, moves_counts_arr, policies_arr, is_terminal_states_arr = arrays

        terminal_states = jax.vmap(env.is_terminal)(states)
        active_games = active & ~terminal_states
        for_terminal = terminal_states & active
        
        new_is_terminal_states = is_terminal_states_arr.at[jnp.arange(batch_size), current_moves_per_game].set(
            jnp.where(for_terminal, True, is_terminal_states_arr[jnp.arange(batch_size), current_moves_per_game])
        )

        rng, search_rng = jax.random.split(rng)
        search_outputs = run_search_batch(states, recurrent_fn_instance, network, model_variables, search_rng, env, num_simulations)
        next_states = jax.vmap(env.step)(states, search_outputs.action)

        # ... (logique de stockage reste identique mais utilise les variables renommées pour la clarté) ...
        new_boards_3d_arr = boards_3d_arr.at[jnp.arange(batch_size), current_moves_per_game].set(
             jnp.where(active[:, None, None, None], states.board, boards_3d_arr[jnp.arange(batch_size), current_moves_per_game])
        )
        current_boards_2d = batch_cube_to_2d(states.board)
        new_boards_2d_arr = boards_2d_arr.at[jnp.arange(batch_size), current_moves_per_game].set(
            jnp.where(active[:, None, None], current_boards_2d, boards_2d_arr[jnp.arange(batch_size), current_moves_per_game])
        )
        new_actual_players_arr = actual_players_arr.at[jnp.arange(batch_size), current_moves_per_game].set(
            jnp.where(active, states.actual_player, actual_players_arr[jnp.arange(batch_size), current_moves_per_game])
        )
        new_black_outs_arr = black_outs_arr.at[jnp.arange(batch_size), current_moves_per_game].set(
            jnp.where(active, states.black_out, black_outs_arr[jnp.arange(batch_size), current_moves_per_game])
        )
        new_white_outs_arr = white_outs_arr.at[jnp.arange(batch_size), current_moves_per_game].set(
            jnp.where(active, states.white_out, white_outs_arr[jnp.arange(batch_size), current_moves_per_game])
        )
        new_moves_counts_arr = moves_counts_arr.at[jnp.arange(batch_size), current_moves_per_game].set(
            jnp.where(active, states.moves_count, moves_counts_arr[jnp.arange(batch_size), current_moves_per_game])
        )
        new_policies_arr = policies_arr.at[jnp.arange(batch_size), current_moves_per_game].set(
            jnp.where(active_games[:, None], jax.nn.softmax(search_outputs.action_weights),
                      jnp.where(for_terminal[:, None], jnp.zeros_like(search_outputs.action_weights),
                                policies_arr[jnp.arange(batch_size), current_moves_per_game]))
        )

        final_next_states = AbaloneState(
            board=jnp.where(active_games[:, None, None, None], next_states.board, states.board),
            actual_player=jnp.where(active_games, next_states.actual_player, states.actual_player),
            black_out=jnp.where(active_games, next_states.black_out, states.black_out),
            white_out=jnp.where(active_games, next_states.white_out, states.white_out),
            moves_count=jnp.where(active_games, next_states.moves_count, states.moves_count)
        )
        new_moves_per_game_updated = jnp.where(active_games, current_moves_per_game + 1, current_moves_per_game)

        new_arrays_updated = (new_boards_3d_arr, new_boards_2d_arr, new_actual_players_arr, new_black_outs_arr,
                       new_white_outs_arr, new_moves_counts_arr, new_policies_arr, new_is_terminal_states)

        return (final_next_states, rng, new_arrays_updated, new_moves_per_game_updated, active_games)

    def cond_fn(carry):
        _, _, _, _, active = carry
        return jnp.any(active)

    arrays_init = (boards_3d, boards_2d, actual_players, black_outs, white_outs, moves_counts, policies, is_terminal_states)
    initial_active = jnp.ones(batch_size, dtype=jnp.bool_)

    final_loop_states, _, final_arrays_loop, final_moves_per_game_loop, _ = jax.lax.while_loop(
        cond_fn,
        game_step,
        (init_states, rng_key, arrays_init, moves_per_game, initial_active)
    )
    
    # ... (extraction et essential_data reste identique) ...
    _, final_boards_2d, final_actual_players, final_black_outs, final_white_outs, final_moves_counts, final_policies, final_terminal_states = final_arrays_loop
    
    essential_data = {
        'boards_2d': final_boards_2d,
        'policies': final_policies,
        'moves_per_game': final_moves_per_game_loop,
        'actual_players': final_actual_players,
        'black_outs': final_black_outs,
        'white_outs': final_white_outs,
        'is_terminal': final_terminal_states,
        'final_black_out': final_loop_states.black_out,
        'final_white_out': final_loop_states.white_out,
        'final_player': final_loop_states.actual_player
    }
    return essential_data


@partial(jax.pmap, axis_name='device', static_broadcasted_argnums=(2, 3, 4))
def generate_parallel_games_pmap(rngs,
                                 model_variables: Dict[str, Any],
                                 network: AbaloneModel,
                                 env: AbaloneEnv,
                                 batch_size_per_device: int,
                                 shaping_factor: float = 0.1):  # NOUVEAU PARAMÈTRE
    """
    Version pmappée de generate_game_mcts_batch pour paralléliser sur plusieurs devices
    """
    # MODIFIÉ: Passer shaping_factor ET model_variables
    return generate_game_mcts_batch(rngs, model_variables, network, env, batch_size_per_device, 
                                   num_simulations=500, shaping_factor=shaping_factor)


def create_optimized_game_generator(num_simulations: int = 500, initial_shaping_factor: float = 0.1):  # NOUVEAU PARAMÈTRE
    """
    Crée une version pmappée de la génération de parties optimisée
    
    Args:
        num_simulations: Nombre de simulations MCTS par action
        initial_shaping_factor: Facteur initial pour le reward shaping  # NOUVEAU
    """
    
    @partial(jax.pmap, axis_name='device', static_broadcasted_argnums=(2, 3, 4))
    def generate_optimized_games_pmap(rng_key,
                                      model_variables: Dict[str, Any],
                                      network: AbaloneModel,
                                      env: AbaloneEnv,
                                      batch_size_per_device: int,
                                      shaping_factor: float = initial_shaping_factor):  # NOUVEAU PARAMÈTRE
        """Version optimisée avec lax.scan et filtrage des parties terminées"""

        # MODIFIÉ: Créer l'instance avec le facteur de shaping
        recurrent_fn_instance = AbaloneMCTSRecurrentFn(env, network, shaping_factor)

        def game_step(carry, _):
            states, rng, current_moves_per_game, active, game_data_loop = carry

            terminal_states = jax.vmap(env.is_terminal)(states)
            active_games = active & ~terminal_states
            rng, search_rng = jax.random.split(rng)

            search_outputs = run_search_batch(states, recurrent_fn_instance, network, model_variables, search_rng, env, num_simulations)
            next_states = jax.vmap(env.step)(states, search_outputs.action)
            current_boards_2d = jax.vmap(cube_to_2d)(states.board)
            
            batch_indices = jnp.arange(batch_size_per_device)
            move_indices = current_moves_per_game
            
            new_game_data = {}
            for key, value_to_store in [
                ('boards_2d', current_boards_2d),
                ('policies', jax.nn.softmax(search_outputs.action_weights)),  # S'assurer que c'est softmaxé
                ('actual_players', states.actual_player),
                ('black_outs', states.black_out),
                ('white_outs', states.white_out),
                ('is_terminal', terminal_states)
            ]:
                mask = active_games
                if len(value_to_store.shape) > 1:  # Adapter le masque pour le broadcast
                    mask = mask.reshape(-1, *([1] * (len(value_to_store.shape) - 1)))
                
                new_game_data[key] = game_data_loop[key].at[batch_indices, move_indices].set(
                    jnp.where(mask, value_to_store, game_data_loop[key][batch_indices, move_indices])
                )
            
            # S'assurer que tous les champs de game_data sont mis à jour ou passés
            for key_original in game_data_loop:
                if key_original not in new_game_data:
                    new_game_data[key_original] = game_data_loop[key_original]

            new_moves_per_game_updated = jnp.where(active_games, current_moves_per_game + 1, current_moves_per_game)

            # Mettre à jour states pour le prochain tour seulement pour les parties actives
            final_next_states = AbaloneState(
                board=jnp.where(active_games[:, None, None, None], next_states.board, states.board),
                actual_player=jnp.where(active_games, next_states.actual_player, states.actual_player),
                black_out=jnp.where(active_games, next_states.black_out, states.black_out),
                white_out=jnp.where(active_games, next_states.white_out, states.white_out),
                moves_count=jnp.where(active_games, next_states.moves_count, states.moves_count)
            )

            return (final_next_states, rng, new_moves_per_game_updated, active_games, new_game_data), None

        init_states = env.reset_batch(rng_key, batch_size_per_device)
        max_moves = 200
        game_data_init = {
            'boards_2d': jnp.zeros((batch_size_per_device, max_moves, 9, 9), dtype=jnp.int8),
            'policies': jnp.zeros((batch_size_per_device, max_moves, env.moves_index['positions'].shape[0]), dtype=jnp.float32),
            'actual_players': jnp.zeros((batch_size_per_device, max_moves), dtype=jnp.int32),
            'black_outs': jnp.zeros((batch_size_per_device, max_moves), dtype=jnp.int32),
            'white_outs': jnp.zeros((batch_size_per_device, max_moves), dtype=jnp.int32),
            'is_terminal': jnp.zeros((batch_size_per_device, max_moves), dtype=jnp.bool_)
        }
        active_init = jnp.ones(batch_size_per_device, dtype=jnp.bool_)
        moves_per_game_init = jnp.zeros(batch_size_per_device, dtype=jnp.int32)

        (final_scan_states, _, final_moves_per_game_scan, _, final_data_scan), _ = jax.lax.scan(
            game_step,
            (init_states, rng_key, moves_per_game_init, active_init, game_data_init),
            None,
            length=max_moves
        )

        essential_data = {
            **final_data_scan,
            'moves_per_game': final_moves_per_game_scan,
            'final_black_out': final_scan_states.black_out,
            'final_white_out': final_scan_states.white_out,
            'final_player': final_scan_states.actual_player
        }
        return essential_data
    
    return generate_optimized_games_pmap