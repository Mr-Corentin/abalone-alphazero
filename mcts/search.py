import mctx
import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, Dict, Any

# Importations locales
from environment.env import AbaloneEnv, AbaloneState
from model.neural_net import AbaloneModel
from core.coord_conversion import cube_to_2d
# Les fonctions de core.py ont été mises à jour pour accepter model_variables
from mcts.core import AbaloneMCTSRecurrentFn, get_root_output_batch


@partial(jax.jit, static_argnames=['recurrent_fn_instance', 'network', 'env', 'num_simulations', 'max_num_considered_actions']) # Renommé recurrent_fn pour éviter confusion
def run_search_batch(states: AbaloneState,
                     recurrent_fn_instance: AbaloneMCTSRecurrentFn, # L'instance de la classe
                     network: AbaloneModel,
                     model_variables: Dict[str, Any], # MODIFIÉ: params -> model_variables
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
    # MODIFIÉ: Passer model_variables à get_root_output_batch
    root = get_root_output_batch(states, network, model_variables, env)

    # Obtenir les masques de mouvements légaux pour le batch
    legal_moves = jax.vmap(env.get_legal_moves)(states)
    invalid_actions = ~legal_moves

    # Lancer la recherche MCTS
    # MODIFIÉ: Passer model_variables à l'argument `params` de mctx.gumbel_muzero_policy,
    # car c'est ce qui sera passé comme premier argument à recurrent_fn_instance.recurrent_fn
    policy_output = mctx.gumbel_muzero_policy(
        params=model_variables, # Ce `params` est pour mctx, il le passera à la recurrent_fn
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn_instance.recurrent_fn, # La méthode de l'instance
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
                             model_variables: Dict[str, Any], # MODIFIÉ: params -> model_variables
                             network: AbaloneModel, # network est l'instance du modèle nn.Module
                             env: AbaloneEnv,
                             batch_size: int,
                             num_simulations: int = 500):
    """
    Génère un batch de parties en utilisant MCTS pour la sélection des actions
    
    Args:
        rng_key: Clé aléatoire JAX
        model_variables: Dictionnaire contenant les 'params' et 'batch_stats' du modèle
        network: Modèle réseau (instance nn.Module)
        env: Environnement du jeu
        batch_size: Nombre de parties à générer en parallèle
        num_simulations: Nombre de simulations MCTS par action
        
    Returns:
        Données des parties générées
    """
    max_moves = 200

    init_states = env.reset_batch(rng_key, batch_size)
    recurrent_fn_instance = AbaloneMCTSRecurrentFn(env, network) # Instance correcte

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
        states, rng, arrays, current_moves_per_game, active = carry # Renommé moves_per_game pour clarté
        boards_3d_arr, boards_2d_arr, actual_players_arr, black_outs_arr, white_outs_arr, moves_counts_arr, policies_arr, is_terminal_states_arr = arrays # Renommé pour clarté

        terminal_states = jax.vmap(env.is_terminal)(states)
        active_games = active & ~terminal_states
        for_terminal = terminal_states & active
        
        new_is_terminal_states = is_terminal_states_arr.at[jnp.arange(batch_size), current_moves_per_game].set(
            jnp.where(for_terminal, True, is_terminal_states_arr[jnp.arange(batch_size), current_moves_per_game])
        )

        rng, search_rng = jax.random.split(rng)
        # MODIFIÉ: Passer model_variables à run_search_batch
        search_outputs = run_search_batch(states, recurrent_fn_instance, network, model_variables, search_rng, env, num_simulations)
        next_states = jax.vmap(env.step)(states, search_outputs.action)

        # ... (logique de stockage reste identique mais utilise les variables renommées pour la clarté) ...
        # Exemple pour new_boards_3d:
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

        final_next_states = AbaloneState( # Renommé pour éviter conflit avec next_states plus haut
            board=jnp.where(active_games[:, None, None, None], next_states.board, states.board),
            actual_player=jnp.where(active_games, next_states.actual_player, states.actual_player),
            black_out=jnp.where(active_games, next_states.black_out, states.black_out),
            white_out=jnp.where(active_games, next_states.white_out, states.white_out),
            moves_count=jnp.where(active_games, next_states.moves_count, states.moves_count)
        )
        new_moves_per_game_updated = jnp.where(active_games, current_moves_per_game + 1, current_moves_per_game) # Renommé

        new_arrays_updated = (new_boards_3d_arr, new_boards_2d_arr, new_actual_players_arr, new_black_outs_arr,
                       new_white_outs_arr, new_moves_counts_arr, new_policies_arr, new_is_terminal_states) # Renommé

        return (final_next_states, rng, new_arrays_updated, new_moves_per_game_updated, active_games)

    def cond_fn(carry):
        _, _, _, _, active = carry
        return jnp.any(active)

    arrays_init = (boards_3d, boards_2d, actual_players, black_outs, white_outs, moves_counts, policies, is_terminal_states) # Renommé
    initial_active = jnp.ones(batch_size, dtype=jnp.bool_)

    final_loop_states, _, final_arrays_loop, final_moves_per_game_loop, _ = jax.lax.while_loop( # Renommé
        cond_fn,
        game_step,
        (init_states, rng_key, arrays_init, moves_per_game, initial_active)
    )
    
    # ... (extraction et essential_data reste identique, utilisant les variables renommées de la sortie du while_loop) ...
    _, final_boards_2d, final_actual_players, final_black_outs, final_white_outs, final_moves_counts, final_policies, final_terminal_states = final_arrays_loop
    
    essential_data = {
        'boards_2d': final_boards_2d,
        'policies': final_policies,
        'moves_per_game': final_moves_per_game_loop, # Utilisez la variable de sortie de la boucle
        'actual_players': final_actual_players,
        'black_outs': final_black_outs,
        'white_outs': final_white_outs,
        'is_terminal': final_terminal_states,
        'final_black_out': final_loop_states.black_out, # Utilisez la variable de sortie de la boucle
        'final_white_out': final_loop_states.white_out, # Utilisez la variable de sortie de la boucle
        'final_player': final_loop_states.actual_player # Utilisez la variable de sortie de la boucle
    }
    return essential_data


# MODIFIÉ: params -> model_variables dans la signature
@partial(jax.pmap, axis_name='device', static_broadcasted_argnums=(2, 3, 4)) # Les indices des argnums statiques restent les mêmes (network, env, batch_size_per_device)
def generate_parallel_games_pmap(rngs,
                                 model_variables: Dict[str, Any], # MODIFIÉ
                                 network: AbaloneModel,
                                 env: AbaloneEnv,
                                 batch_size_per_device: int):
    """
    Version pmappée de generate_game_mcts_batch pour paralléliser sur plusieurs devices
    """
    # MODIFIÉ: Passer model_variables
    return generate_game_mcts_batch(rngs, model_variables, network, env, batch_size_per_device, num_simulations=network.num_simulations_for_search_if_needed_here) # Assurez-vous que num_simulations est bien défini si vous ne le passez pas


def create_optimized_game_generator(num_simulations: int = 500): # num_simulations est défini ici
    """
    Crée une version pmappée de la génération de parties optimisée
    """
    
    # MODIFIÉ: params -> model_variables dans la signature
    @partial(jax.pmap, axis_name='device', static_broadcasted_argnums=(2, 3, 4)) # Les indices des argnums statiques restent les mêmes
    def generate_optimized_games_pmap(rng_key,
                                      model_variables: Dict[str, Any], # MODIFIÉ
                                      network: AbaloneModel,
                                      env: AbaloneEnv,
                                      batch_size_per_device: int):
        """Version optimisée avec lax.scan et filtrage des parties terminées"""

        recurrent_fn_instance = AbaloneMCTSRecurrentFn(env, network) # Doit être défini ici

        def game_step(carry, _):
            states, rng, current_moves_per_game, active, game_data_loop = carry # Renommé

            terminal_states = jax.vmap(env.is_terminal)(states)
            active_games = active & ~terminal_states
            rng, search_rng = jax.random.split(rng)

            # MODIFIÉ: Passer model_variables à run_search_batch
            search_outputs = run_search_batch(states, recurrent_fn_instance, network, model_variables, search_rng, env, num_simulations)
            next_states = jax.vmap(env.step)(states, search_outputs.action)
            current_boards_2d = jax.vmap(cube_to_2d)(states.board)
            
            batch_indices = jnp.arange(batch_size_per_device)
            move_indices = current_moves_per_game
            
            # Mise à jour game_data_loop... (comme avant, mais attention aux noms de variables)
            # Exemple pour 'boards_2d':
            # game_data_loop['boards_2d'] = game_data_loop['boards_2d'].at[batch_indices, move_indices].set(
            #    jnp.where(active_games_mask_for_board, current_boards_2d, game_data_loop['boards_2d'][batch_indices, move_indices])
            # )
            # La logique de mise à jour de game_data doit être revue avec soin ici pour la boucle scan
            # Il est plus simple de construire les tranches à mettre à jour
            
            new_game_data = {}
            for key, value_to_store in [
                ('boards_2d', current_boards_2d),
                ('policies', search_outputs.action_weights), # Assurez-vous que c'est bien softmaxé si besoin
                ('actual_players', states.actual_player),
                ('black_outs', states.black_out),
                ('white_outs', states.white_out),
                ('is_terminal', terminal_states)
            ]:
                mask = active_games
                if len(value_to_store.shape) > 1: # Adapter le masque pour le broadcast
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

            return (final_next_states, rng, new_moves_per_game_updated, active_games, new_game_data), None # Renvoyer None pour la partie "ys" de scan

        init_states = env.reset_batch(rng_key, batch_size_per_device)
        max_moves = 200
        game_data_init = { # Renommé pour clarté
            'boards_2d': jnp.zeros((batch_size_per_device, max_moves, 9, 9), dtype=jnp.int8), # max_moves, pas +1 car scan a une longueur fixe
            'policies': jnp.zeros((batch_size_per_device, max_moves, env.moves_index['positions'].shape[0]), dtype=jnp.float32),
            'actual_players': jnp.zeros((batch_size_per_device, max_moves), dtype=jnp.int32),
            'black_outs': jnp.zeros((batch_size_per_device, max_moves), dtype=jnp.int32),
            'white_outs': jnp.zeros((batch_size_per_device, max_moves), dtype=jnp.int32),
            'is_terminal': jnp.zeros((batch_size_per_device, max_moves), dtype=jnp.bool_)
        }
        active_init = jnp.ones(batch_size_per_device, dtype=jnp.bool_) # Renommé
        moves_per_game_init = jnp.zeros(batch_size_per_device, dtype=jnp.int32) # Renommé

        (final_scan_states, _, final_moves_per_game_scan, _, final_data_scan), _ = jax.lax.scan( # Renommé
            game_step,
            (init_states, rng_key, moves_per_game_init, active_init, game_data_init),
            None, # xs est None car on itère un nombre fixe de fois (max_moves)
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