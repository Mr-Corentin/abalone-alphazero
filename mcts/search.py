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


@partial(jax.jit, static_argnames=['recurrent_fn', 'network', 'env', 'num_simulations', 'max_num_considered_actions'])
def run_search_batch(states: AbaloneState,
                    recurrent_fn: AbaloneMCTSRecurrentFn,
                    network: AbaloneModel,
                    params,
                    rng_key,
                    env: AbaloneEnv,
                    num_simulations: int = 600,
                    max_num_considered_actions: int = 16):
    """
    Version batchée de run_search
    
    Args:
        states: Batch d'états AbaloneState
        recurrent_fn: Fonction récurrente pour MCTS
        network: Modèle réseau
        params: Paramètres du modèle
        rng_key: Clé aléatoire JAX
        env: Environnement du jeu
        num_simulations: Nombre de simulations MCTS
        max_num_considered_actions: Nombre maximum d'actions à considérer
        
    Returns:
        policy_output pour chaque état du batch
    """
    # Obtenir root pour le batch
    root = get_root_output_batch(states, network, params, env)

    # Obtenir les masques de mouvements légaux pour le batch
    legal_moves = jax.vmap(env.get_legal_moves)(states)  # shape: (batch_size, num_actions)
    invalid_actions = ~legal_moves  # shape: (batch_size, num_actions)

    # Lancer la recherche MCTS
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

    # Créer une structure de sortie simplifiée qui ne contient pas l'arbre de recherche
    lightweight_output = type(policy_output)(
        action=policy_output.action,
        action_weights=policy_output.action_weights,
        search_tree=None  
    )

    return lightweight_output


@partial(jax.jit, static_argnames=['env', 'network', 'num_simulations', 'batch_size'])
def generate_game_mcts_batch(rng_key, params, network, env, batch_size, num_simulations=500):
    """
    Génère un batch de parties en utilisant MCTS pour la sélection des actions
    
    Args:
        rng_key: Clé aléatoire JAX
        params: Paramètres du modèle
        network: Modèle réseau
        env: Environnement du jeu
        batch_size: Nombre de parties à générer en parallèle
        num_simulations: Nombre de simulations MCTS par action
        
    Returns:
        Données des parties générées
    """
    max_moves = 200

    # Reset initial pour le batch
    init_states = env.reset_batch(rng_key, batch_size)

    # Initialiser recurrent_fn pour MCTS
    recurrent_fn = AbaloneMCTSRecurrentFn(env, network)

    # Pré-allouer les buffers - maintenant avec des plateaux 2D
    # Supposons que la conversion 3D->2D donne un plateau 9x9
    boards_3d = jnp.zeros((batch_size, max_moves + 1) + init_states.board.shape[1:], dtype=jnp.int8)  # +1 pour l'état terminal
    boards_2d = jnp.zeros((batch_size, max_moves + 1, 9, 9), dtype=jnp.int8)  # Plateaux convertis en 2D, +1 pour l'état terminal
    actual_players = jnp.zeros((batch_size, max_moves + 1), dtype=jnp.int8)
    black_outs = jnp.zeros((batch_size, max_moves + 1), dtype=jnp.int8)
    white_outs = jnp.zeros((batch_size, max_moves + 1), dtype=jnp.int8)
    moves_counts = jnp.zeros((batch_size, max_moves + 1), dtype=jnp.int32)
    policies = jnp.zeros((batch_size, max_moves + 1, env.moves_index['positions'].shape[0]), dtype=jnp.float32)
    is_terminal_states = jnp.zeros((batch_size, max_moves + 1), dtype=jnp.bool_)

    # Tableau pour suivre le nombre de coups par partie
    moves_per_game = jnp.zeros(batch_size, dtype=jnp.int32)

    # Vectoriser la fonction de conversion cube_to_2d
    batch_cube_to_2d = jax.vmap(cube_to_2d)

    def game_step(carry):
        states, rng, arrays, moves_per_game, active = carry
        boards_3d, boards_2d, actual_players, black_outs, white_outs, moves_counts, policies, is_terminal_states = arrays

        # Vérifier les terminaisons
        terminal_states = jax.vmap(env.is_terminal)(states)
        
        # Utiliser active pour masquer, et aussi arrêter si terminal
        active_games = active & ~terminal_states

        # Stocker l'état terminal avant de faire un nouveau mouvement
        # (Ceci enregistre l'état final d'une partie lorsqu'elle se termine)
        for_terminal = terminal_states & active  # seulement les parties qui viennent juste de terminer
        
        # Stocker info terminal pour ces parties
        is_terminal_states = is_terminal_states.at[jnp.arange(batch_size), moves_per_game].set(
            jnp.where(for_terminal, True, is_terminal_states[jnp.arange(batch_size), moves_per_game])
        )

        # MCTS et coups seulement pour les parties actives non-terminales
        rng, search_rng = jax.random.split(rng)
        search_outputs = run_search_batch(states, recurrent_fn, network, params, search_rng, env, num_simulations)

        # Mise à jour des états
        next_states = jax.vmap(env.step)(states, search_outputs.action)

        # Stocker les états 3D
        new_boards_3d = boards_3d.at[jnp.arange(batch_size), moves_per_game].set(
            jnp.where(active[:, None, None, None], states.board, boards_3d[jnp.arange(batch_size), moves_per_game])
        )
        
        # Convertir et stocker les plateaux 2D
        current_boards_2d = batch_cube_to_2d(states.board)
        new_boards_2d = boards_2d.at[jnp.arange(batch_size), moves_per_game].set(
            jnp.where(active[:, None, None], current_boards_2d, boards_2d[jnp.arange(batch_size), moves_per_game])
        )
        
        # Reste du code identique
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

        # Mettre à jour l'état pour le prochain tour (seulement pour les parties actives non-terminales)
        final_states = AbaloneState(
            board=jnp.where(active_games[:, None, None, None], next_states.board, states.board),
            actual_player=jnp.where(active_games, next_states.actual_player, states.actual_player),
            black_out=jnp.where(active_games, next_states.black_out, states.black_out),
            white_out=jnp.where(active_games, next_states.white_out, states.white_out),
            moves_count=jnp.where(active_games, next_states.moves_count, states.moves_count)
        )

        # Incrémenter moves_per_game seulement pour les parties actives non-terminales
        new_moves_per_game = jnp.where(active_games, moves_per_game + 1, moves_per_game)

        new_arrays = (new_boards_3d, new_boards_2d, new_actual_players, new_black_outs,
                    new_white_outs, new_moves_counts, new_policies, is_terminal_states)

        return (final_states, rng, new_arrays, new_moves_per_game, active_games)

    def cond_fn(carry):
        _, _, _, _, active = carry
        return jnp.any(active)

    arrays = (boards_3d, boards_2d, actual_players, black_outs, white_outs, moves_counts, policies, is_terminal_states)
    initial_active = jnp.ones(batch_size, dtype=jnp.bool_)

    final_states, _, final_arrays, final_moves_per_game, _ = jax.lax.while_loop(
        cond_fn,
        game_step,
        (init_states, rng_key, arrays, moves_per_game, initial_active)
    )
    
    # Extraction des tableaux finaux
    _, final_boards_2d, final_actual_players, final_black_outs, final_white_outs, final_moves_counts, final_policies, final_terminal_states = final_arrays
    
    # Création d'un dictionnaire pytree pour optimiser le transfert
    essential_data = {
        'boards_2d': final_boards_2d,          # Plateaux 2D pour chaque mouvement
        'policies': final_policies,            # Politiques MCTS pour chaque mouvement
        'moves_per_game': final_moves_per_game,  # Longueur réelle de chaque partie
        'actual_players': final_actual_players,  # Joueur actif à chaque mouvement
        'black_outs': final_black_outs,        # Billes noires sorties à chaque mouvement
        'white_outs': final_white_outs,        # Billes blanches sorties à chaque mouvement
        'is_terminal': final_terminal_states,  # Indicateur si l'état est terminal
        'final_black_out': final_states.black_out,  # Billes noires sorties à la fin
        'final_white_out': final_states.white_out,  # Billes blanches sorties à la fin
        'final_player': final_states.actual_player  # Dernier joueur actif
    }


    return essential_data


@partial(jax.pmap, axis_name='device', static_broadcasted_argnums=(2, 3, 4))
def generate_parallel_games_pmap(rngs, params, network, env, batch_size_per_device):
    """
    Version pmappée de generate_game_mcts_batch pour paralléliser sur plusieurs devices
    """
    return generate_game_mcts_batch(rngs, params, network, env, batch_size_per_device)



def create_optimized_game_generator(num_simulations=500):
    """
    Crée une version pmappée de la génération de parties optimisée
    
    Args:
        num_simulations: Nombre de simulations MCTS par mouvement
        
    Returns:
        Fonction pmappée pour générer des parties
    """
    
    @partial(jax.pmap, axis_name='device', static_broadcasted_argnums=(2, 3, 4))
    def generate_optimized_games_pmap(rng_key, params, network, env, batch_size_per_device):
        """Version optimisée avec lax.scan et filtrage des parties terminées"""

        def game_step(carry, _):
            states, rng, moves_per_game, active, game_data = carry

            # Ignorer les états déjà terminés
            terminal_states = jax.vmap(env.is_terminal)(states)
            active_games = active & ~terminal_states

            # Générer une nouvelle clé RNG
            rng, search_rng = jax.random.split(rng)

            # Exécuter MCTS uniquement sur les états actifs
            search_outputs = run_search_batch(states, recurrent_fn, network, params, search_rng, env, num_simulations)

            # Appliquer les actions pour obtenir les nouveaux états
            next_states = jax.vmap(env.step)(states, search_outputs.action)

            # Calculer les données pour ce tour
            current_boards_2d = jax.vmap(cube_to_2d)(states.board)
            
            # Mise à jour des tableaux avec approche vectorisée
            batch_indices = jnp.arange(batch_size_per_device)
            move_indices = moves_per_game
            
            # Mettre à jour les données du jeu
            for key, value in [
                ('boards_2d', current_boards_2d),
                ('policies', search_outputs.action_weights),
                ('actual_players', states.actual_player),
                ('black_outs', states.black_out),
                ('white_outs', states.white_out),
                ('is_terminal', terminal_states)
            ]:
                # Créer un masque adapté à la forme de la valeur
                mask = active_games
                if len(value.shape) > 1:
                    reshape_dims = [len(value.shape) - 1]
                    mask = mask.reshape(-1, *([1] * reshape_dims[0]))
                
                # Mettre à jour le tableau
                game_data[key] = game_data[key].at[batch_indices, move_indices].set(
                    jnp.where(mask, value, game_data[key][batch_indices, move_indices])
                )

            # Incrémenter le nombre de coups pour les parties actives
            new_moves_per_game = jnp.where(active_games, moves_per_game + 1, moves_per_game)

            return (next_states, rng, new_moves_per_game, active_games, game_data), None

        # Initialiser l'état récurrent pour MCTS
        recurrent_fn = AbaloneMCTSRecurrentFn(env, network)
        
        # Initialiser les états
        init_states = env.reset_batch(rng_key, batch_size_per_device)

        # Pré-allouer les tableaux de données
        max_moves = 200  # Limiter le nombre maximum de coups
        game_data = {
            'boards_2d': jnp.zeros((batch_size_per_device, max_moves + 1, 9, 9), dtype=jnp.int8),
            'policies': jnp.zeros((batch_size_per_device, max_moves + 1, 1734), dtype=jnp.float32),
            'actual_players': jnp.zeros((batch_size_per_device, max_moves + 1), dtype=jnp.int32),
            'black_outs': jnp.zeros((batch_size_per_device, max_moves + 1), dtype=jnp.int32),
            'white_outs': jnp.zeros((batch_size_per_device, max_moves + 1), dtype=jnp.int32),
            'is_terminal': jnp.zeros((batch_size_per_device, max_moves + 1), dtype=jnp.bool_)
        }

        # Initialiser les parties
        active = jnp.ones(batch_size_per_device, dtype=jnp.bool_)
        moves_per_game = jnp.zeros(batch_size_per_device, dtype=jnp.int32)

        # Exécuter la simulation des parties avec lax.scan (plus efficace que while_loop)
        (final_states, _, final_moves_per_game, _, final_data), _ = jax.lax.scan(
            game_step,
            (init_states, rng_key, moves_per_game, active, game_data),
            None,
            length=max_moves
        )

        # Ajouter les états finaux au dictionnaire de retour
        essential_data = {
            **final_data,
            'moves_per_game': final_moves_per_game,
            'final_black_out': final_states.black_out,
            'final_white_out': final_states.white_out,
            'final_player': final_states.actual_player
        }

        return essential_data
    
    return generate_optimized_games_pmap