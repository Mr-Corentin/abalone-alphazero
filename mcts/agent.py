import jax
import jax.numpy as jnp
from functools import partial
import time

# Importations locales
from environment.env import AbaloneEnv, AbaloneState
from model.neural_net import AbaloneModel
from mcts.core import AbaloneMCTSRecurrentFn
from mcts.search import run_search_batch


# @partial(jax.jit, static_argnames=['network', 'env', 'num_simulations'])
# def get_best_move(state: AbaloneState,
#                  params,
#                  network: AbaloneModel,
#                  env: AbaloneEnv,
#                  num_simulations: int = 600):
#     """
#     Obtient le meilleur coup à jouer dans un état donné selon MCTS+réseau.
    
#     Args:
#         state: État actuel du jeu
#         params: Paramètres du réseau
#         network: Modèle réseau
#         env: Environnement du jeu
#         num_simulations: Nombre de simulations MCTS
        
#     Returns:
#         Index du meilleur coup à jouer
#     """
#     # Créer une clé RNG
#     rng_key = jax.random.PRNGKey(int(time.time() * 1000) % (2**32))

#     # Créer le recurrent_fn pour MCTS
#     recurrent_fn = AbaloneMCTSRecurrentFn(env, network)

#     # Transformer l'état en batch de taille 1
#     batch_state = AbaloneState(
#         board=state.board[None, ...],
#         actual_player=jnp.array([state.actual_player]),
#         black_out=jnp.array([state.black_out]),
#         white_out=jnp.array([state.white_out]),
#         moves_count=jnp.array([state.moves_count])
#     )

#     # Exécuter la recherche MCTS
#     policy_output = run_search_batch(
#         batch_state,
#         recurrent_fn,
#         network,
#         params,
#         rng_key,
#         env,
#         num_simulations=num_simulations
#     )

#     # Récupérer l'action avec le score le plus élevé
#     best_action = policy_output.action[0]

#     return best_action
@partial(jax.jit, static_argnames=['network', 'env', 'num_simulations'])
def get_best_move(state: AbaloneState,
                 params,
                 network: AbaloneModel,
                 env: AbaloneEnv,
                 num_simulations: int = 600,
                 rng_key=None,
                 iteration: int = 10):
    """
    Obtient le meilleur coup à jouer dans un état donné selon MCTS+réseau.
    """
    # Utiliser la clé RNG fournie ou en créer une par défaut
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    # Créer le recurrent_fn pour MCTS
    recurrent_fn = AbaloneMCTSRecurrentFn(env, network)

    # Transformer l'état en batch de taille 1
    batch_state = AbaloneState(
        board=state.board[None, ...],
        history=state.history[None, ...],
        actual_player=jnp.array([state.actual_player]),
        black_out=jnp.array([state.black_out]),
        white_out=jnp.array([state.white_out]),
        moves_count=jnp.array([state.moves_count])
    )

    # Exécuter la recherche MCTS
    policy_output = run_search_batch(
        batch_state,
        recurrent_fn,
        network,
        params,
        rng_key,
        env,
        iteration,
        num_simulations
    )

    # Récupérer l'action avec le score le plus élevé
    best_action = policy_output.action[0]

    return best_action

@partial(jax.jit, static_argnames=['network', 'env', 'num_simulations'])
def get_move_probabilities(state: AbaloneState,
                          params,
                          network: AbaloneModel,
                          env: AbaloneEnv,
                          num_simulations: int = 600,
                          temperature: float = 1.0,
                          iteration: int = 10):
    """
    Retourne les probabilités des coups selon MCTS+réseau.
    Utile pour l'apprentissage ou pour sélectionner un coup de manière stochastique.
    
    Args:
        state: État actuel du jeu
        params: Paramètres du réseau
        network: Modèle réseau
        env: Environnement du jeu
        num_simulations: Nombre de simulations MCTS
        temperature: Température pour contrôler l'exploration (1.0 = plus exploratoire)
        
    Returns:
        Distribution de probabilité sur les coups possibles
    """
    rng_key = jax.random.PRNGKey(int(time.time() * 1000) % (2**32))
    recurrent_fn = AbaloneMCTSRecurrentFn(env, network)
    
    # Transformer l'état en batch de taille 1
    batch_state = AbaloneState(
        board=state.board[None, ...],
        history=state.history[None, ...],
        actual_player=jnp.array([state.actual_player]),
        black_out=jnp.array([state.black_out]),
        white_out=jnp.array([state.white_out]),
        moves_count=jnp.array([state.moves_count])
    )
    
    # Exécuter la recherche MCTS
    policy_output = run_search_batch(
        batch_state,
        recurrent_fn,
        network,
        params,
        rng_key,
        env,
        iteration,
        num_simulations
    )
    
    # Obtenir les poids des actions et appliquer la température
    action_weights = policy_output.action_weights[0]
    if temperature != 1.0:
        # Ajuster par la température
        action_weights = action_weights / temperature
    
    # Convertir en probabilités
    move_probs = jax.nn.softmax(action_weights)
    
    return move_probs


def sample_move(state: AbaloneState, 
                params, 
                network: AbaloneModel, 
                env: AbaloneEnv, 
                rng_key=None, 
                num_simulations: int = 600, 
                temperature: float = 1.0,
                iteration: int = 10):
    """
    Échantillonne un coup selon la distribution de probabilité MCTS.
    Utile pour l'exploration pendant l'entraînement.
    
    Args:
        state: État actuel du jeu
        params: Paramètres du réseau
        network: Modèle réseau
        env: Environnement du jeu
        rng_key: Clé aléatoire JAX (si None, une clé est générée)
        num_simulations: Nombre de simulations MCTS
        temperature: Température pour contrôler l'exploration
        
    Returns:
        Index du coup échantillonné
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(int(time.time() * 1000) % (2**32))
    
    # Obtenir les probabilités des coups
    move_probs = get_move_probabilities(
        state, params, network, env, num_simulations, temperature, iteration
    )
    
    # Échantillonner un coup selon cette distribution
    move_idx = jax.random.choice(rng_key, len(move_probs), p=move_probs)
    
    return move_idx