import jax
import jax.numpy as jnp
from functools import partial
import time # Utilisé pour rng_key, nous y reviendrons

# Importations locales
from environment.env import AbaloneEnv, AbaloneState
from model.neural_net import AbaloneModel 
from mcts.core import AbaloneMCTSRecurrentFn
from mcts.search import run_search_batch   



@partial(jax.jit, static_argnames=['network', 'env', 'num_simulations'])
def get_best_move(state: AbaloneState,
                  model_variables: dict, # MODIFIÉ: params -> model_variables
                  network: AbaloneModel,
                  env: AbaloneEnv,
                  num_simulations: int = 600,
                  rng_key: jax.random.PRNGKey = None): # Bonne pratique: type hint pour rng_key
    """
    Obtient le meilleur coup à jouer dans un état donné selon MCTS+réseau.
    """

    if rng_key is None:
        rng_key = jax.random.PRNGKey(0) 
    recurrent_fn_instance = AbaloneMCTSRecurrentFn(env, network)

    batch_state = AbaloneState(
        board=state.board[None, ...],
        actual_player=jnp.array([state.actual_player]),
        black_out=jnp.array([state.black_out]),
        white_out=jnp.array([state.white_out]),
        moves_count=jnp.array([state.moves_count])
    )

    # Exécuter la recherche MCTS
    policy_output = run_search_batch(
        states=batch_state, 
        recurrent_fn_instance=recurrent_fn_instance, 
        network=network,
        model_variables=model_variables, 
        rng_key=rng_key,
        env=env,
        num_simulations=num_simulations
    )

    best_action = policy_output.action[0]

    return best_action

@partial(jax.jit, static_argnames=['network', 'env', 'num_simulations', 'temperature']) 
def get_move_probabilities(state: AbaloneState,
                           model_variables: dict, 
                           network: AbaloneModel,
                           env: AbaloneEnv,
                           rng_key: jax.random.PRNGKey, 
                           num_simulations: int = 600,
                           temperature: float = 1.0):
    """
    Retourne les probabilités des coups selon MCTS+réseau.
    Utile pour l'apprentissage ou pour sélectionner un coup de manière stochastique.
    
    Args:
        state: État actuel du jeu
        model_variables: Dictionnaire {'params': ..., 'batch_stats': ...}
        network: Modèle réseau
        env: Environnement du jeu
        rng_key: Clé aléatoire JAX pour la recherche MCTS
        num_simulations: Nombre de simulations MCTS
        temperature: Température pour contrôler l'exploration
        
    Returns:
        Distribution de probabilité sur les coups possibles
    """

    recurrent_fn_instance = AbaloneMCTSRecurrentFn(env, network)
    
    batch_state = AbaloneState(
        board=state.board[None, ...],
        actual_player=jnp.array([state.actual_player]),
        black_out=jnp.array([state.black_out]),
        white_out=jnp.array([state.white_out]),
        moves_count=jnp.array([state.moves_count])
    )
    
    policy_output = run_search_batch(
        states=batch_state,
        recurrent_fn_instance=recurrent_fn_instance,
        network=network,
        model_variables=model_variables, 
        rng_key=rng_key, 
        env=env,
        num_simulations=num_simulations
    )
    
    action_weights = policy_output.action_weights[0]

    if temperature == 0.0:
        max_logit = jnp.max(action_weights)
        adjusted_weights = jnp.where(action_weights == max_logit, 1e6, -1e6)
    else:
        adjusted_weights = action_weights / temperature

    move_probs = jax.nn.softmax(adjusted_weights)
    
    return move_probs


@partial(jax.jit, static_argnames=['network', 'env', 'num_simulations', 'temperature'])
def sample_move(state: AbaloneState,
                model_variables: dict,
                network: AbaloneModel,
                env: AbaloneEnv,
                rng_key: jax.random.PRNGKey, 
                num_simulations: int = 600,
                temperature: float = 1.0):
    """
    Échantillonne un coup selon la distribution de probabilité MCTS.
    Cette fonction est maintenant JIT-compilée.

    Args:
        state: État actuel du jeu (Pytree, tracé par JIT)
        model_variables: Dictionnaire {'params': ..., 'batch_stats': ...} (Pytree de JAX arrays, tracé)
        network: Modèle réseau (statique)
        env: Environnement du jeu (statique)
        rng_key: Clé aléatoire JAX (JAX array, tracé)
        num_simulations: Nombre de simulations MCTS (statique)
        temperature: Température pour contrôler l'exploration (statique)

    Returns:
        Index du coup échantillonné
    """

    mcts_rng_key, choice_rng_key = jax.random.split(rng_key)

    move_probs = get_move_probabilities(
        state, model_variables, network, env, mcts_rng_key, num_simulations, temperature
    )

    move_idx = jax.random.choice(choice_rng_key, jnp.arange(move_probs.shape[-1]), p=move_probs)

    return move_idx