import jax
import jax.numpy as jnp
from functools import partial
import time

# Importations locales
from environment.env import AbaloneEnv, AbaloneState
from model.neural_net import AbaloneModel
from mcts.core import AbaloneMCTSRecurrentFn
from mcts.search import run_search_batch


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

    # Get action with highest score
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
    Return move probabilities according to MCTS+network.
    Useful for training or selecting moves stochastically.
    
    Args:
        state: Current game state
        params: Network parameters
        network: Network model
        env: Game environment
        num_simulations: Number of MCTS simulations
        temperature: Temperature to control exploration (1.0 = more exploratory)
        
    Returns:
        Probability distribution over possible moves
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
    Sample move according to MCTS probability distribution.
    Useful for exploration during training.
    
    Args:
        state: Current game state
        params: Network parameters
        network: Network model
        env: Game environment
        rng_key: JAX random key (if None, key is generated)
        num_simulations: Number of MCTS simulations
        temperature: Temperature to control exploration
        
    Returns:
        Index of sampled move
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