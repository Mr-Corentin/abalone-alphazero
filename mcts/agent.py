import jax
import jax.numpy as jnp
from functools import partial
import time

# Importations locales
from environment.env import AbaloneEnv, AbaloneState
from model.neural_net import AbaloneModel
from mcts.core import AbaloneMCTSRecurrentFn
from mcts.search import run_search_batch


@partial(jax.jit, static_argnames=['network', 'env', 'num_simulations', 'max_num_considered_actions'])
def get_best_move(state: AbaloneState,
                 params,
                 network: AbaloneModel,
                 env: AbaloneEnv,
                 num_simulations: int,
                 rng_key=None,
                 max_num_considered_actions: int = 64,
                 iteration: int = 0):
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
        num_simulations=num_simulations,
        max_num_considered_actions=max_num_considered_actions,
        iteration=iteration,
    )

    # Get action with highest score
    best_action = policy_output.action[0]

    return best_action

@partial(jax.jit, static_argnames=['network', 'env', 'num_simulations', 'temperature', 'max_num_considered_actions'])
def get_move_probabilities(state: AbaloneState,
                          params,
                          network: AbaloneModel,
                          env: AbaloneEnv,
                          num_simulations: int,
                          temperature: float = 1.0,
                          max_num_considered_actions: int = 64,
                          iteration: int = 0):
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
        num_simulations=num_simulations,
        max_num_considered_actions=max_num_considered_actions,
        iteration=iteration,
    )
    
    # action_weights est DEJA une distribution de probabilite (mctx applique un
    # softmax sur les logits de recherche completes, et met zero sur les coups
    # illegaux). Le re-passer dans un softmax aplatissait la distribution sur les
    # 1734 actions -- ~97% de la masse allait sur des coups illegaux.
    move_probs = policy_output.action_weights[0]

    if temperature != 1.0:
        # Temperature appliquee a la distribution, pas a des logits : p^(1/T),
        # renormalise. T -> 0 concentre sur le meilleur coup, T grand aplatit.
        # Les coups illegaux ont une proba nulle et le restent.
        support = move_probs > 0
        logits = jnp.where(support, jnp.log(jnp.where(support, move_probs, 1.0)) / temperature, -jnp.inf)
        move_probs = jax.nn.softmax(logits)

    return move_probs


def sample_move(state: AbaloneState, 
                params, 
                network: AbaloneModel, 
                env: AbaloneEnv, 
                rng_key=None,
                num_simulations: int = None,
                temperature: float = 1.0,
                max_num_considered_actions: int = 64,
                iteration: int = 0):
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
        state, params, network, env,
        num_simulations=num_simulations,
        temperature=temperature,
        max_num_considered_actions=max_num_considered_actions,
        iteration=iteration,
    )
    
    # Échantillonner un coup selon cette distribution
    move_idx = jax.random.choice(rng_key, len(move_probs), p=move_probs)
    
    return move_idx