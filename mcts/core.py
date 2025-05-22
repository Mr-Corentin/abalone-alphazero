import mctx
import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Dict, Any # Assurez-vous que Any est bien importé
from functools import partial

# Importations locales
from environment.env import AbaloneEnv, AbaloneState
from model.neural_net import AbaloneModel # Votre AbaloneModel mis à jour
from core.coord_conversion import prepare_input

# REWARD_SCALING_FACTOR = 0.1 

# INITIAL_SHAPING_FACTOR = 0.1  # Ou la valeur de départ que vous préférez
# K_POLYNOMIAL = 2.0  # Pour une décroissance quadratique (lent puis rapide), k > 1
# # TOTAL_TRAINING_EPOCHS sera défini dans votre boucle d'entraînement
# # SHAPING_DURATION_RATIO = 0.3 # 30% de l'entraînement total

# @jax.jit
# def get_dynamic_shaping_factor(
#     current_epoch: jnp.ndarray,
#     total_training_epochs: jnp.ndarray,
#     shaping_duration_ratio: jnp.ndarray,
#     initial_factor: jnp.ndarray,
#     k_polynomial: jnp.ndarray
# ) -> jnp.ndarray:
#     """
#     Version JAX de la fonction - tous les paramètres doivent être des tableaux JAX
#     """
#     shaping_active_epochs = shaping_duration_ratio * total_training_epochs
    
#     progress_t = current_epoch / shaping_active_epochs
#     factor = initial_factor * (1.0 - progress_t**k_polynomial)
    
#     # Utiliser jnp.where au lieu de if/else
#     return jnp.where(current_epoch < shaping_active_epochs, factor, 0.0)
    

# @partial(jax.jit)
# def calculate_reward(current_state: AbaloneState, next_state: AbaloneState, shaping_factor: float) -> float: # Valeur par défaut enlevée pour clarté
#     """
#     Calcule la récompense avec un facteur de shaping configurable et dynamique.
#     """
#     # 1. Reward intermédiaire pour billes sorties
#     black_diff = next_state.black_out - current_state.black_out
#     white_diff = next_state.white_out - current_state.white_out

#     billes_sorties_adv = jnp.where(current_state.actual_player == 1,
#                                    white_diff,
#                                    black_diff)

#     intermediate_reward = billes_sorties_adv * shaping_factor

#     # 2. Reward finale (victoire/défaite) - reste identique
#     is_terminal = (next_state.black_out >= 6) | (next_state.white_out >= 6) | (next_state.moves_count >= 200)

#     raw_final_reward = jnp.where(
#         next_state.white_out >= 6, 1.0,
#         jnp.where(next_state.black_out >= 6, -1.0, 0.0)
#     )

#     final_reward = jnp.where(
#         is_terminal,
#         raw_final_reward * current_state.actual_player,
#         0.0
#     )
#     return intermediate_reward + final_reward

@partial(jax.jit)
def calculate_reward(current_state: AbaloneState, next_state: AbaloneState, current_iteration: int, total_iterations: int) -> float:
    """
    Calcule la récompense avec reward shaping par paliers basé sur l'itération actuelle.
    Reward shaping réduit progressivement et s'arrête à 60% de l'entraînement.
    """
    # 1. Reward intermédiaire pour billes sorties (avec paliers)
    black_diff = next_state.black_out - current_state.black_out
    white_diff = next_state.white_out - current_state.white_out

    billes_sorties_adv = jnp.where(current_state.actual_player == 1,
                                   white_diff,
                                   black_diff)

    # Calcul du facteur de shaping par paliers
    progress = current_iteration / total_iterations
    
    shaping_factor = jnp.where(
        progress < 0.2, 0.1,          # 0-20% : facteur 0.1
        jnp.where(
            progress < 0.4, 0.05,     # 20-40% : facteur 0.05  
            jnp.where(
                progress < 0.6, 0.02, # 40-60% : facteur 0.02
                0.0                   # 60%+ : pas de reward shaping
            )
        )
    )

    intermediate_reward = billes_sorties_adv * shaping_factor

    # 2. Reward finale (victoire/défaite) - reste identique
    is_terminal = (next_state.black_out >= 6) | (next_state.white_out >= 6) | (next_state.moves_count >= 200)

    raw_final_reward = jnp.where(
        next_state.white_out >= 6, 1.0,
        jnp.where(next_state.black_out >= 6, -1.0, 0.0)
    )

    final_reward = jnp.where(
        is_terminal,
        raw_final_reward * current_state.actual_player,
        0.0
    )
    
    return intermediate_reward + final_reward

@partial(jax.jit)
def calculate_discount(state: AbaloneState) -> float:
    """
    Retourne le facteur d'atténuation en utilisant jnp.where
    """
    is_terminal = (state.black_out >= 6) | (state.white_out >= 6) | (state.moves_count >= 200)
    return jnp.where(is_terminal, 0.0, 1.0)

class AbaloneMCTSRecurrentFn:
    """
    Classe pour la fonction récurrente de MCTS, utilisant mctx
    """
    def __init__(self, env: AbaloneEnv, network: AbaloneModel):
        self.env = env
        self.network = network

    @partial(jax.jit, static_argnums=(0,))
    def recurrent_fn(self, model_variables: Dict[str, Any], rng_key, action, embedding):
        """
        Fonction récurrente pour MCTS qui gère un batch d'états
        """
        batch_size = action.shape[0]

        # 1. Construction des états en batch
        current_states = jax.vmap(lambda b: AbaloneState(
            board=embedding['board_3d'][b],
            actual_player=embedding['actual_player'][b],
            black_out=embedding['black_out'][b],
            white_out=embedding['white_out'][b],
            moves_count=embedding['moves_count'][b]
        ))(jnp.arange(batch_size))

        # 2. Application des actions en batch
        next_states = jax.vmap(self.env.step)(current_states, action)

        # 3. Calcul des rewards et discounts en batch
        # Utiliser le facteur de shaping actuel
      
        current_iteration = embedding.get('current_iteration', 0)
        total_iterations = embedding.get('total_iterations', 1000)

        reward = jax.vmap(lambda cs, ns: calculate_reward(cs, ns, current_iteration, total_iterations))(
            current_states, next_states
        )
        discount = jax.vmap(calculate_discount)(next_states)

        # Reste du code reste identique...
        our_marbles = jnp.where(next_states.actual_player == 1,
                              next_states.black_out,
                              next_states.white_out)
        opp_marbles = jnp.where(next_states.actual_player == 1,
                              next_states.white_out,
                              next_states.black_out)

        board_2d, marbles_out = prepare_input(next_states.board, our_marbles, opp_marbles)

        prior_logits, value = self.network.apply(
            model_variables,
            board_2d,
            marbles_out,
            train=False 
        )

        next_embedding = {
            'board_3d': next_states.board,
            'actual_player': next_states.actual_player,
            'black_out': next_states.black_out,
            'white_out': next_states.white_out,
            'moves_count': next_states.moves_count
        }

        return mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=prior_logits,
            value=value
        ), next_embedding

# Modifier la signature pour accepter model_variables
@partial(jax.jit, static_argnames=['network', 'env'])
def get_root_output_batch(states: AbaloneState, network: AbaloneModel, model_variables: Dict[str, Any], env: AbaloneEnv, current_iteration: int = 0, total_iterations: int = 1000):

    """
    Version vectorisée de get_root_output pour traiter un batch d'états

    Args:
        states: Batch d'états AbaloneState (avec batch_size états)
        network: Le réseau de neurones
        model_variables: Dict contenant les 'params' et 'batch_stats' du réseau
        env: L'environnement Abalone
    """
    # Calculer les billes out pour chaque état dans le batch
    our_marbles = jnp.where(states.actual_player == 1,
                            states.black_out,
                            states.white_out)
    opp_marbles = jnp.where(states.actual_player == 1,
                            states.white_out,
                            states.black_out)

    # Préparer les entrées du réseau
    board_2d, marbles_out = prepare_input(states.board, our_marbles, opp_marbles)

    # Obtenir les prédictions du réseau
    prior_logits, value = network.apply(
        model_variables,
        board_2d,
        marbles_out,
        train=False 
    )

    embedding = {
        'board_3d': states.board,
        'actual_player': states.actual_player,
        'black_out': states.black_out,
        'white_out': states.white_out,
        'moves_count': states.moves_count,
        'current_iteration': current_iteration,  
        'total_iterations': total_iterations    
    }

    return mctx.RootFnOutput(
        prior_logits=prior_logits,
        value=value,
        embedding=embedding
    )