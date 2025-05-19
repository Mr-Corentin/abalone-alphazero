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

REWARD_SCALING_FACTOR = 0.1 

# @partial(jax.jit)
# def calculate_reward(current_state: AbaloneState, next_state: AbaloneState) -> float:
#     """
#     Calcule la récompense d'une transition en version canonique
#     """
#     black_diff = next_state.black_out - current_state.black_out
#     white_diff = next_state.white_out - current_state.white_out

#     billes_sorties = jnp.where(current_state.actual_player == 1,
#                                white_diff,
#                                black_diff)

#     return billes_sorties * REWARD_SCALING_FACTOR 


# Dans votre calculate_reward
@partial(jax.jit)
def calculate_reward(current_state: AbaloneState, next_state: AbaloneState) -> float:
    # Reward intermédiaire (comme maintenant)
    # black_diff = next_state.black_out - current_state.black_out  
    # white_diff = next_state.white_out - current_state.white_out
    # intermediate_reward = jnp.where(current_state.actual_player == 1, white_diff, black_diff) * 0.1
    intermediate_reward = 0
    is_terminal = (next_state.black_out >= 6) | (next_state.white_out >= 6) | (next_state.moves_count >= 200)
    final_reward = jnp.where(
        is_terminal,
        jnp.where(
            next_state.white_out >= 6, 1.0,    # Noir gagne
            jnp.where(next_state.black_out >= 6, -1.0, 0.0)  # Blanc gagne / Match nul
        ),
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
        self.network = network # network est une instance de AbaloneModel

    @partial(jax.jit, static_argnums=(0,))
    # Modifier la signature pour accepter model_variables
    def recurrent_fn(self, model_variables: Dict[str, Any], rng_key, action, embedding):
        """
        Fonction récurrente pour MCTS qui gère un batch d'états

        Args:
            model_variables: Dict contenant les 'params' et 'batch_stats' du réseau
            rng_key: Clé JAX RNG
            action: Actions à appliquer (shape: (batch_size,))
            embedding: Dict contenant l'état du batch
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
        reward = jax.vmap(calculate_reward)(current_states, next_states)
        discount = jax.vmap(calculate_discount)(next_states)

        # 4. Préparation des entrées du réseau en batch
        our_marbles = jnp.where(next_states.actual_player == 1,
                                next_states.black_out,
                                next_states.white_out)
        opp_marbles = jnp.where(next_states.actual_player == 1,
                                next_states.white_out,
                                next_states.black_out)

        board_2d, marbles_out = prepare_input(next_states.board, our_marbles, opp_marbles)

        # 5. Évaluation par le réseau
        # MODIFICATION ICI: Utiliser model_variables et ajouter train=False
        prior_logits, value = self.network.apply(
            model_variables, # Contient {'params': ..., 'batch_stats': ...}
            board_2d,
            marbles_out,
            train=False # Important: mode inférence pour MCTS
        )

        # 6. Préparation du prochain embedding
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
def get_root_output_batch(states: AbaloneState, network: AbaloneModel, model_variables: Dict[str, Any], env: AbaloneEnv):
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
        'moves_count': states.moves_count
    }

    return mctx.RootFnOutput(
        prior_logits=prior_logits,
        value=value,
        embedding=embedding
    )