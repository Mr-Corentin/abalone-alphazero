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

        # 1. Construction des états en batch AVEC l'historique
        current_states = jax.vmap(lambda b: AbaloneState(
            board=embedding['board_3d'][b],
            history=embedding['history_3d'][b],  # ← AJOUT DE L'HISTORIQUE
            actual_player=embedding['actual_player'][b],
            black_out=embedding['black_out'][b],
            white_out=embedding['white_out'][b],
            moves_count=embedding['moves_count'][b]
        ))(jnp.arange(batch_size))

        # 2. Application des actions en batch
        next_states = jax.vmap(self.env.step)(current_states, action)

        # 3. Calcul des rewards et discounts en batch
        # Utiliser le facteur de shaping actuel
        current_iteration = embedding['current_iteration'][0]  # Prendre le premier élément
        total_iterations = embedding['total_iterations'][0]    # Prendre le premier élément

        reward = jax.vmap(lambda cs, ns: calculate_reward(cs, ns, current_iteration, total_iterations))(
            current_states, next_states
        )
        discount = jax.vmap(calculate_discount)(next_states)

        # 4. Préparation des entrées réseau AVEC l'historique
        our_marbles = jnp.where(next_states.actual_player == 1,
                              next_states.black_out,
                              next_states.white_out)
        opp_marbles = jnp.where(next_states.actual_player == 1,
                              next_states.white_out,
                              next_states.black_out)

        # ← MISE À JOUR: Ajouter next_states.history
        board_2d, marbles_out = prepare_input(next_states.board, next_states.history, next_states.actual_player, our_marbles, opp_marbles)

        prior_logits, value = self.network.apply(
            model_variables,
            board_2d,
            marbles_out,
            train=False 
        )

        # 5. Embedding suivant AVEC l'historique
        next_embedding = {
            'board_3d': next_states.board,
            'history_3d': next_states.history,  # ← AJOUT DE L'HISTORIQUE
            'actual_player': next_states.actual_player,
            'black_out': next_states.black_out,
            'white_out': next_states.white_out,
            'moves_count': next_states.moves_count,
            'current_iteration': embedding['current_iteration'],  
            'total_iterations': embedding['total_iterations']     
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
    """
    # Calculer les billes out pour chaque état dans le batch
    our_marbles = jnp.where(states.actual_player == 1,
                            states.black_out,
                            states.white_out)
    opp_marbles = jnp.where(states.actual_player == 1,
                            states.white_out,
                            states.black_out)

    # Préparer les entrées du réseau AVEC l'historique
    # ← MISE À JOUR: Ajouter states.history
    board_2d, marbles_out = prepare_input(states.board, states.history, states.actual_player, our_marbles, opp_marbles)

    # Obtenir les prédictions du réseau
    prior_logits, value = network.apply(
        model_variables,
        board_2d,
        marbles_out,
        train=False 
    )

    batch_size = states.board.shape[0]
    current_iteration_batch = jnp.full(batch_size, current_iteration, dtype=jnp.int32)
    total_iterations_batch = jnp.full(batch_size, total_iterations, dtype=jnp.int32)

    # Embedding initial AVEC l'historique
    embedding = {
        'board_3d': states.board,
        'history_3d': states.history,  # ← AJOUT DE L'HISTORIQUE
        'actual_player': states.actual_player,
        'black_out': states.black_out,
        'white_out': states.white_out,
        'moves_count': states.moves_count,
        'current_iteration': current_iteration_batch,  
        'total_iterations': total_iterations_batch     
    }

    return mctx.RootFnOutput(
        prior_logits=prior_logits,
        value=value,
        embedding=embedding
    )