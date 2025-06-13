import mctx
import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Dict, Any
from functools import partial

# Importations locales
from environment.env import AbaloneEnv, AbaloneState
from model.neural_net import AbaloneModel
from core.coord_conversion import prepare_input

@partial(jax.jit)
def calculate_reward(current_state: AbaloneState, next_state: AbaloneState) -> float:
    """
    Calcule la récompense d'une transition en version canonique
    - 1.0 pour gagner la partie
    - 0.1 pour chaque bille adverse sortie
    """
    black_diff = next_state.black_out - current_state.black_out
    white_diff = next_state.white_out - current_state.white_out

    # Billes sorties par le joueur courant
    billes_sorties = jnp.where(current_state.actual_player == 1,
                              white_diff,  # Noir sort des blanches
                              black_diff)  # Blanc sort des noires

    # Récompense pour les billes sorties
    marble_reward = billes_sorties * 0.1

    # Récompense pour gagner la partie (6 billes sorties)
    game_won = (next_state.black_out >= 6) | (next_state.white_out >= 6)
    winning_reward = jnp.where(game_won, 1.0, 0.0)

    return marble_reward + winning_reward

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
    def recurrent_fn(self, params, rng_key, action, embedding):
        """
        Fonction récurrente pour MCTS qui gère un batch d'états

        Args:
            params: Paramètres du réseau
            rng_key: Clé JAX RNG
            action: Actions à appliquer (shape: (batch_size,))
            embedding: Dict contenant l'état du batch
        """
        batch_size = action.shape[0]

        # 1. Construction des états en batch
        current_states = jax.vmap(lambda b: AbaloneState(
            board=embedding['board_3d'][b],
            history=embedding.get('history_3d', jnp.zeros((8,) + embedding['board_3d'][b].shape))[b],
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

        # Pour MCTS, utiliser la version legacy sans historique pour l'instant
        from core.coord_conversion import prepare_input_legacy
        board_2d, marbles_out = prepare_input_legacy(next_states.board, our_marbles, opp_marbles)

        # 5. Évaluation par le réseau
        prior_logits, value = self.network.apply(params, board_2d, marbles_out)

        # 6. Préparation du prochain embedding
        next_embedding = {
            'board_3d': next_states.board,
            'history_3d': next_states.history,
            'actual_player': next_states.actual_player,
            'black_out': next_states.black_out,
            'white_out': next_states.white_out,
            'moves_count': next_states.moves_count
        }

        return mctx.RecurrentFnOutput(
            reward=reward,  # shape: (batch_size,)
            discount=discount,  # shape: (batch_size,)
            prior_logits=prior_logits,  # shape: (batch_size, num_actions)
            value=value  # shape: (batch_size,)
        ), next_embedding


@partial(jax.jit, static_argnames=['network', 'env'])
def get_root_output_batch(states: AbaloneState, network: AbaloneModel, params, env: AbaloneEnv):
    """
    Version vectorisée de get_root_output pour traiter un batch d'états

    Args:
        states: Batch d'états AbaloneState (avec batch_size états)
        network: Le réseau de neurones
        params: Paramètres du réseau
        env: L'environnement Abalone
    """
    # Calculer les billes out pour chaque état dans le batch
    our_marbles = jnp.where(states.actual_player == 1,
                           states.black_out,
                           states.white_out)
    opp_marbles = jnp.where(states.actual_player == 1,
                           states.white_out,
                           states.black_out)

    # Préparer les entrées du réseau - utiliser la version legacy pour MCTS
    from core.coord_conversion import prepare_input_legacy
    board_2d, marbles_out = prepare_input_legacy(states.board, our_marbles, opp_marbles)

    # Obtenir les prédictions du réseau
    prior_logits, value = network.apply(params, board_2d, marbles_out)

    embedding = {
        'board_3d': states.board,  # shape: (batch_size, x, y, z)
        'history_3d': states.history,  # shape: (batch_size, 8, x, y, z)
        'actual_player': states.actual_player,  # shape: (batch_size,)
        'black_out': states.black_out,  # shape: (batch_size,)
        'white_out': states.white_out,  # shape: (batch_size,)
        'moves_count': states.moves_count  # shape: (batch_size,)
    }

    return mctx.RootFnOutput(
        prior_logits=prior_logits,  # shape: (batch_size, num_actions)
        value=value,  # shape: (batch_size,)
        embedding=embedding
    )