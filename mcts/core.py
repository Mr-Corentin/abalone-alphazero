import mctx
import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Dict, Any
from functools import partial

# Importations locales
from environment.env import AbaloneEnv, AbaloneState
from model.neural_net import AbaloneModel
from core.coord_conversion import prepare_input, prepare_input_legacy, cube_to_2d

# @partial(jax.jit)
# def calculate_reward_intermediate(current_state: AbaloneState, next_state: AbaloneState) -> float:
#     """
#     INTERMEDIATE REWARDS VERSION (commented out)
#     Calcule la récompense d'une transition en version canonique
#     - 1.0 pour gagner la partie
#     - 0.1 pour chaque bille adverse sortie
#     """
#     black_diff = next_state.black_out - current_state.black_out
#     white_diff = next_state.white_out - current_state.white_out

#     # Billes sorties par le joueur courant
#     billes_sorties = jnp.where(current_state.actual_player == 1,
#                               white_diff,  # Noir sort des blanches
#                               black_diff)  # Blanc sort des noires

#     # Récompense pour les billes sorties
#     marble_reward = billes_sorties * 0.1

#     # Récompense pour gagner la partie (6 billes sorties)
#     game_won = (next_state.black_out >= 6) | (next_state.white_out >= 6)
#     winning_reward = jnp.where(game_won, 1.0, 0.0)

#     return marble_reward + winning_reward

@partial(jax.jit)
def calculate_reward_terminal_only(current_state: AbaloneState, next_state: AbaloneState) -> float:
    """
    TERMINAL REWARDS ONLY (AlphaZero approach) - VERSION CANONIQUE
    Calcule la récompense d'une transition - rewards seulement à la fin de partie
    - +1.0 pour gagner la partie (depuis la perspective du joueur courant)
    - -1.0 pour perdre la partie (depuis la perspective du joueur courant)
    - 0.0 pour toutes les autres transitions
    
    Note: Utilise la même logique que trainer.py avec outcome * player
    """
    # Vérifier si la partie est terminée
    game_over = (next_state.black_out >= 6) | (next_state.white_out >= 6)
    
    # Si la partie n'est pas finie, pas de reward
    reward = jnp.where(~game_over, 0.0,
        # Si la partie est finie, calculer l'outcome comme dans trainer.py
        jnp.where(
            next_state.white_out >= 6,
            1.0 * current_state.actual_player,   # Black wins → outcome=+1, adjust by actual_player
            jnp.where(
                next_state.black_out >= 6, 
                -1.0 * current_state.actual_player,  # White wins → outcome=-1, adjust by actual_player
                0.0  # Draw (ne devrait pas arriver ici mais par sécurité)
            )
        )
    )
    
    return reward

@partial(jax.jit)
def calculate_reward_with_intermediate(current_state: AbaloneState, next_state: AbaloneState, weight: float = 0.1) -> float:
    """
    INTERMEDIATE REWARDS VERSION - FOR TESTING
    Calcule la récompense avec récompenses intermédiaires pour pousser des billes
    - +1.0 pour gagner la partie
    - +weight pour chaque bille adverse poussée
    - -weight pour chaque bille propre perdue
    - -1.0 pour perdre la partie
    
    Version canonique: toujours depuis la perspective du joueur courant
    """
    # Calculer les changements de billes
    black_diff = next_state.black_out - current_state.black_out
    white_diff = next_state.white_out - current_state.white_out
    
    # Depuis la perspective du joueur courant (canonical)
    # Si actual_player = 1 (Black), on veut pousser des blanches
    # Si actual_player = -1 (White), on veut pousser des noires
    
    opponent_marbles_pushed = jnp.where(
        current_state.actual_player == 1,
        white_diff,  # Black player: pushed white marbles
        black_diff   # White player: pushed black marbles
    )
    
    # Récompenses intermédiaires
    intermediate_reward = weight * opponent_marbles_pushed
    
    # Vérifier si la partie est terminée pour récompense finale
    game_over = (next_state.black_out >= 6) | (next_state.white_out >= 6)
    
    # Récompense finale (terminal)
    terminal_reward = jnp.where(~game_over, 0.0,
        jnp.where(
            next_state.white_out >= 6,
            1.0 * current_state.actual_player,   # Black wins → +1 for black, -1 for white
            jnp.where(
                next_state.black_out >= 6, 
                -1.0 * current_state.actual_player,  # White wins → -1 for black, +1 for white
                0.0
            )
        )
    )
    
    return intermediate_reward + terminal_reward

@partial(jax.jit)
def calculate_reward_curriculum(current_state: AbaloneState, next_state: AbaloneState, iteration: int) -> float:
    """
    CURRICULUM REWARD VERSION
    Switches between intermediate and terminal rewards based on iteration:
    - Iterations 0-4: weight = 0.1 (full intermediate rewards)
    - Iterations 5-9: weight = 0.05 (reduced intermediate rewards)
    - Iterations 10+: weight = 0.0 (terminal only, pure AlphaZero)
    """
    # weight = jnp.where(
    #     iteration < 10,
    #     0.3,  # First 10 iterations: full intermediate
    #     jnp.where(
    #         iteration < 20,
    #         0.15,  # Next 5 iterations: reduced intermediate
    #         0.0    # After iteration 20: terminal only
    #     )
    # )

    weight = jnp.where(
      iteration < 10,
      1.0,   # First 15 iterations: strong intermediate rewards
      jnp.where(
          iteration < 15,
          0.5,   # Iterations 15-30: reduced intermediate rewards
          jnp.where(
              iteration < 30,
              0.1,   # Iterations 30-50: minimal intermediate rewards
              0.0    # After iteration 50: pure AlphaZero
          )
      )
  )

    
    return jnp.where(
        weight > 0.0,
        calculate_reward_with_intermediate(current_state, next_state, weight),
        calculate_reward_terminal_only(current_state, next_state)
    )

# Default function (can be switched for testing)
def calculate_reward(current_state: AbaloneState, next_state: AbaloneState) -> float:
    """Current reward function - can be switched between terminal-only and intermediate"""
    return calculate_reward_terminal_only(current_state, next_state)

@partial(jax.jit)
def calculate_discount(state: AbaloneState) -> float:
    """
    Retourne le facteur d'atténuation en utilisant jnp.where
    """
    is_terminal = (state.black_out >= 6) | (state.white_out >= 6) | (state.moves_count >= 300)
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
        reward = jax.vmap(calculate_reward_terminal_only)(current_states, next_states)
        discount = jax.vmap(calculate_discount)(next_states)

        # 4. Préparation des entrées du réseau en batch
        our_marbles = jnp.where(next_states.actual_player == 1,
                               next_states.black_out,
                               next_states.white_out)
        opp_marbles = jnp.where(next_states.actual_player == 1,
                               next_states.white_out,
                               next_states.black_out)

        # Utiliser la version legacy pour MCTS (compatible avec le réseau initialisé avec history)
        board_2d, marbles_out = prepare_input_legacy(next_states.board, our_marbles, opp_marbles)
        
        # Convertir l'historique en 2D pour le réseau
        history_2d = jax.vmap(jax.vmap(cube_to_2d))(next_states.history)  # (batch, 8, 9, 9)

        # 5. Évaluation par le réseau avec historique
        prior_logits, value = self.network.apply(params, board_2d, marbles_out, history_2d)

        # 6. Préparation du prochain embedding
        next_embedding = {
            'board_3d': next_states.board,
            'history_3d': next_states.history,
            'actual_player': next_states.actual_player,
            'black_out': next_states.black_out,
            'white_out': next_states.white_out,
            'moves_count': next_states.moves_count,
            'iteration': embedding.get('iteration', jnp.zeros_like(next_states.actual_player))  # Propagate iteration array
        }

        return mctx.RecurrentFnOutput(
            reward=reward,  # shape: (batch_size,)
            discount=discount,  # shape: (batch_size,)
            prior_logits=prior_logits,  # shape: (batch_size, num_actions)
            value=-value  # Negate value for correct MCTS backup perspective
        ), next_embedding


@partial(jax.jit, static_argnames=['network', 'env'])
def get_root_output_batch(states: AbaloneState, network: AbaloneModel, params, env: AbaloneEnv, iteration: int = 0):
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
    board_2d, marbles_out = prepare_input_legacy(states.board, our_marbles, opp_marbles)
    
    # Convertir l'historique en 2D pour le réseau
    history_2d = jax.vmap(jax.vmap(cube_to_2d))(states.history)  # (batch, 8, 9, 9)

    # Obtenir les prédictions du réseau avec historique
    prior_logits, value = network.apply(params, board_2d, marbles_out, history_2d)

    embedding = {
        'board_3d': states.board,  # shape: (batch_size, x, y, z)
        'history_3d': states.history,  # shape: (batch_size, 8, x, y, z)
        'actual_player': states.actual_player,  # shape: (batch_size,)
        'black_out': states.black_out,  # shape: (batch_size,)
        'white_out': states.white_out,  # shape: (batch_size,)
        'moves_count': states.moves_count,  # shape: (batch_size,)
        'iteration': jnp.full_like(states.actual_player, iteration)  # shape: (batch_size,)
    }

    return mctx.RootFnOutput(
        prior_logits=prior_logits,  # shape: (batch_size, num_actions)
        value=value,  # shape: (batch_size,)
        embedding=embedding
    )