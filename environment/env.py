import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Dict, NamedTuple
from core.board import initialize_board, create_custom_board
from core.legal_moves import get_legal_moves
from core.moves import move_group_inline, move_group_parallel, move_single_marble
import numpy as np
from core.coord_conversion import compute_coord_map
from functools import partial

class AbaloneState(NamedTuple):
    """État du jeu d'Abalone (version canonique)"""
    board: chex.Array  # Le plateau où le joueur courant est toujours 1
    actual_player: int  # Le joueur réel (1=noir, -1=blanc)
    black_out: int  # Nombre de billes noires sorties
    white_out: int  # Nombre de billes blanches sorties
    moves_count: int

class AbaloneEnv:
    def __init__(self, radius: int = 4):
      self.radius = radius
      self.moves_index = self._load_moves_index()
      self.coord_map = compute_coord_map(radius)

    # RNG arg to parallelize
    def reset(self, rng: chex.PRNGKey) -> AbaloneState:
        """Reset avec une clé RNG pour compatibilité batch."""
        board = initialize_board()  # Peut être randomisé avec rng plus tard
        return AbaloneState(
            board=board,
            actual_player=1,
            black_out=0,
            white_out=0,
            moves_count=0
        )

    @partial(jax.jit, static_argnames=['self', 'batch_size'])
    def reset_batch(self, rng: chex.PRNGKey, batch_size: int = 1) -> AbaloneState:
        """Reset pour un batch d'états"""
        # Initialiser un seul plateau
        single_board = initialize_board(self.radius)  # shape: (size, size, size)

        # Créer batch_size copies du plateau
        # On veut: (batch_size, size, size, size)
        boards = jnp.repeat(single_board[None, ...], batch_size, axis=0)

        return AbaloneState(
            board=boards,  # shape: (batch_size, size, size, size)
            actual_player=jnp.ones(batch_size, dtype=jnp.int32),
            black_out=jnp.zeros(batch_size, dtype=jnp.int32),
            white_out=jnp.zeros(batch_size, dtype=jnp.int32),
            moves_count=jnp.zeros(batch_size, dtype=jnp.int32)
        )

    @partial(jax.jit, static_argnames=['self'])
    def step(self, state: AbaloneState, move_idx: int) -> AbaloneState:
        """Effectue un mouvement et retourne le nouvel état"""
        # Convertir move_idx en entier scalaire
        move_idx = move_idx.astype(jnp.int32).reshape(())

        # Accéder aux données avec JAX
        positions = jnp.array(self.moves_index['positions'])
        direction = jnp.array(self.moves_index['directions'])[move_idx]
        move_type = jnp.array(self.moves_index['move_types'])[move_idx]
        group_size = jnp.array(self.moves_index['group_sizes'])[move_idx]
        position = positions[move_idx]

        def single_marble_case(inputs):
            state, position, direction = inputs
            new_board, _ = move_single_marble(state.board, position[0], direction, self.radius)
            return new_board, 0

        def group_parallel_case(inputs):
            state, position, direction, group_size = inputs
            new_board, _ = move_group_parallel(state.board, position, direction, group_size, self.radius)
            return new_board, 0

        def group_inline_case(inputs):
            state, position, direction, group_size = inputs
            new_board, _, billes_sorties = move_group_inline(state.board, position, direction, group_size, self.radius)
            return new_board, billes_sorties

        # Utiliser switch pour le type de mouvement
        new_board, billes_sorties = jax.lax.switch(
            move_type,
            [
                lambda x: single_marble_case((state, position, direction)),
                lambda x: group_parallel_case((state, position, direction, group_size)),
                lambda x: group_inline_case((state, position, direction, group_size))
            ],
            0
        )

        # S'assurer que actual_player est un scalaire
        actual_player = state.actual_player.reshape(())

        # Mise à jour des billes sorties
        black_out = state.black_out + billes_sorties * (actual_player == -1)
        white_out = state.white_out + billes_sorties * (actual_player == 1)

        return AbaloneState(
            board=-new_board,
            actual_player=-actual_player,
            black_out=black_out,
            white_out=white_out,
            moves_count=state.moves_count + 1
        )

    @partial(jax.jit, static_argnames=['self'])
    def step_batch(self, states: AbaloneState, move_idxs: chex.Array) -> AbaloneState:
        return jax.vmap(self.step)(states, move_idxs)

    # def _load_moves_index(self):
    #     """Charge l'index des mouvements à partir du fichier npz"""
    #     moves_data = np.load('move_map.npz')
    #     return {
    #         'positions': moves_data['positions'],
    #         'directions': moves_data['directions'],
    #         'move_types': moves_data['move_types'],
    #         'group_sizes': moves_data['group_sizes']
    #     }
    
    def _load_moves_index(self):
        """Charge l'index des mouvements à partir du fichier npz"""
        import os
        
        # Construire le chemin vers le fichier dans le dossier data/
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'move_map.npz')
        
        # Charger les données
        moves_data = np.load(data_path)
        
        return {
            'positions': moves_data['positions'],
            'directions': moves_data['directions'],
            'move_types': moves_data['move_types'],
            'group_sizes': moves_data['group_sizes']
        }

    @partial(jax.jit, static_argnames=['self'])
    def get_legal_moves(self, state: AbaloneState) -> chex.Array:
        """Retourne un masque des mouvements légaux"""
        return get_legal_moves(state.board, self.moves_index, self.radius)

    @partial(jax.jit, static_argnames=['self'])
    def get_legal_moves_batch(self, states: AbaloneState) -> chex.Array:
        """Retourne un masque des mouvements légaux pour un batch d'états"""
        return jax.vmap(get_legal_moves)(states.board,
                                        self.moves_index,
                                        self.radius)

    def is_terminal(self, state: AbaloneState) -> bool:
      """Vérifie si l'état est terminal"""
      # Remplacer les or par jnp.logical_or
      return jnp.logical_or(
          jnp.logical_or(
              state.black_out >= 6,
              state.white_out >= 6
          ),
          state.moves_count >= 300
      )
    @partial(jax.jit, static_argnames=['self'])
    def is_terminal_batch(self, states: AbaloneState) -> chex.Array:
        return jax.vmap(lambda s: jnp.logical_or(
            jnp.logical_or(
                s.black_out >= 6,
                s.white_out >= 6
            ),
            s.moves_count >= 300
        ))(states)
    def get_winner(self, state: AbaloneState) -> int:
        """
        Détermine le gagnant

        Returns:
            1 si les noirs gagnent, -1 si les blancs gagnent, 0 si match nul
        """
        if state.white_out >= 6:
            return 1  # Noirs gagnent
        elif state.black_out >= 6:
            return -1  # Blancs gagnent
        elif state.moves_count >= 300:
            return 0  # Match nul
        return 0  # Partie en cours
    @partial(jax.jit, static_argnames=['self'])
    def get_winner_batch(self, states: AbaloneState) -> chex.Array:
        return jax.vmap(lambda s: jnp.where(
            s.white_out >= 6,
            1,  # Noirs gagnent
            jnp.where(
                s.black_out >= 6,
                -1,  # Blancs gagnent
                0  # Match nul ou en cours
            )
        ))(states)

    def is_legal_move(self, state: AbaloneState, move_idx: int) -> bool:
        """Vérifie si un mouvement spécifique est légal"""
        legal_moves = self.get_legal_moves(state)
        return legal_moves[move_idx]

    def get_score(self, state: AbaloneState) -> dict:
        """Retourne le score actuel sous forme de dictionnaire"""
        return {
            'black_out': state.black_out,
            'white_out': state.white_out,
            'moves': state.moves_count
        }

    def get_canonical_state(self, board: chex.Array, actual_player: int) -> chex.Array:
        """
        Convertit un plateau en sa représentation canonique où le joueur à jouer est toujours 1

        Args:
            board: État du plateau
            actual_player: Joueur qui doit jouer (1 ou -1)

        Returns:
            board_canonical: Plateau en représentation canonique
        """
        return jnp.where(actual_player == 1, board, -board)


class AbaloneStateNonCanonical(NamedTuple):
    """État du jeu d'Abalone (version non-canonique)"""
    board: chex.Array  # Le plateau où noir=1, blanc=-1 (fixe)
    current_player: int  # Le joueur qui doit jouer (1=noir, -1=blanc)
    black_out: int  # Nombre de billes noires sorties
    white_out: int  # Nombre de billes blanches sorties
    moves_count: int

class AbaloneEnvNonCanonical(AbaloneEnv):
    def reset(self, rng: chex.PRNGKey) -> AbaloneStateNonCanonical:
        """Reset avec une clé RNG pour compatibilité batch."""
        board = initialize_board()  # Noir=1, blanc=-1
        return AbaloneStateNonCanonical(
            board=board,
            current_player=1,  # Noir commence
            black_out=0,
            white_out=0,
            moves_count=0
        )
    @partial(jax.jit, static_argnames=['self'])
    def step(self, state: AbaloneStateNonCanonical, move_idx: int) -> AbaloneStateNonCanonical:
        """Effectue un mouvement sans changer la représentation du plateau"""
        move_idx = move_idx.astype(jnp.int32).reshape(())
        
        canonical_board = self.get_canonical_state(state.board, state.current_player)
        canonical_state = AbaloneState(
            board=canonical_board,
            actual_player=state.current_player,
            black_out=state.black_out,
            white_out=state.white_out,
            moves_count=state.moves_count
        )
        
        positions = jnp.array(self.moves_index['positions'])
        direction = jnp.array(self.moves_index['directions'])[move_idx]
        move_type = jnp.array(self.moves_index['move_types'])[move_idx]
        group_size = jnp.array(self.moves_index['group_sizes'])[move_idx]
        position = positions[move_idx]
        
        def single_marble_case(inputs):
            state, position, direction = inputs
            new_board, _ = move_single_marble(state.board, position[0], direction, self.radius)
            return new_board, 0

        def group_parallel_case(inputs):
            state, position, direction, group_size = inputs
            new_board, _ = move_group_parallel(state.board, position, direction, group_size, self.radius)
            return new_board, 0

        def group_inline_case(inputs):
            state, position, direction, group_size = inputs
            new_board, _, billes_sorties = move_group_inline(state.board, position, direction, group_size, self.radius)
            return new_board, billes_sorties

        # Utiliser switch pour le type de mouvement
        new_board, billes_sorties = jax.lax.switch(
            move_type,
            [
                lambda x: single_marble_case((canonical_state, position, direction)),
                lambda x: group_parallel_case((canonical_state, position, direction, group_size)),
                lambda x: group_inline_case((canonical_state, position, direction, group_size))
            ],
            0
        )
        
        non_canonical_board = jnp.where(state.current_player == 1, 
                                       new_board, 
                                       -new_board)
        
        black_out = state.black_out + billes_sorties * (state.current_player == -1)
        white_out = state.white_out + billes_sorties * (state.current_player == 1)
        return AbaloneStateNonCanonical(
            board=non_canonical_board,
            current_player=-state.current_player,  # Changer de joueur
            black_out=black_out,
            white_out=white_out,
            moves_count=state.moves_count + 1
        )
    
    @partial(jax.jit, static_argnames=['self'])
    def get_legal_moves(self, state: AbaloneStateNonCanonical) -> chex.Array:
        """Retourne un masque des mouvements légaux"""
        # Convertir en représentation canonique pour utiliser la fonction existante
        canonical_board = self.get_canonical_state(state.board, state.current_player)
        return get_legal_moves(canonical_board, self.moves_index, self.radius)
    
    def is_terminal(self, state: AbaloneStateNonCanonical) -> bool:
        """Vérifie si l'état est terminal"""
        return jnp.logical_or(
            jnp.logical_or(
                state.black_out >= 6,
                state.white_out >= 6
            ),
            state.moves_count >= 300
        )
    
    def get_winner(self, state: AbaloneStateNonCanonical) -> int:
        """Détermine le gagnant"""
        if state.white_out >= 6:
            return 1  # Noirs gagnent
        elif state.black_out >= 6:
            return -1  # Blancs gagnent
        elif state.moves_count >= 300:
            return 0  # Match nul
        return 0  # Partie en cours