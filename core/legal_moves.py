import jax
import jax.numpy as jnp
import chex
from functools import partial
from typing import Dict, Tuple
from core.moves import move_single_marble , move_group_inline, move_group_parallel
from core.core import Direction
import numpy as np

@partial(jax.jit, static_argnames=['radius'])
def create_player_positions_mask(board: chex.Array, radius: int = 4) -> chex.Array:
    """
    Create boolean mask of positions where current player's marbles are located (always 1)
    """
    return board == 1

@partial(jax.jit, static_argnames=['radius'])
def filter_moves_by_positions(player_mask: chex.Array, 
                            moves_index: Dict[str, chex.Array],
                            radius: int = 4) -> chex.Array:
    """
    Create mask of moves where all start positions have player marbles
    """
    def check_move_positions(move_idx):
        positions = moves_index['positions'][move_idx]
        group_size = moves_index['group_sizes'][move_idx]
        
        board_positions = positions + radius
        
        has_pieces = player_mask[board_positions[:, 0],
                               board_positions[:, 1],
                               board_positions[:, 2]]
        
        # Create mask for correct number of positions
        positions_mask = jnp.arange(3) < group_size

        # True if all required positions have our pieces
        return jnp.all(jnp.where(positions_mask, has_pieces, True))
    
    return jax.vmap(check_move_positions)(jnp.arange(len(moves_index['directions'])))


@partial(jax.jit, static_argnames=['radius'])
def check_moves_validity(board: chex.Array,
                        moves_index: Dict[str, chex.Array],
                        filtered_moves: chex.Array,
                        radius: int = 4) -> chex.Array:
    """
    Check which filtered moves are legal according to game rules.

    NOTE: this is the original, slow path, kept only as a reference for tests --
    AbaloneEnv now uses get_legal_moves_fast. The lax.switch below does NOT skip
    branches here: it is applied under vmap with a batched move_type, and JAX
    lowers a batched switch to a select_n over ALL branches. So this evaluates
    1734 x 3 move functions per state.
    """
    def check_move(move_idx):

        # Si le mouvement n'a pas passé le premier filtre, retourner False
        is_filtered = filtered_moves[move_idx]

        positions = moves_index['positions'][move_idx]
        direction = moves_index['directions'][move_idx]
        move_type = moves_index['move_types'][move_idx]
        group_size = moves_index['group_sizes'][move_idx]

        # Définir les branches comme des fonctions
        # Seulement la branche sélectionnée sera évaluée (économie de calcul)
        def branch_single(_):
            _, success = move_single_marble(board, positions[0], direction, radius)
            return success

        def branch_parallel(_):
            _, success = move_group_parallel(board, positions, direction, group_size, radius)
            return success

        def branch_inline(_):
            _, success, _ = move_group_inline(board, positions, direction, group_size, radius)
            return success

        # Switch n'évalue QUE la branche sélectionnée (vs jnp.where qui évalue tout)
        is_valid = jax.lax.switch(
            move_type,
            [branch_single, branch_parallel, branch_inline],
            None
        )

        return jnp.where(is_filtered, is_valid, False)

    return jax.vmap(check_move)(jnp.arange(len(moves_index['directions'])))


@partial(jax.jit, static_argnames=['radius'])
def get_legal_moves(board: chex.Array,
                   moves_index: Dict[str, chex.Array],
                   radius: int = 4) -> chex.Array:
    """
    Determine all legal moves for current player (always 1)
    """
    position_filtered = filter_moves_by_positions(
        create_player_positions_mask(board),  
        moves_index,
        radius
    )
    
    return check_moves_validity(board, moves_index, position_filtered, radius)


def filter_moves_by_positions_for_eval(player_mask: chex.Array, 
                                     moves_index: Dict[str, chex.Array],
                                     radius: int = 4) -> chex.Array:
    """
    Non-vmap version of filter_moves_by_positions for evaluation
    """
    num_moves = len(moves_index['directions'])
    results = np.zeros(num_moves, dtype=bool)
    
    for move_idx in range(num_moves):
        positions = moves_index['positions'][move_idx]
        group_size = moves_index['group_sizes'][move_idx]
        
        board_positions = positions + radius
        
        # Check each required position
        valid = True
        for i in range(group_size):
            pos = board_positions[i]
            if not player_mask[pos[0], pos[1], pos[2]]:
                valid = False
                break
        
        results[move_idx] = valid
    
    return results

def check_moves_validity_for_eval(board: chex.Array,
                                moves_index: Dict[str, chex.Array],
                                filtered_moves: np.ndarray,
                                radius: int = 4) -> np.ndarray:
    """
    Non-vmap version of check_moves_validity for evaluation
    """
    num_moves = len(moves_index['directions'])
    results = np.zeros(num_moves, dtype=bool)
    
    for move_idx in range(num_moves):
        # If move didn't pass first filter, skip to next
        if not filtered_moves[move_idx]:
            continue
        
        positions = moves_index['positions'][move_idx]
        direction = moves_index['directions'][move_idx]
        move_type = moves_index['move_types'][move_idx]
        group_size = moves_index['group_sizes'][move_idx]
        
        # Check based on move type
        if move_type == 0:
            _, success = move_single_marble(board, positions[0], direction, radius)
            results[move_idx] = success
        elif move_type == 1:
            _, success = move_group_parallel(board, positions, direction, group_size, radius)
            results[move_idx] = success
        else:  # move_type == 2
            _, success, _ = move_group_inline(board, positions, direction, group_size, radius)
            results[move_idx] = success
    
    return results

def get_legal_moves_for_eval(board: chex.Array,
                           moves_index: Dict[str, chex.Array],
                           radius: int = 4) -> np.ndarray:
    """
    Non-vmap version of get_legal_moves for evaluation
    """
    player_mask = (board == 1)
    position_filtered = filter_moves_by_positions_for_eval(player_mask, moves_index, radius)
    return check_moves_validity_for_eval(board, moves_index, position_filtered, radius)



@partial(jax.jit, static_argnames=['radius'])
def get_legal_moves_for_single(board, moves_index, radius=4):
    """
    JIT version but non-vectorized for single state.
    Reuses internal functions from get_legal_moves but without vmap.
    """
    # Create player mask
    player_mask = board == 1
    
    # Use internal function from filter_moves_by_positions but apply
    # to each move one by one without vmap
    def check_move_positions(move_idx):
        positions = moves_index['positions'][move_idx]
        group_size = moves_index['group_sizes'][move_idx]
        
        board_positions = positions + radius
        
        has_pieces = player_mask[board_positions[:, 0],
                               board_positions[:, 1],
                               board_positions[:, 2]]
        
        # Create mask for correct number of positions
        positions_mask = jnp.arange(3) < group_size

        # True if all required positions have our pieces
        return jnp.all(jnp.where(positions_mask, has_pieces, True))
    
    # Apply to each move with scan or cumulatively
    num_moves = len(moves_index['directions'])
    position_filtered = jnp.zeros(num_moves, dtype=jnp.bool_)
    
    # Use jax.lax.fori_loop instead of Python loop
    def body_fn(i, filtered):
        filtered = filtered.at[i].set(check_move_positions(i))
        return filtered
        
    position_filtered = jax.lax.fori_loop(
        0, num_moves, body_fn, position_filtered
    )
    
    # Internal check_move function from check_moves_validity
    def check_move(move_idx):
        # Si le mouvement n'a pas passé le premier filtre, retourner False
        is_filtered = position_filtered[move_idx]
        
        positions = moves_index['positions'][move_idx]
        direction = moves_index['directions'][move_idx]
        move_type = moves_index['move_types'][move_idx]
        group_size = moves_index['group_sizes'][move_idx]  
        
        # Vérifier les différents types de mouvements
        _, success_single = move_single_marble(board, positions[0], direction, radius)
        
        _, success_parallel = move_group_parallel(board, positions, direction, group_size, radius)
        
        _, success_inline, _ = move_group_inline(board, positions, direction, group_size, radius)
        
        # Sélectionner le bon résultat selon le type
        is_valid = jnp.where(
            move_type == 0, success_single,
            jnp.where(move_type == 1, success_parallel, success_inline)
        )

        return jnp.where(is_filtered, is_valid, False)
    
    # Apply to each move
    legal_moves = jnp.zeros(num_moves, dtype=jnp.bool_)
    
    def body_fn2(i, legal):
        legal = legal.at[i].set(check_move(i))
        return legal
        
    legal_moves = jax.lax.fori_loop(
        0, num_moves, body_fn2, legal_moves
    )
    
    return legal_moves

# =============================================================================
# Implementation vectorisee de la legalite des coups.
#
# La legalite d'un coup ne depend que du contenu d'au plus 9 cases :
#   - les 3 cases du groupe
#   - leurs 3 destinations
#   - les 3 cases devant la tete du groupe (poussee)
# Ces indices ne dependent QUE du coup, jamais du plateau : on les precalcule
# une fois, puis la legalite des 1734 coups se lit en UN seul gather (1734, 9)
# suivi d'operations tableau pures -- ni branchement, ni scatter.
#
# A comparer a l'ancienne version qui, sous vmap, evalue les 3 branches du
# lax.switch pour les 1734 coups (5202 evaluations de fonction de coup par etat).
# =============================================================================

# Valeur sentinelle pour "hors plateau". Distincte de -1 / 0 / 1.
_OFF_BOARD = 2.0


def build_fast_tables(moves_index, radius: int = 4):
    """
    Precalcule les tables d'indices utilisees par get_legal_moves_fast.

    Le plateau est lu a plat sur (size**3 + 1) elements : le dernier slot est la
    sentinelle "hors plateau". Toute case hors de l'hexagone y est redirigee.
    """
    size = 2 * radius + 1
    sentinel = size ** 3

    positions = np.asarray(moves_index['positions'])      # (N, 3, 3)
    directions = np.asarray(moves_index['directions'])    # (N,)
    move_types = np.asarray(moves_index['move_types'])    # (N,)
    group_sizes = np.asarray(moves_index['group_sizes'])  # (N,)
    n_moves = len(directions)

    dir_vectors = np.array([d.value for d in Direction], dtype=np.int32)  # (6, 3)

    def flat_index(coord):
        """Index a plat, ou sentinelle si la case est hors de l'hexagone."""
        if np.max(np.abs(coord)) > radius:
            return sentinel
        i, j, k = coord + radius
        return int((i * size + j) * size + k)

    group_idx = np.full((n_moves, 3), sentinel, dtype=np.int32)
    dest_idx = np.full((n_moves, 3), sentinel, dtype=np.int32)
    ahead_idx = np.full((n_moves, 3), sentinel, dtype=np.int32)

    for m in range(n_moves):
        g = int(group_sizes[m])
        dv = dir_vectors[directions[m]]
        cells = positions[m][:g].astype(np.int32)

        for j in range(g):
            group_idx[m, j] = flat_index(cells[j])
            dest_idx[m, j] = flat_index(cells[j] + dv)

        if move_types[m] == 2:  # inline : on a besoin des cases devant la tete
            head = cells[np.argmax(cells @ dv)]
            for j in range(3):
                ahead_idx[m, j] = flat_index(head + dv * (j + 1))

    return {
        'group_idx': jnp.asarray(group_idx),
        'dest_idx': jnp.asarray(dest_idx),
        'ahead_idx': jnp.asarray(ahead_idx),
        'group_mask': jnp.asarray(np.arange(3)[None, :] < group_sizes[:, None]),
        'is_inline': jnp.asarray(move_types == 2),
        'group_sizes': jnp.asarray(group_sizes.astype(np.int32)),
        # Cases reellement sur l'hexagone : sert a restaurer les NaN apres un coup.
        'on_board': jnp.asarray(_on_board_mask(radius)),
    }


def _on_board_mask(radius: int = 4):
    size = 2 * radius + 1
    coords = np.indices((size, size, size)).reshape(3, -1).T - radius
    inside = (np.abs(coords).max(axis=1) <= radius) & (coords.sum(axis=1) == 0)
    return inside.reshape(size, size, size)


def _read_cells(board: chex.Array, tables, move_idx=None):
    """
    Aplatit le plateau (+ slot sentinelle) et lit les 9 cases utiles.

    Sans move_idx : lit les 1734 coups d'un coup, tableaux (1734, 3).
    Avec move_idx : lit un seul coup, tableaux (3,).
    Le dernier slot du tableau aplati sert a la fois de valeur "hors plateau"
    en lecture et de poubelle en ecriture.
    """
    flat = jnp.concatenate([
        jnp.nan_to_num(board, nan=_OFF_BOARD).reshape(-1),
        jnp.array([_OFF_BOARD], dtype=board.dtype),
    ])
    if move_idx is None:
        gi, di, ai = tables['group_idx'], tables['dest_idx'], tables['ahead_idx']
        mask, inline, sizes = tables['group_mask'], tables['is_inline'], tables['group_sizes']
    else:
        gi = tables['group_idx'][move_idx]
        di = tables['dest_idx'][move_idx]
        ai = tables['ahead_idx'][move_idx]
        mask = tables['group_mask'][move_idx]
        inline = tables['is_inline'][move_idx]
        sizes = tables['group_sizes'][move_idx]
    return flat, (gi, di, ai), (flat[gi], flat[di], flat[ai]), (mask, inline, sizes)


def _legality(cells, meta):
    """
    Predicat de legalite, ecrit sur le dernier axe pour marcher aussi bien sur
    (1734, 3) que sur (3,).

    Renvoie (legal, pushes_one, pushes_two, marbles_out) ou pushes_one/two
    indiquent le nombre de billes adverses effectivement poussees.
    """
    group, dest, ahead = cells
    mask, is_inline, group_sizes = meta

    # Toutes les cases du groupe portent une bille du joueur courant.
    owned = jnp.all(jnp.where(mask, group == 1, True), axis=-1)

    # Coup simple / broadside : toutes les destinations doivent etre vides.
    # Une destination hors plateau vaut _OFF_BOARD, donc != 0 : rejetee.
    destinations_free = jnp.all(jnp.where(mask, dest == 0, True), axis=-1)

    a1, a2, a3 = ahead[..., 0], ahead[..., 1], ahead[..., 2]

    slides = a1 == 0
    one_opponent = (a1 == -1) & (a2 != -1)
    two_opponents = (a1 == -1) & (a2 == -1) & (a3 != -1)

    # Sumito : strictement plus de billes que l'adversaire, et la case derriere
    # la colonne poussee est vide ou hors plateau (la bille sort).
    pushes_one = one_opponent & (group_sizes >= 2) & ((a2 == 0) | (a2 == _OFF_BOARD))
    pushes_two = two_opponents & (group_sizes >= 3) & ((a3 == 0) | (a3 == _OFF_BOARD))

    inline_ok = slides | pushes_one | pushes_two
    legal = owned & jnp.where(is_inline, inline_ok, destinations_free)

    pushes_one = legal & is_inline & pushes_one
    pushes_two = legal & is_inline & pushes_two
    marbles_out = jnp.where((pushes_one & (a2 == _OFF_BOARD)) |
                            (pushes_two & (a3 == _OFF_BOARD)), 1, 0)

    return legal, pushes_one, pushes_two, marbles_out


@jax.jit
def get_legal_moves_fast(board: chex.Array, tables) -> chex.Array:
    """
    Masque des coups legaux pour le joueur courant (toujours represente par 1).

    Equivalent a get_legal_moves, sans branchement ni ecriture sur le plateau.
    """
    _, _, cells, meta = _read_cells(board, tables)
    return _legality(cells, meta)[0]


@jax.jit
def get_marbles_pushed_out_fast(board: chex.Array, tables) -> chex.Array:
    """Nombre de billes adverses ejectees par chaque coup (0 ou 1)."""
    _, _, cells, meta = _read_cells(board, tables)
    return _legality(cells, meta)[3]


@jax.jit
def apply_move_fast(board: chex.Array, move_idx, tables):
    """
    Applique un coup et renvoie (nouveau_plateau, succes, billes_sorties).

    Meme contrat que move_single_marble / move_group_parallel / move_group_inline
    reunies : un coup illegal laisse le plateau inchange et n'ejecte rien.

    Contrairement au lax.switch de env.step, il n'y a pas de branche : sous vmap
    un switch a index batche est abaisse en select_n qui evalue les TROIS
    fonctions de coup pour chaque etat du batch. Ici on ecrit au plus 8 cases,
    et toute ecriture invalide (case hors groupe, bille ejectee, coup non-inline)
    est redirigee vers le slot poubelle en fin de tableau.
    """
    flat, (gi, di, ai), cells, meta = _read_cells(board, tables, move_idx)
    legal, pushes_one, pushes_two, marbles_out = _legality(cells, meta)

    # 1. les cases de depart se vident, 2. les destinations recoivent nos billes.
    # Pour j >= group_size les deux index valent la poubelle : ecriture sans effet.
    moved = flat.at[gi].set(0.0).at[di].set(1.0)

    # 3. les billes adverses poussees avancent d'une case. Celle qui sort du
    #    plateau a un index poubelle, donc elle disparait sans traitement special.
    a2, a3 = ai[1], ai[2]
    moved = moved.at[a2].set(jnp.where(pushes_one | pushes_two, -1.0, moved[a2]))
    moved = moved.at[a3].set(jnp.where(pushes_two, -1.0, moved[a3]))

    moved = jnp.where(legal, moved, flat)

    new_board = moved[:-1].reshape(board.shape)
    # On avait remplace les NaN par _OFF_BOARD pour lire : on les restaure.
    new_board = jnp.where(tables['on_board'], new_board, jnp.nan)

    return new_board, legal, jnp.where(legal, marbles_out, 0)
