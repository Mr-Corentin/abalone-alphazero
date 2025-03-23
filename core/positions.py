from functools import partial
import jax
import jax.numpy as jnp
import chex
from core.core import Direction, DIRECTIONS, DIR_TO_IDX

@partial(jax.jit)
def get_valid_neighbors(pos: chex.Array, board: chex.Array, radius: int = 4) -> tuple[chex.Array, chex.Array]:
    """
    Trouve les voisins valides d'une position donnée en coordonnées cubiques.
    
    Args:
        pos: Position actuelle [x, y, z]
        board: État du plateau
        radius: Rayon du plateau
    
    Returns:
        tuple[chex.Array, chex.Array]: 
            - masque des directions valides (6,)
            - positions des voisins (6, 3)
    """
    # Calculer tous les voisins possibles
    neighbors = pos[None, :] + DIRECTIONS  # (6, 3)
    
    # Convertir en indices de tableau
    board_pos = neighbors + radius  # Ajouter radius à toutes les coordonnées
    
    # Créer un masque pour les positions valides
    within_bounds = ((board_pos >= 0) & (board_pos < board.shape[0])).all(axis=1)
    
    # Vérifier si les positions sont sur le plateau (non-nan)
    valid_pos = ~jnp.isnan(board[board_pos[:, 0], board_pos[:, 1], board_pos[:, 2]])
    
    # Combiner les masques
    valid_mask = within_bounds & valid_pos
    
    return valid_mask, neighbors

@partial(jax.jit, static_argnames=['radius'])
def is_valid_group(positions: chex.Array, board: chex.Array, radius: int = 4) -> tuple[chex.Array, jnp.int32]:
    """
    Vérifie si un groupe de positions forme un groupe valide
    Args:
        positions: tableau de coordonnées (N, 3) où N est 2 ou 3
        board: état du plateau
        radius: rayon du plateau
    Returns:
        tuple[chex.Array, jnp.int32]: (est_valide, index_direction)
            index_direction correspond à l'index dans DIRECTIONS
    """
    # Vérifier que toutes les positions ont la même couleur
    board_positions = positions + radius
    values = board[board_positions[:, 0], board_positions[:, 1], board_positions[:, 2]]
    same_color = (values[0] != 0) & jnp.all(values == values[0])
    
    # Pour 2 billes
    if len(positions) == 2:
        # Calculer le vecteur entre les deux positions
        diff = positions[1] - positions[0]
        
        # Vérifier si ce vecteur correspond à une direction valide
        is_valid_direction = jnp.any(jnp.all(diff == DIRECTIONS, axis=1))
        direction_idx = jnp.argmax(jnp.all(diff == DIRECTIONS, axis=1))
        
        return same_color & is_valid_direction, direction_idx
    
    # Pour 3 billes
    elif len(positions) == 3:
        # Calculer les vecteurs entre positions consécutives
        diff1 = positions[1] - positions[0]
        diff2 = positions[2] - positions[1]
        
        # Les différences doivent être identiques pour un alignement
        aligned = jnp.all(diff1 == diff2)
        # Vérifier si la première différence correspond à une direction valide
        is_valid_direction = jnp.any(jnp.all(diff1 == DIRECTIONS, axis=1))
        direction_idx = jnp.argmax(jnp.all(diff1 == DIRECTIONS, axis=1))
        
        return same_color & aligned & is_valid_direction, direction_idx
    
    # Si le nombre de positions n'est pas 2 ou 3
    return jnp.array(False), jnp.array(-1)

def test_group(positions: list[tuple[int, int, int]], board: chex.Array):
    """
    Fonction de test pour afficher les résultats de la validation d'un groupe
    Args:
        positions: liste de tuples (x, y, z)
        board: état du plateau
    """
    pos_array = jnp.array(positions)
    is_valid, direction_idx = is_valid_group(pos_array, board)
    
    print(f"\nTest du groupe:")
    for x, y, z in positions:
        print(f"({x}, {y}, {z})")
    
    if is_valid:
        direction = list(Direction)[int(direction_idx)]
        print(f"Groupe valide, aligné dans la direction: {direction.name}")
    else:
        print("Groupe invalide")
    
    # Afficher les valeurs des positions
    print("\nValeurs des positions:")
    for x, y, z in positions:
        value = board[x + 4, y + 4, z + 4]
        content = "●" if value == 1 else "○" if value == -1 else "·"
        print(f"Position ({x}, {y}, {z}): {content}")


@partial(jax.jit)
def analyze_group(positions: chex.Array, board: chex.Array, group_size: int, radius: int = 4) -> tuple[chex.Array, chex.Array, chex.Array]:
    """
    Analyse un groupe de positions pour déterminer sa validité et ses mouvements possibles.
    
    Args:
        positions: Tableau de positions (Nx3) où N est 2 ou 3
        board: État du plateau
        group_size: Nombre réel de billes dans le groupe
        radius: Rayon du plateau
    
    Returns:
        tuple:
            - is_valid: si le groupe est valide
            - inline_dirs: masque des directions possibles pour mouvement en ligne (6,)
            - parallel_dirs: masque des directions possibles pour mouvement parallèle (6,)
    """
    # Vérifier que toutes les billes sont de la même couleur
    # board_positions = positions + radius
    # values = board[board_positions[:, 0], board_positions[:, 1], board_positions[:, 2]]
    # same_color = (values[0] != 0) & jnp.all(values[:group_size] == values[0])


    board_positions = positions + radius
    values = board[board_positions[:, 0], board_positions[:, 1], board_positions[:, 2]]
    values_mask = jnp.arange(values.shape[0]) < group_size
    same_color = (values[0] != 0) & jnp.all(jnp.where(values_mask, values == values[0], True))
    
    # Calculer le vecteur de différence entre billes adjacentes
    diff = positions[1] - positions[0]
    
    # Identifier la direction d'alignement
    is_ew = jnp.all(diff == jnp.array([1, -1, 0])) | jnp.all(diff == jnp.array([-1, 1, 0]))
    is_ne_sw = jnp.all(diff == jnp.array([1, 0, -1])) | jnp.all(diff == jnp.array([-1, 0, 1]))
    is_se_nw = jnp.all(diff == jnp.array([0, -1, 1])) | jnp.all(diff == jnp.array([0, 1, -1]))
    
    # Vérifier l'adjacence
    is_adjacent = is_ew | is_ne_sw | is_se_nw
    
    # Traiter différemment selon la taille du groupe
    def handle_size_3(dummy):
        diff2 = positions[2] - positions[1]
        is_aligned = jnp.all(diff == diff2)
        return same_color & is_adjacent & is_aligned
    def handle_size_2(dummy):
        return same_color & is_adjacent
    
    is_valid = jax.lax.switch(group_size - 2,
                             [handle_size_2, handle_size_3],
                             None)
    
    # Initialiser les masques de direction
    inline_dirs = jnp.zeros(6, dtype=jnp.bool_)
    parallel_dirs = jnp.zeros(6, dtype=jnp.bool_)
    
    # Créer les masques pour chaque type d'alignement
    ew_inline = jnp.zeros(6, dtype=jnp.bool_).at[
        jnp.array([DIR_TO_IDX[Direction.E], DIR_TO_IDX[Direction.W]])
    ].set(True)
    
    ew_parallel = jnp.zeros(6, dtype=jnp.bool_).at[
        jnp.array([DIR_TO_IDX[Direction.NE], DIR_TO_IDX[Direction.SE], 
                  DIR_TO_IDX[Direction.NW], DIR_TO_IDX[Direction.SW]])
    ].set(True)
    
    ne_sw_inline = jnp.zeros(6, dtype=jnp.bool_).at[
        jnp.array([DIR_TO_IDX[Direction.NE], DIR_TO_IDX[Direction.SW]])
    ].set(True)
    
    ne_sw_parallel = jnp.zeros(6, dtype=jnp.bool_).at[
        jnp.array([DIR_TO_IDX[Direction.E], DIR_TO_IDX[Direction.W],
                  DIR_TO_IDX[Direction.SE], DIR_TO_IDX[Direction.NW]])
    ].set(True)
    
    se_nw_inline = jnp.zeros(6, dtype=jnp.bool_).at[
        jnp.array([DIR_TO_IDX[Direction.SE], DIR_TO_IDX[Direction.NW]])
    ].set(True)
    
    se_nw_parallel = jnp.zeros(6, dtype=jnp.bool_).at[
        jnp.array([DIR_TO_IDX[Direction.E], DIR_TO_IDX[Direction.W],
                  DIR_TO_IDX[Direction.NE], DIR_TO_IDX[Direction.SW]])
    ].set(True)
    
    # Assigner les directions selon l'alignement
    inline_dirs = jnp.where(is_ew, ew_inline, 
                  jnp.where(is_ne_sw, ne_sw_inline,
                  jnp.where(is_se_nw, se_nw_inline, inline_dirs)))
    
    parallel_dirs = jnp.where(is_ew, ew_parallel,
                    jnp.where(is_ne_sw, ne_sw_parallel,
                    jnp.where(is_se_nw, se_nw_parallel, parallel_dirs)))
    
    # Masquer les directions si le groupe n'est pas valide
    inline_dirs = jnp.where(is_valid, inline_dirs, jnp.zeros_like(inline_dirs))
    parallel_dirs = jnp.where(is_valid, parallel_dirs, jnp.zeros_like(parallel_dirs))
    
    return is_valid, inline_dirs, parallel_dirs

def print_group_analysis(positions: list[tuple[int, int, int]], board: chex.Array):
    """
    Affiche l'analyse d'un groupe pour debug
    """
    pos_array = jnp.array(positions)
    is_valid, axis_idx, inline_dirs, parallel_dirs = analyze_group(pos_array, board)
    
    print(f"\nAnalyse du groupe:")
    print(f"Positions: {positions}")
    print(f"Valide: {bool(is_valid)}")
    
    if bool(is_valid):
        print("\nMouvements en ligne possibles:")
        inline_dirs = jnp.array(inline_dirs)
        for dir_enum in Direction:
            idx = DIR_TO_IDX[dir_enum]
            if inline_dirs[idx]:
                print(f"- {dir_enum.name}")
        
        print("\nMouvements parallèles possibles:")
        parallel_dirs = jnp.array(parallel_dirs)
        for dir_enum in Direction:
            idx = DIR_TO_IDX[dir_enum]
            if parallel_dirs[idx]:
                print(f"- {dir_enum.name}")

# def analyze_group_debug(positions: list[tuple[int, int, int]], board: chex.Array):
#     """
#     Version debug de analyze_group sans JIT pour voir les valeurs
#     """
#     positions = jnp.array(positions)
    
#     # Calculer le vecteur de différence
#     diff = positions[1] - positions[0]
#     print(f"\nDébug du groupe: {positions}")
#     print(f"Vecteur de différence: {diff}")
    
#     # Pour un groupe aligné E-W, la différence est (1,-1,0) ou (-1,1,0)
#     diff_ew1 = jnp.array([1, -1, 0])
#     diff_ew2 = jnp.array([-1, 1, 0])
#     is_ew = jnp.all(diff == diff_ew1) or jnp.all(diff == diff_ew2)
    
#     # Pour un groupe aligné NE-SW, la différence est (1,0,-1) ou (-1,0,1)
#     diff_ne_sw1 = jnp.array([1, 0, -1])
#     diff_ne_sw2 = jnp.array([-1, 0, 1])
#     is_ne_sw = jnp.all(diff == diff_ne_sw1) or jnp.all(diff == diff_ne_sw2)
    
#     # Pour un groupe aligné SE-NW, la différence est (0,-1,1) ou (0,1,-1)
#     diff_se_nw1 = jnp.array([0, -1, 1])
#     diff_se_nw2 = jnp.array([0, 1, -1])
#     is_se_nw = jnp.all(diff == diff_se_nw1) or jnp.all(diff == diff_se_nw2)
    
#     print(f"Est aligné E-W: {is_ew}")
#     print(f"Est aligné NE-SW: {is_ne_sw}")
#     print(f"Est aligné SE-NW: {is_se_nw}")
    
#     if is_ew:
#         print("Pour un alignement E-W:")
#         print("- Mouvements inline: E, W")
#         print("- Mouvements parallèles devraient être: NE, SE, NW, SW")
#     elif is_ne_sw:
#         print("Pour un alignement NE-SW:")
#         print("- Mouvements inline: NE, SW")
#         print("- Mouvements parallèles devraient être: E, W, SE, NW")
#     elif is_se_nw:
#         print("Pour un alignement SE-NW:")
#         print("- Mouvements inline: SE, NW")
#         print("- Mouvements parallèles devraient être: E, W, NE, SW")

#     return is_ew, is_ne_sw, is_se_nw



