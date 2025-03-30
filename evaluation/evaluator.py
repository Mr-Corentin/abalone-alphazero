import numpy as np
import time
import math
from typing import Dict, List, Tuple, Any
import sys
import os

from environment.env import AbaloneState
from core.board import create_board_mask
from mcts.agent import get_best_move
# sys.path.append(os.path.join(os.path.dirname(__file__), 'abalone-ai'))

# # Le reste de vos imports
# from abalone.grid import Hex, AbaloneGrid
# import abalone.config as config
# from abalone.ai import AI, TT

# Remonter d'un niveau depuis le dossier 'evaluation' pour atteindre la racine du projet
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'abalone-ai'))

# Le reste de vos imports
from abalone.grid import Hex, HexBlock, AbaloneGrid
import abalone.config as config
from abalone.ai import AI, TT



# Vos fonctions de conversion
def their_grid_to_our_state(their_grid, current_player=None, previous_state=None, radius=4):
    """
    Convertit leur AbaloneGrid en notre AbaloneState
    
    Args:
        their_grid: Leur grille (AbaloneGrid)
        current_player: Joueur actuel (config.WHITE ou config.BLACK)
        previous_state: État précédent (optionnel)
        radius: Rayon du plateau (par défaut: 4)
    
    Returns:
        AbaloneState: Notre état
    """
    # Créer un tableau 3D pour notre représentation
    size = 2 * radius + 1
    board = np.full((size, size, size), np.nan)
    center = radius
    
    # Récupérer les positions des billes
    black_positions = their_grid.query.marbles(config.BLACK)
    white_positions = their_grid.query.marbles(config.WHITE)
    
    # Déterminer le joueur actuel
    if current_player is not None:
        # Si le joueur actuel est spécifié, utiliser cette valeur
        actual_player = 1 if current_player == config.BLACK else -1
    else:
        # Comportement par défaut si current_player n'est pas fourni
        actual_player = 1  # Supposons noir par défaut
    
    # Remplir notre plateau avec les positions valides
    valid_mask = create_board_mask(radius)
    board = np.where(valid_mask, 0., board)
    
    # Placer les billes sur le plateau
    for hex_pos in black_positions:
        x, z = hex_pos.x, hex_pos.z
        y = -x - z
        
        array_x = x + center
        array_y = y + center
        array_z = z + center
        
        # Noir = 1 (ou -1 si actual_player == -1)
        board[array_x, array_y, array_z] = 1. if actual_player == 1 else -1.
    
    for hex_pos in white_positions:
        x, z = hex_pos.x, hex_pos.z
        y = -x - z
        
        array_x = x + center
        array_y = y + center
        array_z = z + center
        
        # Blanc = -1 (ou 1 si actual_player == -1)
        board[array_x, array_y, array_z] = -1. if actual_player == 1 else 1.
    
    # Compter les billes sorties
    black_out = 14 - len(black_positions)
    white_out = 14 - len(white_positions)
    
    # Créer l'état
    return AbaloneState(
        board=board,
        actual_player=actual_player,
        black_out=black_out,
        white_out=white_out,
        moves_count=0 if previous_state is None else previous_state.moves_count + 1
    )


def convert_action_to_their_format(action_idx, our_state, their_grid, env, radius=4):
    """
    Convertit un indice d'action de notre système en format (bloc, direction) pour leur système
    """
    # Récupérer les informations sur l'action
    positions = env.moves_index['positions'][action_idx]
    direction_value = env.moves_index['directions'][action_idx]
    group_size = env.moves_index['group_sizes'][action_idx]
    
    
    # Ne prendre que les positions valides selon group_size
    positions = positions[:group_size]
    
    # Mapping des indices de direction aux tuples (x, z)
    direction_mapping = {
        0: (1, -1),   # NE (Nord-Est) -> (1, -1) dans leur système
        1: (1, 0),    # E (Est) -> (1, 0) dans leur système
        2: (0, 1),    # SE (Sud-Est) -> (0, 1) dans leur système
        3: (-1, 1),   # SW (Sud-Ouest) -> (-1, 1) dans leur système
        4: (-1, 0),   # W (Ouest) -> (-1, 0) dans leur système
        5: (0, -1),   # NW (Nord-Ouest) -> (0, -1) dans leur système
    }
    
    # Convertir la valeur de direction
    their_direction = direction_mapping.get(int(direction_value), (0, 0))
    
    # Convertir les positions: prendre simplement les composantes x et z
    their_positions = []
    for pos in positions:
        # Extraire les coordonnées et convertir en int standard
        x = int(pos[0]) if hasattr(pos[0], 'item') else pos[0]
        z = int(pos[2]) if hasattr(pos[2], 'item') else pos[2]
        
        their_positions.append(Hex(x=x, z=z))
    
    # Trier les positions pour la direction
    if their_direction[0] != 0:
        their_positions.sort(key=lambda h: h.x, reverse=(their_direction[0] < 0))
    elif their_direction[1] != 0:
        their_positions.sort(key=lambda h: h.z, reverse=(their_direction[1] < 0))
    
    # Créer un HexBlock avec ces positions
    block = HexBlock(their_positions)
    
    return block, their_direction


def simulate_alphazero_vs_classical(params, network, env, classical_algo="tt_pvs", depth=3, max_moves=300, verbose=False):
    """
    Simule une partie entre AlphaZero et un algorithme classique.
    """
    # Création d'un état initial
    state_dict = {        
        config.WHITE: [
                (0, -4), (1, -4), (2, -4), (3, -4), (4, -4),
            (-1, -3), (0, -3), (1, -3), (2, -3), (3, -3), (4, -3),
                        (0, -2), (1, -2), (2, -2),
        ],
        config.BLACK: [
                        (-2, 2), (-1, 2), (0, 2),
            (-4, 3), (-3, 3), (-2, 3), (-1, 3), (0, 3), (1, 3),
                (-4, 4), (-3, 4), (-2, 4), (-1, 4), (0, 4),
        ]
    }
    
    # Convertir les tuples en Hex
    state_hex = {
        key: [Hex(x=x, z=z) for x, z in positions]
        for key, positions in state_dict.items()
    }
    
    # Créer la grille
    grid = AbaloneGrid(state_hex)
    
    # AlphaZero joue toujours en Noir, l'algorithme classique en Blanc
    alphazero_player = config.BLACK
    classical_player = config.WHITE
    
    if verbose:
        print(f"AlphaZero (Noir) vs {classical_algo} (Blanc), profondeur {depth}")
    
    # Jouer la partie
    move_count = 0
    current_player = config.WHITE  # Blanc commence
    
    # Pour détecter les cycles
    seen_states = {}
    
    while move_count < max_moves and not grid.query.check_win(config.WHITE) and not grid.query.check_win(config.BLACK):
        if verbose and move_count % 10 == 0:
            print(f"Tour {move_count}/{max_moves}", end="\r")
        
        # Vérifier les cycles
        state_signature = str(grid)
        if state_signature in seen_states:
            seen_states[state_signature] += 1
            if seen_states[state_signature] > 3:  # Si on voit le même état 3 fois
                if verbose:
                    print(f"Détection de cycle après {move_count} tours")
                break  # Terminer la partie
        else:
            seen_states[state_signature] = 1
        
        if current_player == alphazero_player:
            # Tour d'AlphaZero
            our_state = their_grid_to_our_state(grid, current_player=current_player)
            action_idx = get_best_move(our_state, params, network, env, num_simulations=50)
            block, direction = convert_action_to_their_format(action_idx, our_state, grid, env)
        else:
            # Tour de l'algorithme classique
            if classical_algo == "minimax":
                _, move = minimax(grid, depth, current_player)
            elif classical_algo == "alphabeta":
                _, move = alphabeta(grid, depth, current_player, -math.inf, math.inf)
            elif classical_algo == "tt_alphabeta":
                #TT.table = {}
                TT.initialize_keys()
                _, move = TT.alphabeta(grid, depth, current_player, -math.inf, math.inf)
            elif classical_algo == "pvs":
                _, move = pvs(grid, current_player, -math.inf, math.inf, depth)
            elif classical_algo == "tt_pvs":
                #TT.table = {}
                TT.initialize_keys()
                _, move = TT.pvs(grid, current_player, -math.inf, math.inf, depth)
            
            block, direction = move
        
        # Appliquer le mouvement
        try:
            grid.move(block, direction)
        except Exception as e:
            if verbose:
                print(f"Erreur lors du mouvement: {e}")
        
        # Passer au joueur suivant
        current_player = config.BLACK if current_player == config.WHITE else config.WHITE
        move_count += 1
    
    # Déterminer le résultat
    if grid.query.check_win(config.WHITE):
        result = "Blanc (Algorithme classique)"
    elif grid.query.check_win(config.BLACK):
        result = "Noir (AlphaZero)"
    else:
        result = "Match nul"
    
    if verbose:
        print(f"Partie terminée après {move_count} tours. Résultat: {result}")
    
    return result


class Evaluator:
    """
    Classe pour évaluer les performances du modèle contre différents algorithmes
    """
    
    def __init__(self, model_params, network, env):
        self.params = model_params
        self.network = network
        self.env = env
    
    def evaluate_against_classical(self, algorithms=None, num_games_per_algo=2, verbose=True):
        """
        Évalue le modèle contre plusieurs algorithmes classiques
        """
        if algorithms is None:
            algorithms = [
                ("tt_alphabeta", 3),
                ("tt_pvs", 3)
            ]
        
        print("\n--- Évaluation du modèle ---")
        
        # Résultats pour chaque algorithme
        results = {}
        
        # Pour chaque algorithme
        for algo_name, depth in algorithms:
            if verbose:
                print(f"\nÉvaluation contre {algo_name} (profondeur {depth})...")
            
            # Jouer plusieurs parties
            wins = 0
            losses = 0
            draws = 0
            
            for game_idx in range(num_games_per_algo):
                if verbose:
                    print(f"Partie {game_idx+1}/{num_games_per_algo}")
                
                # Réinitialiser la table de transposition avant chaque partie
                TT.table = {}
                TT.initialize_keys()
                
                # Jouer une partie
                result = simulate_alphazero_vs_classical(
                    self.params, 
                    self.network, 
                    self.env, 
                    classical_algo=algo_name, 
                    depth=depth,
                    verbose=verbose
                )
                
                # Comptabiliser le résultat
                if result == "Noir (AlphaZero)":
                    wins += 1
                elif result == "Blanc (Algorithme classique)":
                    losses += 1
                else:
                    draws += 1
            
            # Calculer les statistiques
            win_rate = wins / num_games_per_algo
            results[algo_name] = {
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_rate": win_rate
            }
            
            if verbose:
                print(f"Résultats contre {algo_name}:")
                print(f"  Victoires: {wins}/{num_games_per_algo} ({win_rate:.1%})")
                print(f"  Défaites: {losses}/{num_games_per_algo} ({losses/num_games_per_algo:.1%})")
                print(f"  Matchs nuls: {draws}/{num_games_per_algo} ({draws/num_games_per_algo:.1%})")
        
        return results

# Fonction à intégrer dans AbaloneTrainerSync
def evaluate_model(self):
    """
    Évalue le modèle actuel contre plusieurs algorithmes classiques.
    Cette fonction est destinée à être utilisée dans AbaloneTrainerSync.
    """
    evaluator = Evaluator(self.params, self.network, self.env)
    
    # Algorithmes à évaluer
    algos = [
        ("tt_alphabeta", 3),
        ("tt_pvs", 3)
    ]
    
    # Évaluer contre les algorithmes classiques
    results = evaluator.evaluate_against_classical(
        algorithms=algos,
        num_games_per_algo=2,
        verbose=True
    )
    
    # Enregistrer les résultats dans l'historique des métriques
    latest_metrics = self.metrics_history[-1] if self.metrics_history else {}
    eval_metrics = {f"win_rate_{algo}": data["win_rate"] for algo, data in results.items()}
    
    # Mettre à jour les métriques avec les résultats d'évaluation
    if latest_metrics:
        latest_metrics.update(eval_metrics)
    
    return results


if __name__ == '__main__':
    print("hello")