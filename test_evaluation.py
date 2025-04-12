#!/usr/bin/env python3
"""
Script de test d'évaluation alphabeta sur TPU
"""
import os
import time
import argparse
import jax
import jax.numpy as jnp
import numpy as np

from environment.env import AbaloneEnv, AbaloneEnvNonCanonical, AbaloneStateNonCanonical
from core.board import initialize_board
from evaluation.alphabeta.heuristics import alphabeta_pruning

def parse_args():
    parser = argparse.ArgumentParser(description='Test alphabeta sur TPU')
    parser.add_argument('--num_games', type=int, default=2,
                       help='Nombre de parties à simuler')
    parser.add_argument('--depth', type=int, default=3,
                       help='Profondeur de recherche alphabeta')
    parser.add_argument('--verbose', action='store_true',
                       help='Activer les logs détaillés')
    return parser.parse_args()

def initialize_game():
    """Initialise un état de jeu pour test"""
    env = AbaloneEnvNonCanonical()
    board = initialize_board()
    
    # Créer un état initial
    state = AbaloneStateNonCanonical(
        board=board,
        current_player=1,  # Noir commence
        black_out=0,
        white_out=0,
        moves_count=0
    )
    
    return env, state

def run_alphabeta_test(num_games, depth, verbose=False):
    """Exécute un test d'alphabeta sur TPU"""
    # Initialiser JAX pour TPU
    jax.distributed.initialize()
    
    # Obtenir des informations sur l'environnement
    process_id = jax.process_index()
    num_processes = jax.process_count()
    is_main_process = process_id == 0
    devices = jax.local_devices()
    
    if is_main_process:
        print(f"\n=== Test d'évaluation alphabeta sur TPU ===")
        print(f"Nombre de processus: {num_processes}")
        print(f"Dispositifs locaux: {len(devices)}")
        print(f"Profondeur alphabeta: {depth}")
        print(f"Nombre de parties: {num_games}")
    
    env, initial_state = initialize_game()
    
    # Chaque processus exécute num_games parties
    total_time = 0
    total_nodes = 0
    moves_counter = 0
    
    # Traçage initial pour compiler la fonction et éviter les pénalités de première exécution
    _, _ = alphabeta_pruning(
        initial_state, 
        1,  # profondeur minimale pour le traçage 
        float('-inf'),
        float('inf'),
        env,
        radius=env.radius
    )
    
    # Simuler plusieurs parties partielles
    for game_idx in range(num_games):
        if is_main_process and verbose:
            print(f"\nProcessus {process_id}: Partie {game_idx+1}/{num_games}")
        
        # Réinitialiser l'état pour chaque partie
        state = initial_state
        
        # Jouer quelques coups (max 5 par partie)
        for move_num in range(5):
            if is_main_process and verbose:
                print(f"  Coup {move_num+1}/5")
            
            start_time = time.time()
            
            # Mesurer le temps pour alphabeta_pruning
            _, best_move = alphabeta_pruning(
                state, 
                depth,
                float('-inf'), 
                float('inf'),
                env, 
                radius=env.radius
            )
            
            elapsed = time.time() - start_time
            total_time += elapsed
            moves_counter += 1
            
            # Estimation grossière du nombre de nœuds
            estimated_nodes = sum(20**d for d in range(depth+1))
            total_nodes += estimated_nodes
            
            if is_main_process and verbose:
                print(f"    Temps: {elapsed:.3f}s, ~{estimated_nodes/elapsed:.0f} nœuds/sec")
            
            # Appliquer le coup et continuer
            if best_move >= 0:
                state = env.step(state, best_move)
            else:
                if is_main_process and verbose:
                    print("  Aucun coup légal trouvé, fin de la partie")
                break
    
    # Agréger les résultats de tous les processus
    avg_time = total_time / moves_counter if moves_counter > 0 else 0
    nodes_per_sec = total_nodes / total_time if total_time > 0 else 0
    
    # Seulement le processus principal affiche les résultats finaux
    if is_main_process:
        print("\nRésultats (processus principal):")
        print(f"  Nombre de coups joués: {moves_counter}")
        print(f"  Temps total: {total_time:.3f}s")
        print(f"  Temps moyen par coup: {avg_time:.3f}s")
        print(f"  Performance: ~{nodes_per_sec:.0f} nœuds/sec")
    
    return {
        'total_time': total_time,
        'avg_time': avg_time,
        'nodes_per_sec': nodes_per_sec,
        'moves_played': moves_counter
    }

def main():
    args = parse_args()
    
    # Exécuter le test sur TPU
    results = run_alphabeta_test(args.num_games, args.depth, args.verbose)
    

if __name__ == "__main__":
    main()