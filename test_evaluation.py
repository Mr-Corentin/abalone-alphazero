#!/usr/bin/env python3
"""
Script de test d'évaluation avec Evaluator sur TPU
"""
import jax
jax.distributed.initialize()  # Initialisation JAX pour TPU dès le début

import os
import time
import argparse
import jax.numpy as jnp
import numpy as np

from model.neural_net import AbaloneModel
from environment.env import AbaloneEnv
from evaluation.evaluator import Evaluator

def parse_args():
    parser = argparse.ArgumentParser(description='Test d\'évaluation avec Evaluator sur TPU')
    parser.add_argument('--num_games', type=int, default=1,
                      help='Nombre de parties à simuler par algorithme')
    parser.add_argument('--depth', type=int, default=3,
                      help='Profondeur de recherche alphabeta')
    parser.add_argument('--verbose', action='store_true',
                      help='Activer les logs détaillés')
    return parser.parse_args()

def initialize_model_and_env():
    """Initialise le modèle et l'environnement pour le test"""
    # Initialiser un environnement
    env = AbaloneEnv()
    
    # Créer un modèle avec une configuration légère
    model = AbaloneModel(num_filters=32, num_blocks=3)
    
    # Initialiser des paramètres aléatoires
    rng = jax.random.PRNGKey(42)
    sample_board = jnp.zeros((1, 9, 9), dtype=jnp.int8)
    sample_marbles = jnp.zeros((1, 2), dtype=jnp.int8)
    params = model.init(rng, sample_board, sample_marbles)
    
    return params, model, env

def run_evaluator_test(num_games, depth, verbose=False):
    """Exécute un test d'évaluation avec Evaluator"""
    # Obtenir des informations sur l'environnement
    process_id = jax.process_index()
    num_processes = jax.process_count()
    is_main_process = process_id == 0
    devices = jax.local_devices()
    
    if is_main_process:
        print(f"\n=== Test d'évaluation avec Evaluator sur TPU ===")
        print(f"Nombre de processus: {num_processes}")
        print(f"Dispositifs locaux: {len(devices)} ({[d.platform for d in devices]})")
        print(f"Profondeur alphabeta: {depth}")
        print(f"Nombre de parties par algorithme: {num_games}")
    
    # Initialiser le modèle et l'environnement
    params, model, env = initialize_model_and_env()
    
    # Créer l'évaluateur
    evaluator = Evaluator(params, model, env)
    
    # Définir les algorithmes à tester
    algorithms = [
        ("alphabeta_pruning", depth)
    ]
    
    # Mesurer le temps d'exécution
    if is_main_process:
        print("\nDémarrage de l'évaluation...")
        
    start_time = time.time()
    
    # Exécuter l'évaluation uniquement sur le processus principal
    if is_main_process:
        results = evaluator.evaluate_against_classical(
            algorithms=algorithms,
            num_games_per_algo=num_games,
            verbose=verbose
        )
    else:
        # Les autres processus attendent simplement
        # Cela simule le comportement correct où l'évaluation est limitée au processus principal
        time.sleep(1)
        results = {}
    
    elapsed = time.time() - start_time
    
    # Afficher les résultats
    if is_main_process:
        print(f"\nÉvaluation terminée en {elapsed:.3f} secondes")
        
        if results:
            print("\nRésultats d'évaluation:")
            for algo_name, data in results.items():
                win_rate = data["win_rate"]
                print(f"vs {algo_name}: {win_rate:.1%} ({data['wins']}/{data['wins']+data['losses']+data['draws']})")
        
        print(f"\nTemps moyen par partie: {elapsed/num_games:.3f}s")
    
    return elapsed

def main():
    args = parse_args()
    
    # Exécuter le test avec Evaluator
    total_time = run_evaluator_test(args.num_games, args.depth, args.verbose)
    
    # Attendre que tous les processus terminent
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

if __name__ == "__main__":
    main()