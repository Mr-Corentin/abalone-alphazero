#!/usr/bin/env python3
"""
Script de test d'évaluation avec Evaluator (TPU vs CPU forcé)
"""
import os
import time
import argparse
import numpy as np

# Initialiser JAX avec la bonne configuration selon le mode
def initialize_jax(force_cpu=False):
    if force_cpu:
        # Force JAX à utiliser uniquement le CPU
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["TPU_NAME"] = ""
        # Importer JAX avec ces configurations
        import jax
        print(f"JAX configuré pour CPU: {jax.devices()}")
        return jax
    else:
        # Configuration normale pour TPU
        import jax
        jax.distributed.initialize()
        print(f"JAX configuré pour TPU: {jax.devices()}")
        return jax

def parse_args():
    parser = argparse.ArgumentParser(description='Test d\'évaluation avec Evaluator (TPU vs CPU)')
    parser.add_argument('--mode', type=str, default='tpu', choices=['tpu', 'cpu'],
                      help='Mode d\'exécution (tpu ou cpu)')
    parser.add_argument('--num_games', type=int, default=2,
                      help='Nombre de parties à simuler par algorithme')
    parser.add_argument('--depth', type=int, default=3,
                      help='Profondeur de recherche alphabeta')
    parser.add_argument('--verbose', action='store_true',
                      help='Activer les logs détaillés')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Configurer JAX selon le mode
    force_cpu = (args.mode == 'cpu')
    jax = initialize_jax(force_cpu)
    
    # Importer les modules après avoir configuré JAX
    import jax.numpy as jnp
    from model.neural_net import AbaloneModel
    from environment.env import AbaloneEnv
    from evaluation.evaluator import Evaluator
    
    # Obtenir des informations sur l'environnement
    process_id = jax.process_index()
    num_processes = jax.process_count()
    is_main_process = process_id == 0
    devices = jax.local_devices()
    
    if is_main_process:
        print(f"\n=== Test d'évaluation avec Evaluator en mode {args.mode.upper()} ===")
        print(f"Nombre de processus: {num_processes}")
        print(f"Dispositifs locaux: {len(devices)} ({[d.platform for d in devices]})")
        print(f"Profondeur alphabeta: {args.depth}")
        print(f"Nombre de parties par algorithme: {args.num_games}")
    
    # Initialiser un environnement
    env = AbaloneEnv()
    
    # Créer un modèle avec une configuration légère
    model = AbaloneModel(num_filters=32, num_blocks=3)
    
    # Initialiser des paramètres aléatoires
    rng = jax.random.PRNGKey(42)
    sample_board = jnp.zeros((1, 9, 9), dtype=jnp.int8)
    sample_marbles = jnp.zeros((1, 2), dtype=jnp.int8)
    params = model.init(rng, sample_board, sample_marbles)
    
    # Créer l'évaluateur
    evaluator = Evaluator(params, model, env)
    
    # Définir les algorithmes à tester
    algorithms = [
        ("alphabeta_pruning", args.depth)
    ]
    
    # Mesurer le temps d'exécution
    if is_main_process:
        print("\nDémarrage de l'évaluation...")
        
    start_time = time.time()
    
    # Exécuter l'évaluation uniquement sur le processus principal
    if is_main_process:
        results = evaluator.evaluate_against_classical(
            algorithms=algorithms,
            num_games_per_algo=args.num_games,
            verbose=args.verbose
        )
    else:
        # Les autres processus attendent simplement
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
        
        print(f"\nTemps moyen par partie: {elapsed/args.num_games:.3f}s")
    
    # Attendre que tous les processus terminent
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

if __name__ == "__main__":
    main()