#!/usr/bin/env python3
"""
Script de test de l'évaluation dans AbaloneTrainerSync (modes TPU et CPU)
"""
import os
import time
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Test d\'évaluation avec AbaloneTrainerSync')
    parser.add_argument('--mode', type=str, default='tpu', choices=['tpu', 'cpu'],
                      help='Mode d\'exécution (tpu ou cpu)')
    parser.add_argument('--num_games', type=int, default=2,
                      help='Nombre de parties à simuler par algorithme')
    parser.add_argument('--depth', type=int, default=3,
                      help='Profondeur de recherche alphabeta')
    parser.add_argument('--verbose', action='store_true',
                      help='Activer les logs détaillés')
    return parser.parse_args()

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

def main():
    args = parse_args()
    
    # Configurer JAX selon le mode
    force_cpu = (args.mode == 'cpu')
    jax = initialize_jax(force_cpu)
    
    # Importer les modules après avoir configuré JAX
    import jax.numpy as jnp
    from model.neural_net import AbaloneModel
    from environment.env import AbaloneEnv
    from training.trainer import AbaloneTrainerSync
    
    # Obtenir des informations sur l'environnement
    process_id = jax.process_index()
    num_processes = jax.process_count()
    is_main_process = process_id == 0
    devices = jax.local_devices()
    
    if is_main_process:
        print(f"\n=== Test d'évaluation avec AbaloneTrainerSync en mode {args.mode.upper()} ===")
        print(f"Nombre de processus: {num_processes}")
        print(f"Dispositifs locaux: {len(devices)} ({[d.platform for d in devices]})")
        print(f"Profondeur alphabeta: {args.depth} (sera transmise à l'Evaluator)")
        print(f"Nombre de parties par algorithme: {args.num_games}")
    
    # Initialiser le modèle et l'environnement
    network = AbaloneModel(num_filters=32, num_blocks=3)
    env = AbaloneEnv()
    
    # Créer le trainer avec une configuration minimale
    trainer = AbaloneTrainerSync(
        network=network,
        env=env,
        buffer_size=1000,
        batch_size=8,
        value_weight=1.0,
        num_simulations=10,
        initial_lr=0.01,
        checkpoint_path="/tmp/test_checkpoint",
        eval_games=args.num_games  # Important: utiliser le nombre de parties spécifié
    )
    
    # Patch pour modifier la profondeur d'alphabeta dans l'Evaluator
    # Cette modification affecte le code de l'Evaluator généré plus tard
    import types
    import evaluation.alphabeta.heuristics
    original_alphabeta = evaluation.alphabeta.heuristics.alphabeta_pruning
    
    def patched_alphabeta(state, depth, alpha, beta, env, radius=4, original_player=None):
        # Forcer la profondeur à la valeur spécifiée
        return original_alphabeta(state, args.depth, alpha, beta, env, radius, original_player)
    
    evaluation.alphabeta.heuristics.alphabeta_pruning = patched_alphabeta
    
    # Mesurer le temps d'exécution pour l'évaluation
    if is_main_process:
        print("\nDémarrage de l'évaluation...")
    
    # Synchroniser tous les processus avant de démarrer
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
    
    start_time = time.time()
    
    # Exécuter l'évaluation
    results = trainer._evaluate()
    
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
    
    # Nettoyer
    evaluation.alphabeta.heuristics.alphabeta_pruning = original_alphabeta
    
    # Attendre que tous les processus terminent
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

if __name__ == "__main__":
    main()