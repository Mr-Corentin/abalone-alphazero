import os
import sys
import jax
import time

# Ajout du dépôt externe au PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), 'abalone-ai'))

# Imports locaux
from environment.env import AbaloneEnv
from model.neural_net import AbaloneModel
from training.trainer import AbaloneTrainerSync

def main():
    print("Démarrage du test TPU d'AlphaZero pour Abalone")
    
    # Afficher les devices disponibles
    print(f"Devices disponibles: {jax.devices()}")
    print(f"Local devices: {jax.local_devices()}")
    
    # Configuration adaptée pour TPU
    # Paramètres modérés pour un test d'environ 10-15 minutes
    model_filters = 64
    model_blocks = 6
    mcts_simulations = 50
    batch_size = 64
    games_per_device = 4
    games_per_iteration = 16
    training_steps = 20
    num_iterations = 5
    
    print(f"\nConfiguration du test:")
    print(f"- Modèle: {model_filters} filtres, {model_blocks} blocs")
    print(f"- MCTS: {mcts_simulations} simulations par coup")
    print(f"- Batch size: {batch_size}")
    print(f"- {games_per_device} parties par device, {games_per_iteration} parties par itération")
    print(f"- {training_steps} étapes d'entraînement par itération")
    print(f"- {num_iterations} itérations au total\n")
    
    # Créer le modèle et l'environnement
    model = AbaloneModel(
        num_filters=model_filters,
        num_blocks=model_blocks
    )
    env = AbaloneEnv()
    
    # Créer le trainer
    trainer = AbaloneTrainerSync(
        network=model,
        env=env,
        buffer_size=10000,  # Buffer suffisant pour ce test
        batch_size=batch_size,
        value_weight=1.0,
        games_per_device=games_per_device,
        num_simulations=mcts_simulations,
        checkpoint_path="checkpoints/tpu_test_model"
    )
    
    # Lancer un entraînement de test
    start_time = time.time()
    print("Démarrage de l'entraînement sur TPU...")
    
    trainer.train(
        num_iterations=num_iterations,
        games_per_iteration=games_per_iteration,
        training_steps_per_iteration=training_steps,
        eval_frequency=0,  # Pas d'évaluation pour ce test de performance
        save_frequency=5   # Sauvegarder à la fin
    )
    
    elapsed_time = time.time() - start_time
    print(f"\n=== Bilan du test TPU ===")
    print(f"Test TPU terminé en {elapsed_time:.2f} secondes")
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Temps total: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Calcul des metrics de performance
    total_games = trainer.total_games
    total_positions = trainer.total_positions
    avg_game_time = elapsed_time / total_games if total_games > 0 else 0
    avg_positions_per_game = total_positions / total_games if total_games > 0 else 0
    
    print(f"Nombre total de parties générées: {total_games}")
    print(f"Nombre total de positions: {total_positions}")
    print(f"Temps moyen par partie: {avg_game_time:.2f}s")
    print(f"Positions moyennes par partie: {avg_positions_per_game:.1f}")
    
    return 0

if __name__ == "__main__":
    main()