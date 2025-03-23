"""
Script de test local minimal pour valider le pipeline d'entraînement AlphaZero
"""
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
from training.config import get_config

def main():
    print("Démarrage du test local d'AlphaZero pour Abalone")
    
    # Afficher les devices disponibles
    print(f"Devices disponibles: {jax.devices()}")
    
    # Configuration minimale pour CPU
    config = get_config()
    
    # Override avec des valeurs très réduites pour CPU
    config["model"]["num_filters"] = 32
    config["model"]["num_blocks"] = 3
    config["mcts"]["num_simulations"] = 5
    config["training"]["batch_size"] = 8
    config["training"]["games_per_device"] = 1
    config["training"]["games_per_iteration"] = 2
    config["training"]["training_steps_per_iteration"] = 5
    config["training"]["num_iterations"] = 3
    
    # Créer le modèle et l'environnement
    model_config = config["model"]
    model = AbaloneModel(
        num_filters=model_config["num_filters"],
        num_blocks=model_config["num_blocks"]
    )
    env = AbaloneEnv()
    
    # Créer le trainer
    trainer = AbaloneTrainerSync(
        network=model,
        env=env,
        buffer_size=1000,
        batch_size=config["training"]["batch_size"],
        value_weight=config["training"]["value_weight"],
        games_per_device=config["training"]["games_per_device"],
        num_simulations=config["mcts"]["num_simulations"],
        checkpoint_path="checkpoints/test_model"
    )
    
    # Lancer un mini-entraînement
    start_time = time.time()
    print("Démarrage de l'entraînement test...")
    
    trainer.train(
        num_iterations=config["training"]["num_iterations"],
        games_per_iteration=config["training"]["games_per_iteration"],
        training_steps_per_iteration=config["training"]["training_steps_per_iteration"],
        eval_frequency=0,  # Pas d'évaluation pour le test
        save_frequency=0   # Pas de sauvegarde pour le test
    )
    
    elapsed_time = time.time() - start_time
    print(f"Test terminé en {elapsed_time:.2f} secondes")
    
    return 0

if __name__ == "__main__":
    main()