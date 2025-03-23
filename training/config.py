"""
Configuration d'entraînement pour AlphaZero Abalone
"""

# Configuration principale d'entraînement
DEFAULT_CONFIG = {
    # Paramètres du modèle
    "model": {
        "num_filters": 128,
        "num_blocks": 10,
    },
    
    # Paramètres de MCTS
    "mcts": {
        "num_simulations": 600,
        "max_num_considered_actions": 16,
    },
    
    # Paramètres du buffer
    "buffer": {
        "size": 1_000_000,
        "recency_bias": True,
        "recency_temperature": 0.8,
    },
    
    # Paramètres d'entraînement
    "training": {
        "batch_size": 128,
        "value_weight": 1.0,
        "games_per_device": 8,
        "games_per_iteration": 64,
        "training_steps_per_iteration": 100,
        "num_iterations": 1000,
    },
    
    # Paramètres d'optimisation
    "optimizer": {
        "initial_lr": 0.2,
        "momentum": 0.9,
        "lr_schedule": [
            (0.0, 0.2),      # Départ
            (0.3, 0.02),     # Première chute
            (0.6, 0.002),    # Deuxième chute
            (0.85, 0.0002)   # Troisième chute
        ]
    },
    
    # Paramètres de checkpoint
    "checkpoint": {
        "path": "checkpoints/model",
        "save_frequency": 10,
        "eval_frequency": 20,
    }
}

CPU_CONFIG = {
    "model": {"num_filters": 32, "num_blocks": 3},
    "mcts": {"num_simulations": 5, "max_num_considered_actions": 8},
    "buffer": {"size": 1000},
    "training": {
        "batch_size": 8,
        "games_per_device": 1,
        "games_per_iteration": 1,
        "training_steps_per_iteration": 2,
        "num_iterations": 1
    }
}

def get_config():
    """
    Récupère la configuration par défaut
    
    Returns:
        Dictionnaire de configuration
    """
    return DEFAULT_CONFIG