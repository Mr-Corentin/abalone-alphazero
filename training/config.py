"""
Configuration d'entraînement pour AlphaZero Abalone
"""

DEFAULT_CONFIG = {
    "model": {
        "num_filters": 128,
        "num_blocks": 10,
    },
    
    "mcts": {
        "num_simulations": 800,
        "max_num_considered_actions": 16,
    },
    
    
    "training": {
        "batch_size": 128,
        "value_weight": 1.0,
        "games_per_device": 8,
        "games_per_iteration": 64,
        "training_steps_per_iteration": 20,
        "num_iterations": 1000,
    },
    
    "optimizer": {
        "initial_lr": 0.2,
        "momentum": 0.9,
        "lr_schedule": [
            (0.0, 0.2),      
            (0.3, 0.02),     
            (0.6, 0.002),   
            (0.85, 0.0002)   
        ]
    },

    "buffer": {
        "size": 1_000_000,
        "recency_bias": True,
        "recency_temperature": 0.8,
        "use_gcs": False,
        "gcs_dir": "buffer"
    },
    
    "checkpoint": {
        "path": "checkpoints/model",
        "save_frequency": 2,
        "eval_frequency": 5,
    }
}


MINIMAL_CONFIG = {
    "model": {
        "num_filters": 64,
        "num_blocks": 10,
    },
    
    "mcts": {
        "num_simulations": 10,
        "max_num_considered_actions": 16,
    },
    
    
    "training": {
        "batch_size": 128,
        "value_weight": 1.0,
        "games_per_device": 8,
        "games_per_iteration": 16,
        "training_steps_per_iteration": 100,
        "num_iterations": 1000,
    },
    
    "optimizer": {
        "initial_lr": 0.2,
        "momentum": 0.9,
        "lr_schedule": [
            (0.0, 0.2),      
            (0.3, 0.02),     
            (0.6, 0.002),   
            (0.85, 0.0002)   
        ]
    },

    "buffer": {
        "size": 1_000_000,
        "recency_bias": True,
        "recency_temperature": 0.8,
        "use_gcs": False,
        "gcs_dir": "buffer"
    },
    
    "checkpoint": {
        "path": "checkpoints/model",
        "save_frequency": 10,
        "eval_frequency": 2,
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
        "num_iterations": 1,
        "value_weight": 1.0
    },
    "checkpoint": {
        "path": "checkpoints/model_cpu",
        "save_frequency": 1,
        "eval_frequency": 1
    },
    "optimizer": {
        "initial_lr": 0.01,
        "momentum": 0.9
    }
}

def get_config():
    """
    Récupère la configuration par défaut
    
    Returns:
        Dictionnaire de configuration
    """
    return DEFAULT_CONFIG