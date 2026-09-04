"""
Training configuration for AlphaZero Abalone
"""

DEFAULT_CONFIG = {
    "game": {
        # Truncation limit for self-play games. NOT an Abalone rule -- see
        # AbaloneEnv. Lower means faster iterations during early training.
        "max_moves": 150,
        # Past positions fed to the network. 0 = disabled (see AbaloneEnv);
        # set to 8 to restore the previous behaviour.
        "history_length": 0,
    },

    "model": {
        "num_filters": 128,
        "num_blocks": 10,
    },
    
    "mcts": {
        "num_simulations": 600,
        "max_num_considered_actions": 16,
    },
    
    
    "training": {
        "batch_size": 128,
        "value_weight": 1.0,
        "games_per_device": 8,
        "games_per_iteration": 64,
        "training_steps_per_iteration": 20,
        "num_iterations": 100,
    },
    
    "optimizer": {
        "initial_lr": 0.2,
        "momentum": 0.9,
        # L2 weight regularization (c*||theta||^2 in the AlphaZero loss).
        "weight_decay": 1e-4,
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
    },
    
    "logging": {
        "enable_comprehensive_logging": True,
        "metrics_logging_interval": 30
    }
}


MINIMAL_CONFIG = {
    "game": {
        # Truncation limit for self-play games. NOT an Abalone rule -- see
        # AbaloneEnv. Lower means faster iterations during early training.
        "max_moves": 150,
        # Past positions fed to the network. 0 = disabled (see AbaloneEnv);
        # set to 8 to restore the previous behaviour.
        "history_length": 0,
    },

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
        # L2 weight regularization (c*||theta||^2 in the AlphaZero loss).
        "weight_decay": 1e-4,
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
    },
    
    "logging": {
        "enable_comprehensive_logging": True,
        "metrics_logging_interval": 30
    }
}

CPU_CONFIG = {
    "game": {"max_moves": 30, "history_length": 0},

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
    },
    "logging": {
        "enable_comprehensive_logging": False,  # Disabled for CPU testing
        "metrics_logging_interval": 30
    }
}

def get_config():
    """
    Get default configuration
    
    Returns:
        Configuration dictionary
    """
    return DEFAULT_CONFIG