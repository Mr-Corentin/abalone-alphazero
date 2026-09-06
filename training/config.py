"""
Training configuration for AlphaZero Abalone
"""

DEFAULT_CONFIG = {
    "game": {
        # Troncature des parties de self-play. PAS une regle d'Abalone -- voir
        # AbaloneEnv. 200 et non 150 : mesure sur 128 parties, la proportion de
        # parties finissant sur un differentiel de billes NON NUL (donc avec une
        # cible de valeur exploitable) vaut 44% a 100 plis, 66% a 150, 74% a 200,
        # puis PLAFONNE -- 74% a 250, 71% a 300. Au-dela de 200 on paie du calcul
        # pour zero signal en plus, le differentiel se comportant comme une marche
        # aleatoire qui repasse par zero. En dessous on perd du signal.
        "max_moves": 200,
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
        # 64, pas les 16 par defaut de mctx : Abalone offre ~66 coups legaux
        # (52 a la position initiale). Gumbel top-k n'echantillonne que
        # `max_num_considered_actions` coups a la RACINE, et les autres sont
        # ensuite exclus par un -inf dans seq_halving.score_considered -- pas
        # rendus improbables, rendus injouables. A 16, l'agent ne pouvait donc
        # jouer que 25% de ses coups legaux a chaque decision, et ratait 58%
        # des mats en 1 (mesure : 42% de detection a m=16, 77% a 32, 97% a 64).
        # Le budget de simulations est INCHANGE (Sequential Halving repartit les
        # memes `num_simulations`), donc ce reglage ne coute aucun calcul : il
        # donne juste un finaliste a 71 visites au lieu de 97.
        "max_num_considered_actions": 64,
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
        # Voir DEFAULT_CONFIG pour le choix de 200.
        "max_moves": 200,
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
        # Voir DEFAULT_CONFIG : 64 et non 16, cf. le facteur de branchement d'Abalone.
        "max_num_considered_actions": 64,
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