import os
import subprocess
import jax
import jax.numpy as jnp
import pickle
import math
from functools import partial
from typing import List, Dict, Any, Tuple
from environment.env import AbaloneEnv

import logging

# Configuration du logger au début de votre script ou dans __init__
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Process %(process)d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("alphazero.evaluator")

def generate_evaluation_checkpoints(total_iterations: int, num_checkpoints: int = 10) -> List[int]:
    """
    Génère une liste de points de référence bien distribués.
    
    Args:
        total_iterations: Nombre total d'itérations prévues
        num_checkpoints: Nombre approximatif de points de référence souhaités
    
    Returns:
        Liste des itérations à utiliser comme références
    """
    # Définir des échelles logarithmiques pour avoir plus de points près du début
    if total_iterations <= 100:
        # Échelle pour les très courts entraînements
        percentages = [0.1, 0.2, 0.35, 0.5, 0.7, 0.9]
    elif total_iterations <= 1000:
        # Échelle plus fine pour les entraînements courts
        percentages = [
            0.02,  # 2%
            0.05,  # 5%
            0.1,   # 10%
            0.2,   # 20%
            0.35,  # 35%
            0.5,   # 50%
            0.7,   # 70%
            0.85,  # 85%
            0.95,  # 95%
        ]
    else:
        # Échelle plus étendue pour les longs entraînements
        percentages = [
            0.01,  # 1%
            0.03,  # 3%
            0.07,  # 7%
            0.15,  # 15%
            0.25,  # 25%
            0.4,   # 40%
            0.6,   # 60%
            0.8,   # 80%
            0.95,  # 95%
        ]
    
    # Convertir les pourcentages en numéros d'itération
    checkpoints = [int(total_iterations * p) for p in percentages]
    
    # Filtrer les valeurs nulles ou en double
    checkpoints = [cp for cp in checkpoints if cp > 0]
    checkpoints = sorted(list(set(checkpoints)))
    
    return checkpoints

def check_checkpoint_exists(checkpoint_path):
    """Vérifie si un checkpoint existe au chemin spécifié."""
    # Vérifier le système de fichiers local
    if not checkpoint_path.startswith("gs://"):
        return os.path.exists(checkpoint_path)
    
    # Vérifier GCS
    try:
        result = subprocess.run(
            f"gsutil -q stat {checkpoint_path}",
            shell=True,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return result.returncode == 0
    except Exception:
        return False

def download_checkpoint(gcs_path, local_path):
    """Télécharge un checkpoint depuis GCS."""
    try:
        subprocess.run(f"gsutil cp {gcs_path} {local_path}", shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.info(f"Erreur lors du téléchargement du checkpoint {gcs_path}: {e}")
        return False

def load_checkpoint_params(checkpoint_path):
    """Charge les paramètres depuis un checkpoint."""
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        return checkpoint['params']
    except Exception as e:
        logger.info(f"Erreur lors du chargement du checkpoint {checkpoint_path}: {e}")
        return None

class ModelsEvaluator:
    """Classe pour évaluer le modèle actuel contre des versions antérieures."""

    def __init__(self, network, radius=4, num_simulations=50, games_per_model=10):
        """
        Initialise l'évaluateur.
        
        Args:
            network: Le modèle de réseau neuronal
            radius: Rayon du plateau (par défaut: 4)
            num_simulations: Nombre de simulations MCTS par coup
            games_per_model: Nombre de parties à jouer contre chaque modèle
        """
        self.network = network
        self.radius = radius
        self.num_simulations = num_simulations
        self.games_per_model = games_per_model
        
        # Créer un environnement non-canonique pour l'évaluation
        self.env = AbaloneEnv(radius=radius)
        
        # Stocker les dispositifs locaux pour les opérations TPU
        self.devices = jax.local_devices()
        self.num_devices = len(self.devices)
        
        # Créer la fonction d'évaluation
        self.play_evaluation_games = self._create_evaluation_function()

    def _create_evaluation_function(self):
        """Crée une fonction pour jouer des parties entre deux versions du modèle."""
        from mcts.agent import get_best_move
        
        @partial(jax.pmap, axis_name='devices', static_broadcasted_argnums=(3))
        def play_evaluation_games(rng_keys, black_params, white_params, games_per_device):
            """
            Joue des parties d'évaluation entre deux versions du modèle.
            """
            def play_single_game(rng_key):
                # Initialiser l'état
                init_state = self.env.reset(rng_key)
                
                # Créer la fonction de condition pour while_loop
                def cond_fn(carry):
                    state, move_count, current_rng = carry
                    not_terminal = jnp.logical_not(self.env.is_terminal(state))
                    under_max = move_count < 200  # max_moves
                    return jnp.logical_and(not_terminal, under_max)
                
                # Créer le corps de la boucle
                def body_fn(carry):
                    state, move_count, current_rng = carry
                    
                    # Générer une nouvelle clé
                    current_rng, action_key = jax.random.split(current_rng)
                    
                    # Sélectionner l'action en fonction du joueur actuel
                    is_black = state.actual_player == 1 
                    
                    action = jax.lax.cond(
                        is_black,
                        lambda: get_best_move(state, black_params, self.network, self.env, 
                                            self.num_simulations, action_key),
                        lambda: get_best_move(state, white_params, self.network, self.env, 
                                            self.num_simulations, action_key)
                    )
                    
                    # Appliquer l'action
                    next_state = self.env.step(state, action)
                    
                    return next_state, move_count + 1, current_rng
                
                # Exécuter la boucle JAX
                final_state, final_moves, _ = jax.lax.while_loop(
                    cond_fn,
                    body_fn,
                    (init_state, 0, rng_key)
                )
                
                # Déterminer le résultat final
                outcome = jax.lax.cond(
                    final_state.black_out >= 6,
                    lambda: jnp.array(-1, dtype=jnp.int8),  # Black lost (white won)
                    lambda: jax.lax.cond(
                        final_state.white_out >= 6,
                        lambda: jnp.array(1, dtype=jnp.int8),  # Black won
                        lambda: jnp.array(0, dtype=jnp.int8)   # Draw
                    )
                )
                
                return outcome, final_moves
            
            # Générer un lot de parties
            keys = jax.random.split(rng_keys, games_per_device + 1)
            
            # Initialiser les tableaux pour stocker les résultats
            outcomes = jnp.zeros(games_per_device, dtype=jnp.int8)
            move_counts = jnp.zeros(games_per_device, dtype=jnp.int16)
            
            # Exécuter les parties (de manière séquentielle mais compilée)
            for i in range(games_per_device):
                outcome, moves = play_single_game(keys[i])
                outcomes = outcomes.at[i].set(outcome)
                move_counts = move_counts.at[i].set(moves)
            
            # Retourner tous les résultats
            return {
                'outcomes': outcomes,
                'move_counts': move_counts
            }
        
        return play_evaluation_games

    

    
    def evaluate_model_pair(self, current_params, reference_params, games_to_play=None):
        """
        Évalue le modèle actuel contre un modèle de référence.
        
        Args:
            current_params: Paramètres du modèle actuel
            reference_params: Paramètres du modèle de référence
            games_to_play: Nombre de parties à jouer (si None, utilise self.games_per_model)
                
        Returns:
            Dictionnaire avec les résultats d'évaluation
        """
        # Utiliser le nombre spécifié ou la valeur par défaut
        num_games = games_to_play if games_to_play is not None else self.games_per_model
        
        # Préparer les paramètres pour distribution aux dispositifs
        current_params_replicated = jax.device_put_replicated(current_params, self.devices)
        reference_params_replicated = jax.device_put_replicated(reference_params, self.devices)
        
        # Nombre de parties par dispositif
        games_per_device = math.ceil(num_games / self.num_devices)
        
        # Générer des clés aléatoires pour chaque dispositif
        rng_key = jax.random.PRNGKey(42)
        sharded_rngs = jax.random.split(rng_key, self.num_devices)
        sharded_rngs = jax.device_put_sharded(list(sharded_rngs), self.devices)
        
        # Jouer des parties d'évaluation (actuel en tant que noir, référence en tant que blanc)
        logger.info("Parties d'évaluation (modèle actuel en tant que Noir)...")
        results_current_black = self.play_evaluation_games(
            sharded_rngs, 
            current_params_replicated,
            reference_params_replicated,
            games_per_device
        )
        
        # Inverser les rôles pour l'équité
        logger.info("Parties d'évaluation (modèle actuel en tant que Blanc)...")
        new_rng_key = jax.random.fold_in(rng_key, 1000)
        sharded_rngs = jax.random.split(new_rng_key, self.num_devices)
        sharded_rngs = jax.device_put_sharded(list(sharded_rngs), self.devices)
        
        results_current_white = self.play_evaluation_games(
            sharded_rngs, 
            reference_params_replicated,
            current_params_replicated,
            games_per_device
        )
        
        # Récupérer et traiter les résultats
        current_black_results = jax.device_get(results_current_black)
        current_white_results = jax.device_get(results_current_white)
        
        # Calculer les statistiques globales
        total_games = 0
        current_wins = 0
        reference_wins = 0
        draws = 0
        
        # Traiter les résultats du modèle actuel en tant que noir
        for device_results in current_black_results['outcomes']:
            for outcome in device_results:
                if outcome == 0:  # Partie non jouée ou match nul
                    draws += 1
                    total_games += 1
                elif outcome == 1:  # Modèle actuel a gagné (en tant que noir)
                    current_wins += 1
                    total_games += 1
                elif outcome == -1:  # Modèle de référence a gagné (en tant que blanc)
                    reference_wins += 1
                    total_games += 1
        
        # Traiter les résultats du modèle actuel en tant que blanc
        for device_results in current_white_results['outcomes']:
            for outcome in device_results:
                if outcome == 0:  # Partie non jouée ou match nul
                    draws += 1
                    total_games += 1
                elif outcome == -1:  # Modèle actuel a gagné (en tant que blanc)
                    current_wins += 1
                    total_games += 1
                elif outcome == 1:  # Modèle de référence a gagné (en tant que noir)
                    reference_wins += 1
                    total_games += 1
        
        # Calculer le taux de victoire
        win_rate = current_wins / max(1, total_games)
        
        return {
            'total_games': total_games,
            'current_wins': current_wins,
            'reference_wins': reference_wins,
            'draws': draws,
            'win_rate': win_rate
        }