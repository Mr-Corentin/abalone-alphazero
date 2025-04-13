import os
import subprocess
import jax
import jax.numpy as jnp
import pickle
import math
from functools import partial
from typing import List, Dict, Any, Tuple

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
        print(f"Erreur lors du téléchargement du checkpoint {gcs_path}: {e}")
        return False

def load_checkpoint_params(checkpoint_path):
    """Charge les paramètres depuis un checkpoint."""
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        return checkpoint['params']
    except Exception as e:
        print(f"Erreur lors du chargement du checkpoint {checkpoint_path}: {e}")
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
        from environment.env import AbaloneEnvNonCanonical
        self.env = AbaloneEnvNonCanonical(radius=radius)
        
        # Stocker les dispositifs locaux pour les opérations TPU
        self.devices = jax.local_devices()
        self.num_devices = len(self.devices)
        
        # Créer la fonction d'évaluation
        self.play_evaluation_games = self._create_evaluation_function()
    
    def _create_evaluation_function(self):
        """Crée une fonction pour jouer des parties entre deux versions du modèle."""
        from mcts.agent import get_best_move
        
        # @partial(jax.pmap, axis_name='devices')
        # def play_evaluation_games(rng_keys, black_params, white_params, games_per_device):
        @partial(jax.pmap, axis_name='devices', static_broadcasted_argnums=(3))
        def play_evaluation_games(rng_keys, black_params, white_params, games_per_device):
            """
            Joue des parties d'évaluation entre deux versions du modèle.
            
            Args:
                rng_keys: Clés aléatoires pour chaque dispositif
                black_params: Paramètres pour le joueur noir
                white_params: Paramètres pour le joueur blanc
                games_per_device: Nombre de parties à jouer par dispositif
                
            Returns:
                Dictionnaire avec les résultats des parties
            """
            def play_single_game(rng_key):
                """Joue une seule partie entre les modèles noir et blanc."""
                # Initialiser l'état du jeu
                state = self.env.reset(rng_key)
                move_count = 0
                max_moves = 300  # Éviter les parties infinies
                
                while not self.env.is_terminal(state) and move_count < max_moves:
                    # Déterminer quel modèle utiliser en fonction du joueur actuel
                    if state.actual_player == 1:  # Tour du noir
                        action = get_best_move(state, black_params, self.network, self.env, self.num_simulations, rng_key)
                    else:  # Tour du blanc
                        action = get_best_move(state, white_params, self.network, self.env, self.num_simulations, rng_key)
                    
                    # Appliquer l'action choisie
                    state = self.env.step(state, action)
                    
                    # Mettre à jour le compteur de coups et la clé RNG
                    move_count += 1
                    rng_key = jax.random.fold_in(rng_key, move_count)
                
                # Déterminer le résultat
                # 1: Noir gagne
                # -1: Blanc gagne
                # 0: Match nul
                if self.env.is_terminal(state):
                    outcome = self.env.get_winner(state)
                else:
                    # Match nul si nombre max de coups atteint
                    outcome = 0
                
                return outcome, move_count
            
            # Générer un lot de parties
            keys = jax.random.split(rng_keys, games_per_device + 1)
            
            # Initialiser les tableaux pour stocker les résultats
            outcomes = jnp.zeros(games_per_device, dtype=jnp.int8)
            move_counts = jnp.zeros(games_per_device, dtype=jnp.int16)
            
            # Boucle pour chaque partie
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
    
    def evaluate_model_pair(self, current_params, reference_params):
        """
        Évalue le modèle actuel contre un modèle de référence.
        
        Args:
            current_params: Paramètres du modèle actuel
            reference_params: Paramètres du modèle de référence
            
        Returns:
            Dictionnaire avec les résultats d'évaluation
        """
        # Préparer les paramètres pour distribution aux dispositifs
        current_params_replicated = jax.device_put_replicated(current_params, self.devices)
        reference_params_replicated = jax.device_put_replicated(reference_params, self.devices)
        
        # Nombre de parties par dispositif
        games_per_device = math.ceil(self.games_per_model / self.num_devices)
        
        # Générer des clés aléatoires pour chaque dispositif
        rng_key = jax.random.PRNGKey(42)
        sharded_rngs = jax.random.split(rng_key, self.num_devices)
        sharded_rngs = jax.device_put_sharded(list(sharded_rngs), self.devices)
        
        # Jouer des parties d'évaluation (actuel en tant que noir, référence en tant que blanc)
        print("Parties d'évaluation (modèle actuel en tant que Noir)...")
        results_current_black = self.play_evaluation_games(
            sharded_rngs, 
            current_params_replicated,
            reference_params_replicated,
            games_per_device
        )
        
        # Inverser les rôles pour l'équité
        print("Parties d'évaluation (modèle actuel en tant que Blanc)...")
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