import subprocess
import jax
import jax.numpy as jnp
# Import explicite : ce sous-module n'est pas charge par `import jax`.
# Le code marchait par accident, quand une autre dependance l'avait
# importe avant nous -- et echouait sinon (AttributeError).
import jax.experimental.multihost_utils
import pickle
import math
from functools import partial
from typing import List, Dict, Any, Tuple
from environment.env import AbaloneEnv
from utils.sharding import replicate

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
    """
    Vérifie si un checkpoint existe au chemin spécifié.

    `checkpoint_path` peut contenir un wildcard (voir
    AbaloneTrainerSync._get_checkpoint_path, qui insère un `*` pour le
    timestamp). `os.path.exists` et `gsutil stat` traitent tous les deux un
    chemin avec `*` comme un nom littéral -- ni l'un ni l'autre ne
    l'expansent -- donc les deux renvoyaient toujours False et aucune
    évaluation contre un ancien modèle ne se déclenchait jamais. `glob` (local)
    et `gsutil ls` (GCS) expansent bien le wildcard.
    """
    if not checkpoint_path.startswith("gs://"):
        import glob
        return len(glob.glob(checkpoint_path)) > 0

    try:
        result = subprocess.run(
            f"gsutil ls {checkpoint_path}",
            shell=True,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:
        return False

    
def download_checkpoint(gcs_pattern, local_path):
    """Télécharge un checkpoint depuis GCS en gérant les wildcards"""
    try:
        import subprocess
        list_cmd = f"gsutil ls {gcs_pattern}"
        result = subprocess.run(list_cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Impossible de lister les fichiers: {result.stderr}")
            return False
            
        files = result.stdout.strip().split('\n')
        files = [f.strip() for f in files if f.strip()]
        
        if not files:
            logger.error(f"Aucun fichier trouvé pour {gcs_pattern}")
            return False
            
        actual_gcs_path = files[0]
        
        cmd = f"gsutil cp {actual_gcs_path} {local_path}"
        result = subprocess.run(cmd, shell=True)
        
        return result.returncode == 0
        
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement: {e}")
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

    def __init__(self, network, radius=4, num_simulations=50, games_per_model=10,
                 max_moves=200, max_num_considered_actions=64):
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
        self.max_num_considered_actions = max_num_considered_actions

        # Même limite de coups qu'à l'entraînement (auparavant 200 ici contre 300
        # à l'entraînement, ce qui faussait la comparaison des taux de nul).
        self.env = AbaloneEnv(radius=radius, max_moves=max_moves)
        
        # Stocker les dispositifs locaux pour les opérations TPU
        self.devices = jax.local_devices()
        self.num_devices = len(self.devices)
        
        # Créer la fonction d'évaluation
        self.play_evaluation_games = self._create_evaluation_function()

    def _create_evaluation_function(self):
        """
        Crée une fonction pour jouer des parties d'évaluation entre deux modèles.

        Les parties d'un même appel avancent en lock-step : elles démarrent toutes
        au pli 0 avec les Noirs au trait, donc le camp au trait ne dépend que de la
        parité du pli. On choisit les paramètres avec un lax.cond et on lance UNE
        recherche MCTS batchée par pli -- au lieu de l'ancienne version qui
        déroulait une boucle Python sur les parties, avec un MCTS de batch 1 par
        coup et deux branches lax.cond contenant chacune une recherche complète.
        """
        from mcts.core import AbaloneMCTSRecurrentFn
        from mcts.search import run_search_batch

        env = self.env
        network = self.network
        num_simulations = self.num_simulations
        max_considered = self.max_num_considered_actions

        @partial(jax.pmap, axis_name='devices', static_broadcasted_argnums=(3,))
        def play_evaluation_games(rng_key, black_params, white_params, games_per_device):
            recurrent_fn = AbaloneMCTSRecurrentFn(env, network)
            init_states = env.reset_batch(rng_key, games_per_device)

            def cond_fn(carry):
                return jnp.any(carry[2])

            def body_fn(carry):
                states, rng, active, move_counts, ply = carry

                terminal = env.is_terminal_batch(states)
                active_games = active & ~terminal

                rng, search_rng = jax.random.split(rng)

                # Toutes les parties actives sont au même pli : le camp au trait
                # est donné par la parité, pas besoin de brancher par partie.
                params = jax.lax.cond(ply % 2 == 0,
                                      lambda: black_params,
                                      lambda: white_params)

                search_outputs = run_search_batch(
                    states, recurrent_fn, network, params, search_rng, env,
                    iteration=0,
                    num_simulations=num_simulations,
                    max_num_considered_actions=max_considered,
                )

                next_states = jax.vmap(env.step)(states, search_outputs.action)

                # Les parties terminées sont gelées.
                states = jax.tree.map(
                    lambda nxt, cur: jnp.where(
                        active_games.reshape(active_games.shape + (1,) * (nxt.ndim - 1)), nxt, cur),
                    next_states, states)

                return (states, rng, active_games,
                        move_counts + active_games.astype(jnp.int16), ply + 1)

            final_states, _, _, move_counts, _ = jax.lax.while_loop(
                cond_fn, body_fn,
                (init_states, rng_key,
                 jnp.ones(games_per_device, dtype=jnp.bool_),
                 jnp.zeros(games_per_device, dtype=jnp.int16),
                 jnp.int32(0)))

            outcomes = jnp.where(
                final_states.black_out >= 6, jnp.int8(-1),      # les Blancs gagnent
                jnp.where(final_states.white_out >= 6, jnp.int8(1),  # les Noirs gagnent
                          jnp.int8(0))).astype(jnp.int8)

            return {'outcomes': outcomes, 'move_counts': move_counts}

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
        current_params_replicated = replicate(current_params, self.num_devices)
        reference_params_replicated = replicate(reference_params, self.num_devices)
        
        # Nombre de parties par dispositif
        games_per_device = math.ceil(num_games / self.num_devices)
        
        # Générer des clés aléatoires pour chaque dispositif
        rng_key = jax.random.PRNGKey(42)
        # pmap shards the leading axis itself
        sharded_rngs = jax.random.split(rng_key, self.num_devices)
        
        # Jouer des parties d'évaluation (actuel en tant que noir, référence en tant que blanc)
        logger.info("Parties d'évaluation (modèle actuel en tant que Noir)...")
        results_current_black = self.play_evaluation_games(
            sharded_rngs, 
            current_params_replicated,
            reference_params_replicated,
            games_per_device
        )
        
        # Synchroniser entre les deux phases d'évaluation
        jax.experimental.multihost_utils.sync_global_devices("between_eval_rounds")
        
        # Inverser les rôles pour l'équité
        logger.info("Parties d'évaluation (modèle actuel en tant que Blanc)...")
        new_rng_key = jax.random.fold_in(rng_key, 1000)
        sharded_rngs = jax.random.split(new_rng_key, self.num_devices)
        
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