import time
import math
import jax
import jax.numpy as jnp
import numpy as np
import optax
import os
import pickle
from functools import partial

# Importations locales
from model.neural_net import AbaloneModel
from environment.env import AbaloneEnv
from training.replay_buffer import CPUReplayBuffer
from training.loss import train_step_pmap_impl
from mcts.search import generate_parallel_games_pmap
from evaluation.evaluator import evaluate_model



def create_game_generator(num_simulations):
    """
    Crée une version personnalisée de generate_parallel_games_pmap avec num_simulations fixé
    
    Args:
        num_simulations: Nombre de simulations MCTS par mouvement
        
    Returns:
        Fonction pmappée pour générer des parties
    """
    from mcts.search import generate_game_mcts_batch
    
    @partial(jax.pmap, axis_name='device', static_broadcasted_argnums=(2, 3, 4))
    def custom_generate_parallel_games(rngs, params, network, env, batch_size_per_device):
        return generate_game_mcts_batch(rngs, params, network, env, 
                                       batch_size_per_device, 
                                       num_simulations=num_simulations)
    
    return custom_generate_parallel_games


class AbaloneTrainerSync:
    def __init__(self,
                network,
                env,
                buffer_size=1_000_000,
                batch_size=128,
                value_weight=1.0,
                games_per_device=8,
                num_simulations=500,
                recency_bias=True,
                recency_temperature=0.8,
                initial_lr=0.2,
                momentum=0.9,
                lr_schedule=None,
                checkpoint_path="checkpoints/model"):
        """
        Initialise le coordinateur d'entraînement avec approche synchronisée par étapes.
        Utilise SGD avec momentum comme dans l'implémentation originale d'AlphaZero.
        
        Args:
            network: Le réseau de neurones
            env: L'environnement de jeu
            buffer_size: Taille du buffer de replay
            batch_size: Taille du batch pour l'entraînement
            value_weight: Poids de la perte de valeur
            games_per_device: Nombre de parties par dispositif TPU
            num_simulations: Nombre de simulations MCTS par mouvement
            recency_bias: Si True, utilise l'échantillonnage biaisé vers les données récentes
            recency_temperature: Température pour le biais de récence
            initial_lr: Learning rate initial (0.2 comme dans AlphaZero)
            momentum: Momentum pour SGD (0.9 standard)
            lr_schedule: Liste de tuples (pourcentage_iteration, learning_rate) ou None
            checkpoint_path: Chemin pour sauvegarder les checkpoints
        """
        # Configuration TPU
        #self.devices = jax.devices('tpu')
        # if jax.device_count('tpu') > 0:
        #     self.devices = jax.devices('tpu')
        # elif jax.device_count('gpu') > 0:
        #     self.devices = jax.devices('gpu')
        # else:
        #     self.devices = jax.devices('cpu')
        # self.num_devices = len(self.devices)
        # print(f"Utilisation de {self.num_devices} cœurs TPU en mode synchronisé par étapes")

        try:
            # Essayer d'abord les TPU (Google Cloud)
            self.devices = jax.devices('tpu')
            self.device_type = 'tpu'
        except RuntimeError:
            try:
                # Ensuite les GPU
                self.devices = jax.devices('gpu')
                self.device_type = 'gpu'
            except RuntimeError:
                # Enfin, utiliser les CPU
                self.devices = jax.devices('cpu')
                self.device_type = 'cpu'
        
        self.num_devices = len(self.devices)
        print(f"Utilisation de {self.num_devices} cœurs {self.device_type.upper()} en mode synchronisé par étapes")
            
        # Stocker les configurations
        self.network = network
        self.env = env
        self.batch_size = batch_size
        self.value_weight = value_weight
        self.checkpoint_path = checkpoint_path
        self.games_per_device = games_per_device
        self.num_simulations = num_simulations
        self.recency_bias = recency_bias
        self.recency_temperature = recency_temperature
        
        # Configuration du learning rate et de l'optimiseur
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.momentum = momentum
        
        # Schedule par défaut d'AlphaZero si non spécifié
        if lr_schedule is None:
            self.lr_schedule = [
                (0.0, initial_lr),      # Départ
                (0.3, initial_lr/10),   # Première chute: 0.2 -> 0.02
                (0.6, initial_lr/100),  # Deuxième chute: 0.02 -> 0.002
                (0.85, initial_lr/1000) # Troisième chute: 0.002 -> 0.0002
            ]
        else:
            self.lr_schedule = lr_schedule
            
        print(f"Optimiseur: SGD+momentum")
        print(f"Learning rate initial: {self.initial_lr}, Momentum: {self.momentum}")
        print(f"Schedule du learning rate: {self.lr_schedule}")
        
        # Afficher la configuration d'échantillonnage
        if self.recency_bias:
            print(f"Utilisation du biais de récence avec température {self.recency_temperature}")
        else:
            print("Utilisation de l'échantillonnage uniforme")
        
        # Initialiser l'optimiseur SGD
        self.optimizer = optax.sgd(learning_rate=self.initial_lr, momentum=self.momentum)
        
        # Initialiser les paramètres et l'état d'optimisation
        rng = jax.random.PRNGKey(42)
        sample_board = jnp.zeros((1, 9, 9), dtype=jnp.int8)
        sample_marbles = jnp.zeros((1, 2), dtype=jnp.int8)
        self.params = network.init(rng, sample_board, sample_marbles)
        self.opt_state = self.optimizer.init(self.params)
        
        # Créer le buffer de replay
        self.buffer = CPUReplayBuffer(buffer_size)
        
        # Statistiques
        self.iteration = 0
        self.total_games = 0
        self.total_positions = 0
        self.metrics_history = []
        
        # Configurer les fonctions JAX
        self._setup_jax_functions()
        
    def _update_learning_rate(self, iteration_percentage):
        """
        Met à jour le learning rate selon le schedule défini
        
        Args:
            iteration_percentage: Pourcentage d'avancement dans l'entraînement [0.0, 1.0]
            
        Returns:
            Nouveau learning rate
        """
        # Trouver le learning rate approprié pour le pourcentage d'itération actuel
        new_lr = self.initial_lr  # Valeur par défaut
        
        for threshold, lr in self.lr_schedule:
            if iteration_percentage >= threshold:
                new_lr = lr
        
        # Si le LR a changé, recréer l'optimiseur
        if new_lr != self.current_lr:
            print(f"Learning rate mis à jour: {self.current_lr} -> {new_lr}")
            self.current_lr = new_lr
            
            # Créer un nouvel optimiseur avec le nouveau learning rate
            self.optimizer = optax.sgd(learning_rate=self.current_lr, momentum=self.momentum)
            
            # Réinitialiser l'état de l'optimiseur
            self.opt_state = self.optimizer.init(self.params)
            
            # Mettre à jour les fonctions JAX qui utilisent l'optimiseur
            self.optimizer_update_pmap = jax.pmap(
                lambda g, o, p: self.optimizer.update(g, o, p),
                axis_name='batch',
                devices=self.devices
            )
        
        return new_lr
        
    def _setup_jax_functions(self):
        """Configure les fonctions JAX pour la génération et l'entraînement."""
        # Utiliser la fonction factory pour créer une version avec num_simulations fixé
        self.generate_games_pmap = create_game_generator(self.num_simulations)
      
        # Fonction d'entraînement parallèle (utilise tous les cœurs TPU)
        self.train_step_pmap = jax.pmap(
            partial(train_step_pmap_impl, network=self.network, value_weight=self.value_weight),
            axis_name='batch',
            devices=self.devices
        )
      
        # Fonction de mise à jour des paramètres avec l'optimiseur
        self.optimizer_update_pmap = jax.pmap(
            lambda g, o, p: self.optimizer.update(g, o, p),
            axis_name='batch',
            devices=self.devices
        )
      
    def train(self, num_iterations=100, games_per_iteration=64, 
             training_steps_per_iteration=100, eval_frequency=10, 
             save_frequency=10):
        """
        Lance l'entraînement avec approche par étapes.
        
        Args:
            num_iterations: Nombre total d'itérations
            games_per_iteration: Nombre de parties à générer par itération
            training_steps_per_iteration: Nombre d'étapes d'entraînement par itération
            eval_frequency: Fréquence d'évaluation (en itérations)
            save_frequency: Fréquence de sauvegarde (en itérations)
        """
        # Initialiser le timer global
        start_time_global = time.time()
        
        # RNG principal
        rng_key = jax.random.PRNGKey(42)
        
        for iteration in range(num_iterations):
            self.iteration = iteration
            
            # Mettre à jour le learning rate selon le schedule
            iteration_percentage = iteration / num_iterations
            self._update_learning_rate(iteration_percentage)
            
            print(f"\n=== Itération {iteration+1}/{num_iterations} (LR: {self.current_lr}) ===")
            
            # 1. Phase de génération
            rng_key, gen_key = jax.random.split(rng_key)
            t_start = time.time()
            
            # Générer le nombre de parties demandé
            games_data = self._generate_games(gen_key, games_per_iteration)
            
            t_gen = time.time() - t_start
            print(f"Génération: {games_per_iteration} parties en {t_gen:.2f}s ({games_per_iteration/t_gen:.1f} parties/s)")
            
            # 2. Mise à jour du buffer
            t_start = time.time()
            positions_added = self._update_buffer(games_data)
            t_buffer = time.time() - t_start
            
            self.total_positions += positions_added
            print(f"Buffer mis à jour: +{positions_added} positions (total: {self.buffer.size})")
            
            # 3. Phase d'entraînement
            rng_key, train_key = jax.random.split(rng_key)
            t_start = time.time()
            
            metrics = self._train_network(train_key, training_steps_per_iteration)
            
            t_train = time.time() - t_start
            print(f"Entraînement: {training_steps_per_iteration} étapes en {t_train:.2f}s ({training_steps_per_iteration/t_train:.1f} étapes/s)")
            
            # Afficher les métriques
            print(f"  Loss totale: {metrics['total_loss']:.4f}")
            print(f"  Loss policy: {metrics['policy_loss']:.4f}, Loss value: {metrics['value_loss']:.4f}")
            print(f"  Précision policy: {metrics['policy_accuracy']:.2%}")
            
            # 4. Évaluation périodique
            if eval_frequency > 0 and (iteration + 1) % eval_frequency == 0:
                self._evaluate()
            
            # 5. Sauvegarde périodique
            if save_frequency > 0 and (iteration + 1) % save_frequency == 0:
                self._save_checkpoint()
        
        # Sauvegarde finale
        self._save_checkpoint(is_final=True)
        
        # Statistiques globales
        total_time = time.time() - start_time_global
        print(f"\n=== Entraînement terminé ===")
        print(f"Parties générées: {self.total_games}")
        print(f"Positions totales: {self.total_positions}")
        print(f"Durée totale: {total_time:.1f}s ({num_iterations/total_time:.2f} itérations/s)")
    
    def _generate_games(self, rng_key, num_games):
        """
        Génère des parties en parallèle sur tous les cœurs TPU.
        
        Args:
            rng_key: Clé aléatoire JAX
            num_games: Nombre de parties à générer
            
        Returns:
            Données des parties générées
        """
        # Déterminer combien de parties par cœur
        games_per_core = math.ceil(num_games / self.num_devices)
        total_games = games_per_core * self.num_devices
        
        # Préparer les RNGs pour chaque cœur
        sharded_rngs = jax.random.split(rng_key, self.num_devices)
        
        # Répliquer les paramètres pour pmap
        sharded_params = jax.tree_util.tree_map(
            lambda x: jnp.array([x] * self.num_devices),
            self.params
        )
        
        # Générer les parties
        games_data_pmap = self.generate_games_pmap(
            sharded_rngs, sharded_params, self.network, self.env, games_per_core
        )
        
        # Récupérer les données sur CPU
        games_data = jax.device_get(games_data_pmap)
        
        # Mettre à jour le compteur
        self.total_games += total_games
        
        return games_data
    
    def _update_buffer(self, games_data):
        """
        Met à jour le buffer de replay avec les nouvelles parties.
        
        Args:
            games_data: Données des parties générées
            
        Returns:
            Nombre de positions ajoutées au buffer
        """
        positions_added = 0
        
        # Pour chaque dispositif
        for device_idx in range(self.num_devices):
            device_data = jax.tree_util.tree_map(
                lambda x: x[device_idx],
                games_data
            )
            
            # Pour chaque partie générée sur ce dispositif
            games_per_device = len(device_data['moves_per_game'])
            for game_idx in range(games_per_device):
                game_length = int(device_data['moves_per_game'][game_idx])
                if game_length == 0:
                    continue
                    
                # Extraire les données pour cette partie
                boards_2d = device_data['boards_2d'][game_idx][:game_length+1]  # +1 pour inclure l'état terminal potentiel
                policies = device_data['policies'][game_idx][:game_length+1]
                actual_players = device_data['actual_players'][game_idx][:game_length+1]
                black_outs = device_data['black_outs'][game_idx][:game_length+1]
                white_outs = device_data['white_outs'][game_idx][:game_length+1]
                
                # Déterminer le résultat final
                final_black_out = device_data['final_black_out'][game_idx]
                final_white_out = device_data['final_white_out'][game_idx]
                
                if final_black_out >= 6:
                    outcome = -1  # Blancs gagnent
                elif final_white_out >= 6:
                    outcome = 1   # Noirs gagnent
                else:
                    outcome = 0   # Match nul
                    
                # Ajouter chaque position au buffer
                for move_idx in range(game_length):
                    # Calculer les billes sorties pour le joueur courant
                    player = actual_players[move_idx]
                    our_marbles = np.where(player == 1,
                                         black_outs[move_idx],
                                         white_outs[move_idx])
                    opp_marbles = np.where(player == 1,
                                         white_outs[move_idx],
                                         black_outs[move_idx])
                    marbles_out = np.array([our_marbles, opp_marbles], dtype=np.int8)
                    
                    # Ajuster pour le point de vue du joueur courant
                    outcome_for_player = outcome * player
                    
                    # Stocker dans le buffer
                    self.buffer.add(
                        boards_2d[move_idx],
                        marbles_out,
                        policies[move_idx],
                        outcome_for_player,
                        player,
                        game_id=self.total_games + game_idx,  # ID unique pour cette partie
                        move_num=move_idx
                    )
                    
                    positions_added += 1
                    
        return positions_added
    
    def _train_network(self, rng_key, num_steps):
        """
        Entraîne le réseau sur des batches du buffer.
        
        Args:
            rng_key: Clé aléatoire JAX
            num_steps: Nombre d'étapes d'entraînement
            
        Returns:
            Métriques moyennes sur toutes les étapes
        """
        # Métriques cumulatives
        cumulative_metrics = None
        
        for step in range(num_steps):
            # Échantillonner un grand batch pour paralléliser
            total_batch_size = self.batch_size * self.num_devices
            
            # Utiliser l'échantillonnage avec biais de récence si activé
            if self.recency_bias:
                batch_data = self.buffer.sample_with_recency_bias(
                    total_batch_size, 
                    temperature=self.recency_temperature, 
                    rng_key=rng_key
                )
            else:
                batch_data = self.buffer.sample(total_batch_size, rng_key)
                
            rng_key = jax.random.fold_in(rng_key, step)
            
            # Convertir en JAX arrays
            boards = jnp.array(batch_data['board'])
            marbles = jnp.array(batch_data['marbles_out'])
            policies = jnp.array(batch_data['policy'])
            values = jnp.array(batch_data['outcome'])
            
            # Diviser les données en chunks pour chaque cœur
            boards = boards.reshape(self.num_devices, -1, *boards.shape[1:])
            marbles = marbles.reshape(self.num_devices, -1, *marbles.shape[1:])
            policies = policies.reshape(self.num_devices, -1, *policies.shape[1:])
            values = values.reshape(self.num_devices, -1, *values.shape[1:])
            
            # Répliquer les paramètres pour pmap
            params_sharded = jax.device_put_replicated(self.params, self.devices)
            opt_state_sharded = jax.device_put_replicated(self.opt_state, self.devices)
            
            # Exécuter l'étape d'entraînement parallèle
            loss, grads = self.train_step_pmap(params_sharded, (boards, marbles), policies, values)
            
            # Mettre à jour les paramètres avec l'optimiseur
            updates, new_opt_state = self.optimizer_update_pmap(grads, opt_state_sharded, params_sharded)
            new_params = jax.tree_map(lambda p, u: p + u, params_sharded, updates)
            
            # Récupérer les résultats du premier dispositif
            self.params = jax.tree_map(lambda x: x[0], new_params)
            self.opt_state = jax.tree_map(lambda x: x[0], new_opt_state)
            
            # Agréger les métriques
            step_metrics = {k: float(jnp.mean(v)) for k, v in loss.items()}
            
            if cumulative_metrics is None:
                cumulative_metrics = step_metrics
            else:
                cumulative_metrics = {k: cumulative_metrics[k] + step_metrics[k] for k in step_metrics}
        
        # Calculer la moyenne des métriques
        avg_metrics = {k: v / num_steps for k, v in cumulative_metrics.items()}
        
        # Enregistrer les métriques
        avg_metrics['iteration'] = self.iteration
        avg_metrics['learning_rate'] = self.current_lr
        avg_metrics['buffer_size'] = self.buffer.size
        avg_metrics['total_games'] = self.total_games
        self.metrics_history.append(avg_metrics)
        
        return avg_metrics
    
    def _evaluate(self):
        """
        Évalue le modèle actuel contre plusieurs algorithmes classiques.
        """
        return evaluate_model(self)
    
    def _save_checkpoint(self, is_final=False):
        """
        Sauvegarde un checkpoint du modèle
        
        Args:
            is_final: Si True, indique que c'est le checkpoint final
        """
        prefix = "final" if is_final else f"iter{self.iteration}"
        
        checkpoint = {
            'params': self.params,
            'opt_state': self.opt_state,
            'iteration': self.iteration,
            'current_lr': self.current_lr,
            'metrics': self.metrics_history,
            'total_games': self.total_games,
            'total_positions': self.total_positions
        }
        
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        
        # Vérifier si c'est un chemin GCS
        is_gcs = self.checkpoint_path.startswith('gs://')
        
        if is_gcs:
            # Sauvegarder sur GCS
            import subprocess
            
            # Sauvegarder d'abord localement
            local_path = f"/tmp/{prefix}.pkl"
            with open(local_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            # Puis envoyer vers GCS
            gcs_path = f"{self.checkpoint_path}_{prefix}.pkl"
            subprocess.run(f"gsutil cp {local_path} {gcs_path}", shell=True)
            
            print(f"Checkpoint sauvegardé: {gcs_path}")
            
            # Supprimer le fichier local
            os.remove(local_path)
        else:
            # Sauvegarder localement
            filename = f"{self.checkpoint_path}_{prefix}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            print(f"Checkpoint sauvegardé: {filename}")
        
    def load_checkpoint(self, checkpoint_path):
        """
        Charge un checkpoint précédemment sauvegardé
        
        Args:
            checkpoint_path: Chemin vers le fichier de checkpoint
        """
        # Vérifier si c'est un chemin GCS
        is_gcs = checkpoint_path.startswith('gs://')
        
        if is_gcs:
            # Télécharger depuis GCS
            import subprocess
            
            local_path = "/tmp/checkpoint.pkl"
            subprocess.run(f"gsutil cp {checkpoint_path} {local_path}", shell=True)
            
            with open(local_path, 'rb') as f:
                checkpoint = pickle.load(f)
                
            # Supprimer le fichier local
            os.remove(local_path)
        else:
            # Charger localement
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
        
        # Restaurer l'état
        self.params = checkpoint['params']
        self.opt_state = checkpoint['opt_state']
        self.iteration = checkpoint['iteration']
        self.current_lr = checkpoint['current_lr']
        self.metrics_history = checkpoint['metrics']
        self.total_games = checkpoint['total_games']
        self.total_positions = checkpoint['total_positions']
        
        print(f"Checkpoint chargé: {checkpoint_path}")
        print(f"Itération: {self.iteration}, Positions: {self.total_positions}")