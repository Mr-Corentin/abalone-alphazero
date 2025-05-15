import time
import math
import jax
import jax.numpy as jnp
import numpy as np
import optax
import os
import pickle
from functools import partial
from tensorboardX import SummaryWriter
import datetime

# Local imports
from model.neural_net import AbaloneModel
from environment.env import AbaloneEnv
from training.replay_buffer import CPUReplayBuffer
from training.loss import train_step_pmap_impl
from mcts.search import generate_parallel_games_pmap, create_optimized_game_generator
from evaluation.evaluator import Evaluator
from utils.game_storage import convert_games_batch, GameLogger, LocalGameLogger



import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Process %(process)d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("alphazero.trainer")

class AbaloneTrainerSync:
    """
    Coordinateur d'entraînement AlphaZero pour le jeu Abalone avec approche synchronisée.
    Supporte les environnements multi-processus et multi-hôtes TPU/GPU.
    """

    def __init__(self,
            network,
            env,
            buffer_size=1_000_000,
            batch_size=128,
            value_weight=1.0,
            num_simulations=500,
            recency_bias=True,
            recency_temperature=0.8,
            initial_lr=0.2,
            momentum=0.9,
            lr_schedule=None,
            checkpoint_path="checkpoints/model",
            log_dir=None,
            gcs_bucket=None,
            save_games=True,
            games_buffer_size=64,
            games_flush_interval=200,
            use_gcs_buffer=False,
            gcs_buffer_dir='buffer',
            eval_games=5,
            verbose=True):

        """
        Initialise le coordinateur d'entraînement avec approche synchronisée par étapes.
        Utilise SGD avec momentum comme dans l'implémentation originale d'AlphaZero.

        Args:
            network: Modèle de réseau neuronal
            env: Environnement de jeu
            buffer_size: Taille du tampon de rejeu
            batch_size: Taille du batch d'entraînement
            value_weight: Poids de la perte de valeur
            num_simulations: Nombre de simulations MCTS par coup
            recency_bias: Si True, utilise un échantillonnage biaisé par récence
            recency_temperature: Température pour le biais de récence
            initial_lr: Taux d'apprentissage initial (0.2 comme dans AlphaZero)
            momentum: Momentum pour SGD (0.9 standard)
            lr_schedule: Liste de tuples (pourcentage_itération, taux_apprentissage) ou None
            checkpoint_path: Chemin pour sauvegarder les points de contrôle
            log_dir: Chemin pour les logs tensorboard
            gcs_bucket: Nom du bucket GCS pour stocker les parties (si None, stockage local)
            save_games: Si True, sauvegarde les parties jouées pour analyse future
            games_buffer_size: Nombre de parties à accumuler avant de sauvegarder
            games_flush_interval: Intervalle en secondes pour sauvegarder les parties
            use_gcs_buffer: Si True, utilise un buffer sur GCS
            gcs_buffer_dir: Dossier dans le bucket GCS pour le buffer
            eval_games: Nombre de parties à jouer pour l'évaluation
            verbose: Si True, affiche les logs détaillés
        """
        # Configuration des appareils
        self.global_devices = jax.devices()
        self.num_global_devices = len(self.global_devices)
        self.devices = jax.local_devices()
        self.num_devices = len(self.devices)
        self.process_id = jax.process_index()
        self.num_processes = jax.process_count()
        self.is_main_process = self.process_id == 0
        self.verbose = verbose and self.is_main_process

        # Identifier le type d'appareil
        if self.devices[0].platform == 'tpu':
            self.device_type = 'tpu'
        elif self.devices[0].platform == 'gpu':
            self.device_type = 'gpu'
        else:
            self.device_type = 'cpu'
            
        logger.info(f"Process {self.process_id+1}/{self.num_processes}: "
              f"Using {self.num_devices} local {self.device_type.upper()} cores from {self.num_global_devices} total")

        # Stockage des configurations
        self.network = network
        self.env = env
        self.batch_size = batch_size
        self.value_weight = value_weight
        self.checkpoint_path = checkpoint_path
        self.num_simulations = num_simulations
        self.recency_bias = recency_bias
        self.recency_temperature = recency_temperature
        self.save_games = save_games
        self.eval_games = eval_games

        # Configuration du taux d'apprentissage et de l'optimiseur
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.momentum = momentum

        # Planning AlphaZero par défaut si non spécifié
        if lr_schedule is None:
            self.lr_schedule = [
                (0.0, initial_lr),      # Début
                (0.3, initial_lr/10),   # Premier palier: 0.2 -> 0.02
                (0.6, initial_lr/100),  # Deuxième palier: 0.02 -> 0.002
                (0.85, initial_lr/1000) # Troisième palier: 0.002 -> 0.0002
            ]
        else:
            self.lr_schedule = lr_schedule
            
        if self.verbose:
            logger.info(f"Optimizer: SGD+momentum")
            logger.info(f"Initial learning rate: {self.initial_lr}, Momentum: {self.momentum}")
            logger.info(f"Learning rate schedule: {self.lr_schedule}")
            if self.recency_bias:
                logger.info(f"Using recency bias with temperature {self.recency_temperature}")
            else:
                logger.info("Using uniform sampling")

        # Initialisation du buffer
        if use_gcs_buffer and gcs_bucket:
            if self.verbose:
                logger.info(f"Utilisation d'un buffer global sur GCS: {gcs_bucket}/{gcs_buffer_dir}")
            from training.replay_buffer import GCSReplayBufferSync
            self.buffer = GCSReplayBufferSync(
                bucket_name=gcs_bucket,
                buffer_dir=gcs_buffer_dir,
                max_local_size=buffer_size // 10,
                max_buffer_size=20_000_000,
                buffer_cleanup_threshold=0.95,
                recency_enabled=recency_bias,
                recency_temperature=recency_temperature,
                cleanup_temperature=2.0,
                log_level='INFO' if self.verbose else 'WARNING'
            )
        else:
            if self.verbose:
                logger.info(f"Utilisation d'un buffer local de taille {buffer_size}")
            self.buffer = CPUReplayBuffer(buffer_size)

        # Initialisation de l'optimiseur SGD
        self.optimizer = optax.sgd(learning_rate=self.initial_lr, momentum=self.momentum)

        # Initialisation des paramètres et de l'état d'optimisation
        rng = jax.random.PRNGKey(42)
        sample_board = jnp.zeros((1, 9, 9), dtype=jnp.int8)
        sample_marbles = jnp.zeros((1, 2), dtype=jnp.int8)
        self.params = network.init(rng, sample_board, sample_marbles)
        self.opt_state = self.optimizer.init(self.params)

        # Statistiques
        self.iteration = 0
        self.total_games = 0
        self.total_positions = 0
        self.metrics_history = []

        # Configuration des logs Tensorboard
        self._setup_tensorboard(log_dir)

        # Initialisation du logger de jeu
        if self.save_games:
            self._setup_game_logger(gcs_bucket, games_buffer_size, games_flush_interval)

        # Configuration des fonctions JAX
        self._setup_jax_functions()
        
        # Configuration de l'évaluation
        self.eval_enabled = False

    def _setup_tensorboard(self, log_dir):
        """Configure le logging TensorBoard"""
        if log_dir is None:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.log_dir = os.path.join("logs", f"abalone_az_{current_time}")
        else:
            self.log_dir = log_dir

        self.writer = SummaryWriter(self.log_dir)
        
        if self.verbose:
            is_gcs = self.log_dir.startswith('gs://')
            if is_gcs:
                bucket_name = self.log_dir.split('/', 3)[2]
                log_path = '/'.join(self.log_dir.split('/')[3:])
                logger.info(f"TensorBoard logs: {self.log_dir}")
                logger.info(f"Pour visualiser: tensorboard --logdir=gs://{bucket_name}/{log_path}")
            else:
                abs_log_dir = os.path.abspath(self.log_dir)
                logger.info(f"TensorBoard logs: {abs_log_dir}")
                logger.info(f"Pour visualiser: tensorboard --logdir={abs_log_dir}")

    def _setup_game_logger(self, gcs_bucket, buffer_size, flush_interval):
        """Configure le logger de parties"""
        if gcs_bucket:
            if self.verbose:
                logger.info(f"Stockage des parties dans GCS: {gcs_bucket}")
            self.game_logger = GameLogger(
                bucket_name=gcs_bucket,
                process_id=self.process_id,
                buffer_size=buffer_size,
                flush_interval=flush_interval
            )
        else:
            games_dir = os.path.join(self.log_dir, "games")
            if self.verbose:
                logger.info(f"Stockage local des parties: {games_dir}")
            self.game_logger = LocalGameLogger(
                output_dir=games_dir,
                buffer_size=buffer_size,
                flush_interval=flush_interval
            )

    def enable_evaluation(self, enable=True):
        """Active ou désactive l'évaluation"""
        self.eval_enabled = enable
        if self.verbose:
            logger.info(f"Évaluation {'activée' if enable else 'désactivée'}")

    def _update_learning_rate(self, iteration_percentage):
        """
        Met à jour le taux d'apprentissage selon le planning défini

        Args:
            iteration_percentage: Pourcentage de progression de l'entraînement [0.0, 1.0]

        Returns:
            Nouveau taux d'apprentissage
        """
        # Trouver le taux d'apprentissage approprié pour le pourcentage d'itération actuel
        new_lr = self.initial_lr  # Valeur par défaut

        for threshold, lr in self.lr_schedule:
            if iteration_percentage >= threshold:
                new_lr = lr

        if new_lr != self.current_lr:
            if self.verbose:
                logger.info(f"Learning rate updated: {self.current_lr} -> {new_lr}")
            self.current_lr = new_lr

            # Créer un nouvel optimiseur avec écrêtage de gradient
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.sgd(learning_rate=self.current_lr, momentum=self.momentum)
            )

            # Réinitialiser l'état de l'optimiseur
            self.opt_state = self.optimizer.init(self.params)

            # Mettre à jour les fonctions JAX qui utilisent l'optimiseur
            self.optimizer_update_pmap = jax.pmap(
                lambda g, o, p: self.optimizer.update(g, o, p),
                axis_name='devices',
                devices=self.devices
            )

        return new_lr

    def _setup_jax_functions(self):
        """Configure les fonctions JAX pour la génération et l'entraînement."""
        # Utiliser notre générateur optimisé
        self.generate_games_pmap = create_optimized_game_generator(self.num_simulations)

        # Ajouter l'écrêtage de gradient à l'optimiseur
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Limiter la norme du gradient à 1.0
            optax.sgd(learning_rate=self.current_lr, momentum=self.momentum)
        )

        # Réinitialiser l'état de l'optimiseur
        self.opt_state = self.optimizer.init(self.params)

        # Configurer les fonctions de traitement parallèle
        self.train_step_pmap = jax.pmap(
            partial(train_step_pmap_impl, network=self.network, value_weight=self.value_weight),
            axis_name='devices',
            devices=self.devices
        )

        self.optimizer_update_pmap = jax.pmap(
            lambda g, o, p: self.optimizer.update(g, o, p),
            axis_name='devices',
            devices=self.devices
        )
        
        self.sum_across_devices = jax.pmap(
            lambda x: jax.lax.psum(x, axis_name='devices'),
            axis_name='devices',
            devices=self.devices
        )

    def train(self, num_iterations=100, games_per_iteration=64,
            training_steps_per_iteration=100, save_frequency=10):
        """
        Démarre l'entraînement avec approche synchronisée par étapes.
        L'évaluation est maintenant déclenchée par les checkpoints de référence.

        Args:
            num_iterations: Nombre total d'itérations
            games_per_iteration: Nombre de parties à générer par itération
            training_steps_per_iteration: Nombre d'étapes d'entraînement par itération
            save_frequency: Fréquence de sauvegarde régulière (en itérations)
        """
        # Initialiser le timer global
        start_time_global = time.time()

        # Initialiser la clé aléatoire spécifique au processus
        seed_base = 42
        process_specific_seed = seed_base + (self.process_id * 1000)
        rng_key = jax.random.PRNGKey(process_specific_seed)

        # Déterminer les itérations de référence pour tout l'entraînement
        from evaluation.models_evaluator import generate_evaluation_checkpoints
        self.reference_iterations = generate_evaluation_checkpoints(num_iterations)
        
        if self.verbose:
            logger.info(f"Itérations de référence planifiées: {self.reference_iterations}")
            logger.info(f"Évaluation {'activée' if self.eval_enabled else 'désactivée'}")

        try:
            for iteration in range(num_iterations):
                self.iteration = iteration

                # Mettre à jour le taux d'apprentissage selon le planning
                iteration_percentage = iteration / num_iterations
                self._update_learning_rate(iteration_percentage)
                
                if self.verbose:
                    logger.info(f"\n=== Itération {iteration+1}/{num_iterations} (LR: {self.current_lr}) ===")

                # 1. Phase de génération
                rng_key, gen_key = jax.random.split(rng_key)
                t_start = time.time()
                games_data = self._generate_games(gen_key, games_per_iteration)
                t_gen = time.time() - t_start

                if self.verbose:
                    logger.info(f"Génération: {games_per_iteration} parties en {t_gen:.2f}s ({games_per_iteration/t_gen:.1f} parties/s)")

                # 2. Mise à jour du buffer
                jax.experimental.multihost_utils.sync_global_devices("post_generation")
                t_start = time.time()
                positions_added = self._update_buffer(games_data)
                t_buffer = time.time() - t_start
                self.total_positions += positions_added

                # Afficher les infos du buffer
                if self.verbose:
                    if hasattr(self.buffer, 'gcs_index'):
                        # Buffer GCS
                        logger.info(f"Buffer mis à jour: +{positions_added} positions")
                        logger.info(f"  - Cache local: {self.buffer.local_size} positions")
                        logger.info(f"  - Total estimé: {self.buffer.total_size} positions")
                    else:
                        # Buffer local
                        logger.info(f"Buffer mis à jour: +{positions_added} positions (total: {self.buffer.size})")

                # 3. Phase d'entraînement
                rng_key, train_key = jax.random.split(rng_key)
                t_start = time.time()
                metrics = self._train_network(train_key, training_steps_per_iteration)
                t_train = time.time() - t_start
                
                if self.verbose:
                    logger.info(f"Entraînement: {training_steps_per_iteration} étapes en {t_train:.2f}s ({training_steps_per_iteration/t_train:.1f} étapes/s)")
                    logger.info(f"  Perte totale: {metrics['total_loss']:.4f}")
                    logger.info(f"  Perte politique: {metrics['policy_loss']:.4f}, Perte valeur: {metrics['value_loss']:.4f}")
                    logger.info(f"  Précision politique: {metrics['policy_accuracy']}%")

                # 4. Gestion des checkpoints de référence
                if iteration in self.reference_iterations and self.is_main_process:
                    if self.verbose:
                        logger.info(f"\nItération {iteration}: Checkpoint de référence détecté")
                    
                    # Sauvegarder le modèle de référence
                    self._save_checkpoint(is_reference=True)
                    
                    # Évaluer si activé et pas à l'itération 0
                    if self.eval_enabled and iteration > 0:
                        eval_start = time.time()
                        if self.verbose:
                            logger.info("Évaluation déclenchée par nouveau modèle de référence...")
                        self.evaluate_against_previous_models(num_iterations)
                        eval_time = time.time() - eval_start
                        if self.verbose:
                            logger.info(f"Évaluation terminée en {eval_time:.2f}s")
                
                # 5. Sauvegarde périodique standard (non-référence)
                elif save_frequency > 0 and (iteration + 1) % save_frequency == 0 and self.is_main_process:
                    # Sauvegarde normale (pas un checkpoint de référence)
                    if self.verbose:
                        logger.info("\nSauvegarde périodique du checkpoint...")
                    self._save_checkpoint(is_reference=False)

            # Sauvegarde finale si pas déjà fait
            if self.is_main_process:
                final_is_reference = (num_iterations - 1) in self.reference_iterations
                self._save_checkpoint(is_final=True, is_reference=final_is_reference)
                
                if self.metrics_history:
                    final_metrics = self.metrics_history[-1]
                    
                    # Métriques d'entraînement
                    training_metrics = {k: v for k, v in final_metrics.items() 
                                    if k in ['total_loss', 'policy_loss', 'value_loss', 
                                            'policy_accuracy', 'value_sign_match']}
                    self._log_metrics_to_tensorboard(training_metrics, "training")
                    
                    # Taux d'apprentissage
                    self._log_metrics_to_tensorboard({"learning_rate": self.current_lr}, "training")
                    
                    # Statistiques du buffer et des parties
                    buffer_stats = {}
                    if hasattr(self.buffer, 'gcs_index'):
                        buffer_stats["buffer_size_total"] = self.buffer.total_size
                        buffer_stats["buffer_size_local"] = self.buffer.local_size
                    else:
                        buffer_stats["buffer_size"] = self.buffer.size
                    
                    buffer_stats["total_games_local"] = self.total_games
                    buffer_stats["total_games_global"] = self.total_games * self.num_processes
                    self._log_metrics_to_tensorboard(buffer_stats, "stats")
                    
                    eval_metrics = {k: v for k, v in final_metrics.items() 
                                if k.startswith('win_rate_vs_iter_') or k == 'avg_win_rate_vs_prev'}
                    if eval_metrics:
                        self._log_metrics_to_tensorboard(eval_metrics, "eval_vs_prev")

        finally:
            jax.experimental.multihost_utils.sync_global_devices("pre_close_resources")
            self.writer.close()

            if self.save_games and hasattr(self, 'game_logger'):
                self.game_logger.stop()

            if hasattr(self.buffer, 'close'):
                if self.verbose and self.is_main_process:
                    logger.info("Fermeture du buffer GCS...")
                self.buffer.close()

            total_time = time.time() - start_time_global
            if self.verbose:
                logger.info(f"\n=== Entraînement terminé ===")
                logger.info(f"Parties générées: {self.total_games}")
                logger.info(f"Positions totales: {self.total_positions}")
                logger.info(f"Durée totale: {total_time:.1f}s ({num_iterations/total_time:.2f} itérations/s)")

    def _generate_games(self, rng_key, num_games):
        """
        Génère des parties en parallèle sur les cœurs TPU locaux.

        Args:
            rng_key: Clé aléatoire JAX
            num_games: Nombre de parties à générer

        Returns:
            Données des parties générées
        """
        # Calculer le nombre de jeux par processus et par dispositif
        games_per_process = math.ceil(num_games / self.num_processes)
        batch_size_per_device = math.ceil(games_per_process / self.num_devices)

        # Nombre total de jeux générés par ce processus
        local_total_games = batch_size_per_device * self.num_devices

        # Préparation et distribution des clés aléatoires
        sharded_rngs = jax.random.split(rng_key, self.num_devices)
        sharded_rngs = jax.device_put_sharded(list(sharded_rngs), self.devices)

        # Répliquer les paramètres sur les dispositifs locaux
        sharded_params = jax.device_put_replicated(self.params, self.devices)

        # Génération des parties avec version optimisée
        games_data_pmap = self.generate_games_pmap(
            sharded_rngs,
            sharded_params,
            self.network,
            self.env,
            batch_size_per_device
        )

        # Récupérer les données sur CPU
        games_data = jax.device_get(games_data_pmap)
        

        # Mettre à jour le compteur avec les jeux locaux
        self.total_games += local_total_games

        # Enregistrer les parties pour analyse si activé
        if self.save_games and hasattr(self, 'game_logger'):
            # Générer un préfixe incluant l'ID de processus pour éviter les conflits
            game_id_prefix = f"iter{self.iteration}_p{self.process_id}"

            # Convertir les parties dans un format adapté à l'analyse
            converted_games = convert_games_batch(
                games_data,
                self.env,
                base_game_id=game_id_prefix,
                model_iteration=self.iteration
            )

            # Envoyer les parties au logger qui les écrira de manière asynchrone
            self.game_logger.log_games_batch(converted_games)

        return games_data
    def _log_metrics_to_tensorboard(self, metrics_dict, prefix="training"):
        """
        Centralise l'écriture des métriques dans TensorBoard
        
        Args:
            metrics_dict: Dictionnaire de métriques à enregistrer
            prefix: Préfixe pour organiser les métriques (ex: "training", "eval", "stats")
        """
        if not self.is_main_process:
            return
            
        for name, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"{prefix}/{name}", value, self.iteration)

    def _update_buffer(self, games_data):
        """
        Met à jour le buffer d'expérience avec les nouvelles parties générées.
        
        Args:
            games_data: Données des parties générées
            
        Returns:
            int: Nombre de positions ajoutées au buffer
        """
        positions_added = 0
        
        # Identifier si nous utilisons un buffer GCS
        using_gcs_buffer = hasattr(self.buffer, 'flush_to_gcs')
        
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
                boards_2d = device_data['boards_2d'][game_idx][:game_length+1]
                policies = device_data['policies'][game_idx][:game_length+1]
                actual_players = device_data['actual_players'][game_idx][:game_length+1]
                black_outs = device_data['black_outs'][game_idx][:game_length+1]
                white_outs = device_data['white_outs'][game_idx][:game_length+1]
                
                # Déterminer le résultat final
                final_black_out = device_data['final_black_out'][game_idx]
                final_white_out = device_data['final_white_out'][game_idx]
                
                if final_black_out >= 6:
                    outcome = -1  # White wins
                elif final_white_out >= 6:
                    outcome = 1   # Black wins
                else:
                    outcome = 0   # Draw
                
                # Générer un identifiant unique pour cette partie
                if using_gcs_buffer:
                    # Pour GCS, commencer une nouvelle partie
                    game_id = self.buffer.start_new_game()
                else:
                    # Pour le buffer local, utiliser un nombre séquentiel
                    game_id = self.total_games + game_idx
                    
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
                    
                    # Ajuster pour la perspective du joueur courant
                    outcome_for_player = outcome * player
                    
                    # Stocker dans le buffer avec métadonnées
                    self.buffer.add(
                        boards_2d[move_idx],
                        marbles_out,
                        policies[move_idx],
                        outcome_for_player,
                        player,
                        game_id=game_id,
                        move_num=move_idx,
                        iteration=self.iteration,
                        model_version=self.total_games
                    )
                    
                    positions_added += 1
        
        # Si buffer GCS, effectuer le flush
        if using_gcs_buffer:
            positions_flushed = self.buffer.flush_to_gcs()
            if self.verbose:
                stats = self.buffer.get_stats()
                fill_percentage = stats["fill_percentage"]
                logger.info(f"  {positions_flushed} positions écrites sur GCS, buffer à {fill_percentage:.1f}% de capacité")
                
                if stats.get("cleanup_operations", 0) > 0:
                    logger.info(f"  Nettoyage effectué: {stats['files_removed']} fichiers supprimés")
        
        return positions_added

    def _train_network(self, rng_key, num_steps):
        """
        Entraîne le réseau sur des batchs depuis le buffer.
        Compatible avec les deux types de buffer (local et GCS).

        Args:
            rng_key: Clé aléatoire JAX
            num_steps: Nombre d'étapes d'entraînement

        Returns:
            Métriques moyennes sur toutes les étapes
        """
        buffer_empty = ((hasattr(self.buffer, 'local_size') and self.buffer.local_size == 0) and 
                        (not hasattr(self.buffer, 'total_size') or self.buffer.total_size == 0)) or \
                    (hasattr(self.buffer, 'size') and self.buffer.size == 0)
        
        if buffer_empty:
            if self.verbose:
                logger.info("Buffer vide, impossible d'entraîner le réseau.")
            return {'total_loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0,
                    'policy_accuracy': 0.0, 'value_sign_match': 0.0}

        # Identifier si nous utilisons un buffer GCS
        using_gcs_buffer = hasattr(self.buffer, 'gcs_index')

        # Réplication des paramètres et état d'optimisation sur les dispositifs
        params_sharded = jax.device_put_replicated(self.params, self.devices)
        opt_state_sharded = jax.device_put_replicated(self.opt_state, self.devices)

        # Cumul des métriques pour ce processus
        cumulative_metrics = None
        steps_completed = 0

        for step in range(num_steps):
            total_batch_size = self.batch_size * self.num_devices

            # Échantillonnage depuis le buffer
            try:
                if using_gcs_buffer:
                    batch_data = self.buffer.sample(total_batch_size, rng_key=rng_key)
                else:
                    if self.recency_bias:
                        batch_data = self.buffer.sample_with_recency_bias(
                            total_batch_size,
                            temperature=self.recency_temperature,
                            rng_key=rng_key
                        )
                    else:
                        batch_data = self.buffer.sample(total_batch_size, rng_key=rng_key)
            except ValueError as e:
                if self.verbose:
                    logger.warning(f"Erreur lors de l'échantillonnage: {e}")
                if steps_completed == 0:
                    continue
                else:
                    break

            rng_key = jax.random.fold_in(rng_key, step)

            # Préparation des données
            boards = jnp.array(batch_data['board'])
            marbles = jnp.array(batch_data['marbles_out'])
            policies = jnp.array(batch_data['policy'])
            values = jnp.array(batch_data['outcome'])

            # Reshaping pour distribution sur les dispositifs
            boards = boards.reshape(self.num_devices, -1, *boards.shape[1:])
            marbles = marbles.reshape(self.num_devices, -1, *marbles.shape[1:])
            policies = policies.reshape(self.num_devices, -1, *policies.shape[1:])
            values = values.reshape(self.num_devices, -1, *values.shape[1:])

            # Exécution de l'étape d'entraînement
            metrics_sharded, grads_averaged = self.train_step_pmap(
                params_sharded, (boards, marbles), policies, values
            )

            # Application des mises à jour
            updates, opt_state_sharded = self.optimizer_update_pmap(
                grads_averaged, opt_state_sharded, params_sharded
            )
            params_sharded = jax.tree.map(lambda p, u: p + u, params_sharded, updates)

            # Agréger les métriques localement pour cette étape
            step_metrics = {k: float(jnp.mean(v)) for k, v in metrics_sharded.items()}

            if cumulative_metrics is None:
                cumulative_metrics = step_metrics
            else:
                cumulative_metrics = {k: cumulative_metrics[k] + step_metrics[k] for k in step_metrics}
                
            steps_completed += 1

        # Récupérer les paramètres mis à jour
        self.params = jax.tree.map(lambda x: x[0], params_sharded)
        self.opt_state = jax.tree.map(lambda x: x[0], opt_state_sharded)

        # Si aucune étape d'entraînement n'a été effectuée
        if cumulative_metrics is None or steps_completed == 0:
            return {'total_loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0,
                    'policy_accuracy': 0.0, 'value_sign_match': 0.0}

        # Calculer les métriques moyennes
        avg_metrics = {k: v / steps_completed for k, v in cumulative_metrics.items()}

        # Enregistrer dans TensorBoard si processus principal
        if self.is_main_process:
            # Utiliser la fonction centralisée au lieu des appels directs
            self._log_metrics_to_tensorboard(avg_metrics, "training")
            self._log_metrics_to_tensorboard({"learning_rate": self.current_lr}, "training")
            
            # Statistiques du buffer et des parties
            buffer_stats = {}
            if using_gcs_buffer:
                buffer_stats["buffer_size_total"] = self.buffer.total_size
                buffer_stats["buffer_size_local"] = self.buffer.local_size
            else:
                buffer_stats["buffer_size"] = self.buffer.size
            
            buffer_stats["total_games_local"] = self.total_games
            buffer_stats["total_games_global"] = self.total_games * self.num_processes
            self._log_metrics_to_tensorboard(buffer_stats, "stats")

        # Enregistrer l'historique des métriques
        local_metrics_record = avg_metrics.copy()
        local_metrics_record['iteration'] = self.iteration
        local_metrics_record['learning_rate'] = self.current_lr
        
        # Ajouter les bonnes métriques de taille du buffer
        if using_gcs_buffer:
            local_metrics_record['buffer_size_total'] = self.buffer.total_size
            local_metrics_record['buffer_size_local'] = self.buffer.local_size
        else:
            local_metrics_record['buffer_size'] = self.buffer.size
            
        local_metrics_record['total_games_local'] = self.total_games
        self.metrics_history.append(local_metrics_record)

        return avg_metrics
    
    def _save_checkpoint(self, is_final=False, is_reference=False):
        """
        Sauvegarde un point de contrôle du modèle

        Args:
            is_final: Si True, indique que c'est le point de contrôle final
            is_reference: Si True, indique que c'est un checkpoint de référence pour l'évaluation
        """
        if not self.is_main_process:
            return
            
        if is_final:
            prefix = "final"
        elif is_reference:
            prefix = f"ref_iter{self.iteration}"
        else:
            prefix = f"iter{self.iteration}"

        checkpoint = {
            'params': self.params,
            'opt_state': self.opt_state,
            'iteration': self.iteration,
            'current_lr': self.current_lr,
            'metrics': self.metrics_history,
            'total_games': self.total_games,
            'total_positions': self.total_positions,
            'is_reference': is_reference
        }

        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)

        # Vérifier si c'est un chemin GCS
        is_gcs = self.checkpoint_path.startswith('gs://')

        if is_gcs:
            # Sauvegarder sur GCS
            import subprocess

            # Sauvegarder localement d'abord
            local_path = f"/tmp/{prefix}.pkl"
            with open(local_path, 'wb') as f:
                pickle.dump(checkpoint, f)

            # Puis envoyer vers GCS
            gcs_path = f"{self.checkpoint_path}_{prefix}.pkl"
            subprocess.run(f"gsutil cp {local_path} {gcs_path}", shell=True)
            
            if self.verbose:
                checkpoint_type = "de référence" if is_reference else "standard"
                logger.info(f"Checkpoint {checkpoint_type} sauvegardé: {gcs_path}")

            # Supprimer le fichier local
            os.remove(local_path)
        else:
            # Sauvegarder localement
            filename = f"{self.checkpoint_path}_{prefix}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(checkpoint, f)
                
            if self.verbose:
                checkpoint_type = "de référence" if is_reference else "standard"
                logger.info(f"Checkpoint {checkpoint_type} sauvegardé: {filename}")

    def load_checkpoint(self, checkpoint_path):
        """
        Charge un point de contrôle précédemment sauvegardé

        Args:
            checkpoint_path: Chemin vers le fichier de point de contrôle
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
        
        if self.verbose:
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            logger.info(f"Iteration: {self.iteration}, Positions: {self.total_positions}")

            
    def evaluate_against_previous_models(self, total_iterations, num_reference_models=8):
        """
        Évalue le modèle actuel contre des versions précédentes.

        Args:
            total_iterations: Nombre total d'itérations prévues pour l'entraînement
            num_reference_models: Nombre approximatif de modèles de référence à utiliser

        Returns:
            Dictionary contenant les résultats d'évaluation
        """
        from evaluation.models_evaluator import (
            generate_evaluation_checkpoints,
            check_checkpoint_exists,
            download_checkpoint,
            load_checkpoint_params,
            ModelsEvaluator
        )

        current_iter = self.iteration

        # Générer les itérations de référence
        target_references = generate_evaluation_checkpoints(total_iterations, num_reference_models)

        # Filtrer pour ne garder que celles qui sont disponibles et antérieures à l'itération actuelle
        available_refs = []
        for ref in target_references:
            if ref < current_iter:
                # Vérifier si le checkpoint existe
                ref_path = self._get_checkpoint_path(ref)
                if check_checkpoint_exists(ref_path):
                    available_refs.append(ref)

        if not available_refs:
            if self.verbose:
                logger.info("Aucun modèle précédent disponible pour l'évaluation")
            return {}

        games_per_model = 16 
        local_games_per_model = games_per_model // self.num_processes
        
        if self.verbose:
            logger.info(f"\n=== Évaluation contre modèles précédents (itération actuelle: {current_iter}) ===")
            logger.info(f"Itérations sélectionnées: {available_refs}")
        logger.info(f"Processus {self.process_id}: jouera {local_games_per_model} parties par modèle")

        evaluator = ModelsEvaluator(
            network=self.network,
            num_simulations=500,  
            games_per_model=games_per_model
        )

        current_params = self.params
        local_results = {}

        for ref_iter in available_refs:
            if self.verbose:
                logger.info(f"\nÉvaluation contre le modèle de l'itération {ref_iter}...")

            ref_path = self._get_checkpoint_path(ref_iter)

            local_path = f"/tmp/ref_model_{ref_iter}.pkl"
            if ref_path.startswith("gs://"):
                if not download_checkpoint(ref_path, local_path):
                    if self.verbose:
                        logger.info(f"Échec du téléchargement du checkpoint pour l'itération {ref_iter}, on passe")
                    continue
            else:
                local_path = ref_path

            ref_params = load_checkpoint_params(local_path)
            if ref_params is None:
                if self.verbose:
                    logger.info(f"Échec du chargement des paramètres pour l'itération {ref_iter}, on passe")
                continue

            eval_results = evaluator.evaluate_model_pair(
                current_params,
                ref_params,
                games_to_play=local_games_per_model
            )

            local_results[ref_iter] = eval_results

        # Agréger les résultats de tous les processus
        all_results = self._aggregate_evaluation_results(local_results, available_refs)

        # Afficher et enregistrer les résultats
        if self.verbose:
            for ref_iter, ref_results in all_results.items():
                win_rate = ref_results['win_rate']
                logger.info(f"Résultats vs iter {ref_iter}: {win_rate:.1%} taux de victoire")
                logger.info(f"  Victoires: {ref_results['current_wins']}, Défaites: {ref_results['reference_wins']}, Nuls: {ref_results['draws']}")

                # Utiliser la fonction centralisée
                self._log_metrics_to_tensorboard({
                    f"win_rate_iter_{ref_iter}": win_rate,
                    f"games_iter_{ref_iter}": ref_results['total_games']
                }, "eval_vs_prev")

            if all_results:
                # Calculer le taux de victoire moyen sur tous les modèles de référence
                avg_win_rate = sum(res['win_rate'] for res in all_results.values()) / len(all_results)
                self._log_metrics_to_tensorboard({"avg_win_rate": avg_win_rate}, "eval_vs_prev")

                # Ajouter un résumé à l'historique des métriques
                if self.metrics_history and self.iteration > 0:
                    latest_metrics = self.metrics_history[-1]
                    latest_metrics['avg_win_rate_vs_prev'] = avg_win_rate

                    # Stocker les taux de victoire individuels
                    for ref_iter, ref_results in all_results.items():
                        latest_metrics[f'win_rate_vs_iter_{ref_iter}'] = ref_results['win_rate']
                        
            logger.info(f"\n=== Évaluation terminée ===")
            
        return all_results

    def _aggregate_evaluation_results(self, local_results, model_iterations):
        """
        Agrège les résultats d'évaluation de tous les processus.

        Args:
            local_results: Résultats locaux de ce processus
            model_iterations: Liste des itérations de modèle évaluées

        Returns:
            Résultats agrégés de tous les processus
        """
        # Créer des tableaux pour chaque métrique et chaque modèle
        aggregated_data = {}

        for ref_iter in model_iterations:
            # Initialiser avec les résultats locaux ou des zéros
            if ref_iter in local_results:
                ref_data = local_results[ref_iter]
                aggregated_data[ref_iter] = jnp.array([
                    ref_data['total_games'],
                    ref_data['current_wins'],
                    ref_data['reference_wins'],
                    ref_data['draws']
                ])
            else:
                aggregated_data[ref_iter] = jnp.zeros(4, dtype=jnp.int32)

        # Convertir en un seul grand tableau pour faciliter la communication collective
        all_models_data = jnp.stack([aggregated_data[ref] for ref in model_iterations])

        # Répliquer les données sur tous les dispositifs locaux
        replicated_data = jnp.repeat(all_models_data[None, :, :], self.num_devices, axis=0)
        devices_data = jax.device_put_sharded(list(replicated_data), self.devices)

        # Exécuter la somme collective
        summed_data = self.sum_across_devices(devices_data)

        # Récupérer les résultats agrégés (prendre juste le premier dispositif)
        global_results = jax.device_get(summed_data)[0]

        # Reconstruire le format de résultat original
        final_results = {}
        for i, ref_iter in enumerate(model_iterations):
            total_games = int(global_results[i][0])
            if total_games > 0:  # Si ce modèle a été évalué
                final_results[ref_iter] = {
                    'total_games': total_games,
                    'current_wins': int(global_results[i][1]),
                    'reference_wins': int(global_results[i][2]),
                    'draws': int(global_results[i][3]),
                    'win_rate': float(global_results[i][1] / total_games)
                }

        return final_results



    def _get_checkpoint_path(self, iteration):
        """Obtient le chemin vers un checkpoint pour l'itération donnée."""
        prefix = f"ref_iter{iteration}"
        
        if self.checkpoint_path.startswith("gs://"):
            base_path = self.checkpoint_path
            if '_' in base_path:
                base_path = base_path.rsplit('_', 1)[0]
            

            return f"{base_path}_*_{prefix}.pkl"
        else:
            base_path = self.checkpoint_path
            if '_' in base_path:
                base_path = base_path.rsplit('_', 1)[0]
            return f"{base_path}_*_{prefix}.pkl"