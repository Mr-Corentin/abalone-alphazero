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


class AbaloneTrainerSync:

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
            games_flush_interval=300,
            use_gcs_buffer=False, 
            gcs_buffer_dir='buffer',
            eval_games=5): 
    
        """
        Initialize the training coordinator with step-synchronized approach.
        Uses SGD with momentum as in the original AlphaZero implementation.
        
        Args:
            network: Neural network model
            env: Game environment
            buffer_size: Replay buffer size
            batch_size: Training batch size
            value_weight: Weight of the value loss
            num_simulations: Number of MCTS simulations per move
            recency_bias: If True, use recency-biased sampling
            recency_temperature: Temperature for recency bias
            initial_lr: Initial learning rate (0.2 as in AlphaZero)
            momentum: Momentum for SGD (0.9 standard)
            lr_schedule: List of tuples (iteration_percentage, learning_rate) or None
            checkpoint_path: Path to save checkpoints
            log_dir: Path for tensorboard logs
            gcs_bucket: GCS bucket name for storing games (if None, local storage)
            save_games: If True, save played games for future analysis
            games_buffer_size: Number of games to accumulate before saving
            games_flush_interval: Interval in seconds to save games
        """
        # Stocker les dispositifs globaux pour la configuration
        self.global_devices = jax.devices()
        self.num_global_devices = len(self.global_devices)
        
        # Mais utiliser uniquement les dispositifs locaux pour les opérations
        self.devices = jax.local_devices()
        self.num_devices = len(self.devices)
        
        self.process_id = jax.process_index()
        self.num_processes = jax.process_count()
        self.is_main_process = self.process_id == 0

        
        # Identifier le type d'appareil
        if self.devices[0].platform == 'tpu':
            self.device_type = 'tpu'
        elif self.devices[0].platform == 'gpu':
            self.device_type = 'gpu'
        else:
            self.device_type = 'cpu'
        
        print(f"Process {self.process_id+1}/{self.num_processes}: "
              f"Using {self.num_devices} local {self.device_type.upper()} cores from {self.num_global_devices} total")
            
        # Store configurations
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
        
        # Learning rate and optimizer configuration
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.momentum = momentum
        
        # Default AlphaZero schedule if not specified
        if lr_schedule is None:
            self.lr_schedule = [
                (0.0, initial_lr),      # Start
                (0.3, initial_lr/10),   # First drop: 0.2 -> 0.02
                (0.6, initial_lr/100),  # Second drop: 0.02 -> 0.002
                (0.85, initial_lr/1000) # Third drop: 0.002 -> 0.0002
            ]
        else:
            self.lr_schedule = lr_schedule
            
        print(f"Optimizer: SGD+momentum")
        print(f"Initial learning rate: {self.initial_lr}, Momentum: {self.momentum}")
        print(f"Learning rate schedule: {self.lr_schedule}")
        
        # Display sampling configuration
        if self.recency_bias:
            print(f"Using recency bias with temperature {self.recency_temperature}")
        else:
            print("Using uniform sampling")
        if use_gcs_buffer and gcs_bucket:
            print(f"Utilisation d'un buffer global sur GCS: {gcs_bucket}/{gcs_buffer_dir}")
            from training.replay_buffer import GCSReplayBuffer
            self.buffer = GCSReplayBuffer(
                bucket_name=gcs_bucket,
                buffer_dir=gcs_buffer_dir,
                max_local_size=buffer_size // 10,  # Cache local plus petit
                recency_enabled=recency_bias,
                recency_temperature=recency_temperature
            )
        else:
            print(f"Utilisation d'un buffer local de taille {buffer_size}")
            self.buffer = CPUReplayBuffer(buffer_size)
        
        # Initialize SGD optimizer
        self.optimizer = optax.sgd(learning_rate=self.initial_lr, momentum=self.momentum)
        
        # Initialize parameters and optimization state
        #rng = jax.random.PRNGKey(42)
        rng = jax.random.PRNGKey(42)

        sample_board = jnp.zeros((1, 9, 9), dtype=jnp.int8)
        sample_marbles = jnp.zeros((1, 2), dtype=jnp.int8)
        self.params = network.init(rng, sample_board, sample_marbles)
        self.opt_state = self.optimizer.init(self.params)
        
        
        # Statistics
        self.iteration = 0
        self.total_games = 0
        self.total_positions = 0
        self.metrics_history = []

        # Tensorboard logging
        if log_dir is None:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.log_dir = os.path.join("logs", f"abalone_az_{current_time}")
        else:
            self.log_dir = log_dir
            
        self.writer = SummaryWriter(self.log_dir)
        is_gcs = self.log_dir.startswith('gs://')
        if is_gcs:
            bucket_name = self.log_dir.split('/', 3)[2]  # Extrait le nom du bucket
            log_path = '/'.join(self.log_dir.split('/')[3:])  # Chemin dans le bucket
            print(f"TensorBoard logs: {self.log_dir}")
            print(f"Pour visualiser les logs:")
            print(f"  1. tensorboard --logdir=gs://{bucket_name}/{log_path}")
            print(f"  2. ou avec proxy: python -m tensorboard.main --logdir={self.log_dir} --port=6006")
        else:
            # Obtenir le chemin absolu pour les logs locaux
            abs_log_dir = os.path.abspath(self.log_dir)
            print(f"TensorBoard logs: {abs_log_dir}")
            print(f"Pour visualiser les logs: tensorboard --logdir={abs_log_dir}")
                
        # Initialize game logger
        if self.save_games:
            if gcs_bucket:
                print(f"Storing games in GCS: {gcs_bucket}")
                self.game_logger = GameLogger(
                    bucket_name=gcs_bucket,
                    process_id=self.process_id,
                    buffer_size=games_buffer_size,
                    flush_interval=games_flush_interval
                )
            else:
                games_dir = os.path.join(self.log_dir, "games")
                print(f"Local game storage: {games_dir}")
                self.game_logger = LocalGameLogger(
                    output_dir=games_dir,
                    buffer_size=games_buffer_size,
                    flush_interval=games_flush_interval
                )
        
        # Configure JAX functions
        self._setup_jax_functions()
    def set_evaluation_options(self, enable=True, frequency=10, num_games=1):
        """
        Configure evaluation options
        
        Args:
            enable: If True, evaluation will be performed during training
            frequency: Evaluation frequency (iterations)
            num_games: Number of games to play against each algorithm
        """
        self.eval_enabled = enable
        self.eval_frequency = frequency
        self.eval_games = num_games
        
        print(f"Evaluation {'enabled' if enable else 'disabled'}")
        if enable:
            print(f"  Frequency: Every {frequency} iterations")
            print(f"  Games per algorithm: {num_games}")
        
    def _update_learning_rate(self, iteration_percentage):
        """
        Update learning rate according to the defined schedule
        
        Args:
            iteration_percentage: Percentage of training progress [0.0, 1.0]
            
        Returns:
            New learning rate
        """
        # Find appropriate learning rate for current iteration percentage
        new_lr = self.initial_lr  # Default value
        
        for threshold, lr in self.lr_schedule:
            if iteration_percentage >= threshold:
                new_lr = lr
        
        if new_lr != self.current_lr:
            print(f"Learning rate updated: {self.current_lr} -> {new_lr}")
            self.current_lr = new_lr
            
            # Create new optimizer with gradient clipping
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(1.0),  # Maintain gradient clipping
                optax.sgd(learning_rate=self.current_lr, momentum=self.momentum)
            )
            
            # Reset optimizer state
            self.opt_state = self.optimizer.init(self.params)
            
            # Update JAX functions that use the optimizer
            self.optimizer_update_pmap = jax.pmap(
                lambda g, o, p: self.optimizer.update(g, o, p),
                axis_name='devices',
                devices=self.devices
            )
        
        return new_lr
    
    def _setup_jax_functions(self):
        """Configure JAX functions for generation and training."""
        # Use our optimized generator instead of the old one
        self.generate_games_pmap = create_optimized_game_generator(self.num_simulations)
        
        # Add gradient clipping to the optimizer
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Limit gradient norm to 1.0
            optax.sgd(learning_rate=self.current_lr, momentum=self.momentum)
        )
        
        # Reset optimizer state
        self.opt_state = self.optimizer.init(self.params)


        self.train_step_pmap = jax.pmap(
            partial(train_step_pmap_impl, network=self.network, value_weight=self.value_weight),
            axis_name='devices',  
            devices=self.devices
        )
        
        # Parameter update function with optimizer
        self.optimizer_update_pmap = jax.pmap(
            lambda g, o, p: self.optimizer.update(g, o, p),
            axis_name='devices',
            devices=self.devices  # Uniquement les dispositifs locaux
    )
      
    def train(self, num_iterations=100, games_per_iteration=64, 
            training_steps_per_iteration=100, eval_frequency=10, 
            save_frequency=10):
        """
        Démarre l'entraînement avec approche synchronisée par étapes.
        Compatible avec le buffer GCS.
        
        Args:
            num_iterations: Nombre total d'itérations
            games_per_iteration: Nombre de parties à générer par itération
            training_steps_per_iteration: Nombre d'étapes d'entraînement par itération
            eval_frequency: Fréquence d'évaluation (en itérations)
            save_frequency: Fréquence de sauvegarde (en itérations)
        """
        # Initialiser le timer global
        start_time_global = time.time()
        
        #rng_key = jax.random.PRNGKey(42)
        seed_base = 42
        process_specific_seed = seed_base + (self.process_id * 1000)
        rng_key = jax.random.PRNGKey(process_specific_seed)
        
        try:
            for iteration in range(num_iterations):
                self.iteration = iteration
                
                # Mettre à jour le taux d'apprentissage selon le planning
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
                
                # Afficher les infos du buffer appropriées
                if hasattr(self.buffer, 'gcs_index'):
                    # Buffer GCS
                    print(f"Buffer mis à jour: +{positions_added} positions")
                    print(f"  - Cache local: {self.buffer.local_size} positions")
                    print(f"  - Total estimé: {self.buffer.total_size} positions")
                else:
                    # Buffer local
                    print(f"Buffer mis à jour: +{positions_added} positions (total: {self.buffer.size})")
                
                # 3. Phase d'entraînement
                rng_key, train_key = jax.random.split(rng_key)
                t_start = time.time()
                
                metrics = self._train_network(train_key, training_steps_per_iteration)
                
                t_train = time.time() - t_start
                print(f"Entraînement: {training_steps_per_iteration} étapes en {t_train:.2f}s ({training_steps_per_iteration/t_train:.1f} étapes/s)")
                
                # Afficher les métriques
                print(f"  Perte totale: {metrics['total_loss']:.4f}")
                print(f"  Perte politique: {metrics['policy_loss']:.4f}, Perte valeur: {metrics['value_loss']:.4f}")
                print(f"  Précision politique: {metrics['policy_accuracy']}")
                
    
                # # 4. Évaluation périodique
                # if eval_frequency > 0 and (iteration + 1) % eval_frequency == 0:
                #     #if self.is_main_process:  # Uniquement sur le processus principal
                #     eval_start = time.time()
                #     print("\nExécution de l'évaluation...")
                #     self._evaluate()
                #     eval_time = time.time() - eval_start
                #     print(f"Évaluation terminée en {eval_time:.2f}s")
                # 4. Évaluation périodique contre les modèles précédents
                if eval_frequency > 0 and (iteration + 1) % eval_frequency == 0:
                    eval_start = time.time()
                    print("\nExécution de l'évaluation contre modèles précédents...")
                    # Passer le nombre total d'itérations prévues
                    self.evaluate_against_previous_models(num_iterations)
                    eval_time = time.time() - eval_start
                    print(f"Évaluation terminée en {eval_time:.2f}s")
            
                # 5. Sauvegarde périodique
                if save_frequency > 0 and (iteration + 1) % save_frequency == 0:
                    self._save_checkpoint()
            
            # Sauvegarde finale
            self._save_checkpoint(is_final=True)
            
        finally:
            # Assurer que les ressources sont libérées
            self.writer.close()
            
            # Fermer proprement le GameLogger si présent
            if self.save_games and hasattr(self, 'game_logger'):
                self.game_logger.stop()
                
            # Fermer proprement le buffer GCS si présent
            if hasattr(self.buffer, 'close'):
                print("Fermeture du buffer GCS...")
                self.buffer.close()
            
            # Statistiques globales
            total_time = time.time() - start_time_global
            print(f"\n=== Entraînement terminé ===")
            print(f"Parties générées: {self.total_games}")
            print(f"Positions totales: {self.total_positions}")
            print(f"Durée totale: {total_time:.1f}s ({num_iterations/total_time:.2f} itérations/s)")

    def _generate_games(self, rng_key, num_games):
        """
        Generate games in parallel on local TPU cores with optimized version.
        
        Args:
            rng_key: JAX random key
            num_games: Number of games to generate
            
        Returns:
            Generated games data
        """
        # Calculer le nombre de jeux par worker et par dispositif
        games_per_process = math.ceil(num_games / self.num_processes)
        batch_size_per_device = math.ceil(games_per_process / self.num_devices)
        
        # Nombre total de jeux générés par ce processus
        local_total_games = batch_size_per_device * self.num_devices
        
        # Total time measurement
        t_total_start = time.time()
        
        # Prepare RNGs for each core and distribute directly
        t_prep_start = time.time()
        # Utiliser uniquement le nombre de dispositifs locaux
        sharded_rngs = jax.random.split(rng_key, self.num_devices)
        sharded_rngs = jax.device_put_sharded(list(sharded_rngs), self.devices)
        
        # Replicate parameters directly on local devices
        sharded_params = jax.device_put_replicated(self.params, self.devices)
        t_prep_end = time.time()
        
        # Generate games with optimized version
        t_gen_start = time.time()
        games_data_pmap = self.generate_games_pmap(
            sharded_rngs, 
            sharded_params, 
            self.network, 
            self.env, 
            batch_size_per_device
        )
        t_gen_end = time.time()
        
        # Retrieve data on CPU
        t_fetch_start = time.time()
        games_data = jax.device_get(games_data_pmap)
        t_fetch_end = time.time()
        
        # Mettre à jour le compteur avec les jeux locaux
        self.total_games += local_total_games
        
        # Calculate and display times
        t_prep = t_prep_end - t_prep_start
        t_gen = t_gen_end - t_gen_start
        t_fetch = t_fetch_end - t_fetch_start
        t_total = time.time() - t_total_start
        
        # Calculer le nombre total de jeux générés par tous les processus (pour l'affichage)
        global_total_games = local_total_games * self.num_processes
        
        if self.is_main_process or self.num_processes == 1:
            print(f"  Preparation: {t_prep:.3f}s")
            print(f"  Generation: {t_gen:.3f}s (local: {local_total_games/t_gen:.1f} games/s, "
                f"estimated global: {global_total_games/t_gen:.1f} games/s)")
            print(f"  Retrieval: {t_fetch:.3f}s ({t_fetch/local_total_games*1000:.1f} ms/game)")
        
        # Record games for analysis if enabled
        if self.save_games and hasattr(self, 'game_logger'):
            t_convert_start = time.time()
            
            # Generate a prefix that includes the process ID to avoid conflicts
            game_id_prefix = f"iter{self.iteration}_p{self.process_id}"
            
            # Convert games to a format suitable for analysis
            converted_games = convert_games_batch(
                games_data,
                self.env,
                base_game_id=game_id_prefix,
                model_iteration=self.iteration
            )
            
            # Send games to logger which will write them asynchronously
            self.game_logger.log_games_batch(converted_games)
            
            t_convert = time.time() - t_convert_start
            if self.is_main_process or self.num_processes == 1:
                print(f"  Conversion for storage: {t_convert:.3f}s")
        
        return games_data
    
    def _update_buffer(self, games_data):
        """
        Met à jour le buffer d'expérience avec les nouvelles parties générées.
        Fonctionne avec les deux types de buffer (local et GCS).
        
        Args:
            games_data: Données des parties générées
            
        Returns:
            int: Nombre de positions ajoutées au buffer
        """
        positions_added = 0
        
        # Identifier si nous utilisons un buffer GCS
        using_gcs_buffer = hasattr(self.buffer, 'gcs_index')
        
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
                        model_version=self.total_games  # Utiliser total_games comme proxy pour la version
                    )
                    
                    positions_added += 1
        
        return positions_added
    
    # def _train_network(self, rng_key, num_steps):
    #     """
    #     Entraîne le réseau sur des batchs depuis le buffer.
    #     Compatible avec les deux types de buffer (local et GCS).
        
    #     Args:
    #         rng_key: Clé aléatoire JAX
    #         num_steps: Nombre d'étapes d'entraînement
            
    #     Returns:
    #         Métriques moyennes sur toutes les étapes (moyenne globale si multi-processus)
    #     """
    #     # Vérifier si le buffer est vide
    #     if (hasattr(self.buffer, 'local_size') and self.buffer.local_size == 0) or \
    #     (hasattr(self.buffer, 'size') and self.buffer.size == 0):
    #         print("Buffer vide, impossible d'entraîner le réseau.")
    #         # Retourner des métriques nulles
    #         return {'total_loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0, 
    #                 'policy_accuracy': 0.0, 'value_sign_match': 0.0}
                    
    #     # Identifier si nous utilisons un buffer GCS
    #     using_gcs_buffer = hasattr(self.buffer, 'gcs_index')
    #     gcs_index_available = False
    #     if using_gcs_buffer:
    #         gcs_index_available = bool(self.buffer.gcs_index) 
    #         if not gcs_index_available:
    #             print("Avertissement: Index GCS non disponible ou vide au début de cette phase d'entraînement. Utilisation du cache local si possible.")
        
    #     # Cumul des métriques pour ce processus sur les étapes
    #     cumulative_metrics = None
        
    #     for step in range(num_steps):
    #         total_batch_size = self.batch_size * self.num_devices
            
    #         if using_gcs_buffer:
    #             try:
    #                 batch_data = self.buffer.sample(total_batch_size, rng_key=rng_key)
    #             except ValueError as e:
    #                 print(f"Erreur lors de l'échantillonnage: {e}")
    #                 # Si GCS pas encore disponible, attendre et sauter cette étape
    #                 if step == 0:
    #                     print("Attente de données sur GCS pour l'entraînement...")
    #                     time.sleep(10)
    #                     continue
    #                 else:
    #                     # Arrêter l'entraînement si on a déjà fait quelques étapes
    #                     break
    #         else:
    #             # Pour le buffer local classique
    #             if self.recency_bias:
    #                 batch_data = self.buffer.sample_with_recency_bias(
    #                     total_batch_size,
    #                     temperature=self.recency_temperature,
    #                     rng_key=rng_key
    #                 )
    #             else:
    #                 batch_data = self.buffer.sample(total_batch_size, rng_key=rng_key)
            
    #         rng_key = jax.random.fold_in(rng_key, step)
            
    #         boards = jnp.array(batch_data['board'])
    #         marbles = jnp.array(batch_data['marbles_out'])
    #         policies = jnp.array(batch_data['policy'])
    #         values = jnp.array(batch_data['outcome'])
            
    #         # Reshape des données en chunks pour chaque dispositif local
    #         # Shape devient: (num_local_devices, batch_size_per_device, ...)
    #         boards = boards.reshape(self.num_devices, -1, *boards.shape[1:])
    #         marbles = marbles.reshape(self.num_devices, -1, *marbles.shape[1:])
    #         policies = policies.reshape(self.num_devices, -1, *policies.shape[1:])
    #         values = values.reshape(self.num_devices, -1, *values.shape[1:])
            
    #         # Répliquer les paramètres et l'état de l'optimiseur sur les dispositifs locaux pour pmap
    #         params_sharded = jax.device_put_replicated(self.params, self.devices)
    #         opt_state_sharded = jax.device_put_replicated(self.opt_state, self.devices)
            
    #         # Exécuter l'étape d'entraînement parallèle sur les dispositifs locaux
    #         metrics_sharded, grads_averaged = self.train_step_pmap(
    #             params_sharded, (boards, marbles), policies, values
    #         )
    #         #print(f"metrics_sharded['policy_accuracy']: {metrics_sharded['policy_accuracy']}")

            
    #         # Mettre à jour les paramètres avec les gradients moyennés
    #         updates, new_opt_state = self.optimizer_update_pmap(
    #             grads_averaged, opt_state_sharded, params_sharded
    #         )
    #         new_params = jax.tree_map(lambda p, u: p + u, params_sharded, updates)
            
    #         # Récupérer les paramètres et l'état de l'optimiseur depuis le premier dispositif
    #         self.params = jax.tree_map(lambda x: x[0], new_params)
    #         self.opt_state = jax.tree_map(lambda x: x[0], new_opt_state)
            
    #         # Agréger les métriques localement pour cette étape
    #         step_metrics = {k: float(jnp.mean(v)) for k, v in metrics_sharded.items()}
    #         #print(f"step_metrics['policy_accuracy']: {step_metrics['policy_accuracy']}")
            
    #         # Cumuler les métriques sur les étapes pour ce processus
    #         if cumulative_metrics is None:
    #             cumulative_metrics = step_metrics
    #         else:
    #             cumulative_metrics = {k: cumulative_metrics[k] + step_metrics[k] for k in step_metrics}
        
    #     # Si aucune étape d'entraînement n'a été effectuée
    #     if cumulative_metrics is None:
    #         return {'total_loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0, 
    #                 'policy_accuracy': 0.0, 'value_sign_match': 0.0}
                    
    #     steps_completed = num_steps
    #     avg_metrics = {k: v / steps_completed for k, v in cumulative_metrics.items()}

    #     if self.is_main_process:
    #         for metric_name, metric_value in avg_metrics.items():
    #             self.writer.add_scalar(f"training/{metric_name}", metric_value, self.iteration)
            
    #         # Enregistrer des informations supplémentaires utiles
    #         self.writer.add_scalar("training/learning_rate", self.current_lr, self.iteration)
            
    #         # Choisir la bonne métrique de taille du buffer
    #         if using_gcs_buffer:
    #             buffer_size = self.buffer.total_size
    #             self.writer.add_scalar("stats/buffer_size_total", buffer_size, self.iteration)
    #             self.writer.add_scalar("stats/buffer_size_local", self.buffer.local_size, self.iteration)
    #         else:
    #             buffer_size = self.buffer.size
    #             self.writer.add_scalar("stats/buffer_size", buffer_size, self.iteration)
                
    #         self.writer.add_scalar("stats/total_games_local", self.total_games, self.iteration)
        
    #     # Enregistrer l'historique des métriques
    #     local_metrics_record = avg_metrics.copy()
    #     local_metrics_record['iteration'] = self.iteration
    #     local_metrics_record['learning_rate'] = self.current_lr
        
    #     # Enregistrer la bonne métrique de taille du buffer
    #     if using_gcs_buffer:
    #         local_metrics_record['buffer_size_total'] = self.buffer.total_size
    #         local_metrics_record['buffer_size_local'] = self.buffer.local_size
    #     else:
    #         local_metrics_record['buffer_size'] = self.buffer.size
            
    #     local_metrics_record['total_games_local'] = self.total_games
    #     self.metrics_history.append(local_metrics_record)

    #     return avg_metrics

    def _train_network(self, rng_key, num_steps):
        """
        Entraîne le réseau sur des batchs depuis le buffer.
        Version optimisée pour environnement multi-host avec maintien des paramètres sur dispositif.
        Compatible avec les deux types de buffer (local et GCS).
        
        Args:
            rng_key: Clé aléatoire JAX
            num_steps: Nombre d'étapes d'entraînement
            
        Returns:
            Métriques moyennes sur toutes les étapes (moyenne globale si multi-processus)
        """
        # Vérifier si le buffer est vide
        if (hasattr(self.buffer, 'local_size') and self.buffer.local_size == 0) or \
        (hasattr(self.buffer, 'size') and self.buffer.size == 0):
            print("Buffer vide, impossible d'entraîner le réseau.")
            # Retourner des métriques nulles
            return {'total_loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0, 
                    'policy_accuracy': 0.0, 'value_sign_match': 0.0}
                    
        # Identifier si nous utilisons un buffer GCS
        using_gcs_buffer = hasattr(self.buffer, 'gcs_index')
        gcs_index_available = False
        if using_gcs_buffer:
            gcs_index_available = bool(self.buffer.gcs_index) 
            if not gcs_index_available:
                print("Avertissement: Index GCS non disponible ou vide au début de cette phase d'entraînement. Utilisation du cache local si possible.")
        
        # Initialiser les paramètres et l'état de l'optimiseur sur les dispositifs une seule fois
        # au début de la séquence d'entraînement
        params_sharded = jax.device_put_replicated(self.params, self.devices)
        opt_state_sharded = jax.device_put_replicated(self.opt_state, self.devices)
        
        # Cumul des métriques pour ce processus sur les étapes
        cumulative_metrics = None
        
        for step in range(num_steps):
            total_batch_size = self.batch_size * self.num_devices
            
            if using_gcs_buffer:
                try:
                    batch_data = self.buffer.sample(total_batch_size, rng_key=rng_key)
                except ValueError as e:
                    print(f"Erreur lors de l'échantillonnage: {e}")
                    # Si GCS pas encore disponible, attendre et sauter cette étape
                    if step == 0:
                        print("Attente de données sur GCS pour l'entraînement...")
                        time.sleep(10)
                        continue
                    else:
                        # Arrêter l'entraînement si on a déjà fait quelques étapes
                        break
            else:
                # Pour le buffer local classique
                if self.recency_bias:
                    batch_data = self.buffer.sample_with_recency_bias(
                        total_batch_size,
                        temperature=self.recency_temperature,
                        rng_key=rng_key
                    )
                else:
                    batch_data = self.buffer.sample(total_batch_size, rng_key=rng_key)
            
            rng_key = jax.random.fold_in(rng_key, step)
            
            boards = jnp.array(batch_data['board'])
            marbles = jnp.array(batch_data['marbles_out'])
            policies = jnp.array(batch_data['policy'])
            values = jnp.array(batch_data['outcome'])
            
            # Reshape des données en chunks pour chaque dispositif local
            # Shape devient: (num_local_devices, batch_size_per_device, ...)
            boards = boards.reshape(self.num_devices, -1, *boards.shape[1:])
            marbles = marbles.reshape(self.num_devices, -1, *marbles.shape[1:])
            policies = policies.reshape(self.num_devices, -1, *policies.shape[1:])
            values = values.reshape(self.num_devices, -1, *values.shape[1:])
            
            # Exécuter l'étape d'entraînement parallèle avec les paramètres déjà sur dispositif
            metrics_sharded, grads_averaged = self.train_step_pmap(
                params_sharded, (boards, marbles), policies, values
            )
            
            # Mettre à jour directement les versions shardées
            updates, opt_state_sharded = self.optimizer_update_pmap(
                grads_averaged, opt_state_sharded, params_sharded
            )
            params_sharded = jax.tree_map(lambda p, u: p + u, params_sharded, updates)
            
            # Agréger les métriques localement pour cette étape
            step_metrics = {k: float(jnp.mean(v)) for k, v in metrics_sharded.items()}
            
            # Cumuler les métriques sur les étapes pour ce processus
            if cumulative_metrics is None:
                cumulative_metrics = step_metrics
            else:
                cumulative_metrics = {k: cumulative_metrics[k] + step_metrics[k] for k in step_metrics}
        
        # À la fin de toutes les étapes, récupérer les paramètres mis à jour
        # pour les stocker dans l'état de l'objet
        self.params = jax.tree_map(lambda x: x[0], params_sharded)
        self.opt_state = jax.tree_map(lambda x: x[0], opt_state_sharded)
        
        # Si aucune étape d'entraînement n'a été effectuée
        if cumulative_metrics is None:
            return {'total_loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0, 
                    'policy_accuracy': 0.0, 'value_sign_match': 0.0}
                    
        steps_completed = num_steps
        avg_metrics = {k: v / steps_completed for k, v in cumulative_metrics.items()}

        if self.is_main_process:
            for metric_name, metric_value in avg_metrics.items():
                self.writer.add_scalar(f"training/{metric_name}", metric_value, self.iteration)
            
            # Enregistrer des informations supplémentaires utiles
            self.writer.add_scalar("training/learning_rate", self.current_lr, self.iteration)
            
            # Choisir la bonne métrique de taille du buffer
            if using_gcs_buffer:
                buffer_size = self.buffer.total_size
                self.writer.add_scalar("stats/buffer_size_total", buffer_size, self.iteration)
                self.writer.add_scalar("stats/buffer_size_local", self.buffer.local_size, self.iteration)
            else:
                buffer_size = self.buffer.size
                self.writer.add_scalar("stats/buffer_size", buffer_size, self.iteration)
                
            self.writer.add_scalar("stats/total_games_local", self.total_games, self.iteration)
        
        # Enregistrer l'historique des métriques
        local_metrics_record = avg_metrics.copy()
        local_metrics_record['iteration'] = self.iteration
        local_metrics_record['learning_rate'] = self.current_lr
        
        # Enregistrer la bonne métrique de taille du buffer
        if using_gcs_buffer:
            local_metrics_record['buffer_size_total'] = self.buffer.total_size
            local_metrics_record['buffer_size_local'] = self.buffer.local_size
        else:
            local_metrics_record['buffer_size'] = self.buffer.size
            
        local_metrics_record['total_games_local'] = self.total_games
        self.metrics_history.append(local_metrics_record)

        return avg_metrics
    
    def _save_checkpoint(self, is_final=False):
        """
        Save a model checkpoint
        
        Args:
            is_final: If True, indicates this is the final checkpoint
        """
        if not self.is_main_process:
            return
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
        
        # Create directory if needed
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        
        # Check if it's a GCS path
        is_gcs = self.checkpoint_path.startswith('gs://')
        
        if is_gcs:
            # Save to GCS
            import subprocess
            
            # Save locally first
            local_path = f"/tmp/{prefix}.pkl"
            with open(local_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            # Then send to GCS
            gcs_path = f"{self.checkpoint_path}_{prefix}.pkl"
            subprocess.run(f"gsutil cp {local_path} {gcs_path}", shell=True)
            
            print(f"Checkpoint saved: {gcs_path}")
            
            # Delete local file
            os.remove(local_path)
        else:
            # Save locally
            filename = f"{self.checkpoint_path}_{prefix}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            print(f"Checkpoint saved: {filename}")
        
    def load_checkpoint(self, checkpoint_path):
        """
        Load a previously saved checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        # Check if it's a GCS path
        is_gcs = checkpoint_path.startswith('gs://')
        
        if is_gcs:
            # Download from GCS
            import subprocess
            
            local_path = "/tmp/checkpoint.pkl"
            subprocess.run(f"gsutil cp {checkpoint_path} {local_path}", shell=True)
            
            with open(local_path, 'rb') as f:
                checkpoint = pickle.load(f)
                
            # Delete local file
            os.remove(local_path)
        else:
            # Load locally
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
        
        # Restore state
        self.params = checkpoint['params']
        self.opt_state = checkpoint['opt_state']
        self.iteration = checkpoint['iteration']
        self.current_lr = checkpoint['current_lr']
        self.metrics_history = checkpoint['metrics']
        self.total_games = checkpoint['total_games']
        self.total_positions = checkpoint['total_positions']
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Iteration: {self.iteration}, Positions: {self.total_positions}")

    def initialize_existing_buffer(self, gcs_bucket, gcs_buffer_dir='buffer', 
                            recency_bias=None, recency_temperature=None,
                            max_local_size=None):
        """
        Initialise un buffer GCS existant pour poursuivre un entraînement.
        
        Args:
            gcs_bucket: Nom du bucket GCS contenant le buffer
            gcs_buffer_dir: Dossier dans le bucket pour le buffer
            recency_bias: Activer/désactiver le biais de récence (None pour conserver la valeur actuelle)
            recency_temperature: Température pour le biais de récence
            max_local_size: Taille maximale du cache local
            
        Returns:
            bool: True si l'initialisation a réussi
        """
        try:
            from training.replay_buffer import GCSReplayBuffer
            
            # Conserver les paramètres actuels si non spécifiés
            if recency_bias is None:
                recency_bias = self.recency_bias
            
            if recency_temperature is None:
                recency_temperature = self.recency_temperature
                
            if max_local_size is None:
                # Par défaut, utiliser 10% de la taille du buffer spécifiée initialement
                if hasattr(self.buffer, 'max_local_size'):
                    max_local_size = self.buffer.max_local_size
                else:
                    max_local_size = 100000  # Valeur par défaut
            
            # Fermer le buffer existant si c'est un GCSReplayBuffer
            if hasattr(self.buffer, 'close'):
                print("Fermeture du buffer existant...")
                self.buffer.close()
            
            print(f"Initialisation du buffer GCS existant: {gcs_bucket}/{gcs_buffer_dir}")
            
            # Créer le nouveau buffer
            self.buffer = GCSReplayBuffer(
                bucket_name=gcs_bucket,
                buffer_dir=gcs_buffer_dir,
                max_local_size=max_local_size,
                recency_enabled=recency_bias,
                recency_temperature=recency_temperature
            )
            
            # Forcer la mise à jour de l'index
            self.buffer._update_gcs_index()
            
            # Attendre un peu que l'indexation se termine
            time.sleep(2)
            
            # Vérifier si des données sont disponibles
            if self.buffer.gcs_index:
                print(f"Buffer GCS initialisé avec succès")
                print(f"  - Itérations disponibles: {len(self.buffer.gcs_index)}")
                print(f"  - Nombre estimé d'exemples: {self.buffer.total_size}")
                return True
            else:
                print("Avertissement: Aucune donnée trouvée dans le buffer GCS.")
                print("Le buffer est prêt mais vide.")
                return True
        
        except Exception as e:
            print(f"Erreur lors de l'initialisation du buffer GCS: {e}")
            # Revenir au buffer original en cas d'erreur
            if not hasattr(self.buffer, 'gcs_index'):
                print("Conservation du buffer local original.")
            return False
        
    # def _evaluate(self):
    #     """
    #     Évalue le modèle actuel contre des algorithmes classiques.
    #     Exécuté uniquement sur le processus principal.
    #     """
    
    #     evaluator = Evaluator(self.params, self.network, self.env)
        
    #     algorithms = [
    #         ("alphabeta_pruning", 3)  
    #     ]
        
    #     results = evaluator.evaluate_against_classical(
    #         algorithms=algorithms,
    #         num_games_per_algo=self.eval_games,
    #         verbose=True
    #     )
        
        # for algo_name, data in results.items():
        #     win_rate = data["win_rate"]
        #     self.writer.add_scalar(f"evaluation/win_rate_{algo_name}", win_rate, self.iteration)
        #     self.writer.add_scalar(f"evaluation/wins_{algo_name}", data["wins"], self.iteration)
        #     self.writer.add_scalar(f"evaluation/losses_{algo_name}", data["losses"], self.iteration)
        #     self.writer.add_scalar(f"evaluation/draws_{algo_name}", data["draws"], self.iteration)
        
        # print("\n=== Résultats d'évaluation ===")
        # for algo_name, data in results.items():
        #     win_rate = data["win_rate"]
        #     print(f"vs {algo_name}: {win_rate:.1%} ({data['wins']}/{data['wins']+data['losses']+data['draws']})")
        
        # return results
    
    def evaluate_against_previous_models(self, total_iterations, num_reference_models=8):
        """
        Évalue le modèle actuel contre des versions précédentes sur TPU.
        Exécuté uniquement sur le processus principal.
        
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
        
        # Ne faire l'évaluation que sur le processus principal
        # if not self.is_main_process:
        #     return {}
        
        current_iter = self.iteration
        results = {}
        
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
            print("Aucun modèle précédent disponible pour l'évaluation")
            return {}
        
        print(f"\n=== Évaluation contre modèles précédents (itération actuelle: {current_iter}) ===")
        print(f"Itérations sélectionnées: {available_refs}")
        
        # Initialiser l'évaluateur
        evaluator = ModelsEvaluator(
            network=self.network,
            num_simulations=max(20, self.num_simulations // 2),  # Réduire le nombre de simulations pour l'évaluation
            games_per_model=4  # 10 parties par modèle (5 comme noir, 5 comme blanc)
        )
        
        # Obtenir les paramètres du modèle actuel
        current_params = self.params
        
        # Évaluer contre chaque modèle de référence
        for ref_iter in available_refs:
            print(f"\nÉvaluation contre le modèle de l'itération {ref_iter}...")
            
            # Construire le chemin du checkpoint
            ref_path = self._get_checkpoint_path(ref_iter)
            
            # Télécharger le checkpoint s'il est sur GCS
            local_path = f"/tmp/ref_model_{ref_iter}.pkl"
            if ref_path.startswith("gs://"):
                if not download_checkpoint(ref_path, local_path):
                    print(f"Échec du téléchargement du checkpoint pour l'itération {ref_iter}, on passe")
                    continue
            else:
                local_path = ref_path
            
            # Charger les paramètres du modèle de référence
            ref_params = load_checkpoint_params(local_path)
            if ref_params is None:
                print(f"Échec du chargement des paramètres pour l'itération {ref_iter}, on passe")
                continue
            
            # Évaluer le modèle actuel contre le modèle de référence
            eval_results = evaluator.evaluate_model_pair(current_params, ref_params)
            
            # Stocker les résultats
            results[ref_iter] = eval_results
            
            # Afficher les résultats
            win_rate = eval_results['win_rate']
            print(f"Résultats vs iter {ref_iter}: {win_rate:.1%} taux de victoire")
            print(f"  Victoires: {eval_results['current_wins']}, Défaites: {eval_results['reference_wins']}, Nuls: {eval_results['draws']}")
            
            # Enregistrer dans TensorBoard
            self.writer.add_scalar(f"eval_vs_prev/win_rate_iter_{ref_iter}", win_rate, self.iteration)
            self.writer.add_scalar(f"eval_vs_prev/games_iter_{ref_iter}", eval_results['total_games'], self.iteration)
        
        # Enregistrer les tendances de performance globales
        if results:
            # Calculer le taux de victoire moyen sur tous les modèles de référence
            avg_win_rate = sum(res['win_rate'] for res in results.values()) / len(results)
            self.writer.add_scalar("eval_vs_prev/avg_win_rate", avg_win_rate, self.iteration)
            
            # Ajouter un résumé à l'historique des métriques
            if self.metrics_history and self.iteration > 0:
                latest_metrics = self.metrics_history[-1]
                latest_metrics['avg_win_rate_vs_prev'] = avg_win_rate
                
                # Stocker les taux de victoire individuels
                for ref_iter, ref_results in results.items():
                    latest_metrics[f'win_rate_vs_iter_{ref_iter}'] = ref_results['win_rate']
        
        print(f"\n=== Évaluation terminée ===")
        return results

    def _get_checkpoint_path(self, iteration):
        """Obtient le chemin vers un checkpoint pour l'itération donnée."""
        # Le format correspond à votre méthode save_checkpoint
        prefix = f"iter{iteration}"
        
        if self.checkpoint_path.startswith("gs://"):
            return f"{self.checkpoint_path}_{prefix}.pkl"
        else:
            return f"{self.checkpoint_path}_{prefix}.pkl"