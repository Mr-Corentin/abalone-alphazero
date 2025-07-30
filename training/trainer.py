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
from utils.game_storage import convert_games_batch, GameLogger, LocalGameLogger
from utils.gcs_metrics_logger import SimpleGCSLogger, LocalMetricsLogger, IterationMetricsAggregator



import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Process %(process)d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("alphazero.trainer")

class AbaloneTrainerSync:
    """
    AlphaZero training coordinator for Abalone game with synchronized approach.
    Supports multi-process and multi-host TPU/GPU environments.
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
            verbose=True,
            enable_comprehensive_logging=True,
            metrics_logging_interval=30):

        """
        Initialize training coordinator with step-by-step synchronized approach.
        Uses SGD with momentum as in original AlphaZero implementation.

        Args:
            network: Neural network model
            env: Game environment
            buffer_size: Replay buffer size
            batch_size: Training batch size
            value_weight: Value loss weight
            num_simulations: Number of MCTS simulations per move
            recency_bias: If True, uses recency-biased sampling
            recency_temperature: Temperature for recency bias
            initial_lr: Initial learning rate (0.2 as in AlphaZero)
            momentum: Momentum for SGD (0.9 standard)
            lr_schedule: List of tuples (iteration_percentage, learning_rate) or None
            checkpoint_path: Path to save checkpoints
            log_dir: Path for tensorboard logs
            gcs_bucket: GCS bucket name to store games (if None, local storage)
            save_games: If True, save played games for future analysis
            games_buffer_size: Number of games to accumulate before saving
            games_flush_interval: Interval in seconds to save games
            use_gcs_buffer: If True, use buffer on GCS
            gcs_buffer_dir: Folder in GCS bucket for buffer
            eval_games: Number of games to play for evaluation
            verbose: If True, display detailed logs
        """
        # Device configuration
        self.global_devices = jax.devices()
        self.num_global_devices = len(self.global_devices)
        self.devices = jax.local_devices()
        self.num_devices = len(self.devices)
        self.process_id = jax.process_index()
        self.num_processes = jax.process_count()
        self.is_main_process = self.process_id == 0
        self.verbose = verbose and self.is_main_process

        # Identify device type
        if self.devices[0].platform == 'tpu':
            self.device_type = 'tpu'
        elif self.devices[0].platform == 'gpu':
            self.device_type = 'gpu'
        else:
            self.device_type = 'cpu'
            
        logger.info(f"Process {self.process_id+1}/{self.num_processes}: "
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
        self.enable_comprehensive_logging = enable_comprehensive_logging
        self.metrics_logging_interval = metrics_logging_interval
        self.gcs_bucket = gcs_bucket

        # Learning rate and optimizer configuration
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.momentum = momentum

        # Default AlphaZero schedule if not specified
        if lr_schedule is None:
            self.lr_schedule = [
                (0.0, initial_lr),      # Start
                (0.3, initial_lr/10),   # First step: 0.2 -> 0.02
                (0.6, initial_lr/100),  # Second step: 0.02 -> 0.002
                (0.85, initial_lr/1000) # Third step: 0.002 -> 0.0002
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

        # Buffer initialization
        if use_gcs_buffer and gcs_bucket:
            if self.verbose:
                logger.info(f"Using global GCS buffer: {gcs_bucket}/{gcs_buffer_dir}")
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
                logger.info(f"Using local buffer of size {buffer_size}")
            self.buffer = CPUReplayBuffer(buffer_size)

        # SGD optimizer initialization
        self.optimizer = optax.sgd(learning_rate=self.initial_lr, momentum=self.momentum)

        # Parameter and optimization state initialization
        rng = jax.random.PRNGKey(42)
        sample_board = jnp.zeros((1, 9, 9), dtype=jnp.int8)
        sample_marbles = jnp.zeros((1, 2), dtype=jnp.int8)
        sample_history = jnp.zeros((1, 8, 9, 9), dtype=jnp.int8)  # 8 history positions
        self.params = network.init(rng, sample_board, sample_marbles, sample_history)
        self.opt_state = self.optimizer.init(self.params)

        # Statistics
        self.iteration = 0
        self.total_games = 0
        self.total_positions = 0
        self.metrics_history = []

        # TensorBoard logging configuration
        self._setup_tensorboard(log_dir)

        # Game logger initialization
        if self.save_games:
            self._setup_game_logger(gcs_bucket, games_buffer_size, games_flush_interval)

        # Generate a single session_id for all workers to ensure consistency
        # Only the main process generates the session_id, then broadcasts it to all processes
        if self.is_main_process:
            session_id_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            session_id_timestamp = None
        
        # Broadcast the session_id to all processes to ensure consistency
        session_id_array = jnp.array([ord(c) for c in session_id_timestamp] if session_id_timestamp else [0] * 15)
        session_id_array = jax.device_put(session_id_array, self.devices[0])
        
        # Use JAX collective communication to broadcast from process 0 to all processes
        broadcasted_session_id = jax.experimental.multihost_utils.broadcast_one_to_all(session_id_array, is_source=self.is_main_process)
        
        # Convert back to string
        broadcasted_chars = jax.device_get(broadcasted_session_id)
        if broadcasted_chars[0] != 0:  # Check if we received valid data
            self.session_id = ''.join(chr(int(c)) for c in broadcasted_chars if c != 0)
        else:
            # Fallback to process-specific timestamp if broadcast failed
            self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Metrics logger initialization
        self._setup_metrics_logger(gcs_bucket, enable_comprehensive_logging, self.session_id)

        # Metrics aggregator initialization (main process only)
        self._setup_metrics_aggregator(gcs_bucket, enable_comprehensive_logging)

        # JAX functions configuration
        self._setup_jax_functions()
        
        # Evaluation configuration
        self.eval_enabled = False

    def _setup_tensorboard(self, log_dir):
        """Configure TensorBoard logging"""
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
                logger.info(f"To visualize: tensorboard --logdir=gs://{bucket_name}/{log_path}")
            else:
                abs_log_dir = os.path.abspath(self.log_dir)
                logger.info(f"TensorBoard logs: {abs_log_dir}")
                logger.info(f"To visualize: tensorboard --logdir={abs_log_dir}")

    def _setup_game_logger(self, gcs_bucket, buffer_size, flush_interval):
        """Configure game logger"""
        if gcs_bucket:
            if self.verbose:
                logger.info(f"Storing games in GCS: {gcs_bucket}")
            self.game_logger = GameLogger(
                bucket_name=gcs_bucket,
                process_id=self.process_id,
                buffer_size=buffer_size,
                flush_interval=flush_interval
            )
        else:
            games_dir = os.path.join(self.log_dir, "games")
            if self.verbose:
                logger.info(f"Local game storage: {games_dir}")
            self.game_logger = LocalGameLogger(
                output_dir=games_dir,
                buffer_size=buffer_size,
                flush_interval=flush_interval
            )

    def _setup_metrics_logger(self, gcs_bucket, enable_comprehensive_logging, session_id):
        """Configure metrics logger for detailed tracking"""
        if not enable_comprehensive_logging:
            self.metrics_logger = None
            if self.verbose:
                logger.info("Comprehensive logging disabled")
            return
        
        if gcs_bucket:
            if self.verbose:
                logger.info(f"Comprehensive metrics logging to GCS bucket: {gcs_bucket}")
            self.metrics_logger = SimpleGCSLogger(
                bucket_name=gcs_bucket,
                process_id=self.process_id,
                session_id=session_id
            )
        else:
            metrics_dir = os.path.join(self.log_dir, "metrics")
            if self.verbose:
                logger.info(f"Comprehensive metrics logging locally: {metrics_dir}")
            self.metrics_logger = LocalMetricsLogger(
                log_dir=metrics_dir,
                process_id=self.process_id,
                session_id=session_id
            )

    def _setup_metrics_aggregator(self, gcs_bucket, enable_comprehensive_logging):
        """Configure metrics aggregator to consolidate logs by iteration"""
        if not enable_comprehensive_logging or not self.is_main_process:
            self.metrics_aggregator = None
            return
        
        if gcs_bucket:
            if self.verbose:
                logger.info(f"Metrics aggregator enabled for GCS bucket: {gcs_bucket}")
            self.metrics_aggregator = IterationMetricsAggregator(bucket_name=gcs_bucket)
        else:
            metrics_dir = os.path.join(self.log_dir, "metrics")
            if self.verbose:
                logger.info(f"Metrics aggregator enabled for local directory: {metrics_dir}")
            self.metrics_aggregator = IterationMetricsAggregator(log_dir=metrics_dir)

    def enable_evaluation(self, enable=True):
        """Enable or disable evaluation"""
        self.eval_enabled = enable
        if self.verbose:
            logger.info(f"Evaluation {'enabled' if enable else 'disabled'}")

    def _update_learning_rate(self, iteration_percentage):
        """
        Update learning rate according to defined schedule

        Args:
            iteration_percentage: Training progress percentage [0.0, 1.0]

        Returns:
            New learning rate
        """
        # Find appropriate learning rate for current iteration percentage
        new_lr = self.initial_lr  # Default value

        for threshold, lr in self.lr_schedule:
            if iteration_percentage >= threshold:
                new_lr = lr

        if new_lr != self.current_lr:
            if self.verbose:
                logger.info(f"Learning rate updated: {self.current_lr} -> {new_lr}")
            self.current_lr = new_lr

            # Create new optimizer with gradient clipping
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(1.0),
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
        # Use our optimized generator
        self.generate_games_pmap = create_optimized_game_generator(self.num_simulations)

        # Add gradient clipping to optimizer
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Limit gradient norm to 1.0
            optax.sgd(learning_rate=self.current_lr, momentum=self.momentum)
        )

        # Reset optimizer state
        self.opt_state = self.optimizer.init(self.params)

        # Configure parallel processing functions
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
        Start training with step-by-step synchronized approach.
        Evaluation is now triggered by reference checkpoints.

        Args:
            num_iterations: Total number of iterations
            games_per_iteration: Number of games to generate per iteration
            training_steps_per_iteration: Number of training steps per iteration
            save_frequency: Regular save frequency (in iterations)
        """
        # Initialize global timer
        start_time_global = time.time()

        # Initialize process-specific random key
        seed_base = 42
        process_specific_seed = seed_base + (self.process_id * 1000)
        rng_key = jax.random.PRNGKey(process_specific_seed)

        # Determine reference iterations for entire training
        from evaluation.models_evaluator import generate_evaluation_checkpoints
        self.reference_iterations = generate_evaluation_checkpoints(num_iterations)
        
        if self.verbose:
            logger.info(f"Planned reference iterations: {self.reference_iterations}")
            logger.info(f"Evaluation {'enabled' if self.eval_enabled else 'disabled'}")
        
        # Initial log for each process
        logger.info(f"Process {self.process_id}: Starting training")

        try:
            for iteration in range(num_iterations):
                self.iteration = iteration
                iter_start_time = time.time()
                
                logger.info(f"Process {self.process_id}: Starting iteration {iteration+1}")

                # Update learning rate according to schedule
                iteration_percentage = iteration / num_iterations
                self._update_learning_rate(iteration_percentage)
                
                if self.verbose:
                    logger.info(f"\n=== Itération {iteration+1}/{num_iterations} (LR: {self.current_lr}) ===")

                # Synchronization at iteration start
                jax.experimental.multihost_utils.sync_global_devices(f"iter_{iteration}_start")
                logger.info(f"Process {self.process_id}: Synchronized at start of iteration {iteration+1}")

                # 1. Generation phase
                gen_start_time = time.time()
                logger.info(f"Process {self.process_id}: Starting generation for iteration {iteration+1}")
                
                rng_key, gen_key = jax.random.split(rng_key)
                t_start = time.time()
                games_data = self._generate_games(gen_key, games_per_iteration)
                t_gen = time.time() - t_start

                logger.info(f"Process {self.process_id}: Finished generation for iteration {iteration+1} in {t_gen:.2f}s")
                
                if self.verbose:
                    logger.info(f"Generation: {games_per_iteration} games in {t_gen:.2f}s ({games_per_iteration/t_gen:.1f} games/s)")

                # 2. Buffer update
                logger.info(f"Process {self.process_id}: Waiting for post-generation synchronization")
                jax.experimental.multihost_utils.sync_global_devices(f"post_generation_iter_{iteration}")
                logger.info(f"Process {self.process_id}: Post-generation synchronization complete")
                
                buffer_start_time = time.time()
                logger.info(f"Process {self.process_id}: Starting buffer update for iteration {iteration+1}")
                
                t_start = time.time()
                positions_added = self._update_buffer(games_data)
                t_buffer = time.time() - t_start
                self.total_positions += positions_added

                logger.info(f"Process {self.process_id}: Finished buffer update for iteration {iteration+1} in {t_buffer:.2f}s")

                # Display buffer info
                if self.verbose:
                    if hasattr(self.buffer, 'gcs_index'):
                        # GCS Buffer
                        logger.info(f"Buffer updated: +{positions_added} positions")
                        logger.info(f"  - Local cache: {self.buffer.local_size} positions")
                        logger.info(f"  - Estimated total: {self.buffer.total_size} positions")
                    else:
                        # Local buffer
                        logger.info(f"Buffer updated: +{positions_added} positions (total: {self.buffer.size})")

                # Synchronization after buffer update
                jax.experimental.multihost_utils.sync_global_devices(f"post_buffer_update_iter_{iteration}")
                logger.info(f"Process {self.process_id}: Synchronized after buffer update")

                # 3. Training phase
                train_start_time = time.time()
                logger.info(f"Process {self.process_id}: Starting training for iteration {iteration+1}")
                
                rng_key, train_key = jax.random.split(rng_key)
                t_start = time.time()
                metrics = self._train_network(train_key, training_steps_per_iteration)
                t_train = time.time() - t_start
                
                logger.info(f"Process {self.process_id}: Finished training for iteration {iteration+1} in {t_train:.2f}s")
                
                if self.verbose:
                    logger.info(f"Training: {training_steps_per_iteration} steps in {t_train:.2f}s ({training_steps_per_iteration/t_train:.1f} steps/s)")
                    logger.info(f"  Total loss: {metrics['total_loss']:.4f}")
                    logger.info(f"  Policy loss: {metrics['policy_loss']:.4f}, Value loss: {metrics['value_loss']:.4f}")
                    logger.info(f"  Policy accuracy: {metrics['policy_accuracy']}%")

                # Log timing metrics for this iteration
                if self.metrics_logger:
                    iter_total_time = time.time() - iter_start_time
                    self.metrics_logger.log_timing_metrics(
                        iteration=iteration,
                        generation_time=t_gen,
                        buffer_update_time=t_buffer,
                        training_time=t_train,
                        total_iteration_time=iter_total_time,
                        games_per_sec=games_per_iteration / t_gen if t_gen > 0 else 0,
                        steps_per_sec=training_steps_per_iteration / t_train if t_train > 0 else 0
                    )

                # Synchronization after training
                jax.experimental.multihost_utils.sync_global_devices(f"post_training_iter_{iteration}")
                logger.info(f"Process {self.process_id}: Synchronized after training")

                # 4. Reference checkpoint management
                if iteration in self.reference_iterations:
                    # ALL processes enter this block
                    logger.info(f"Process {self.process_id}: Processing reference checkpoint for iteration {iteration+1}")
                    
                    if self.is_main_process:
                        if self.verbose:
                            logger.info(f"\nIteration {iteration}: Reference checkpoint detected")
                        
                        # Save reference model (main process only)
                        self._save_checkpoint(is_reference=True)
                        logger.info(f"Process {self.process_id}: Reference checkpoint save complete")
                    
                    # Synchronize ALL processes after save
                    logger.info(f"Process {self.process_id}: Waiting for post-checkpoint synchronization")
                    jax.experimental.multihost_utils.sync_global_devices(f"post_checkpoint_save_iter_{iteration}")
                    logger.info(f"Process {self.process_id}: Synchronized after checkpoint save")
                    
                    # Evaluate if enabled and not at iteration 0
                    if self.eval_enabled and iteration > 0:
                        # ALL processes participate in evaluation
                        eval_start = time.time()
                        if self.verbose:
                            logger.info("Evaluation triggered by new reference model...")
                        
                        logger.info(f"Process {self.process_id}: Starting evaluation for iteration {iteration+1}")
                        
                        # Call evaluation for ALL processes
                        self.evaluate_against_previous_models(num_iterations)
                        
                        eval_time = time.time() - eval_start
                        logger.info(f"Process {self.process_id}: Finished evaluation for iteration {iteration+1} in {eval_time:.2f}s")
                        
                        if self.verbose:
                            logger.info(f"Evaluation complete in {eval_time:.2f}s")
                
                # 5. Standard periodic save (non-reference)
                elif save_frequency > 0 and (iteration + 1) % save_frequency == 0:
                    logger.info(f"Process {self.process_id}: Processing periodic save for iteration {iteration+1}")
                    
                    if self.is_main_process:
                        # Normal save (not a reference checkpoint)
                        if self.verbose:
                            logger.info("\nPeriodic checkpoint save...")
                        self._save_checkpoint(is_reference=False)
                        logger.info(f"Process {self.process_id}: Periodic save complete")
                    
                    # Synchronize all processes after save
                    logger.info(f"Process {self.process_id}: Waiting for post-save synchronization")
                    jax.experimental.multihost_utils.sync_global_devices(f"post_regular_checkpoint_iter_{iteration}")
                    logger.info(f"Process {self.process_id}: Synchronized after periodic save")

                # IMPORTANT: Synchronization at end of each iteration
                iter_time = time.time() - iter_start_time
                logger.info(f"Process {self.process_id}: Iteration {iteration+1} completed in {iter_time:.2f}s")
                logger.info(f"Process {self.process_id}: Waiting for end-of-iteration synchronization")
                jax.experimental.multihost_utils.sync_global_devices(f"end_of_iteration_{iteration}")
                logger.info(f"Process {self.process_id}: Synchronized at end of iteration {iteration+1}")

                # Synchronize before metrics aggregation to ensure all workers finished writing logs
                logger.info(f"Process {self.process_id}: Waiting for synchronization before metrics aggregation")
                jax.experimental.multihost_utils.sync_global_devices(f"before_metrics_aggregation_{iteration}")
                logger.info(f"Process {self.process_id}: Synchronized before metrics aggregation")

                # Metrics aggregation at end of iteration (main process only)
                if self.metrics_aggregator and self.metrics_logger:
                    try:
                        # Allow extra time for all logs to be written
                        time.sleep(2.0)
                        
                        # Use the new consolidated readable summary method instead of separate JSON files
                        self.metrics_aggregator.write_consolidated_readable_summary(
                            iteration=iteration,
                            num_workers=self.num_processes,
                            session_id=self.metrics_logger.session_id
                        )
                        logger.info(f"Process {self.process_id}: Consolidated summary written for iteration {iteration+1}")
                    except Exception as e:
                        logger.error(f"Error during metrics aggregation for iteration {iteration+1}: {e}")

            # Final save
            final_is_reference = (num_iterations - 1) in self.reference_iterations
            logger.info(f"Process {self.process_id}: Preparing for end of training")
            
            if self.is_main_process:
                self._save_checkpoint(is_final=True, is_reference=final_is_reference)
                logger.info(f"Process {self.process_id}: Final save complete")
            
            # Synchronize before end
            logger.info(f"Process {self.process_id}: Waiting for final synchronization")
            jax.experimental.multihost_utils.sync_global_devices("post_final_save")
            logger.info(f"Process {self.process_id}: Final synchronization complete")
                
            if self.metrics_history and self.is_main_process:
                final_metrics = self.metrics_history[-1]
                
                # Training metrics
                training_metrics = {k: v for k, v in final_metrics.items() 
                                if k in ['total_loss', 'policy_loss', 'value_loss', 
                                        'policy_accuracy', 'value_sign_match']}
                self._log_metrics_to_tensorboard(training_metrics, "training")
                
                # Learning rate
                self._log_metrics_to_tensorboard({"learning_rate": self.current_lr}, "training")
                
                # Buffer and game statistics
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
            logger.info(f"Process {self.process_id}: Finalizing resources")
            jax.experimental.multihost_utils.sync_global_devices("pre_close_resources")
            logger.info(f"Process {self.process_id}: Final resource synchronization complete")
            
            if self.is_main_process:
                self.writer.close()
                logger.info(f"Process {self.process_id}: TensorBoard writer closed")

            if self.save_games and hasattr(self, 'game_logger'):
                self.game_logger.stop()
                logger.info(f"Process {self.process_id}: Game logger stopped")

            # Write final summary and close metrics logger
            if self.metrics_logger and self.is_main_process:
                summary_data = {
                    'session_completed': True,
                    'total_iterations': num_iterations,
                    'total_games': self.total_games,
                    'total_positions': self.total_positions,
                    'final_metrics': self.metrics_history[-1] if self.metrics_history else {},
                    'device_info': {
                        'process_id': self.process_id,
                        'num_processes': self.num_processes,
                        'device_type': self.device_type,
                        'num_devices': self.num_devices
                    }
                }
                self.metrics_logger.write_summary_log(summary_data)
                logger.info(f"Process {self.process_id}: Metrics logger summary written")

            if hasattr(self.buffer, 'close'):
                logger.info(f"Process {self.process_id}: Closing buffer")
                self.buffer.close()
                logger.info(f"Process {self.process_id}: Buffer closed")

            total_time = time.time() - start_time_global
            logger.info(f"Process {self.process_id}: Training completed in {total_time:.2f}s")
            
            if self.verbose:
                logger.info(f"\n=== Training Complete ===")
                logger.info(f"Games generated: {self.total_games}")
                logger.info(f"Total positions: {self.total_positions}")
                logger.info(f"Total duration: {total_time:.1f}s ({num_iterations/total_time:.2f} iterations/s)")

    def _generate_games(self, rng_key, num_games):
        """
        Generate games in parallel on local TPU cores.

        Args:
            rng_key: JAX random key
            num_games: Number of games to generate

        Returns:
            Generated game data
        """
        games_per_process = math.ceil(num_games / self.num_processes)
        batch_size_per_device = math.ceil(games_per_process / self.num_devices)

        local_total_games = batch_size_per_device * self.num_devices

        sharded_rngs = jax.random.split(rng_key, self.num_devices)
        sharded_rngs = jax.device_put_sharded(list(sharded_rngs), self.devices)

        sharded_params = jax.device_put_replicated(self.params, self.devices)

        games_data_pmap = self.generate_games_pmap(
            sharded_rngs,
            sharded_params,
            self.network,
            self.env,
            batch_size_per_device,
            self.iteration
        )

        games_data = jax.device_get(games_data_pmap)
        

        self.total_games += local_total_games

        # Log generation metrics
        if self.metrics_logger:
            positions_generated = 0
            total_game_lengths = 0
            games_completed = 0
            
            white_marble_counts = {i: 0 for i in range(7)}  # 0-6 marbles out
            black_marble_counts = {i: 0 for i in range(7)}  # 0-6 marbles out
            
            white_wins = 0
            black_wins = 0
            draws = 0
            
            for device_idx in range(self.num_devices):
                device_data = jax.tree_util.tree_map(lambda x: x[device_idx], games_data)
                games_per_device = len(device_data['moves_per_game'])
                
                for game_idx in range(games_per_device):
                    game_length = int(device_data['moves_per_game'][game_idx])
                    if game_length > 0:
                        games_completed += 1
                        total_game_lengths += game_length
                        positions_generated += game_length + 1  # +1 for initial position
                        
                        # Count marble outs
                        final_white_out = int(device_data['final_white_out'][game_idx])
                        final_black_out = int(device_data['final_black_out'][game_idx])
                        
                        # Ensure counts are within valid range (0-6)
                        final_white_out = min(max(final_white_out, 0), 6)
                        final_black_out = min(max(final_black_out, 0), 6)
                        
                        # Count actual winners
                        if final_black_out >= 6:
                            white_wins += 1  # White won (pushed out 6 black marbles)
                        elif final_white_out >= 6:
                            black_wins += 1  # Black won (pushed out 6 white marbles)
                        else:
                            draws += 1  # Draw or max moves reached
                        
                        white_marble_counts[final_white_out] += 1
                        black_marble_counts[final_black_out] += 1
            
            mean_plays_per_game = total_game_lengths / games_completed if games_completed > 0 else 0
            
            # Calculate proportions
            white_marble_proportions = {k: v / games_completed if games_completed > 0 else 0 
                                      for k, v in white_marble_counts.items()}
            black_marble_proportions = {k: v / games_completed if games_completed > 0 else 0 
                                      for k, v in black_marble_counts.items()}
            
            # Calculate win rates
            total_finished_games = white_wins + black_wins + draws
            white_win_rate = white_wins / total_finished_games if total_finished_games > 0 else 0
            black_win_rate = black_wins / total_finished_games if total_finished_games > 0 else 0
            draw_rate = draws / total_finished_games if total_finished_games > 0 else 0
            
            self.metrics_logger.log_generation_metrics(
                iteration=self.iteration,
                positions_generated=positions_generated,
                games_generated=games_completed,
                games_requested=local_total_games,
                mean_plays_per_game=mean_plays_per_game,
                total_games_so_far=self.total_games,
                total_positions_so_far=self.total_positions,
                white_marble_counts=white_marble_counts,
                black_marble_counts=black_marble_counts,
                white_marble_proportions=white_marble_proportions,
                black_marble_proportions=black_marble_proportions,
                white_wins=white_wins,
                black_wins=black_wins,
                draws=draws,
                white_win_rate=white_win_rate,
                black_win_rate=black_win_rate,
                draw_rate=draw_rate
            )

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
        logger.info(f"Début entrée update buffer")
        positions_added = 0
        
        using_gcs_buffer = hasattr(self.buffer, 'flush_to_gcs')
        
        for device_idx in range(self.num_devices):
            device_data = jax.tree_util.tree_map(
                lambda x: x[device_idx],
                games_data
            )
            
            games_per_device = len(device_data['moves_per_game'])
            for game_idx in range(games_per_device):
                game_length = int(device_data['moves_per_game'][game_idx])
                if game_length == 0:
                    continue
                    
                boards_2d = device_data['boards_2d'][game_idx][:game_length+1]
                policies = device_data['policies'][game_idx][:game_length+1]
                actual_players = device_data['actual_players'][game_idx][:game_length+1]
                black_outs = device_data['black_outs'][game_idx][:game_length+1]
                white_outs = device_data['white_outs'][game_idx][:game_length+1]
                if 'history_2d' in device_data:
                    history_2d = device_data['history_2d'][game_idx][:game_length+1]
                else:
                    history_2d = np.zeros((game_length+1, 8, 9, 9), dtype=np.int8)
                
                final_black_out = device_data['final_black_out'][game_idx]
                final_white_out = device_data['final_white_out'][game_idx]
                
                if final_black_out >= 6:
                    outcome = -1  # White wins
                elif final_white_out >= 6:
                    outcome = 1   # Black wins
                else:
                    outcome = 0   # Draw
                
                if using_gcs_buffer:
                    game_id = self.buffer.start_new_game()
                else:
                    game_id = self.total_games + game_idx
                    
                for move_idx in range(game_length):
                    player = actual_players[move_idx]
                    our_marbles = np.where(player == 1,
                                        black_outs[move_idx],
                                        white_outs[move_idx])
                    opp_marbles = np.where(player == 1,
                                        white_outs[move_idx],
                                        black_outs[move_idx])
                    marbles_out = np.array([our_marbles, opp_marbles], dtype=np.int8)
                    
                    outcome_for_player = outcome * player
                    
                    self.buffer.add(
                        boards_2d[move_idx],
                        marbles_out,
                        policies[move_idx],
                        outcome_for_player,
                        player,
                        history=history_2d[move_idx],  
                        game_id=game_id,
                        move_num=move_idx,
                        iteration=self.iteration,
                        model_version=self.total_games
                    )
                    
                    positions_added += 1
        
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

        using_gcs_buffer = hasattr(self.buffer, 'gcs_index')

        params_sharded = jax.device_put_replicated(self.params, self.devices)
        opt_state_sharded = jax.device_put_replicated(self.opt_state, self.devices)

        cumulative_metrics = None
        steps_completed = 0

        for step in range(num_steps):
            total_batch_size = self.batch_size * self.num_devices

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

            boards = jnp.array(batch_data['board'])
            marbles = jnp.array(batch_data['marbles_out'])
            policies = jnp.array(batch_data['policy'])
            values = jnp.array(batch_data['outcome'])
            if 'history' in batch_data:
                history = jnp.array(batch_data['history'])
            else:
                history = jnp.zeros((boards.shape[0], 8, 9, 9), dtype=jnp.int8)

            boards = boards.reshape(self.num_devices, -1, *boards.shape[1:])
            marbles = marbles.reshape(self.num_devices, -1, *marbles.shape[1:])
            history = history.reshape(self.num_devices, -1, *history.shape[1:])
            policies = policies.reshape(self.num_devices, -1, *policies.shape[1:])
            values = values.reshape(self.num_devices, -1, *values.shape[1:])

            metrics_sharded, grads_averaged = self.train_step_pmap(
                params_sharded, (boards, marbles, history), policies, values
            )

            updates, opt_state_sharded = self.optimizer_update_pmap(
                grads_averaged, opt_state_sharded, params_sharded
            )
            params_sharded = jax.tree.map(lambda p, u: p + u, params_sharded, updates)

            step_metrics = {k: float(jnp.mean(v)) for k, v in metrics_sharded.items()}

            if cumulative_metrics is None:
                cumulative_metrics = step_metrics
            else:
                cumulative_metrics = {k: cumulative_metrics[k] + step_metrics[k] for k in step_metrics}
                
            steps_completed += 1

        self.params = jax.tree.map(lambda x: x[0], params_sharded)
        self.opt_state = jax.tree.map(lambda x: x[0], opt_state_sharded)

        if cumulative_metrics is None or steps_completed == 0:
            return {'total_loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0,
                    'policy_accuracy': 0.0, 'value_sign_match': 0.0}

        avg_metrics = {k: v / steps_completed for k, v in cumulative_metrics.items()}

        if self.is_main_process:
            self._log_metrics_to_tensorboard(avg_metrics, "training")
            self._log_metrics_to_tensorboard({"learning_rate": self.current_lr}, "training")

            buffer_stats = {}
            if using_gcs_buffer:
                buffer_stats["buffer_size_total"] = self.buffer.total_size
                buffer_stats["buffer_size_local"] = self.buffer.local_size
            else:
                buffer_stats["buffer_size"] = self.buffer.size
            
            buffer_stats["total_games_local"] = self.total_games
            buffer_stats["total_games_global"] = self.total_games * self.num_processes
            self._log_metrics_to_tensorboard(buffer_stats, "stats")

        local_metrics_record = avg_metrics.copy()
        local_metrics_record['iteration'] = self.iteration
        local_metrics_record['learning_rate'] = self.current_lr
        
        if using_gcs_buffer:
            local_metrics_record['buffer_size_total'] = self.buffer.total_size
            local_metrics_record['buffer_size_local'] = self.buffer.local_size
        else:
            local_metrics_record['buffer_size'] = self.buffer.size
            
        local_metrics_record['total_games_local'] = self.total_games
        self.metrics_history.append(local_metrics_record)

        # Log training metrics
        if self.metrics_logger:
            # Add buffer information
            buffer_info = {}
            if using_gcs_buffer:
                buffer_info['buffer_size_total'] = self.buffer.total_size
                buffer_info['buffer_size_local'] = self.buffer.local_size
            else:
                buffer_info['buffer_size'] = self.buffer.size
            
            self.metrics_logger.log_training_metrics(
                iteration=self.iteration,
                total_loss=avg_metrics.get('total_loss', 0.0),
                policy_loss=avg_metrics.get('policy_loss', 0.0),
                value_loss=avg_metrics.get('value_loss', 0.0),
                policy_accuracy=avg_metrics.get('policy_accuracy', 0.0),
                value_sign_match=avg_metrics.get('value_sign_match', 0.0),
                learning_rate=self.current_lr,
                training_steps_completed=steps_completed,
                training_steps_requested=num_steps,
                **buffer_info
            )

        return avg_metrics
    

    def _save_checkpoint(self, is_final=False, is_reference=False):
        """Sauvegarde un point de contrôle du modèle"""
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

        is_gcs = self.checkpoint_path.startswith('gs://')

        if is_gcs:
            import subprocess

            local_path = f"/tmp/{prefix}.pkl"
            with open(local_path, 'wb') as f:
                pickle.dump(checkpoint, f)

            if hasattr(self, 'training_timestamp'):
                timestamp = self.training_timestamp
            else:
                self.training_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                timestamp = self.training_timestamp
                
            base_path = self.checkpoint_path
            if '_' in base_path and base_path.count('_') > 1:
                parts = base_path.split('_')
                base_path = '_'.join(parts[:-1]) if parts[-1].replace('-', '').isdigit() else base_path
                
            gcs_path = f"{base_path}_{timestamp}_{prefix}.pkl"
            subprocess.run(f"gsutil cp {local_path} {gcs_path}", shell=True)
            
            if self.verbose:
                checkpoint_type = "de référence" if is_reference else "standard"
                logger.info(f"Checkpoint {checkpoint_type} sauvegardé: {gcs_path}")

            os.remove(local_path)
        else:
            if hasattr(self, 'training_timestamp'):
                timestamp = self.training_timestamp
            else:
                self.training_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                timestamp = self.training_timestamp
                
            base_path = self.checkpoint_path
            if '_' in base_path and base_path.count('_') > 1:
                parts = base_path.split('_')
                base_path = '_'.join(parts[:-1]) if parts[-1].replace('-', '').isdigit() else base_path
                
            filename = f"{base_path}_{timestamp}_{prefix}.pkl"
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
        is_gcs = checkpoint_path.startswith('gs://')

        if is_gcs:
            import subprocess

            local_path = "/tmp/checkpoint.pkl"
            subprocess.run(f"gsutil cp {checkpoint_path} {local_path}", shell=True)

            with open(local_path, 'rb') as f:
                checkpoint = pickle.load(f)

            os.remove(local_path)
        else:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)

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

        target_references = generate_evaluation_checkpoints(total_iterations, num_reference_models)


        sync_refs = [ref for ref in target_references if ref < current_iter]
        
        available_refs = []
        for ref in sync_refs:
            ref_path = self._get_checkpoint_path(ref)
            if check_checkpoint_exists(ref_path):
                available_refs.append(ref)

        jax.experimental.multihost_utils.sync_global_devices("pre_evaluation")

        local_results = {}

        evaluator = None
        current_params = self.params
        local_games_per_model = 0

        if not available_refs:
            if self.verbose:
                logger.info("Aucun modèle précédent disponible pour l'évaluation")
        else:
            if len(available_refs) == 1:
                games_per_model = max(16, self.num_processes * 2)  
            elif len(available_refs) < 4:
                games_per_model = 32  
            else:
                games_per_model = 16  

            local_games_per_model = games_per_model // self.num_processes
            
            if self.verbose:
                logger.info(f"\n=== Évaluation contre modèles précédents (itération actuelle: {current_iter}) ===")
                logger.info(f"Itérations sélectionnées: {available_refs}")
                logger.info(f"Parties totales par modèle: {games_per_model}")
            logger.info(f"Processus {self.process_id}: jouera {local_games_per_model} parties par modèle")

            evaluator = ModelsEvaluator(
                network=self.network,
                num_simulations=500,  
                games_per_model=games_per_model
            )

      
        for ref_iter in sync_refs:

            jax.experimental.multihost_utils.sync_global_devices(f"pre_eval_iter_{ref_iter}")

            if ref_iter in available_refs:
                if self.verbose:
                    logger.info(f"\nÉvaluation contre le modèle de l'itération {ref_iter}...")

                ref_path = self._get_checkpoint_path(ref_iter)
                local_path = f"/tmp/ref_model_{ref_iter}.pkl"
                eval_success = False
                
                if ref_path.startswith("gs://"):
                    if not download_checkpoint(ref_path, local_path):
                        if self.verbose:
                            logger.info(f"Échec du téléchargement du checkpoint pour l'itération {ref_iter}, on passe")
                    else:
                        eval_success = True
                else:
                    local_path = ref_path
                    eval_success = True

                if eval_success:
                    ref_params = load_checkpoint_params(local_path)
                    if ref_params is None:
                        if self.verbose:
                            logger.info(f"Échec du chargement des paramètres pour l'itération {ref_iter}, on passe")
                        eval_success = False
                    else:
                        if evaluator is not None:
                            eval_results = evaluator.evaluate_model_pair(
                                current_params,
                                ref_params,
                                games_to_play=local_games_per_model
                            )
                            local_results[ref_iter] = eval_results
            else:

                if self.verbose:
                    logger.info(f"Modèle itération {ref_iter} non disponible pour ce worker, participation sync seulement")
            
            jax.experimental.multihost_utils.sync_global_devices(f"post_eval_iter_{ref_iter}")
        
        jax.experimental.multihost_utils.sync_global_devices("pre_aggregation")
        
        all_results = self._aggregate_evaluation_results(local_results, sync_refs)

        jax.experimental.multihost_utils.sync_global_devices("post_aggregation")

        if self.verbose and all_results:
            total_wins = sum(res['current_wins'] for res in all_results.values())
            total_games = sum(res['total_games'] for res in all_results.values())
            global_win_rate = total_wins / total_games if total_games > 0 else 0
            
            for ref_iter, ref_results in all_results.items():
                win_rate = ref_results['win_rate']
                logger.info(f"Résultats vs iter {ref_iter}: {win_rate:.1%} taux de victoire")
                logger.info(f"  Victoires: {ref_results['current_wins']}, Défaites: {ref_results['reference_wins']}, Nuls: {ref_results['draws']}")

                self._log_metrics_to_tensorboard({
                    f"model_comparison/iter{current_iter}_vs_iter{ref_iter}": win_rate,
                }, "eval_results")
            
            self._log_metrics_to_tensorboard({
                f"global_performance/iter{current_iter}": global_win_rate,
                f"global_performance/wins": total_wins,
                f"global_performance/games": total_games,
            }, "eval_results")
            
            logger.info(f"Win rate global: {global_win_rate:.1%} ({total_wins}/{total_games} victoires)")
            
            if self.metrics_history and current_iter > 0:
                latest_metrics = self.metrics_history[-1]
                latest_metrics['global_win_rate'] = global_win_rate
                
                for ref_iter, ref_results in all_results.items():
                    latest_metrics[f'win_rate_vs_iter{ref_iter}'] = ref_results['win_rate']
                    
            logger.info(f"\n=== Évaluation terminée ===")
        
        if all_results and self.metrics_logger and self.is_main_process:
            total_wins = sum(res['current_wins'] for res in all_results.values())
            total_games = sum(res['total_games'] for res in all_results.values())
            global_win_rate = total_wins / total_games if total_games > 0 else 0
            
            eval_metrics = {
                'global_win_rate': global_win_rate,
                'total_eval_games': total_games,
                'total_eval_wins': total_wins,
                'num_models_evaluated': len(all_results)
            }
            
            for ref_iter, ref_results in all_results.items():
                eval_metrics[f'win_rate_vs_iter{ref_iter}'] = ref_results['win_rate']
            
            self.metrics_logger.log_evaluation_metrics(
                iteration=current_iter,
                **eval_metrics
            )
        
        jax.experimental.multihost_utils.sync_global_devices("post_evaluation")
            
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
        aggregated_data = {}

        for ref_iter in model_iterations:
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

        if not model_iterations:
            return {}
        all_models_data = jnp.stack([aggregated_data[ref] for ref in model_iterations])

        replicated_data = jnp.repeat(all_models_data[None, :, :], self.num_devices, axis=0)
        devices_data = jax.device_put_sharded(list(replicated_data), self.devices)

        summed_data = self.sum_across_devices(devices_data)

        global_results = jax.device_get(summed_data)[0]

        final_results = {}
        for i, ref_iter in enumerate(model_iterations):
            total_games = int(global_results[i][0])
            if total_games > 0:  
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
        
        base_path = self.checkpoint_path
        if '_' in base_path and base_path.count('_') > 1:
            parts = base_path.split('_')
            base_path = '_'.join(parts[:-1]) if parts[-1].replace('-', '').isdigit() else base_path
        
        return f"{base_path}_*_{prefix}.pkl"
