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
from evaluation.evaluator import evaluate_model
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
                games_flush_interval=300):
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
        try:
            # Try TPU first (Google Cloud)
            self.devices = jax.devices('tpu')
            self.device_type = 'tpu'
        except RuntimeError:
            try:
                # Then GPU
                self.devices = jax.devices('gpu')
                self.device_type = 'gpu'
            except RuntimeError:
                # Finally, use CPU
                self.devices = jax.devices('cpu')
                self.device_type = 'cpu'
        
        self.num_devices = len(self.devices)
        print(f"Using {self.num_devices} {self.device_type.upper()} cores in step-synchronized mode")
            
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
        
        # Initialize SGD optimizer
        self.optimizer = optax.sgd(learning_rate=self.initial_lr, momentum=self.momentum)
        
        # Initialize parameters and optimization state
        rng = jax.random.PRNGKey(42)
        sample_board = jnp.zeros((1, 9, 9), dtype=jnp.int8)
        sample_marbles = jnp.zeros((1, 2), dtype=jnp.int8)
        self.params = network.init(rng, sample_board, sample_marbles)
        self.opt_state = self.optimizer.init(self.params)
        
        # Create replay buffer
        self.buffer = CPUReplayBuffer(buffer_size)
        
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
            
        print(f"TensorBoard logs: {self.log_dir}")
        self.writer = SummaryWriter(self.log_dir)
        
        # Initialize game logger
        if self.save_games:
            if gcs_bucket:
                print(f"Storing games in GCS: {gcs_bucket}")
                self.game_logger = GameLogger(
                    bucket_name=gcs_bucket,
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
                axis_name='batch',
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
        
        # Parallel training function
        self.train_step_pmap = jax.pmap(
            partial(train_step_pmap_impl, network=self.network, value_weight=self.value_weight),
            axis_name='batch',
            devices=self.devices
        )
        
        # Parameter update function with optimizer
        self.optimizer_update_pmap = jax.pmap(
            lambda g, o, p: self.optimizer.update(g, o, p),
            axis_name='batch',
            devices=self.devices
        )
      
    def train(self, num_iterations=100, games_per_iteration=64, 
             training_steps_per_iteration=100, eval_frequency=10, 
             save_frequency=10):
        """
        Start training with step approach.
        
        Args:
            num_iterations: Total number of iterations
            games_per_iteration: Number of games to generate per iteration
            training_steps_per_iteration: Number of training steps per iteration
            eval_frequency: Evaluation frequency (in iterations)
            save_frequency: Save frequency (in iterations)
        """
        # Initialize global timer
        start_time_global = time.time()
        
        # Main RNG
        rng_key = jax.random.PRNGKey(42)
        
        try:
            for iteration in range(num_iterations):
                self.iteration = iteration
                
                # Update learning rate according to schedule
                iteration_percentage = iteration / num_iterations
                self._update_learning_rate(iteration_percentage)
                
                print(f"\n=== Iteration {iteration+1}/{num_iterations} (LR: {self.current_lr}) ===")
                
                # 1. Generation phase
                rng_key, gen_key = jax.random.split(rng_key)
                t_start = time.time()
                
                # Generate requested number of games
                games_data = self._generate_games(gen_key, games_per_iteration)
                
                t_gen = time.time() - t_start
                print(f"Generation: {games_per_iteration} games in {t_gen:.2f}s ({games_per_iteration/t_gen:.1f} games/s)")
                
                # 2. Buffer update
                t_start = time.time()
                positions_added = self._update_buffer(games_data)
                t_buffer = time.time() - t_start
                
                self.total_positions += positions_added
                print(f"Buffer updated: +{positions_added} positions (total: {self.buffer.size})")
                
                # 3. Training phase
                rng_key, train_key = jax.random.split(rng_key)
                t_start = time.time()
                
                metrics = self._train_network(train_key, training_steps_per_iteration)
                
                t_train = time.time() - t_start
                print(f"Training: {training_steps_per_iteration} steps in {t_train:.2f}s ({training_steps_per_iteration/t_train:.1f} steps/s)")
                
                # Display metrics
                print(f"  Total loss: {metrics['total_loss']:.4f}")
                print(f"  Policy loss: {metrics['policy_loss']:.4f}, Value loss: {metrics['value_loss']:.4f}")
                print(f"  Policy accuracy: {metrics['policy_accuracy']:.2%}")
                
                # 4. Periodic evaluation
                if eval_frequency > 0 and (iteration + 1) % eval_frequency == 0:
                    self._evaluate()
                
                # 5. Periodic saving
                if save_frequency > 0 and (iteration + 1) % save_frequency == 0:
                    self._save_checkpoint()
            
            # Final save
            self._save_checkpoint(is_final=True)
            
        finally:
            # Ensure resources are released
            self.writer.close()
            if self.save_games and hasattr(self, 'game_logger'):
                self.game_logger.stop()
            
            # Global statistics
            total_time = time.time() - start_time_global
            print(f"\n=== Training completed ===")
            print(f"Games generated: {self.total_games}")
            print(f"Total positions: {self.total_positions}")
            print(f"Total duration: {total_time:.1f}s ({num_iterations/total_time:.2f} iterations/s)")

    
    def _generate_games(self, rng_key, num_games):
        """
        Generate games in parallel on all TPU cores with optimized version.
        
        Args:
            rng_key: JAX random key
            num_games: Number of games to generate
            
        Returns:
            Generated games data
        """
        # Determine how many games per core
        batch_size_per_device = math.ceil(num_games / self.num_devices)
        total_games = batch_size_per_device * self.num_devices
        
        # Total time measurement
        t_total_start = time.time()
        
        # Prepare RNGs for each core and distribute directly
        t_prep_start = time.time()
        sharded_rngs = jax.random.split(rng_key, self.num_devices)
        sharded_rngs = jax.device_put_sharded(list(sharded_rngs), self.devices)
        
        # Replicate parameters directly on devices
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
        
        # Update counter
        self.total_games += total_games
        
        # Calculate and display times
        t_prep = t_prep_end - t_prep_start
        t_gen = t_gen_end - t_gen_start
        t_fetch = t_fetch_end - t_fetch_start
        t_total = time.time() - t_total_start
        
        print(f"  Preparation: {t_prep:.3f}s")
        print(f"  Generation: {t_gen:.3f}s ({total_games/t_gen:.1f} games/s)")
        print(f"  Retrieval: {t_fetch:.3f}s ({t_fetch/total_games*1000:.1f} ms/game)")
        
        # Record games for analysis if enabled
        if self.save_games and hasattr(self, 'game_logger'):
            t_convert_start = time.time()
            
            # Convert games to a format suitable for analysis
            converted_games = convert_games_batch(
                games_data,
                self.env,
                base_game_id=f"iter{self.iteration}",
                model_iteration=self.iteration
            )
            
            # Send games to logger which will write them asynchronously
            self.game_logger.log_games_batch(converted_games)
            
            t_convert = time.time() - t_convert_start
            print(f"  Conversion for storage: {t_convert:.3f}s")
        
        return games_data
    
    def _update_buffer(self, games_data):
        """Update buffer with newly generated games"""
        positions_added = 0
        
        # For each device
        for device_idx in range(self.num_devices):
            device_data = jax.tree_util.tree_map(
                lambda x: x[device_idx],
                games_data
            )
            
            # For each game generated on this device
            games_per_device = len(device_data['moves_per_game'])
            for game_idx in range(games_per_device):
                game_length = int(device_data['moves_per_game'][game_idx])
                if game_length == 0:
                    continue
                    
                # Extract data for this game
                boards_2d = device_data['boards_2d'][game_idx][:game_length+1]
                policies = device_data['policies'][game_idx][:game_length+1]
                actual_players = device_data['actual_players'][game_idx][:game_length+1]
                black_outs = device_data['black_outs'][game_idx][:game_length+1]
                white_outs = device_data['white_outs'][game_idx][:game_length+1]
                
                # Determine final result
                final_black_out = device_data['final_black_out'][game_idx]
                final_white_out = device_data['final_white_out'][game_idx]
                
                if final_black_out >= 6:
                    outcome = -1  # White wins
                elif final_white_out >= 6:
                    outcome = 1   # Black wins
                else:
                    outcome = 0   # Draw
                    
                # Add each position to buffer
                for move_idx in range(game_length):
                    # Calculate marbles out for current player
                    player = actual_players[move_idx]
                    our_marbles = np.where(player == 1,
                                        black_outs[move_idx],
                                        white_outs[move_idx])
                    opp_marbles = np.where(player == 1,
                                        white_outs[move_idx],
                                        black_outs[move_idx])
                    marbles_out = np.array([our_marbles, opp_marbles], dtype=np.int8)
                    
                    # Adjust for current player's perspective
                    outcome_for_player = outcome * player
                    
                    # Store in buffer with metadata
                    self.buffer.add(
                        boards_2d[move_idx],
                        marbles_out,
                        policies[move_idx],
                        outcome_for_player,
                        player,
                        game_id=self.total_games + game_idx,
                        move_num=move_idx,
                        iteration=self.iteration,
                        model_version=self.total_games  # Use total_games as proxy for version
                    )
                    
                    positions_added += 1
                    
        return positions_added
    
    def _train_network(self, rng_key, num_steps):
        """
        Train the network on batches from the buffer.
        
        Args:
            rng_key: JAX random key
            num_steps: Number of training steps
            
        Returns:
            Average metrics over all steps
        """
        # Cumulative metrics
        cumulative_metrics = None
        
        for step in range(num_steps):
            # Sample a large batch for parallelization
            total_batch_size = self.batch_size * self.num_devices
            
            # Use recency-biased sampling if enabled
            if self.recency_bias:
                batch_data = self.buffer.sample_with_recency_bias(
                    total_batch_size, 
                    temperature=self.recency_temperature, 
                    rng_key=rng_key
                )
            else:
                batch_data = self.buffer.sample(total_batch_size, rng_key)
                
            rng_key = jax.random.fold_in(rng_key, step)
            
            # Convert to JAX arrays
            boards = jnp.array(batch_data['board'])
            marbles = jnp.array(batch_data['marbles_out'])
            policies = jnp.array(batch_data['policy'])
            values = jnp.array(batch_data['outcome'])
            
            # Split data into chunks for each core
            boards = boards.reshape(self.num_devices, -1, *boards.shape[1:])
            marbles = marbles.reshape(self.num_devices, -1, *marbles.shape[1:])
            policies = policies.reshape(self.num_devices, -1, *policies.shape[1:])
            values = values.reshape(self.num_devices, -1, *values.shape[1:])
            
            # Replicate parameters for pmap
            params_sharded = jax.device_put_replicated(self.params, self.devices)
            opt_state_sharded = jax.device_put_replicated(self.opt_state, self.devices)
            
            # Execute parallel training step
            loss, grads = self.train_step_pmap(params_sharded, (boards, marbles), policies, values)
            
            # Update parameters with optimizer
            updates, new_opt_state = self.optimizer_update_pmap(grads, opt_state_sharded, params_sharded)
            new_params = jax.tree_map(lambda p, u: p + u, params_sharded, updates)
            
            # Retrieve results from first device
            self.params = jax.tree_map(lambda x: x[0], new_params)
            self.opt_state = jax.tree_map(lambda x: x[0], new_opt_state)
            
            # Aggregate metrics
            step_metrics = {k: float(jnp.mean(v)) for k, v in loss.items()}
            
            if cumulative_metrics is None:
                cumulative_metrics = step_metrics
            else:
                cumulative_metrics = {k: cumulative_metrics[k] + step_metrics[k] for k in step_metrics}
        
        # Calculate average metrics
        avg_metrics = {k: v / num_steps for k, v in cumulative_metrics.items()}

        for metric_name, metric_value in avg_metrics.items():
            self.writer.add_scalar(f"training/{metric_name}", metric_value, self.iteration)
        
        # Add current learning rate
        self.writer.add_scalar("training/learning_rate", self.current_lr, self.iteration)
        self.writer.add_scalar("stats/buffer_size", self.buffer.size, self.iteration)
        self.writer.add_scalar("stats/total_games", self.total_games, self.iteration)
        
        # Record metrics
        avg_metrics['iteration'] = self.iteration
        avg_metrics['learning_rate'] = self.current_lr
        avg_metrics['buffer_size'] = self.buffer.size
        avg_metrics['total_games'] = self.total_games
        self.metrics_history.append(avg_metrics)
        
        return avg_metrics
    
    
    def _evaluate(self):
        """Evaluate current model against classical algorithms."""
        results = evaluate_model(self)
        
        # Record evaluation results
        for algo_name, data in results.items():
            self.writer.add_scalar(f"evaluation/win_rate_{algo_name}", data["win_rate"], self.iteration)
        
        return results
    
    def _save_checkpoint(self, is_final=False):
        """
        Save a model checkpoint
        
        Args:
            is_final: If True, indicates this is the final checkpoint
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