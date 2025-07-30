import numpy as np
import jax
import jax.numpy as jnp
import time
import os
import tensorflow as tf
from google.cloud import storage
from typing import Dict, List, Tuple, Any, Optional
import math
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Process %(process)d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("alphazero.buffer")

class CPUReplayBuffer:
    def __init__(self, capacity, board_size=9, action_space=1734, history_length=8):
        self.capacity = capacity
        self.size = 0
        self.position = 0
        self.history_length = history_length

        self.buffer = {
            'board': np.zeros((capacity, board_size, board_size), dtype=np.int8),
            'marbles_out': np.zeros((capacity, 2), dtype=np.int8),
            'policy': np.zeros((capacity, action_space), dtype=np.float32),
            'outcome': np.zeros(capacity, dtype=np.int8),
            'player': np.zeros(capacity, dtype=np.int8),
            'history': np.zeros((capacity, history_length, board_size, board_size), dtype=np.int8),  # 2D history
            'game_id': np.zeros(capacity, dtype=np.int32),  # Unique game ID
            'move_num': np.zeros(capacity, dtype=np.int16),  # Move number in game
            'iteration': np.zeros(capacity, dtype=np.int32),  # Training iteration
            'model_version': np.zeros(capacity, dtype=np.int32)  # Model version
        }

        self.current_game_id = 0  

    def add(self, board, marbles_out, policy, outcome, player, history=None, game_id=None, move_num=0, 
            iteration=0, model_version=0):
        """Add individual transition to buffer"""
        idx = self.position

        # Convert to numpy if needed
        if hasattr(board, 'device'):  # Detect if it's a JAX array
            board = np.array(board)
        if hasattr(marbles_out, 'device'):
            marbles_out = np.array(marbles_out)
        if hasattr(policy, 'device'):
            policy = np.array(policy)
        if hasattr(history, 'device') and history is not None:
            history = np.array(history)

        # If game_id not provided, increment internal counter
        if game_id is None:
            game_id = self.current_game_id

        # Store data
        self.buffer['board'][idx] = board
        self.buffer['marbles_out'][idx] = marbles_out
        self.buffer['policy'][idx] = policy
        self.buffer['outcome'][idx] = outcome
        self.buffer['player'][idx] = player
        self.buffer['game_id'][idx] = game_id
        self.buffer['move_num'][idx] = move_num
        self.buffer['iteration'][idx] = iteration
        self.buffer['model_version'][idx] = model_version
        
        # Store history if provided
        if history is not None:
            self.buffer['history'][idx] = history
        else:
            # Empty history by default
            self.buffer['history'][idx] = np.zeros((self.history_length, board.shape[0], board.shape[1]), dtype=np.int8)

        # Update counters
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, batch):
        """Add batch of transitions"""
        batch_size = batch['board'].shape[0]

        # Check for optional fields
        has_game_id = 'game_id' in batch
        has_move_num = 'move_num' in batch
        has_iteration = 'iteration' in batch
        has_model_version = 'model_version' in batch
        has_history = 'history' in batch

        for i in range(batch_size):
            game_id = batch['game_id'][i] if has_game_id else None
            move_num = batch['move_num'][i] if has_move_num else 0
            iteration = batch['iteration'][i] if has_iteration else 0
            model_version = batch['model_version'][i] if has_model_version else 0
            history = batch['history'][i] if has_history else None

            self.add(
                batch['board'][i],
                batch['marbles_out'][i],
                batch['policy'][i],
                batch['outcome'][i],
                batch['player'][i],
                history=history,
                game_id=game_id,
                move_num=move_num,
                iteration=iteration,
                model_version=model_version
            )

    def start_new_game(self):
        """Incrémente l'ID de partie pour commencer une nouvelle partie"""
        self.current_game_id += 1
        return self.current_game_id

    def sample(self, batch_size, rng_key=None):
        """Échantillonne un batch aléatoire"""
        if self.size == 0:
            raise ValueError("Buffer vide, impossible d'échantillonner")

        if rng_key is None:
            indices = np.random.randint(0, self.size, size=batch_size)
        else:
            indices = jax.random.randint(
                rng_key, shape=(batch_size,), minval=0, maxval=self.size
            ).astype(np.int32)
            indices = np.array(indices)

        batch = {
            'board': self.buffer['board'][indices],
            'marbles_out': self.buffer['marbles_out'][indices],
            'policy': self.buffer['policy'][indices],
            'history': self.buffer['history'][indices],
            'outcome': self.buffer['outcome'][indices],
            'player': self.buffer['player'][indices],
            'iteration': self.buffer['iteration'][indices],
            'model_version': self.buffer['model_version'][indices]
        }

        return batch

    def sample_with_recency_bias(self, batch_size, temperature=1.0, rng_key=None):
        """Échantillonne avec priorité aux données récentes"""
        if self.size == 0:
            raise ValueError("Buffer vide, impossible d'échantillonner")

        # Calculate weights based on recency
        if self.position == 0 and self.size == self.capacity:
            # Full circular buffer, position 0 is most recent
            indices = np.arange(self.size)
        else:
            # Position is index of next element to write
            indices = np.arange(self.size)
            # Reorganize so higher indices are most recent
            indices = (indices + self.capacity - self.position) % self.capacity

        # Higher indices correspond to more recent entries
        recency_weights = np.exp((indices / self.size) * temperature)
        sampling_probs = recency_weights / np.sum(recency_weights)

        # Sample with these probabilities
        if rng_key is None:
            sampled_indices = np.random.choice(
                self.size, size=batch_size, p=sampling_probs, replace=True
            )
        else:
            sampled_indices = np.array(jax.random.choice(
                rng_key, self.size, shape=(batch_size,), p=sampling_probs, replace=True
            ))

        # Retrieve samples
        actual_indices = np.arange(self.size)[sampled_indices]
 
        batch = {
            'board': self.buffer['board'][actual_indices],
            'marbles_out': self.buffer['marbles_out'][actual_indices],
            'policy': self.buffer['policy'][actual_indices],
            'history': self.buffer['history'][actual_indices],
            'outcome': self.buffer['outcome'][actual_indices],
            'player': self.buffer['player'][actual_indices],
            'iteration': self.buffer['iteration'][actual_indices],
            'model_version': self.buffer['model_version'][actual_indices]
        }

        return batch
    

logger = logging.getLogger("alphazero.buffer")
class GCSReplayBufferSync:
    """
    Synchronous experience buffer using Google Cloud Storage as primary storage.
    - Enables experience sharing between multiple TPU nodes
    - Maintains fixed size with recency-based sampling
    - Synchronous version for simplicity and reliability
    """
    def __init__(self, 
                bucket_name: str,
                buffer_dir: str = 'buffer',
                max_local_size: int = 10000,
                max_buffer_size: int = 20_000_000,
                buffer_cleanup_threshold: float = 0.95,
                board_size: int = 9,
                action_space: int = 1734,
                history_length: int = 8,
                recency_enabled: bool = True,
                recency_temperature: float = 0.8,
                cleanup_temperature: float = 2.0,
                log_level: str = 'INFO'):
        """
        Initialize synchronous experience buffer based on GCS.
        
        Args:
            bucket_name: GCS bucket name
            buffer_dir: Folder in bucket to store data
            max_local_size: Maximum local cache size
            max_buffer_size: Maximum global buffer size (in number of positions)
            buffer_cleanup_threshold: Fill threshold triggering cleanup (between 0 and 1)
            board_size: Board size (default: 9 for Abalone 2D)
            action_space: Number of possible actions
            recency_enabled: Enable sampling with recency bias
            recency_temperature: Temperature for recency bias for sampling
            cleanup_temperature: Temperature for sampling during cleanup
            log_level: Logging level ('INFO', 'DEBUG', 'WARNING')
        """
        self.bucket_name = bucket_name
        self.buffer_dir = buffer_dir
        self.max_local_size = max_local_size
        self.max_buffer_size = max_buffer_size
        self.buffer_cleanup_threshold = buffer_cleanup_threshold
        self.board_size = board_size
        self.action_space = action_space
        self.history_length = history_length
        self.recency_enabled = recency_enabled
        self.recency_temperature = recency_temperature
        self.cleanup_temperature = cleanup_temperature
        
        # Configure log level
        self.verbose = log_level == 'DEBUG'
        self.log_level = log_level
        
        # Process and host identifier to avoid conflicts
        self.process_id = jax.process_index()
        self.host_id = f"{os.uname().nodename}_{self.process_id}"
        
        # Local data cache
        self.local_buffer = {
            'board': np.zeros((max_local_size, board_size, board_size), dtype=np.int8),
            'marbles_out': np.zeros((max_local_size, 2), dtype=np.int8),
            'policy': np.zeros((max_local_size, action_space), dtype=np.float32),
            'outcome': np.zeros(max_local_size, dtype=np.int8),
            'player': np.zeros(max_local_size, dtype=np.int8),
            'history': np.zeros((max_local_size, history_length, board_size, board_size), dtype=np.int8),
            'game_id': np.zeros(max_local_size, dtype=np.int32),
            'move_num': np.zeros(max_local_size, dtype=np.int16),
            'iteration': np.zeros(max_local_size, dtype=np.int32),
            'model_version': np.zeros(max_local_size, dtype=np.int32)
        }
        
        # Buffer metadata
        self.local_size = 0
        self.position = 0
        self.current_game_id = 0
        self.total_size = 0  # Total size including GCS
        
        # Initialize GCS client
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        
        # Index of available data on GCS
        self.gcs_index = {}
        self.gcs_file_metadata = {}
        self.last_index_update = 0
        self.index_update_interval = 30  # Seconds before forcing index update
        
        # Initialize index from GCS
        self._update_gcs_index()
        
        # Statistics
        self.metrics = {
            "samples_served": 0,
            "files_added": 0,
            "files_removed": 0,
            "cleanup_operations": 0
        }
        
        logger.info(f"GCSReplayBufferSync initialized - Max buffer size: {self.max_buffer_size} positions")
    
    def add(self, board, marbles_out, policy, outcome, player, history=None, game_id=None, move_num=0, 
            iteration=0, model_version=0):
        """Add individual transition to buffer"""
        # Convert to numpy if needed
        if hasattr(board, 'device'):
            board = np.array(board)
        if hasattr(marbles_out, 'device'):
            marbles_out = np.array(marbles_out)
        if hasattr(policy, 'device'):
            policy = np.array(policy)
        if hasattr(history, 'device') and history is not None:
            history = np.array(history)
        
        # If game_id not provided, increment internal counter
        if game_id is None:
            game_id = self.current_game_id
        
        # Store in local cache
        idx = self.position
        self.local_buffer['board'][idx] = board
        self.local_buffer['marbles_out'][idx] = marbles_out
        self.local_buffer['policy'][idx] = policy
        self.local_buffer['outcome'][idx] = outcome
        self.local_buffer['player'][idx] = player
        self.local_buffer['game_id'][idx] = game_id
        self.local_buffer['move_num'][idx] = move_num
        self.local_buffer['iteration'][idx] = iteration
        self.local_buffer['model_version'][idx] = model_version
        
        # Store history if provided
        if history is not None:
            self.local_buffer['history'][idx] = history
        else:
            # Empty history by default
            self.local_buffer['history'][idx] = np.zeros((self.history_length, board.shape[0], board.shape[1]), dtype=np.int8)
        
        # Update counters
        self.position = (self.position + 1) % self.max_local_size
        self.local_size = min(self.local_size + 1, self.max_local_size)
        self.total_size += 1
    
    def add_batch(self, batch):
        """Add batch of transitions"""
        batch_size = batch['board'].shape[0]
        
        # Check for optional fields
        has_game_id = 'game_id' in batch
        has_move_num = 'move_num' in batch
        has_iteration = 'iteration' in batch
        has_model_version = 'model_version' in batch
        has_history = 'history' in batch
        
        for i in range(batch_size):
            game_id = batch['game_id'][i] if has_game_id else None
            move_num = batch['move_num'][i] if has_move_num else 0
            iteration = batch['iteration'][i] if has_iteration else 0
            model_version = batch['model_version'][i] if has_model_version else 0
            history = batch['history'][i] if has_history else None
            
            self.add(
                batch['board'][i],
                batch['marbles_out'][i],
                batch['policy'][i],
                batch['outcome'][i],
                batch['player'][i],
                history=history,
                game_id=game_id,
                move_num=move_num,
                iteration=iteration,
                model_version=model_version
            )
    
    def flush_to_gcs(self):
        """Écrit synchroniquement le contenu du buffer local sur GCS."""
        if self.local_size == 0:
            return 0  
        
        logger.info(f"Début du flush vers GCS: {self.local_size} positions à écrire")
        # else:
            
        #     logger.info(f"Flush vers GCS: {self.local_size} positions")
        
        # Prepare local buffer data
        data_to_write = {}
        for key in self.local_buffer:
            data_to_write[key] = self.local_buffer[key][:self.local_size].copy()
        
        # Counter for written positions
        total_written = 0
        files_created = 0
        
        # Generate unique batch ID
        timestamp = int(time.time())
        batch_id = f"{self.host_id}_{timestamp}"
        
        iterations = np.unique(data_to_write['iteration'])
        
        # Write data for each iteration
        for iteration in iterations:
            # Filter data for this iteration
            iter_mask = data_to_write['iteration'] == iteration
            if not np.any(iter_mask):
                continue
            
            # Create subset for this iteration
            iter_data = {k: v[iter_mask] for k, v in data_to_write.items()}
            positions_in_iter = iter_data['board'].shape[0]
            
            # Create path in bucket
            iter_path = f"{self.buffer_dir}/iteration_{iteration}"
            file_path = f"{iter_path}/{batch_id}.tfrecord"
            
            
            logger.info(f"Writing {positions_in_iter} positions for iteration {iteration}")
            
            # Write in TFRecord format
            example_count = self._write_tfrecord(file_path, iter_data)
            total_written += example_count
            files_created += 1
            
            # Update local index
            if iteration not in self.gcs_index:
                self.gcs_index[iteration] = []
            self.gcs_index[iteration].append(file_path)
            
            # Store file metadata
            self.gcs_file_metadata[file_path] = {
                'size': example_count,
                'timestamp': timestamp,
                'iteration': iteration
            }
            
            self.metrics["files_added"] += 1
        
        # Reset local buffer after writing
        self.local_size = 0
        self.position = 0
        
        # Update total buffer size
        self._update_total_size()
        
        # Update index after writing
        self._update_gcs_index(force=False)  # Light update
            
        # Check if cleanup necessary
        if self.total_size > self.max_buffer_size * self.buffer_cleanup_threshold:
            self._cleanup_buffer()
        
        if self.verbose:
            logger.info(f"Flush completed: {total_written} positions in {files_created} files")
            
        return total_written
    
    def _write_tfrecord(self, file_path: str, data: Dict[str, np.ndarray]):
        """Écrit les données en format TFRecord sur GCS avec métadonnées de comptage"""
        temp_path = f"/tmp/{os.path.basename(file_path)}"
        
        example_count = len(data['board'])
        
        with tf.io.TFRecordWriter(temp_path) as writer:
            for i in range(example_count):
                # Create TF example with features
                example = tf.train.Example(features=tf.train.Features(feature={
                    'board': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[data['board'][i].tobytes()])),
                    'marbles_out': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[data['marbles_out'][i].tobytes()])),
                    'policy': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[data['policy'][i].tobytes()])),
                    'history': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[data['history'][i].tobytes()])),
                    'outcome': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[data['outcome'][i]])),
                    'player': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[data['player'][i]])),
                    'game_id': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[data['game_id'][i]])),
                    'move_num': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[data['move_num'][i]])),
                    'iteration': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[data['iteration'][i]])),
                    'model_version': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[data['model_version'][i]]))
                }))
                writer.write(example.SerializeToString())
        
        # Create iteration folder if necessary (for GCS)
        iter_dir = os.path.dirname(file_path)
        try:
            # Check if folder already exists
            check_blob = self.bucket.blob(f"{iter_dir}/.placeholder")
            if not check_blob.exists():
                # Create folder marker
                placeholder = self.bucket.blob(f"{iter_dir}/.placeholder")
                placeholder.upload_from_string("")
        except Exception as e:
            logger.warning(f"Cannot verify/create folder {iter_dir}: {e}")
        
        # Upload file
        blob = self.bucket.blob(file_path)
        blob.metadata = {'example_count': str(example_count)}
        blob.upload_from_filename(temp_path)
        
        # Clean up
        os.remove(temp_path)
        
        return example_count
    
    def _update_gcs_index(self, force=False):
        """
        Met à jour l'index des fichiers disponibles sur GCS.
        
        Args:
            force: Si True, force une mise à jour complète même si récemment mise à jour
        
        Returns:
            bool: True si l'index a été mis à jour avec succès
        """
        current_time = time.time()
        
        # Check if update is necessary
        if not force and (current_time - self.last_index_update) < self.index_update_interval:
            return True  # No update needed
        
        try:
            # List all blobs in buffer folder
            prefix = f"{self.buffer_dir}/"
            if self.verbose:
                logger.info(f"Updating GCS index for {prefix}")
            
            blobs = list(self.bucket.list_blobs(prefix=prefix))
            
            if not blobs:
                # Check if folder exists
                check_blob = self.bucket.blob(f"{prefix}.placeholder")
                if not check_blob.exists() and self.verbose:
                    logger.warning(f"Folder {prefix} may not exist")
                return False
            
            new_index = {}
            new_metadata = {}
            tfrecord_files_found = 0
            total_examples = 0
            
            for blob in blobs:
                path = blob.name
                if not path.endswith('.tfrecord'):
                    continue
                
                tfrecord_files_found += 1
                parts = path.split('/')
                if len(parts) >= 3 and parts[-2].startswith('iteration_'):
                    iteration = int(parts[-2].replace('iteration_', ''))
                    
                    if iteration not in new_index:
                        new_index[iteration] = []
                    
                    new_index[iteration].append(path)
                    
                    try:
                        # Expected format: {host_id}_{timestamp}.tfrecord
                        file_basename = os.path.basename(path)
                        timestamp_part = file_basename.split('_')[-1].split('.')[0]
                        timestamp = int(timestamp_part)
                    except (IndexError, ValueError):
                        # Fallback if format not recognized
                        timestamp = int(blob.time_created.timestamp()) if hasattr(blob, 'time_created') else 0
                    
                    # Get number of examples from metadata
                    if hasattr(blob, 'metadata') and blob.metadata and 'example_count' in blob.metadata:
                        example_count = int(blob.metadata['example_count'])
                    else:
                        # If no metadata, estimate (will be corrected on loading)
                        example_count = 1000
                    
                    total_examples += example_count
                    
                    new_metadata[path] = {
                        'size': example_count,
                        'timestamp': timestamp,
                        'iteration': iteration
                    }
            
            # Update index only if it contains data
            if tfrecord_files_found > 0:
                # Replace index and metadata
                self.gcs_index = new_index
                self.gcs_file_metadata = new_metadata
                self.total_size = total_examples + self.local_size
                self.last_index_update = current_time
                
                if self.verbose:
                    iterations_found = list(new_index.keys())
                    logger.info(f"GCS index: {tfrecord_files_found} files, {len(iterations_found)} iterations, {total_examples} positions")
                
                return True
            elif self.verbose:
                logger.warning("No TFRecord files found in buffer folder")
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating GCS index: {e}")
            return False
    
    def _update_total_size(self):
        """Met à jour la taille totale du buffer en comptant les exemples dans les métadonnées"""
        total = 0
        
        # Count from file metadata
        for file_path, metadata in self.gcs_file_metadata.items():
            total += metadata['size']
        
        # Add local buffer
        total += self.local_size
        
        # Update total
        self.total_size = total
        
        return total
    
    def _cleanup_buffer(self):
        """
        Nettoie le buffer lorsqu'il dépasse sa taille maximale.
        Utilise une distribution de probabilité décroissante basée sur l'âge
        pour décider quels fichiers supprimer.
        """
        # If buffer is empty or under limit, do nothing
        if self.total_size <= self.max_buffer_size:
            return
        
        # Calculate how many examples need to be removed
        overflow = self.total_size - int(self.max_buffer_size * 0.8)  # Target 80% fill
        if overflow <= 0:
            return
        
        self.metrics["cleanup_operations"] += 1
        logger.info(f"Buffer cleanup: need to remove {overflow}/{self.total_size} positions")
        
        # Collect all files with their metadata
        all_files = []
        
        for iteration, files in self.gcs_index.items():
            for file_path in files:
                if file_path in self.gcs_file_metadata:
                    metadata = self.gcs_file_metadata[file_path]
                    all_files.append((file_path, metadata))
        
        # Exit if no files
        if not all_files:
            return
        
        # Sort by timestamp (oldest to newest)
        all_files.sort(key=lambda x: x[1]['timestamp'])
        
        # Normalize ages (0 = oldest, 1 = newest)
        if len(all_files) > 1:
            oldest_time = all_files[0][1]['timestamp']
            newest_time = all_files[-1][1]['timestamp']
            time_range = max(1, newest_time - oldest_time)
            
            for i in range(len(all_files)):
                file_path, metadata = all_files[i]
                timestamp = metadata['timestamp']
                age_normalized = 1.0 - ((timestamp - oldest_time) / time_range)  # 1 = oldest, 0 = newest
                all_files[i] = (file_path, metadata, age_normalized)
        else:
            # Single file, give it age of 0.5
            file_path, metadata = all_files[0]
            all_files[0] = (file_path, metadata, 0.5)
        
        # Calculate deletion probabilities with temperature
        probabilities = []
        for _, _, age in all_files:
            # Older age (more ancient), higher probability
            prob = math.exp(age * self.cleanup_temperature)
            probabilities.append(prob)
        
        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        else:
            # Fallback to uniform distribution
            probabilities = [1.0 / len(all_files)] * len(all_files)
        
        # Select files to remove until reaching limit
        examples_removed = 0
        files_to_remove = []
        
        # Create copy for sampling without replacement
        remaining_files = list(range(len(all_files)))
        remaining_probs = probabilities.copy()
        
        while examples_removed < overflow and remaining_files:
            # Normalize remaining probabilities
            total_prob = sum(remaining_probs)
            if total_prob <= 0:
                break
            
            norm_probs = [p / total_prob for p in remaining_probs]
            
            # Select file according to distribution
            idx = np.random.choice(len(remaining_files), p=norm_probs)
            file_idx = remaining_files[idx]
            file_path, metadata, _ = all_files[file_idx]
            file_size = metadata['size']
            
            # Add to removal list
            files_to_remove.append(file_path)
            examples_removed += file_size
            
            # Remove from remaining candidates list
            del remaining_files[idx]
            del remaining_probs[idx]
        
        # Remove selected files
        removed_count = 0
        for file_path in files_to_remove:
            try:
                blob = self.bucket.blob(file_path)
                blob.delete()
                
                # Update index
                iteration = self.gcs_file_metadata[file_path]['iteration']
                if iteration in self.gcs_index and file_path in self.gcs_index[iteration]:
                    self.gcs_index[iteration].remove(file_path)
                    
                    # If this iteration has no more files, remove from index
                    if not self.gcs_index[iteration]:
                        del self.gcs_index[iteration]
                
                # Clean metadata
                if file_path in self.gcs_file_metadata:
                    del self.gcs_file_metadata[file_path]
                
                removed_count += 1
                self.metrics["files_removed"] += 1
                
            except Exception as e:
                logger.warning(f"Error deleting {file_path}: {e}")
        
        # Update total size
        self._update_total_size()
        
        logger.info(f"Cleanup completed: {removed_count} files removed, new size: {self.total_size}")
    
    def _parse_tfrecord(self, example):
        """Parse un exemple TFRecord en dictionnaire numpy"""
        # Define feature schema
        feature_description = {
            'board': tf.io.FixedLenFeature([], tf.string),
            'marbles_out': tf.io.FixedLenFeature([], tf.string),
            'policy': tf.io.FixedLenFeature([], tf.string),
            'history': tf.io.FixedLenFeature([], tf.string),
            'outcome': tf.io.FixedLenFeature([], tf.int64),
            'player': tf.io.FixedLenFeature([], tf.int64),
            'game_id': tf.io.FixedLenFeature([], tf.int64),
            'move_num': tf.io.FixedLenFeature([], tf.int64),
            'iteration': tf.io.FixedLenFeature([], tf.int64),
            'model_version': tf.io.FixedLenFeature([], tf.int64)
        }
        
        parsed = tf.io.parse_single_example(example, feature_description)
        
        return {
            'board': tf.io.decode_raw(parsed['board'], tf.int8).numpy().reshape(self.board_size, self.board_size),
            'marbles_out': tf.io.decode_raw(parsed['marbles_out'], tf.int8).numpy().reshape(2),
            'policy': tf.io.decode_raw(parsed['policy'], tf.float32).numpy().reshape(self.action_space),
            'history': tf.io.decode_raw(parsed['history'], tf.int8).numpy().reshape(self.history_length, self.board_size, self.board_size),
            'outcome': parsed['outcome'].numpy(),
            'player': parsed['player'].numpy(),
            'game_id': parsed['game_id'].numpy(),
            'move_num': parsed['move_num'].numpy(),
            'iteration': parsed['iteration'].numpy(),
            'model_version': parsed['model_version'].numpy()
        }
    
    def sample(self, batch_size, rng_key=None):
        """
        Échantillonne un batch de transitions du buffer global sur GCS.
        
        Args:
            batch_size: Nombre d'exemples à échantillonner
            rng_key: Clé JAX pour la génération de nombres aléatoires
            
        Returns:
            Dict contenant les données échantillonnées
        """
        # Check if index needs to be updated
        current_time = time.time()
        if current_time - self.last_index_update > self.index_update_interval:
            self._update_gcs_index()
        
        has_valid_data = bool(self.gcs_index)
        
        # If no GCS data or inaccessible, use local buffer
        if not has_valid_data:
            if self.local_size == 0:
                raise ValueError("Empty buffer (no local or GCS data)")
            
            # Sampling from local buffer
            if rng_key is None:
                local_indices = np.random.randint(0, self.local_size, size=batch_size)
            else:
                local_indices = jax.random.randint(
                    rng_key, shape=(batch_size,), minval=0, maxval=self.local_size
                ).astype(np.int32)
                local_indices = np.array(local_indices)
            
            result = {}
            for k in self.local_buffer:
                result[k] = self.local_buffer[k][local_indices]
            
            self.metrics["samples_served"] += batch_size
            return result
        
        # Sampling from GCS
        try:
            result = self._sample_from_gcs(batch_size, rng_key)
            self.metrics["samples_served"] += batch_size
            return result
        except Exception as e:
            logger.warning(f"Error sampling from GCS, fallback to local buffer: {e}")
            
            # Fallback to local buffer if available
            if self.local_size > 0:
                if rng_key is None:
                    local_indices = np.random.randint(0, self.local_size, size=batch_size)
                else:
                    local_indices = jax.random.randint(
                        rng_key, shape=(batch_size,), minval=0, maxval=self.local_size
                    ).astype(np.int32)
                    local_indices = np.array(local_indices)
                
                result = {}
                for k in self.local_buffer:
                    result[k] = self.local_buffer[k][local_indices]
                
                return result
            else:
                raise ValueError("GCS sampling failed and local buffer empty")
            
    def _sample_from_gcs(self, n_samples, rng_key=None):
        """Échantillonne des exemples depuis GCS avec biais de récence."""
        # Build distribution for sampling iterations
        iterations = sorted(list(self.gcs_index.keys()))
        if not iterations:
            return {}
        
        # Apply recency bias if enabled
        if self.recency_enabled and len(iterations) > 1:
            # Normalize iterations between 0 and 1
            min_iter = min(iterations)
            max_iter = max(iterations)
            range_iter = max(1, max_iter - min_iter)
            
            # Calculate weights with temperature
            weights = [(iter_num - min_iter) / range_iter for iter_num in iterations]
            weights = [np.exp(w * self.recency_temperature) for w in weights]
            total_weight = sum(weights)
            probs = [w / total_weight for w in weights]
        else:
            # Uniform distribution
            probs = [1.0 / len(iterations)] * len(iterations)
        
        # Select iterations
        if rng_key is None:
            selected_iters = np.random.choice(
                iterations, 
                size=min(3, len(iterations)), 
                p=probs, 
                replace=True
            )
        else:
            rng_key, subkey = jax.random.split(rng_key)
            selected_iters = jax.random.choice(
                subkey, 
                np.array(iterations), 
                shape=(min(3, len(iterations)),),
                p=np.array(probs),
                replace=True
            )
            selected_iters = np.array(selected_iters)
        
        # Collect examples from each selected iteration
        all_examples = []
        examples_per_iter = n_samples // len(selected_iters) + 1
        
        for iter_num in selected_iters:
            files = self.gcs_index[iter_num]
            if not files:
                continue
            
            # Randomly select some files for diversity
            num_files_to_sample = min(2, len(files))
            if rng_key is None:
                file_indices = np.random.choice(len(files), size=num_files_to_sample, replace=False)
            else:
                rng_key, subkey = jax.random.split(rng_key)
                file_indices = jax.random.choice(
                    subkey, 
                    len(files), 
                    shape=(num_files_to_sample,), 
                    replace=False
                )
                file_indices = np.array(file_indices)
            
            # Distribute examples among selected files
            examples_per_file = examples_per_iter // num_files_to_sample + 1
            
            for file_idx in file_indices:
                file_path = files[int(file_idx)]
                
                # Load and sample examples from this file
                try:
                    examples = self._load_examples_from_gcs(file_path, examples_per_file)
                    all_examples.extend(examples)
                    
                    # If we have enough examples, stop sampling
                    if len(all_examples) >= n_samples:
                        break
                except Exception as e:
                    logger.warning(f"Error loading examples from {file_path}: {e}")
            
            # If we have enough examples, stop sampling
            if len(all_examples) >= n_samples:
                break
        
        # Handle case where we don't have enough examples
        if not all_examples:
            raise ValueError("No examples could be loaded from GCS")
            
        if len(all_examples) < n_samples:
            # Duplicate existing examples to reach requested size
            if all_examples:  
                indices_to_duplicate = np.random.choice(
                    len(all_examples), size=n_samples-len(all_examples), replace=True)
                
                for idx in indices_to_duplicate:
                    all_examples.append(all_examples[idx])
        elif len(all_examples) > n_samples:
            # Truncate if too many examples
            all_examples = all_examples[:n_samples]
        
        # Consolidate examples into single dict
        result = {}
        for k in all_examples[0].keys():
            result[k] = np.array([ex[k] for ex in all_examples])
        
        return result

    def _load_examples_from_gcs(self, file_path, max_examples):
        """Charge des exemples depuis un fichier TFRecord sur GCS"""
        blob = self.bucket.blob(file_path)
        temp_path = f"/tmp/{os.path.basename(file_path)}"
        blob.download_to_filename(temp_path)
        
        raw_dataset = tf.data.TFRecordDataset(temp_path)
        
        examples = []
        for i, raw_example in enumerate(raw_dataset):
            if i >= max_examples:
                break
            example = self._parse_tfrecord(raw_example)
            examples.append(example)
        
        os.remove(temp_path)
        
        return examples
    
    def sample_with_recency_bias(self, batch_size, temperature=None, rng_key=None):
        """
        Échantillonne avec biais de récence depuis GCS.
        
        Args:
            batch_size: Nombre d'exemples à échantillonner
            temperature: Température pour le biais de récence (None pour utiliser la valeur par défaut)
            rng_key: Clé JAX pour la génération de nombres aléatoires
            
        Returns:
            Dict contenant les données échantillonnées
        """
        original_temp = self.recency_temperature
        if temperature is not None:
            self.recency_temperature = temperature
        
        result = self.sample(batch_size, rng_key)
        
        if temperature is not None:
            self.recency_temperature = original_temp
        
        return result
    
    def start_new_game(self):
        """Incrémente l'ID de partie pour commencer une nouvelle partie"""
        self.current_game_id += 1
        return self.current_game_id
    
    def get_stats(self):
        """Renvoie des statistiques sur le buffer"""
        stats = {
            "total_size": self.total_size,
            "local_size": self.local_size,
            "max_size": self.max_buffer_size,
            "fill_percentage": 100 * self.total_size / self.max_buffer_size if self.max_buffer_size > 0 else 0,
            "iterations": len(self.gcs_index),
            "files": sum(len(files) for files in self.gcs_index.values()),
        }
        
        stats.update(self.metrics)
        
        return stats
    
    def close(self):
        """Ferme proprement le buffer et assure que toutes les données sont écrites"""
        if self.local_size > 0:
            positions_flushed = self.flush_to_gcs()
            logger.info(f"Final flush: {positions_flushed} positions written to GCS")
        else :
            logger.info(f"Local empty")
        
        logger.info(f"GCS buffer closed. Total: {self.total_size} positions")
    
    def __del__(self):
        """Destructeur pour assurer la fermeture propre"""
        try:
            self.close()
        except:
            pass