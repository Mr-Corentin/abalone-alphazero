import time
import json
from typing import Dict, Any, Optional
from google.cloud import storage
from datetime import datetime
import logging

logger = logging.getLogger("alphazero.gcs_metrics_logger")

class SimpleGCSLogger:
    """
    Simple sequential metrics logger for AlphaZero training.
    Logs immediately when called - no threading or buffering complexity.
    """
    
    def __init__(self, bucket_name: str, process_id: int, session_id: Optional[str] = None):
        """
        Initialize the simple GCS logger.
        
        Args:
            bucket_name: GCS bucket name for storing logs
            process_id: Worker/process ID for this logger instance
            session_id: Unique session identifier (timestamp if None)
        """
        self.bucket_name = bucket_name
        self.process_id = process_id
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize GCS client once
        try:
            self.client = storage.Client()
            self.bucket = self.client.bucket(bucket_name)
            self.enabled = True
            logger.info(f"GCS Logger initialized for worker {process_id}, session {self.session_id}")
        except Exception as e:
            logger.warning(f"Failed to initialize GCS client: {e}. Logging disabled.")
            self.enabled = False
    
    def log_generation_metrics(self, iteration: int, **metrics):
        """Log generation phase metrics immediately."""
        if not self.enabled:
            return
            
        log_data = {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'session_id': self.session_id,
            'worker_id': self.process_id,
            'iteration': iteration,
            'phase': 'generation',
            **metrics
        }
        
        self._write_log('generation', iteration, log_data)
    
    def log_training_metrics(self, iteration: int, **metrics):
        """Log training phase metrics immediately."""
        if not self.enabled:
            return
            
        log_data = {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'session_id': self.session_id,
            'worker_id': self.process_id,
            'iteration': iteration,
            'phase': 'training',
            **metrics
        }
        
        self._write_log('training', iteration, log_data)
    
    def log_timing_metrics(self, iteration: int, **metrics):
        """Log timing metrics immediately."""
        if not self.enabled:
            return
            
        log_data = {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'session_id': self.session_id,
            'worker_id': self.process_id,
            'iteration': iteration,
            'phase': 'timing',
            **metrics
        }
        
        self._write_log('timing', iteration, log_data)
    
    def log_evaluation_metrics(self, iteration: int, **metrics):
        """Log evaluation metrics immediately."""
        if not self.enabled:
            return
            
        log_data = {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'session_id': self.session_id,
            'worker_id': self.process_id,
            'iteration': iteration,
            'phase': 'evaluation',
            **metrics
        }
        
        self._write_log('evaluation', iteration, log_data)
    
    def _write_log(self, log_type: str, iteration: int, data: Dict[str, Any]):
        """Write a single log entry to GCS."""
        try:
            # Create filename: training_logs/generation/worker_0_iter_5.json
            filename = f"training_logs/{log_type}/worker_{self.process_id}_iter_{iteration}.json"
            
            # Upload to GCS
            blob = self.bucket.blob(filename)
            blob.upload_from_string(json.dumps(data, indent=2), content_type='application/json')
            
        except Exception as e:
            logger.error(f"Failed to write {log_type} log to GCS: {e}")
    
    def write_summary_log(self, summary_data: Dict[str, Any]):
        """Write a session summary log."""
        if not self.enabled:
            return
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_logs/summaries/session_{self.session_id}_worker_{self.process_id}_{timestamp}.json"
            
            blob = self.bucket.blob(filename)
            blob.upload_from_string(json.dumps(summary_data, indent=2), content_type='application/json')
            
            logger.info(f"Session summary written to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to write session summary: {e}")


class LocalMetricsLogger:
    """Local file version for development without GCS."""
    
    def __init__(self, log_dir: str, process_id: int, session_id: Optional[str] = None):
        import os
        
        self.log_dir = log_dir
        self.process_id = process_id
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log directories
        for subdir in ['generation', 'training', 'timing', 'evaluation', 'summaries']:
            os.makedirs(os.path.join(log_dir, 'training_logs', subdir), exist_ok=True)
        
        logger.info(f"Local Logger initialized for worker {process_id}, session {self.session_id}")
    
    def log_generation_metrics(self, iteration: int, **metrics):
        """Log generation phase metrics to local file."""
        log_data = {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'session_id': self.session_id,
            'worker_id': self.process_id,
            'iteration': iteration,
            'phase': 'generation',
            **metrics
        }
        
        self._write_log('generation', iteration, log_data)
    
    def log_training_metrics(self, iteration: int, **metrics):
        """Log training phase metrics to local file."""
        log_data = {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'session_id': self.session_id,
            'worker_id': self.process_id,
            'iteration': iteration,
            'phase': 'training',
            **metrics
        }
        
        self._write_log('training', iteration, log_data)
    
    def log_timing_metrics(self, iteration: int, **metrics):
        """Log timing metrics to local file."""
        log_data = {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'session_id': self.session_id,
            'worker_id': self.process_id,
            'iteration': iteration,
            'phase': 'timing',
            **metrics
        }
        
        self._write_log('timing', iteration, log_data)
    
    def log_evaluation_metrics(self, iteration: int, **metrics):
        """Log evaluation metrics to local file."""
        log_data = {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'session_id': self.session_id,
            'worker_id': self.process_id,
            'iteration': iteration,
            'phase': 'evaluation',
            **metrics
        }
        
        self._write_log('evaluation', iteration, log_data)
    
    def _write_log(self, log_type: str, iteration: int, data: Dict[str, Any]):
        """Write a single log entry to local file."""
        import os
        
        try:
            filename = os.path.join(self.log_dir, 'training_logs', log_type, f"worker_{self.process_id}_iter_{iteration}.json")
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to write {log_type} log locally: {e}")
    
    def write_summary_log(self, summary_data: Dict[str, Any]):
        """Write a session summary log."""
        import os
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.log_dir, 'training_logs', 'summaries', 
                                  f"session_{self.session_id}_worker_{self.process_id}_{timestamp}.json")
            
            with open(filename, 'w') as f:
                json.dump(summary_data, f, indent=2)
                
            logger.info(f"Session summary written to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to write session summary: {e}")


class IterationMetricsAggregator:
    """
    Aggregates metrics from all workers at the end of each iteration into consolidated files.
    Only runs on the main process to avoid conflicts.
    """
    
    def __init__(self, bucket_name: str = None, log_dir: str = None):
        """
        Initialize the aggregator.
        
        Args:
            bucket_name: GCS bucket name (if None, uses local files)
            log_dir: Local log directory (if GCS not used)
        """
        self.bucket_name = bucket_name
        self.log_dir = log_dir
        self.use_gcs = bucket_name is not None
        
        if self.use_gcs:
            try:
                self.client = storage.Client()
                self.bucket = self.client.bucket(bucket_name)
                self.enabled = True
                logger.info(f"Iteration aggregator initialized for GCS bucket: {bucket_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize GCS client for aggregation: {e}")
                self.enabled = False
        else:
            self.enabled = True
            logger.info(f"Iteration aggregator initialized for local directory: {log_dir}")
        
        # Initialize the consolidated log file
        self.consolidated_log_path = self._get_consolidated_log_path()
        self._initialize_consolidated_log()
    
    def aggregate_iteration_metrics(self, iteration: int, num_workers: int, session_id: str):
        """
        Aggregate all worker metrics for a specific iteration into consolidated files.
        
        Args:
            iteration: The iteration number to aggregate
            num_workers: Total number of workers
            session_id: Session ID for filtering logs
        """
        if not self.enabled:
            return
        
        try:
            # Aggregate each metric type
            generation_data = self._collect_worker_data('generation', iteration, num_workers, session_id)
            training_data = self._collect_worker_data('training', iteration, num_workers, session_id)
            timing_data = self._collect_worker_data('timing', iteration, num_workers, session_id)
            evaluation_data = self._collect_worker_data('evaluation', iteration, num_workers, session_id)
            
            # Create consolidated files
            if generation_data:
                self._write_consolidated_file('generation', iteration, generation_data, session_id)
            if training_data:
                self._write_consolidated_file('training', iteration, training_data, session_id)
            if timing_data:
                self._write_consolidated_file('timing', iteration, timing_data, session_id)
            if evaluation_data:
                self._write_consolidated_file('evaluation', iteration, evaluation_data, session_id)
            
            logger.info(f"Aggregated metrics for iteration {iteration} from {len(generation_data)} workers")
            
        except Exception as e:
            logger.error(f"Failed to aggregate iteration {iteration} metrics: {e}")
    
    def _collect_worker_data(self, metric_type: str, iteration: int, num_workers: int, session_id: str):
        """Collect data from all workers for a specific metric type and iteration."""
        all_data = []
        
        for worker_id in range(num_workers):
            try:
                if self.use_gcs:
                    # Download from GCS
                    blob_name = f"training_logs/{metric_type}/worker_{worker_id}_iter_{iteration}.json"
                    blob = self.bucket.blob(blob_name)
                    if blob.exists():
                        content = blob.download_as_text()
                        data = json.loads(content)
                        # Verify it's from the correct session
                        if data.get('session_id') == session_id:
                            all_data.append(data)
                else:
                    # Read from local file
                    import os
                    filename = os.path.join(self.log_dir, 'training_logs', metric_type, 
                                          f"worker_{worker_id}_iter_{iteration}.json")
                    if os.path.exists(filename):
                        with open(filename, 'r') as f:
                            data = json.load(f)
                            # Verify it's from the correct session
                            if data.get('session_id') == session_id:
                                all_data.append(data)
                                
            except Exception as e:
                logger.debug(f"Could not read {metric_type} data for worker {worker_id}, iteration {iteration}: {e}")
                continue
        
        return all_data
    
    def _write_consolidated_file(self, metric_type: str, iteration: int, data_list, session_id: str):
        """Write consolidated data to a single file."""
        if not data_list:
            return
        
        # Calculate aggregated statistics
        consolidated = {
            'session_id': session_id,
            'iteration': iteration,
            'metric_type': metric_type,
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'num_workers': len(data_list),
            'workers_data': data_list,
            'aggregated_stats': self._calculate_stats(data_list, metric_type)
        }
        
        filename = f"training_logs/consolidated/{metric_type}_iter_{iteration}.json"
        
        try:
            if self.use_gcs:
                # Upload to GCS
                blob = self.bucket.blob(filename)
                blob.upload_from_string(json.dumps(consolidated, indent=2), 
                                      content_type='application/json')
            else:
                # Write to local file
                import os
                os.makedirs(os.path.join(self.log_dir, 'training_logs', 'consolidated'), exist_ok=True)
                local_filename = os.path.join(self.log_dir, filename)
                with open(local_filename, 'w') as f:
                    json.dump(consolidated, f, indent=2)
            
            logger.debug(f"Written consolidated {metric_type} file for iteration {iteration}")
            
        except Exception as e:
            logger.error(f"Failed to write consolidated {metric_type} file: {e}")
    
    def _calculate_stats(self, data_list, metric_type: str):
        """Calculate aggregated statistics for the metric type."""
        if not data_list:
            return {}
        
        stats = {}
        
        if metric_type == 'generation':
            # Sum across workers
            stats['total_positions_generated'] = sum(d.get('positions_generated', 0) for d in data_list)
            stats['total_games_generated'] = sum(d.get('games_generated', 0) for d in data_list)
            stats['total_games_requested'] = sum(d.get('games_requested', 0) for d in data_list)
            
            # Average across workers
            mean_plays = [d.get('mean_plays_per_game', 0) for d in data_list if d.get('mean_plays_per_game', 0) > 0]
            if mean_plays:
                stats['avg_mean_plays_per_game'] = sum(mean_plays) / len(mean_plays)
                stats['min_mean_plays_per_game'] = min(mean_plays)
                stats['max_mean_plays_per_game'] = max(mean_plays)
            
            # Aggregate marble counts across workers
            white_total_counts = {i: 0 for i in range(7)}
            black_total_counts = {i: 0 for i in range(7)}
            
            for data in data_list:
                white_counts = data.get('white_marble_counts', {})
                black_counts = data.get('black_marble_counts', {})
                
                for i in range(7):
                    white_total_counts[i] += white_counts.get(str(i), white_counts.get(i, 0))
                    black_total_counts[i] += black_counts.get(str(i), black_counts.get(i, 0))
            
            stats['white_marble_counts'] = white_total_counts
            stats['black_marble_counts'] = black_total_counts
            
            # Calculate proportions from total counts
            total_games = stats['total_games_generated']
            if total_games > 0:
                stats['white_marble_proportions'] = {i: count / total_games for i, count in white_total_counts.items()}
                stats['black_marble_proportions'] = {i: count / total_games for i, count in black_total_counts.items()}
            
            # Aggregate win/loss statistics across workers
            stats['total_white_wins'] = sum(d.get('white_wins', 0) for d in data_list)
            stats['total_black_wins'] = sum(d.get('black_wins', 0) for d in data_list)
            stats['total_draws'] = sum(d.get('draws', 0) for d in data_list)
            
            # Calculate average win rates across workers
            win_rates = [d.get('white_win_rate', 0) for d in data_list if 'white_win_rate' in d]
            if win_rates:
                stats['avg_white_win_rate'] = sum(win_rates) / len(win_rates)
            
            win_rates = [d.get('black_win_rate', 0) for d in data_list if 'black_win_rate' in d]
            if win_rates:
                stats['avg_black_win_rate'] = sum(win_rates) / len(win_rates)
            
            draw_rates = [d.get('draw_rate', 0) for d in data_list if 'draw_rate' in d]
            if draw_rates:
                stats['avg_draw_rate'] = sum(draw_rates) / len(draw_rates)
        
        elif metric_type == 'training':
            # Average losses and accuracies across workers
            numeric_fields = ['total_loss', 'policy_loss', 'value_loss', 'policy_accuracy', 'value_sign_match']
            for field in numeric_fields:
                values = [d.get(field, 0) for d in data_list if field in d]
                if values:
                    stats[f'avg_{field}'] = sum(values) / len(values)
                    stats[f'min_{field}'] = min(values)
                    stats[f'max_{field}'] = max(values)
                    stats[f'std_{field}'] = (sum((x - stats[f'avg_{field}'])**2 for x in values) / len(values))**0.5
            
            # Sum training steps
            stats['total_training_steps_completed'] = sum(d.get('training_steps_completed', 0) for d in data_list)
            stats['total_training_steps_requested'] = sum(d.get('training_steps_requested', 0) for d in data_list)
        
        elif metric_type == 'timing':
            # Sum times across workers for total workload
            time_fields = ['generation_time', 'buffer_update_time', 'training_time', 'total_iteration_time']
            for field in time_fields:
                values = [d.get(field, 0) for d in data_list if field in d]
                if values:
                    stats[f'total_{field}'] = sum(values)
                    stats[f'avg_{field}'] = sum(values) / len(values)
                    stats[f'min_{field}'] = min(values)
                    stats[f'max_{field}'] = max(values)
            
            # Performance rates
            games_per_sec = [d.get('games_per_sec', 0) for d in data_list if d.get('games_per_sec', 0) > 0]
            if games_per_sec:
                stats['total_games_per_sec'] = sum(games_per_sec)
                stats['avg_games_per_sec'] = sum(games_per_sec) / len(games_per_sec)
        
        elif metric_type == 'evaluation':
            # Aggregate evaluation results
            if data_list:
                # Take the evaluation data from the main process (worker 0)
                main_eval_data = next((d for d in data_list if d.get('worker_id') == 0), data_list[0] if data_list else {})
                
                # Copy over the evaluation metrics
                for key, value in main_eval_data.items():
                    if key.startswith('win_rate_vs_iter') or key in ['global_win_rate', 'total_eval_games', 'total_eval_wins']:
                        stats[key] = value
                        
                # Calculate summary stats
                win_rates = [value for key, value in main_eval_data.items() if key.startswith('win_rate_vs_iter')]
                if win_rates:
                    stats['avg_win_rate_vs_previous'] = sum(win_rates) / len(win_rates)
                    stats['min_win_rate_vs_previous'] = min(win_rates)
                    stats['max_win_rate_vs_previous'] = max(win_rates)
                    stats['num_models_evaluated'] = len(win_rates)
        
        return stats
    
    def _get_consolidated_log_path(self):
        """Get the path for the consolidated readable log file."""
        if self.use_gcs:
            return "training_logs/consolidated_metrics.txt"
        else:
            import os
            return os.path.join(self.log_dir, "training_metrics_summary.txt")
    
    def _initialize_consolidated_log(self):
        """Initialize the consolidated log file with header, cleaning any existing content."""
        if not self.enabled:
            return
        
        header = """AlphaZero Training Metrics Summary
=================================
Format: Each iteration shows generation, training, and timing metrics from all workers
Last updated: {datetime}

""".format(datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        try:
            if self.use_gcs:
                blob = self.bucket.blob(self.consolidated_log_path)
                # Always overwrite the file to clean previous training data
                blob.upload_from_string(header, content_type='text/plain')
                logger.info(f"Consolidated log file cleaned and initialized: {self.consolidated_log_path}")
            else:
                import os
                # Always overwrite the file to clean previous training data
                with open(self.consolidated_log_path, 'w') as f:
                    f.write(header)
                logger.info(f"Consolidated log file cleaned and initialized: {self.consolidated_log_path}")
        except Exception as e:
            logger.error(f"Failed to initialize consolidated log: {e}")
    
    def write_consolidated_readable_summary(self, iteration: int, num_workers: int, session_id: str):
        """
        Write a human-readable summary for the iteration and clean up individual worker files.
        This replaces the need for many individual JSON files with one readable text file.
        """
        if not self.enabled:
            return
        
        try:
            # Collect data from all workers
            generation_data = self._collect_worker_data('generation', iteration, num_workers, session_id)
            training_data = self._collect_worker_data('training', iteration, num_workers, session_id)
            timing_data = self._collect_worker_data('timing', iteration, num_workers, session_id)
            evaluation_data = self._collect_worker_data('evaluation', iteration, num_workers, session_id)
            
            # Calculate aggregated stats
            gen_stats = self._calculate_stats(generation_data, 'generation')
            train_stats = self._calculate_stats(training_data, 'training')
            time_stats = self._calculate_stats(timing_data, 'timing')
            eval_stats = self._calculate_stats(evaluation_data, 'evaluation') if evaluation_data else {}
            
            # Create readable summary
            summary = self._format_readable_summary(iteration, session_id, gen_stats, train_stats, time_stats, eval_stats, num_workers, generation_data, training_data, timing_data, evaluation_data)
            
            # Append to consolidated file
            self._append_to_consolidated_log(summary)
            
            # Clean up individual worker files
            cleanup_types = ['generation', 'training', 'timing']
            if evaluation_data:
                cleanup_types.append('evaluation')
            self._cleanup_worker_files(iteration, num_workers, cleanup_types)
            
            logger.info(f"Consolidated readable summary written for iteration {iteration}")
            
        except Exception as e:
            logger.error(f"Failed to write consolidated readable summary for iteration {iteration}: {e}")
    
    def _format_readable_summary(self, iteration: int, session_id: str, gen_stats: dict, train_stats: dict, time_stats: dict, eval_stats: dict, num_workers: int, generation_data: list, training_data: list, timing_data: list, evaluation_data: list = None):
        """Format the metrics into a human-readable text summary."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        summary = f"""
{'='*60}
ITERATION {iteration} - {timestamp}
Session: {session_id} | Workers: {num_workers}
{'='*60}

GENERATION METRICS:
  • Total Games Generated: {gen_stats.get('total_games_generated', 0):,}
  • Total Positions: {gen_stats.get('total_positions_generated', 0):,}
  • Average Plays per Game: {gen_stats.get('avg_mean_plays_per_game', 0):.1f}
  • Min/Max Plays per Game: {gen_stats.get('min_mean_plays_per_game', 0):.1f} / {gen_stats.get('max_mean_plays_per_game', 0):.1f}

WIN/LOSS DISTRIBUTION:
  • White Wins: {gen_stats.get('total_white_wins', 0):,} ({gen_stats.get('avg_white_win_rate', 0)*100:.1f}%)
  • Black Wins: {gen_stats.get('total_black_wins', 0):,} ({gen_stats.get('avg_black_win_rate', 0)*100:.1f}%)
  • Draws: {gen_stats.get('total_draws', 0):,} ({gen_stats.get('avg_draw_rate', 0)*100:.1f}%)

MARBLE OUT DISTRIBUTION:
  • White Marbles Out: {self._format_marble_distribution(gen_stats.get('white_marble_counts', {}), gen_stats.get('white_marble_proportions', {}))}
  • Black Marbles Out: {self._format_marble_distribution(gen_stats.get('black_marble_counts', {}), gen_stats.get('black_marble_proportions', {}))}

TRAINING METRICS:
  • Total Loss: {train_stats.get('avg_total_loss', 0):.4f} (±{train_stats.get('std_total_loss', 0):.4f})
  • Policy Loss: {train_stats.get('avg_policy_loss', 0):.4f} (±{train_stats.get('std_policy_loss', 0):.4f})
  • Value Loss: {train_stats.get('avg_value_loss', 0):.4f} (±{train_stats.get('std_value_loss', 0):.4f})
  • Policy Accuracy: {train_stats.get('avg_policy_accuracy', 0)*100:.1f}% (±{train_stats.get('std_policy_accuracy', 0)*100:.1f}%)
  • Value Sign Match: {train_stats.get('avg_value_sign_match', 0)*100:.1f}% (±{train_stats.get('std_value_sign_match', 0)*100:.1f}%)
  • Training Steps: {train_stats.get('total_training_steps_completed', 0):,}

TIMING METRICS:
  • Total Generation Time: {time_stats.get('total_generation_time', 0):.1f}s (avg: {time_stats.get('avg_generation_time', 0):.1f}s)
  • Total Training Time: {time_stats.get('total_training_time', 0):.1f}s (avg: {time_stats.get('avg_training_time', 0):.1f}s)
  • Total Iteration Time: {time_stats.get('total_total_iteration_time', 0):.1f}s
  • Games per Second: {time_stats.get('total_games_per_sec', 0):.1f} (avg: {time_stats.get('avg_games_per_sec', 0):.1f}){self._format_evaluation_section(eval_stats, evaluation_data)}

WORKER BREAKDOWN:
{self._format_worker_breakdown(generation_data, training_data, timing_data)}
"""
        return summary
    
    def _format_worker_breakdown(self, generation_data: list, training_data: list, timing_data: list):
        """Format per-worker metrics breakdown."""
        breakdown = "  Generation Time & Loss by Worker:\n"
        
        # Create a mapping of worker_id to data
        worker_data = {}
        
        # Collect generation data by worker
        for gen_data in generation_data:
            worker_id = gen_data.get('worker_id', 'unknown')
            if worker_id not in worker_data:
                worker_data[worker_id] = {}
            worker_data[worker_id]['generation'] = gen_data
            
        # Collect training data by worker
        for train_data in training_data:
            worker_id = train_data.get('worker_id', 'unknown')
            if worker_id not in worker_data:
                worker_data[worker_id] = {}
            worker_data[worker_id]['training'] = train_data
            
        # Collect timing data by worker
        for time_data in timing_data:
            worker_id = time_data.get('worker_id', 'unknown')
            if worker_id not in worker_data:
                worker_data[worker_id] = {}
            worker_data[worker_id]['timing'] = time_data
        
        # Format each worker's data
        for worker_id in sorted(worker_data.keys()):
            data = worker_data[worker_id]
            
            # Get generation time
            gen_time = data.get('timing', {}).get('generation_time', 0)
            
            # Get losses
            total_loss = data.get('training', {}).get('total_loss', 0)
            policy_loss = data.get('training', {}).get('policy_loss', 0)
            value_loss = data.get('training', {}).get('value_loss', 0)
            
            # Get games generated
            games_gen = data.get('generation', {}).get('games_generated', 0)
            
            breakdown += f"    Worker {worker_id}: Gen={gen_time:.1f}s | Games={games_gen} | Loss={total_loss:.4f} (P:{policy_loss:.4f}, V:{value_loss:.4f})\n"
        
        return breakdown
    
    def _format_marble_distribution(self, counts_dict, proportions_dict):
        """Format marble distribution in a compact readable format."""
        if not counts_dict and not proportions_dict:
            return "No data"
        
        parts = []
        for i in range(7):
            count = counts_dict.get(i, counts_dict.get(str(i), 0))
            proportion = proportions_dict.get(i, proportions_dict.get(str(i), 0))
            if count > 0:
                parts.append(f"{i}:{count}({proportion:.1%})")
        
        return " ".join(parts) if parts else "No games completed"
    
    def _format_evaluation_section(self, eval_stats: dict, evaluation_data: list = None):
        """Format evaluation metrics section if evaluation occurred."""
        if not eval_stats or not evaluation_data:
            return ""
        
        section = "\n\nEVALUATION METRICS:"
        
        # Global win rate
        global_win_rate = eval_stats.get('global_win_rate')
        if global_win_rate is not None:
            section += f"\n  • Global Win Rate: {global_win_rate:.1%}"
        
        # Number of models evaluated
        num_models = eval_stats.get('num_models_evaluated', 0)
        if num_models > 0:
            section += f"\n  • Models Evaluated: {num_models}"
            
            # Average win rate vs previous models
            avg_win_rate = eval_stats.get('avg_win_rate_vs_previous')
            min_win_rate = eval_stats.get('min_win_rate_vs_previous')
            max_win_rate = eval_stats.get('max_win_rate_vs_previous')
            
            if avg_win_rate is not None:
                section += f"\n  • Avg Win Rate vs Previous: {avg_win_rate:.1%}"
                section += f"\n  • Win Rate Range: {min_win_rate:.1%} - {max_win_rate:.1%}"
        
        # Detailed results per model
        individual_results = []
        for key, value in eval_stats.items():
            if key.startswith('win_rate_vs_iter'):
                iter_num = key.replace('win_rate_vs_iter', '')
                individual_results.append(f"iter{iter_num}:{value:.1%}")
        
        if individual_results:
            section += f"\n  • Individual Results: {' '.join(individual_results)}"
        
        # Total games and wins
        total_games = eval_stats.get('total_eval_games')
        total_wins = eval_stats.get('total_eval_wins')
        if total_games and total_wins:
            section += f"\n  • Total Evaluation Games: {total_games} (wins: {total_wins})"
        
        return section
    
    def _append_to_consolidated_log(self, summary: str):
        """Append the summary to the consolidated log file."""
        try:
            if self.use_gcs:
                # Download existing content, append, and re-upload
                blob = self.bucket.blob(self.consolidated_log_path)
                if blob.exists():
                    existing_content = blob.download_as_text()
                else:
                    existing_content = ""
                
                updated_content = existing_content + summary
                blob.upload_from_string(updated_content, content_type='text/plain')
            else:
                # Append to local file
                with open(self.consolidated_log_path, 'a', encoding='utf-8') as f:
                    f.write(summary)
                    
        except Exception as e:
            logger.error(f"Failed to append to consolidated log: {e}")
    
    def _cleanup_worker_files(self, iteration: int, num_workers: int, metric_types: list):
        """Clean up individual worker files after consolidation."""
        for metric_type in metric_types:
            for worker_id in range(num_workers):
                try:
                    if self.use_gcs:
                        blob_name = f"training_logs/{metric_type}/worker_{worker_id}_iter_{iteration}.json"
                        blob = self.bucket.blob(blob_name)
                        if blob.exists():
                            blob.delete()
                    else:
                        import os
                        filename = os.path.join(self.log_dir, 'training_logs', metric_type, 
                                              f"worker_{worker_id}_iter_{iteration}.json")
                        if os.path.exists(filename):
                            os.remove(filename)
                            
                except Exception as e:
                    logger.debug(f"Could not clean up worker file {metric_type}/worker_{worker_id}_iter_{iteration}.json: {e}")
                    continue