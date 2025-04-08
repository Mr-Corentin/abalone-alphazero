#!/usr/bin/env python3
"""
Main script to launch AlphaZero training for Abalone
"""

import os
import sys
import json
import argparse
import time
import jax
import datetime

from model.neural_net import AbaloneModel
from environment.env import AbaloneEnv
from training.trainer import AbaloneTrainerSync
from training.config import DEFAULT_CONFIG, CPU_CONFIG, get_config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='AlphaZero for Abalone')
    
    # General options
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'play'],
                      help='Mode: train, evaluate or play')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to a configuration file (JSON)')
    parser.add_argument('--cpu-only', action='store_true',
                       help='Force CPU usage (minimal configuration)')
    
    # Training options
    parser.add_argument('--iterations', type=int, default=None,
                       help='Number of training iterations')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--games-per-iter', type=int, default=None,
                       help='Number of games per iteration')
    parser.add_argument('--buffer-size', type=int, default=None,
                       help='Size of the replay buffer')
    parser.add_argument('--checkpoint-path', type=str, default=None,
                       help='Path to save checkpoints')
    parser.add_argument('--log-dir', type=str, default=None,
                       help='Directory for TensorBoard logs')
    parser.add_argument('--gcs-bucket', type=str, default=None,
                       help='GCS bucket name to store games')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint to load for resuming training')
    
    # Model options
    parser.add_argument('--num-filters', type=int, default=None,
                       help='Number of filters in the network')
    parser.add_argument('--num-blocks', type=int, default=None,
                       help='Number of residual blocks')

    return parser.parse_args()


def get_merged_config(args):
    """Combine configuration from defaults and command line arguments"""
    # Load base configuration
    if args.cpu_only:
        config = CPU_CONFIG.copy()
    else:
        config = get_config().copy()
    
    # Load from file if specified
    if args.config:
        with open(args.config, 'r') as f:
            file_config = json.load(f)
        
        # Merge file config
        for category, options in file_config.items():
            if category in config:
                config[category].update(options)
            else:
                config[category] = options
    
    # Override with command line arguments
    if args.iterations:
        config['training']['num_iterations'] = args.iterations
    
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    if args.games_per_iter:
        config['training']['games_per_iteration'] = args.games_per_iter
    
    if args.buffer_size:
        config['buffer']['size'] = args.buffer_size
    
    if args.num_filters:
        config['model']['num_filters'] = args.num_filters
    
    if args.num_blocks:
        config['model']['num_blocks'] = args.num_blocks
    
    if args.checkpoint_path:
        config['checkpoint']['path'] = args.checkpoint_path
    
    # If using GCS bucket, update paths for cloud storage
    if args.gcs_bucket:
        bucket_path = f"gs://{args.gcs_bucket}"
        if not args.checkpoint_path:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            config['checkpoint']['path'] = f"{bucket_path}/checkpoints/model_{timestamp}"
        
        if not args.log_dir:
            if 'log_dir' not in config:
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                args.log_dir = f"{bucket_path}/logs/abalone_az_{timestamp}"
    
    return config


def display_config_summary(config):
    """Display a summary of the configuration"""
    print("\n=== Configuration ===")
    print(f"Model: {config['model']['num_filters']} filters, {config['model']['num_blocks']} blocks")
    print(f"Buffer: {config['buffer']['size']} positions")
    print(f"Training: {config['training']['num_iterations']} iterations, {config['training']['games_per_iteration']} games/iter")
    print(f"Batch: {config['training']['batch_size']}, {config['training']['training_steps_per_iteration']} steps/iter")
    print(f"MCTS: {config['mcts']['num_simulations']} simulations per action")
    print(f"Checkpoints: {config['checkpoint']['path']}")


def display_hardware_info():
    """Display information about available hardware"""
    try:
        # Try TPU first
        tpu_devices = jax.devices('tpu')
        print(f"\n=== Hardware: {len(tpu_devices)} TPU cores detected ===")
        return
    except RuntimeError:
        pass
    
    try:
        # Then GPU
        gpu_devices = jax.devices('gpu')
        print(f"\n=== Hardware: {len(gpu_devices)} GPUs detected ===")
        return
    except RuntimeError:
        pass
    
    # Finally CPU
    cpu_devices = jax.devices('cpu')
    print(f"\n=== Hardware: {len(cpu_devices)} CPU cores detected ===")


# Dans la fonction create_trainer du main.py:

def create_trainer(config, args):
    """Create and configure the trainer"""
    # Create the model
    network = AbaloneModel(
        num_filters=config['model']['num_filters'],
        num_blocks=config['model']['num_blocks']
    )
    
    # Create the environment
    env = AbaloneEnv()
    
    # Get evaluation parameters
    eval_games = config.get('evaluation', {}).get('num_games', 2)
    
    # Create the trainer
    trainer = AbaloneTrainerSync(
        network=network,
        env=env,
        buffer_size=config['buffer']['size'],
        batch_size=config['training']['batch_size'],
        value_weight=config['training']['value_weight'],
        num_simulations=config['mcts']['num_simulations'],
        recency_bias=config['buffer'].get('recency_bias', True),
        recency_temperature=config['buffer'].get('recency_temperature', 0.8),
        initial_lr=config['optimizer']['initial_lr'],
        momentum=config['optimizer']['momentum'],
        lr_schedule=config['optimizer'].get('lr_schedule', None),
        checkpoint_path=config['checkpoint']['path'],
        log_dir=args.log_dir,
        gcs_bucket=args.gcs_bucket,
        save_games=True,
        eval_games=eval_games
    )
    
    # Load checkpoint if specified
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    return trainer



def main():
    """Main entry point"""
    args = parse_args()
    config = get_merged_config(args)
    
    display_hardware_info()
    display_config_summary(config)
    
    if args.mode == 'train':
        trainer = create_trainer(config, args)
        
        # Launch training with evaluation configuration
        print("\n=== Starting training ===")
        
        # Get evaluation frequency (0 means disabled)
        eval_frequency = config['checkpoint']['eval_frequency']
        if args.no_eval:
            eval_frequency = 0
            print("Evaluation disabled")
        else:
            eval_games = config.get('evaluation', {}).get('num_games', 5)
            print(f"Evaluation: Every {eval_frequency} iterations, {eval_games} games per algorithm")
        
        trainer.train(
            num_iterations=config['training']['num_iterations'],
            games_per_iteration=config['training']['games_per_iteration'],
            training_steps_per_iteration=config['training']['training_steps_per_iteration'],
            eval_frequency=eval_frequency,
            save_frequency=config['checkpoint']['save_frequency']
        )
    
    elif args.mode == 'eval':
        # To implement: model evaluation
        print("Evaluation mode not implemented. Use the Evaluator class directly.")
    
    elif args.mode == 'play':
        # To implement: game interface
        print("Play mode not implemented.")


if __name__ == "__main__":
    main()