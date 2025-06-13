import jax
jax.distributed.initialize() 

import os
import sys
import json
import argparse
import time
import datetime
import warnings
warnings.filterwarnings("ignore")
from model.neural_net import AbaloneModel
from environment.env import AbaloneEnv
from training.trainer import AbaloneTrainerSync
from training.config import DEFAULT_CONFIG, CPU_CONFIG, get_config

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Process %(process)d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("alphazero.main")

IS_MAIN_PROCESS = jax.process_index() == 0

def main_process_log(message, level=logging.INFO):
    """Journalise uniquement si c'est le processus principal"""
    if IS_MAIN_PROCESS:
        if level == logging.INFO:
            logger.info(message)
        elif level == logging.WARNING:
            logger.warning(message)
        elif level == logging.ERROR:
            logger.error(message)
        elif level == logging.DEBUG:
            logger.debug(message)

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
    parser.add_argument('--no-eval', action='store_true',
                   help='Disable evaluation during training')
    parser.add_argument('--use-gcs-buffer', action='store_true',
                   help='Use a global buffer on Google Cloud Storage')
    parser.add_argument('--gcs-buffer-dir', type=str, default='buffer',
                    help='Directory in the GCS bucket for the buffer')
    parser.add_argument('--verbose', action='store_true',
                   help='Enable verbose output')
    parser.add_argument('--enable-comprehensive-logging', action='store_true', default=True,
                   help='Enable comprehensive metrics logging to GCS/local files')
    parser.add_argument('--disable-comprehensive-logging', action='store_true',
                   help='Disable comprehensive metrics logging')
    
    # Model options
    parser.add_argument('--num-filters', type=int, default=None,
                       help='Number of filters in the network')
    parser.add_argument('--num-blocks', type=int, default=None,
                       help='Number of residual blocks')
    
    # MCTS options  
    parser.add_argument('--num-simulations', type=int, default=None,
                       help='Number of MCTS simulations per action')

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
    
    if args.num_simulations:
        config['mcts']['num_simulations'] = args.num_simulations
    
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
    
    if args.use_gcs_buffer:
        config['buffer']['use_gcs'] = True
        
    if args.gcs_buffer_dir:
        config['buffer']['gcs_dir'] = args.gcs_buffer_dir
    
    return config


def display_config_summary(config):
    """Display a summary of the configuration"""
    main_process_log("\n=== Configuration ===")
    main_process_log(f"Model: {config['model']['num_filters']} filters, {config['model']['num_blocks']} blocks")
    main_process_log(f"Buffer: {config['buffer']['size']} positions")
    main_process_log(f"Training: {config['training']['num_iterations']} iterations, {config['training']['games_per_iteration']} games/iter")
    main_process_log(f"Batch: {config['training']['batch_size']}, {config['training']['training_steps_per_iteration']} steps/iter")
    main_process_log(f"MCTS: {config['mcts']['num_simulations']} simulations per action")
    main_process_log(f"Checkpoints: {config['checkpoint']['path']}")
    
    # Show logging configuration
    logging_config = config.get('logging', {})
    if logging_config.get('enable_comprehensive_logging', True):
        main_process_log(f"Comprehensive logging: Enabled")
    else:
        main_process_log(f"Comprehensive logging: Disabled")


def display_hardware_info():
    """Display information about available hardware"""
    local_device_count = jax.local_device_count()
    global_device_count = jax.device_count()
    process_index = jax.process_index()
    process_count = jax.process_count()

    # Information matérielle - affichée pour tous les processus
    logger.info(f"Process {process_index+1}/{process_count} - Local devices: {local_device_count}")
    
    # Information globale - affichée uniquement par le processus principal
    if IS_MAIN_PROCESS:
        main_process_log(f"\n=== Hardware Configuration ===")
        main_process_log(f"Total devices across all processes: {global_device_count}")

    # Détails sur le type de matériel - affichés par chaque processus
    local_devices = jax.local_devices()
    if not local_devices:
        logger.error("Platform: No local devices found!")
        return

    first_device = local_devices[0]
    platform_msg = f"Process {process_index+1}/{process_count} - "
    
    if first_device.platform == 'tpu':
        platform_msg += f"Platform: TPU ({first_device.device_kind})"
    elif first_device.platform == 'gpu':
        platform_msg += f"Platform: GPU ({first_device.device_kind})"
    else:
        platform_msg += f"Platform: {first_device.platform}"
    
    logger.info(platform_msg)


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
        eval_games=eval_games,
        use_gcs_buffer=args.use_gcs_buffer,
        gcs_buffer_dir=args.gcs_buffer_dir,
        verbose=args.verbose,
        enable_comprehensive_logging=(args.enable_comprehensive_logging and not args.disable_comprehensive_logging) if hasattr(args, 'enable_comprehensive_logging') else config.get('logging', {}).get('enable_comprehensive_logging', True))

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
        
        main_process_log("\n=== Starting training ===")
        
        # Configuration de l'évaluation
        if args.no_eval:
            main_process_log("Evaluation disabled")
        else:
            eval_games = config.get('evaluation', {}).get('num_games', 5)
            main_process_log(f"Evaluation: Automatic at reference checkpoints, {eval_games} games per model")
            
            # Activer l'évaluation
            trainer.enable_evaluation(enable=True)
        
        trainer.train(
            num_iterations=config['training']['num_iterations'],
            games_per_iteration=config['training']['games_per_iteration'],
            training_steps_per_iteration=config['training']['training_steps_per_iteration'],
            save_frequency=config['checkpoint']['save_frequency']
        )
    
    elif args.mode == 'eval':
        main_process_log("Evaluation mode not implemented. Use the Evaluator class directly.")
    
    elif args.mode == 'play':
        main_process_log("Play mode not implemented.")


if __name__ == "__main__":
    main()