import os

import jax

# jax.distributed.initialize() sets up a multi-host TPU pod. On a single host
# (Colab, one TPU VM, CPU, GPU) it is unnecessary and can fail or hang, so allow
# skipping it. Default behaviour is unchanged for pod runs.
if os.environ.get("ABALONE_NO_DISTRIBUTED", "").lower() not in ("1", "true", "yes"):
    jax.distributed.initialize()

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
    parser.add_argument('--training-steps', type=int, default=None,
                       help='Number of training steps per iteration')
    parser.add_argument('--checkpoint-path', type=str, default=None,
                       help='Path to save checkpoints')
    parser.add_argument('--log-dir', type=str, default=None,
                       help='Directory for TensorBoard logs')
    parser.add_argument('--gcs-bucket', type=str, default=None,
                       help='GCS bucket name to store games')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint to load for resuming training')
    parser.add_argument('--save-frequency', type=int, default=None,
                       help='Iterations between periodic checkpoint saves')
    parser.add_argument('--vertex-tensorboard-id', type=str, default=None,
                       help='Vertex AI TensorBoard instance ID to stream metrics to (optional)')
    parser.add_argument('--gcp-project', type=str, default=None,
                       help='GCP project for --vertex-tensorboard-id (defaults to the ambient ADC project)')
    parser.add_argument('--gcp-location', type=str, default='europe-west4',
                       help='Region of the Vertex AI TensorBoard instance')
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
    parser.add_argument('--max-considered-actions', type=int, default=None,
                       help='Root actions sampled by Gumbel MuZero (default 64). '
                            'Actions outside this set are unplayable at the root, '
                            'so it must stay comparable to the branching factor '
                            '(~66 legal moves in Abalone). Costs no extra compute: '
                            'Sequential Halving splits the same --num-simulations.')

    # Game options
    parser.add_argument('--max-moves', type=int, default=None,
                       help='Move limit before a game is truncated (default 200)')
    parser.add_argument('--history-length', type=int, default=None,
                       help='Past positions fed to the network (0 = disabled)')
    parser.add_argument('--win-threshold', type=int, default=None,
                       help='Marbles to push out to win (default 3; the real game is 6). '
                            'Curriculum knob: raise it by hand when the CURRICULUM block '
                            'of the iteration summary says the criterion is met, resuming '
                            'from a checkpoint and passing --reset-buffer.')
    parser.add_argument('--curriculum-target', type=int, default=None,
                       help='Threshold the curriculum stops at (default 6, the real '
                            'rule of Abalone). The curriculum exists only to reach it.')
    parser.add_argument('--no-curriculum', action='store_true',
                       help='Pin --win-threshold instead of letting the trainer raise it '
                            'automatically as the agent learns to convert.')
    parser.add_argument('--reset-buffer', action='store_true',
                       help='Empty the replay buffer at startup. Use it whenever you resume '
                            'a checkpoint with a different --win-threshold: the stored value '
                            'targets were computed under the old threshold and are wrong '
                            'under the new one.')

    return parser.parse_args()


def get_merged_config(args):
    """Combine configuration from defaults and command line arguments"""
    # Load base configuration
    import copy
    if args.cpu_only:
        config = copy.deepcopy(CPU_CONFIG)
    else:
        config = copy.deepcopy(get_config())
    
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

    if args.max_considered_actions:
        config['mcts']['max_num_considered_actions'] = args.max_considered_actions

    if args.max_moves is not None:
        config.setdefault('game', {})['max_moves'] = args.max_moves

    if args.history_length is not None:
        config.setdefault('game', {})['history_length'] = args.history_length

    if args.win_threshold is not None:
        config.setdefault('game', {})['win_threshold'] = args.win_threshold

    if args.curriculum_target is not None:
        config.setdefault('game', {})['curriculum_target'] = args.curriculum_target

    if args.no_curriculum:
        config.setdefault('game', {})['curriculum_enabled'] = False
    
    if args.training_steps:
        config['training']['training_steps_per_iteration'] = args.training_steps
    
    if args.checkpoint_path:
        config['checkpoint']['path'] = args.checkpoint_path

    if args.save_frequency:
        config['checkpoint']['save_frequency'] = args.save_frequency
    
    # If using GCS bucket, update paths for cloud storage
    if args.gcs_bucket:
        bucket_path = "gs://{}".format(args.gcs_bucket)
        if not args.checkpoint_path:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            config['checkpoint']['path'] = "{}/checkpoints/model_{}".format(bucket_path, timestamp)
        
        if not args.log_dir:
            if 'log_dir' not in config:
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                args.log_dir = "{}/logs/abalone_az_{}".format(bucket_path, timestamp)
    
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
    main_process_log(f"MCTS: {config['mcts']['num_simulations']} simulations per action, "
                     f"{config['mcts'].get('max_num_considered_actions', 64)} root actions considered")
    game_cfg = config.get('game', {})
    main_process_log(f"Game: move limit {game_cfg.get('max_moves', 200)}, "
                     f"history length {game_cfg.get('history_length', 0)}, "
                     f"win threshold {game_cfg.get('win_threshold', 3)} marbles "
                     f"(real game: 6)")
    if game_cfg.get('curriculum_enabled', True):
        main_process_log(f"Curriculum: automatic, raising the threshold up to "
                         f"{game_cfg.get('curriculum_target', 6)} marbles")
    else:
        main_process_log("Curriculum: disabled, threshold pinned")
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
    game_cfg = config.get('game', {})
    env = AbaloneEnv(
        max_moves=game_cfg.get('max_moves', 200),
        history_length=game_cfg.get('history_length', 0),
        win_threshold=game_cfg.get('win_threshold', 3),
    )

    # Get evaluation parameters
    eval_games = config.get('evaluation', {}).get('num_games', 2)

    # Create the trainer
    logger.info(f"Process {jax.process_index()+1}/{jax.process_count()} - Using {config['mcts']['num_simulations']} MCTS simulations")
    trainer = AbaloneTrainerSync(
        network=network,
        env=env,
        buffer_size=config['buffer']['size'],
        batch_size=config['training']['batch_size'],
        value_weight=config['training']['value_weight'],
        num_simulations=config['mcts']['num_simulations'],
        max_num_considered_actions=config['mcts'].get('max_num_considered_actions', 64),
        eval_simulations=config['mcts'].get('eval_simulations'),
        recency_bias=config['buffer'].get('recency_bias', True),
        recency_temperature=config['buffer'].get('recency_temperature', 0.8),
        initial_lr=config['optimizer']['initial_lr'],
        momentum=config['optimizer']['momentum'],
        weight_decay=config['optimizer'].get('weight_decay', 1e-4),
        lr_schedule=config['optimizer'].get('lr_schedule', None),
        checkpoint_path=config['checkpoint']['path'],
        log_dir=args.log_dir,
        gcs_bucket=args.gcs_bucket,
        save_games=True,
        eval_games=eval_games,
        use_gcs_buffer=args.use_gcs_buffer,
        gcs_buffer_dir=args.gcs_buffer_dir,
        verbose=args.verbose,
        enable_comprehensive_logging=(args.enable_comprehensive_logging and not args.disable_comprehensive_logging) if hasattr(args, 'enable_comprehensive_logging') else config.get('logging', {}).get('enable_comprehensive_logging', True),
        vertex_tensorboard_id=args.vertex_tensorboard_id,
        gcp_project=args.gcp_project,
        gcp_location=args.gcp_location,
        curriculum_enabled=game_cfg.get('curriculum_enabled', True),
        curriculum_target=game_cfg.get('curriculum_target', 6))

    # Load checkpoint if specified
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    # Apres le checkpoint : load_checkpoint() avertit si le palier a change,
    # et --reset-buffer est justement la reponse a cet avertissement.
    if args.reset_buffer:
        trainer.buffer.clear(tag="wt%d" % trainer.env.win_threshold)
        main_process_log("Replay buffer vide sur demande (--reset-buffer)")

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