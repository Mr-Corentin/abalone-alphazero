# AlphaZero for Abalone

A high-performance implementation of AlphaZero for the Abalone board game, built with JAX and optimized for TPU training with multi-host support.

## Overview

This project implements the AlphaZero algorithm specifically for Abalone, a strategic board game where players aim to push their opponent's marbles off the board. The implementation is designed to run efficiently on TPU clusters with distributed training capabilities

### Key Features

- **JAX-based neural network**: Fast, scalable training with automatic differentiation
- **Monte Carlo Tree Search (MCTS)**: Efficient tree search using the mctx library
- **Multi-host TPU training**: Optimized for distributed training across TPU pods
- **Self-play data generation**: Generates training data through self-play games
- **Cloud storage integration**: Google Cloud Storage for distributed training data

## Architecture

### Core Components

- **Environment** (`environment/`): Abalone game logic and state management
- **Neural Network** (`model/`): ResNet-based architecture for board evaluation
- **MCTS** (`mcts/`): Monte Carlo Tree Search implementation using mctx
- **Training** (`training/`): Self-play data generation and model training
- **Evaluation** (`evaluation/`): Model testing framework

### Game Representation

- **3D board representation**: Hexagonal board mapped to 3D coordinate system
- **Move encoding**: Comprehensive move representation supporting pushes and formations
- **State features**: Board position, player turn, marble counts, and game history

## Installation

### Requirements

- Python 3.8+
- JAX/JAXLib (TPU version)
- See `requirements.txt` for complete dependencies

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd abalone-alphazero

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training a Model

Basic training command:

```bash
python main.py --mode train --iterations 100 --games-per-iter 64
```

Advanced training with custom configuration:

```bash
python main.py --mode train \
    --iterations 200 \
    --games-per-iter 128 \
    --batch-size 256 \
    --num-simulations 800 \
    --checkpoint-path checkpoints/my_model \
    --log-dir logs/training
```

### Multi-Host TPU Training

For multi-host TPU training, run the training script on each TPU worker:

```bash
# Run this command on each TPU host
python main.py --mode train --gcs-bucket your-bucket-name
```

### Configuration Options

#### Model Architecture
- `--num-filters`: Number of convolutional filters (default: 128)
- `--num-blocks`: Number of residual blocks (default: 10)

#### MCTS Parameters
- `--num-simulations`: MCTS simulations per move (default: 600)

#### Training Parameters
- `--batch-size`: Training batch size (default: 128)
- `--games-per-iter`: Self-play games per iteration (default: 64)
- `--training-steps`: Training steps per iteration (default: 20)

#### Cloud Storage
- `--gcs-bucket`: Google Cloud Storage bucket for distributed training
- `--use-gcs-buffer`: Use cloud-based replay buffer

### Playing Against the Model

```bash
python play_game.py --model-path checkpoints/model_final.pkl
```

## Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Model filters | 128 | Convolutional filter count |
| Model blocks | 10 | Residual block count |
| MCTS simulations | 600 | Simulations per action |
| Batch size | 128 | Training batch size |
| Games per iteration | 64 | Self-play games per training iteration |
| Buffer size | 1,000,000 | Replay buffer capacity |
| Learning rate | 0.2 | Initial learning rate |

## Training Process

### Self-Play Loop

1. **Game Generation**: Generate self-play games using current model
2. **Data Collection**: Store game positions, moves, and outcomes
3. **Network Training**: Train neural network on collected data
4. **Model Evaluation**: Test new model against previous versions
5. **Checkpoint Saving**: Save model if performance improves

## Advanced Features

### Cloud Integration

- **Google Cloud Storage**: Store checkpoints and training data
- **Comprehensive logging**: Track metrics and game data
- **Distributed replay buffer**: Share training data across workers

### Reward Functions

The implementation supports multiple reward strategies:

- **Terminal-only rewards**: Standard AlphaZero approach (default)
- **Intermediate rewards**: Rewards for marble pushes
- **Curriculum learning**: Gradually transition between reward strategies

## Project Structure

```
abalone-alphazero/
├── README.md
├── requirements.txt
├── main.py                     # Main training script
├── play_game.py               # Interactive game playing
├── environment/               # Abalone game logic
│   ├── __init__.py
│   ├── abalone_env.py        # Game environment implementation
│   ├── board.py              # Board representation and logic
│   └── moves.py              # Move generation and validation
├── model/                     # Neural network architecture
│   ├── __init__.py
│   ├── network.py            # ResNet-based model
│   └── utils.py              # Model utilities
├── mcts/                      # Monte Carlo Tree Search
│   ├── __init__.py
│   ├── search.py             # MCTS implementation using mctx
│   └── node.py               # Tree node structure
├── training/                  # Training pipeline
│   ├── __init__.py
│   ├── self_play.py          # Self-play game generation
│   ├── trainer.py            # Model training logic
│   └── buffer.py             # Replay buffer management
├── evaluation/                # Model evaluation
│   ├── __init__.py
│   ├── evaluator.py          # Model testing framework
│   └── metrics.py            # Performance metrics
├── utils/                     # Shared utilities
│   ├── __init__.py
│   ├── logging.py            # Logging utilities
│   └── storage.py            # GCS integration
├── config/                    # Configuration files
│   ├── __init__.py
│   └── default_config.py     # Default parameters
├── checkpoints/               # Model checkpoints
└── logs/                      # Training logs
```

## Acknowledgments

This project was developed with support from the TPU Research Cloud (TRC) program, which provided access to TPU resources for training and experimentation.

- **mctx**: DeepMind's MCTS library for JAX, used for efficient tree search implementation
- **JAX**: Google's JAX library for high-performance machine learning
- **TPU Research Cloud**: For providing computational resources
