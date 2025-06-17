import numpy as np
import time
import math
from typing import Dict, List, Tuple, Any
import sys
import os
import jax
from environment.env import AbaloneState, AbaloneEnvNonCanonical, AbaloneStateNonCanonical
from core.board import create_board_mask, display_board, create_custom_board
from mcts.agent import get_best_move
from evaluation.alphabeta.heuristics import alphabeta_pruning


class EvaluationGameState:
    """Helper class to maintain game state during evaluation"""
    def __init__(self, board, current_player, black_out=0, white_out=0, moves_count=0):
        self.board = board
        self.current_player = current_player  # 1 for black, -1 for white
        self.black_out = black_out
        self.white_out = white_out
        self.moves_count = moves_count
        
    def copy(self):
        return EvaluationGameState(
            board=np.copy(self.board),
            current_player=self.current_player,
            black_out=self.black_out,
            white_out=self.white_out,
            moves_count=self.moves_count
        )
    
    def to_non_canonical_state(self):
        """Converts to AbaloneStateNonCanonical for evaluation functions"""
        return AbaloneStateNonCanonical(
            board=self.board,
            current_player=self.current_player,
            black_out=self.black_out,
            white_out=self.white_out,
            moves_count=self.moves_count
        )
    
def simulate_alphazero_vs_classical(params, network, env, classical_algo="alphabeta_pruning", depth=3, max_moves=250, verbose=False):
    """
    Simulates a game between AlphaZero and a classical algorithm.
    
    Args:
        params: AlphaZero model parameters
        network: Neural network model
        env: Game environment
        classical_algo: Algorithm to use ("alphabeta_pruning")
        depth: Search depth for classical algorithm
        max_moves: Maximum number of moves before draw
        verbose: Whether to display game progress
        
    Returns:
        Game result as string
    """
    # Create non-canonical environment for simulation
    env_non_canonical = AbaloneEnvNonCanonical(radius=env.radius)
    
    # Initialize non-canonical state
    rng = jax.random.PRNGKey(42)
    game_state = env_non_canonical.reset(rng)
    
    if verbose:
        print(f"AlphaZero (Black) vs {classical_algo} (White), depth {depth}")
        display_board(game_state.board)
    
    # Play the game
    move_count = 0
    
    while move_count < max_moves and game_state.black_out < 6 and game_state.white_out < 6:
        if verbose and move_count % 10 == 0:
            print(f"Move {move_count}/{max_moves}", end="\r")
        
        # Determine who plays
        if game_state.current_player == 1:  # AlphaZero's turn (Black)
            # Convert to canonical state for AlphaZero
            canonical_board = env.get_canonical_state(game_state.board, game_state.current_player)
            canonized_state = AbaloneState(
                board=canonical_board,
                actual_player=game_state.current_player,
                black_out=game_state.black_out,
                white_out=game_state.white_out,
                moves_count=game_state.moves_count
            )
            
            # Get AlphaZero's move
            action_idx = get_best_move(canonized_state, params, network, env, num_simulations=50)
            
            # Apply move with non-canonical environment
            game_state = env_non_canonical.step(game_state, action_idx)
            
        else:  # Classical algorithm's turn (White)
            # Use non-canonical state directly for alpha-beta
            _, action_idx = alphabeta_pruning(
                game_state,
                depth,
                float('-inf'), 
                float('inf'),
                env_non_canonical, 
                radius=env.radius
            )
            
            # Apply move
            game_state = env_non_canonical.step(game_state, action_idx)
        
        move_count += 1
        
        if verbose and move_count % 10 == 0:
            print(f"\nState after {move_count} moves:")
            display_board(game_state.board)
            print(f"Marbles out - Black: {game_state.black_out}, White: {game_state.white_out}")
    
    # Determine result
    if game_state.white_out >= 6:
        result = "Black (AlphaZero)"
    elif game_state.black_out >= 6:
        result = "White (Classical Algorithm)"
    else:
        result = "Draw"
    
    if verbose:
        print(f"Game ended after {move_count} moves. Result: {result}")
    
    return result


class Evaluator:
    """
    Class for evaluating model performance against different algorithms
    """
    
    def __init__(self, model_params, network, env):
        self.params = model_params
        self.network = network
        self.env = env
    
    def evaluate_against_classical(self, algorithms=None, num_games_per_algo=2, verbose=True):
        """
        Evaluates the model against several classical algorithms
        
        Args:
            algorithms: List of (algorithm_name, depth) tuples
            num_games_per_algo: Number of games to play against each algorithm
            verbose: Whether to display detailed progress
            
        Returns:
            Dictionary of results by algorithm
        """
        if algorithms is None:
            algorithms = [
                ("alphabeta_pruning", 3)
            ]
        
        print("\n--- Model Evaluation ---")
        
        results = {}
        
        for algo_name, depth in algorithms:
            if verbose:
                print(f"\nEvaluating against {algo_name} (depth {depth})...")
            
            # Play multiple games
            wins = 0
            losses = 0
            draws = 0
            
            for game_idx in range(num_games_per_algo):
                if verbose:
                    print(f"Game {game_idx+1}/{num_games_per_algo}")
                
                # Play a game
                result = simulate_alphazero_vs_classical(
                    self.params, 
                    self.network, 
                    self.env, 
                    classical_algo=algo_name, 
                    depth=depth,
                    verbose=verbose
                )
                
                # Count result
                if result == "Black (AlphaZero)":
                    wins += 1
                elif result == "White (Classical Algorithm)":
                    losses += 1
                else:
                    draws += 1
            
            # Calculate statistics
            win_rate = wins / num_games_per_algo
            results[algo_name] = {
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_rate": win_rate
            }
            
            if verbose:
                print(f"Results against {algo_name}:")
                print(f"  Wins: {wins}/{num_games_per_algo} ({win_rate:.1%})")
                print(f"  Losses: {losses}/{num_games_per_algo} ({losses/num_games_per_algo:.1%})")
                print(f"  Draws: {draws}/{num_games_per_algo} ({draws/num_games_per_algo:.1%})")
        
        return results



def evaluate_model(self, num_games_per_algo=5):
    """
    Evaluates the current model against classical algorithms.
    This function is intended to be used within AbaloneTrainerSync.
    
    Args:
        num_games_per_algo: Number of games to play against each algorithm
        
    Returns:
        Dictionary with evaluation results
    """
    evaluator = Evaluator(self.params, self.network, self.env)
    
    algos = [
        ("alphabeta_pruning", 3)
    ]
    
    results = evaluator.evaluate_against_classical(
        algorithms=algos,
        num_games_per_algo=num_games_per_algo,
        verbose=True
    )
    
    latest_metrics = self.metrics_history[-1] if self.metrics_history else {}
    eval_metrics = {f"win_rate_{algo}": data["win_rate"] for algo, data in results.items()}
    
    # Update metrics with evaluation results
    if latest_metrics:
        latest_metrics.update(eval_metrics)
    
    return results


if __name__ == '__main__':
    print("Evaluator module loaded successfully")