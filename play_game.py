#!/usr/bin/env python3
"""
Interactive game interface for Abalone AlphaZero
Play against random agent or trained model
"""

import random
import numpy as np
import jax
import jax.numpy as jnp
import pickle
import time
from environment.env import AbaloneEnv, AbaloneState
from core.board import display_board
from core.core import DIRECTIONS
from model.neural_net import AbaloneModel
from core.coord_conversion import prepare_input_legacy, cube_to_2d
from mcts.agent import get_best_move

class GameInterface:
    def __init__(self):
        self.env = AbaloneEnv()
        self.model = None
        self.model_params = None
        self.mcts_simulations = 800
    
    def display_board(self, state):
        """Display the current board state using the proper Abalone display function"""
        print("\nCurrent board state:")
        print("  Player to move:", "Black (●)" if state.actual_player == 1 else "White (○)")
        print("  Score - Black out:", state.black_out, "White out:", state.white_out)
        print("  Moves played:", state.moves_count)
        
        # Use the built-in display function from core.board
        display_board(state.board, self.env.radius)
    
    def load_model(self, model_path: str, num_filters: int = 128, num_blocks: int = 10):
        """Load a trained model from pickle file"""
        try:
            print(f"Loading model from {model_path}...")
            
            # Load checkpoint
            with open(model_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            # Extract model parameters
            if 'params' in checkpoint:
                self.model_params = checkpoint['params']
                print(f"Loaded model from iteration {checkpoint.get('iteration', 'unknown')}")
            else:
                print("Warning: No 'params' key found, trying to use checkpoint directly")
                self.model_params = checkpoint
            
            # Initialize model with the correct architecture
            self.model = AbaloneModel(
                num_actions=1734,
                num_filters=num_filters,
                num_blocks=num_blocks
            )
            
            print(f"Model loaded successfully!")
            print(f"Architecture: {num_filters} filters, {num_blocks} blocks")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.model_params = None
            return False
    
    def set_mcts_simulations(self, num_simulations: int = 800):
        """Set the number of MCTS simulations"""
        self.mcts_simulations = num_simulations
        print(f"MCTS simulations set to: {num_simulations}")
    
    def get_human_move(self, state):
        """Get move input from human player with coordinate-based selection"""
        legal_moves = self.env.get_legal_moves(state)
        legal_indices = np.where(np.array(legal_moves))[0]
        
        print(f"\nLegal moves available: {len(legal_indices)}")
        print("Move input options:")
        print("  'coord' - Select by coordinates and direction")
        print("  'index' - Select by move index")
        print("  'help'  - Show all legal moves")
        print("  'quit'  - Exit game")
        
        while True:
            try:
                user_input = input("> ").strip().lower()
                
                if user_input == 'quit':
                    return None
                elif user_input == 'help':
                    self.show_move_details(legal_indices)
                    continue
                elif user_input == 'coord':
                    move_idx = self.get_move_by_coordinates(state, legal_indices)
                    if move_idx is not None:
                        return move_idx
                    continue
                elif user_input == 'index':
                    move_idx = self.get_move_by_index(legal_indices)
                    if move_idx is not None:
                        return move_idx
                    continue
                else:
                    # Try to parse as direct index
                    try:
                        move_idx = int(user_input)
                        if move_idx in legal_indices:
                            return move_idx
                        else:
                            print(f"Invalid move {move_idx}. Use 'help' to see legal moves.")
                    except ValueError:
                        print("Please enter 'coord', 'index', 'help', or 'quit'")
                    
            except KeyboardInterrupt:
                return None
    
    def show_move_details(self, legal_indices):
        """Show detailed information about legal moves"""
        print("\nLegal moves (showing first 20):")
        print("Format: [Index] Position -> Direction (Type)")
        
        for i, move_idx in enumerate(legal_indices[:20]):
            if i >= 20:
                break
                
            try:
                position = self.env.moves_index['positions'][move_idx]
                direction = self.env.moves_index['directions'][move_idx]
                move_type = self.env.moves_index['move_types'][move_idx]
                group_size = self.env.moves_index['group_sizes'][move_idx]
                
                # Convert direction index to direction name
                direction_names = ['NE', 'E', 'SE', 'SW', 'W', 'NW']
                dir_name = direction_names[direction] if direction < 6 else f"Dir{direction}"
                
                # Convert move type to readable format
                type_names = ['Single', 'Parallel', 'Inline']
                type_name = type_names[move_type] if move_type < 3 else f"Type{move_type}"
                
                # Format position (only show first position for groups)
                pos_str = f"({position[0][0]},{position[0][1]},{position[0][2]})"
                if group_size > 1:
                    pos_str += f" (+{group_size-1})"
                
                print(f"[{move_idx:3d}] {pos_str} -> {dir_name} ({type_name})")
                
            except (IndexError, KeyError):
                print(f"[{move_idx:3d}] <error reading move data>")
        
        if len(legal_indices) > 20:
            print(f"... and {len(legal_indices) - 20} more moves")
        print()
    
    def get_move_by_index(self, legal_indices):
        """Get move by entering index directly"""
        print(f"Legal move indices: {legal_indices[:15]}{'...' if len(legal_indices) > 15 else ''}")
        try:
            move_idx = int(input("Enter move index: ").strip())
            if move_idx in legal_indices:
                return move_idx
            else:
                print(f"Invalid move index {move_idx}")
                return None
        except ValueError:
            print("Please enter a valid number")
            return None
    
    def get_move_by_coordinates(self, state, legal_indices):
        """Get move by selecting marbles and direction"""
        print("\n=== Coordinate-based Move Selection ===")
        
        # Step 1: Show player's marbles
        player_marbles = self.get_player_marbles(state)
        print(f"Your marbles ({len(player_marbles)} total):")
        for i, (x, y, z) in enumerate(player_marbles):
            print(f"  {i+1:2d}. ({x:2d},{y:2d},{z:2d})")
        
        # Step 2: Get marble selection
        selected_marbles = self.select_marbles(player_marbles)
        if not selected_marbles:
            return None
            
        # Step 3: Get direction
        direction = self.select_direction()
        if direction is None:
            return None
            
        # Step 4: Find matching move
        move_idx = self.find_move_index(selected_marbles, direction, legal_indices)
        if move_idx is not None:
            print(f"Selected move: {move_idx}")
            return move_idx
        else:
            print("No valid move found for this marble/direction combination")
            return None
    
    def get_player_marbles(self, state):
        """Get positions of current player's marbles"""
        marbles = []
        radius = self.env.radius
        board = np.array(state.board)
        
        for x in range(-radius, radius + 1):
            for y in range(-radius, radius + 1):
                for z in range(-radius, radius + 1):
                    if x + y + z == 0:  # Valid cubic coordinate
                        array_pos = (x + radius, y + radius, z + radius)
                        if not np.isnan(board[array_pos]) and board[array_pos] == 1:
                            marbles.append((x, y, z))
        
        return sorted(marbles)
    
    def select_marbles(self, player_marbles):
        """Let user select marbles for the move"""
        print("\nSelect marbles to move:")
        print("  Enter marble numbers (e.g., '1' for single, '1,2,3' for group)")
        print("  Or enter coordinates directly (e.g., '(0,1,-1)' or '(0,1,-1),(1,0,-1)')")
        
        while True:
            try:
                user_input = input("Select marbles: ").strip()
                if not user_input:
                    continue
                
                # Try parsing as coordinate format
                if '(' in user_input and ')' in user_input:
                    selected = self.parse_coordinate_input(user_input)
                    if selected:
                        # Validate coordinates are player's marbles
                        valid_selection = []
                        for coord in selected:
                            if coord in player_marbles:
                                valid_selection.append(coord)
                            else:
                                print(f"Coordinate {coord} is not your marble")
                        if valid_selection:
                            return valid_selection
                    continue
                
                # Try parsing as marble indices
                indices = [int(x.strip()) for x in user_input.split(',')]
                selected = []
                for idx in indices:
                    if 1 <= idx <= len(player_marbles):
                        selected.append(player_marbles[idx - 1])
                    else:
                        print(f"Invalid marble number {idx}. Must be 1-{len(player_marbles)}")
                        break
                else:
                    if len(selected) <= 3:
                        return selected
                    else:
                        print("Maximum 3 marbles can be selected")
                        
            except ValueError:
                print("Invalid input format")
            except KeyboardInterrupt:
                return None
    
    def parse_coordinate_input(self, input_str):
        """Parse coordinate input like '(0,1,-1)' or '(0,1,-1),(1,0,-1)'"""
        try:
            import re
            # Find all coordinate patterns
            pattern = r'\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)'
            matches = re.findall(pattern, input_str)
            
            coords = []
            for match in matches:
                x, y, z = int(match[0]), int(match[1]), int(match[2])
                if x + y + z == 0:  # Valid cubic coordinate
                    coords.append((x, y, z))
                else:
                    print(f"Invalid cubic coordinate ({x},{y},{z}): x+y+z must equal 0")
                    return None
            
            return coords
        except:
            print("Invalid coordinate format. Use: (x,y,z) or (x1,y1,z1),(x2,y2,z2)")
            return None
    
    def select_direction(self):
        """Let user select movement direction"""
        directions = ['NE', 'E', 'SE', 'SW', 'W', 'NW']
        
        print("\nSelect direction:")
        for i, direction in enumerate(directions):
            print(f"  {i+1}. {direction}")
        
        while True:
            try:
                user_input = input("Enter direction (1-6 or name): ").strip()
                
                # Try parsing as number
                if user_input.isdigit():
                    dir_idx = int(user_input) - 1
                    if 0 <= dir_idx < 6:
                        return dir_idx
                    else:
                        print("Please enter 1-6")
                        continue
                
                # Try parsing as direction name
                user_input = user_input.upper()
                if user_input in directions:
                    return directions.index(user_input)
                
                print("Invalid direction. Enter 1-6 or direction name (NE, E, SE, SW, W, NW)")
                
            except KeyboardInterrupt:
                return None
    
    def find_move_index(self, selected_marbles, direction, legal_indices):
        """Find the move index that matches the selected marbles and direction"""
        # Sort selected marbles for consistent comparison
        selected_set = set(selected_marbles)
        
        for move_idx in legal_indices:
            # Get move data
            move_positions = self.env.moves_index['positions'][move_idx]
            move_direction = self.env.moves_index['directions'][move_idx]
            move_group_size = self.env.moves_index['group_sizes'][move_idx]
            
            # Check if direction matches
            if move_direction != direction:
                continue
            
            # Get positions from move (only first group_size positions are used)
            move_coords = set()
            for i in range(move_group_size):
                pos = tuple(move_positions[i])
                move_coords.add(pos)
            
            # Check if selected marbles match move positions
            if selected_set == move_coords:
                return move_idx
        
        return None
    
    def get_random_move(self, state):
        """Get random move"""
        legal_moves = self.env.get_legal_moves(state)
        legal_indices = np.where(np.array(legal_moves))[0]
        
        if len(legal_indices) == 0:
            return None
        
        return random.choice(legal_indices)
    
    
    def get_model_move(self, state):
        """Get move from trained model"""
        if self.model is None or self.model_params is None:
            print("No model loaded, using random move")
            return self.get_random_move(state)
        
        try:
            # Calculate marbles out from current player's perspective
            if state.actual_player == 1:
                # Black's turn: our marbles out = black_out, opponent's = white_out
                our_marbles = state.black_out
                opponent_marbles = state.white_out
            else:
                # White's turn: our marbles out = white_out, opponent's = black_out  
                our_marbles = state.white_out
                opponent_marbles = state.black_out
            
            # Use prepare_input_legacy (consistent with training/MCTS)
            board_2d, marbles_out = prepare_input_legacy(
                board_3d=state.board,
                our_marbles_out=our_marbles,
                opponent_marbles_out=opponent_marbles,
                radius=self.env.radius
            )
            
            # Convert history: (8, 9, 9, 9) -> (1, 8, 9, 9)
            history_2d = jax.vmap(lambda h: cube_to_2d(h, self.env.radius))(state.history)
            history_2d = history_2d[None, ...]  # Add batch dimension
            
            # Get model prediction
            prior_logits, value = self.model.apply(
                self.model_params, 
                board_2d,      # (1, 9, 9)
                marbles_out,   # (1, 2) 
                history_2d     # (1, 8, 9, 9)
            )
            
            # Get legal moves and mask invalid moves
            legal_moves = self.env.get_legal_moves(state)
            legal_mask = jnp.array(legal_moves)
            
            # Mask invalid moves with large negative values
            masked_logits = jnp.where(legal_mask, prior_logits[0], -1e9)
            
            # Select move with highest probability
            move_idx = jnp.argmax(masked_logits)
            
            # Display model evaluation
            print(f"Model evaluation: {float(value[0]):.3f}")
            
            # Show top 3 moves for transparency
            legal_indices = jnp.where(legal_mask)[0]
            if len(legal_indices) > 1:
                legal_probs = jax.nn.softmax(masked_logits[legal_indices])
                top_indices = jnp.argsort(legal_probs)[-3:][::-1]  # Top 3 in descending order
                
                print("Top model moves:")
                for i, idx in enumerate(top_indices[:3]):
                    move_id = legal_indices[idx]
                    prob = legal_probs[idx]
                    print(f"  {i+1}. Move {move_id}: {prob:.1%}")
            
            return int(move_idx)
            
        except Exception as e:
            print(f"Error getting model move: {e}")
            print("Falling back to random move")
            return self.get_random_move(state)
    
    def get_mcts_move(self, state):
        """Get move from MCTS search (true AlphaZero)"""
        if self.model is None or self.model_params is None:
            print("No model loaded, using random move")
            return self.get_random_move(state)
        
        try:
            print(f"🧠 MCTS thinking... ({self.mcts_simulations} simulations)")
            
            # Create RNG key
            rng_key = jax.random.PRNGKey(int(time.time() * 1000) % (2**32))
            
            # CANONICALIZE HISTORY FOR MCTS
            # The network expects canonical history where current player's pieces are always 1
            canonical_history = jnp.where(
                state.actual_player == 1, 
                state.history,      # If actual_player == 1, history is already correct
                -state.history      # If actual_player == -1, flip history so current player sees pieces as 1
            )
            
            # Create state with canonical history for MCTS
            canonical_state = AbaloneState(
                board=state.board,           # Board is already canonical
                history=canonical_history,   # Now history is canonical too
                actual_player=state.actual_player,
                black_out=state.black_out,
                white_out=state.white_out,
                moves_count=state.moves_count
            )
            
            # Get MCTS move using the canonical state
            move = get_best_move(
                state=canonical_state,      # ← Use canonical state
                params=self.model_params,
                network=self.model,
                env=self.env,
                num_simulations=self.mcts_simulations,
                rng_key=rng_key,
                iteration=10
            )
            
            print(f"MCTS completed {self.mcts_simulations} simulations")
            print(f"MCTS chose move: {move}")
            
            return int(move)
            
        except Exception as e:
            print(f"Error getting MCTS move: {e}")
            print("Falling back to network-only move")
            return self.get_model_move(state)
    
    def play_game(self, player1_type="human", player2_type="random", model_path=None):
        """
        Play a game between two agents
        
        Args:
            player1_type: "human", "random", "model", or "mcts" (plays as Black)
            player2_type: "human", "random", "model", or "mcts" (plays as White)
            model_path: path to model file (required if using "model" or "mcts")
        """
        # Load model if needed
        if (player1_type in ["model", "mcts"] or player2_type in ["model", "mcts"]):
            if model_path is None:
                print("Model path required for model/MCTS players")
                return
            
            if not self.load_model(model_path):
                print("Failed to load model, falling back to random for AI players")
                player1_type = "random" if player1_type in ["model", "mcts"] else player1_type
                player2_type = "random" if player2_type in ["model", "mcts"] else player2_type
        
        # Initialize game
        rng = jax.random.PRNGKey(random.randint(0, 1000000))
        state = self.env.reset(rng)
        
        print(f"Starting game: {player1_type} (Black) vs {player2_type} (White)")
        
        move_count = 0
        while not self.env.is_terminal(state) and move_count < 500:
            self.display_board(state)
            
            # Determine current player type
            current_player_type = player1_type if state.actual_player == 1 else player2_type
            player_name = "Black" if state.actual_player == 1 else "White"
            
            print(f"{player_name}'s turn ({current_player_type})")
            
            # Get move based on player type
            if current_player_type == "human":
                move = self.get_human_move(state)
                if move is None:  # Player quit
                    print("Game ended by player")
                    return
            elif current_player_type == "random":
                move = self.get_random_move(state)
                print(f"Random player chose move: {move}")
            elif current_player_type == "model":
                move = self.get_model_move(state)
                print(f"Model chose move: {move}")
            elif current_player_type == "mcts":
                move = self.get_mcts_move(state)
                print(f"MCTS chose move: {move}")
            else:
                print(f"Unknown player type: {current_player_type}")
                return
            
            if move is None:
                print("No legal moves available!")
                break
            
            # Execute move
            state = self.env.step(state, move)
            move_count += 1
            
            # Small delay for readability
            if current_player_type != "human":
                # Auto-play for MCTS vs Random, pause for other combinations
                if not (player1_type in ["mcts", "random"] and player2_type in ["mcts", "random"]):
                    input("Press Enter to continue...")
                else:
                    time.sleep(0.5)  # Brief pause to see the moves
        
        # Display final result
        self.display_board(state)
        winner = self.env.get_winner(state)
        
        if winner == 1:
            print("🎉 Black wins!")
        elif winner == -1:
            print("🎉 White wins!")
        else:
            print("🤝 Draw!")
        
        print(f"Game ended after {move_count} moves")
        print(f"Final score - Black out: {state.black_out}, White out: {state.white_out}")

def main():
    """Main game loop with menu"""
    interface = GameInterface()
    
    while True:
        print("\n" + "="*50)
        print("ABALONE ALPHAZERO - GAME INTERFACE")
        print("="*50)
        print("1. Human vs Random")
        print("2. Human vs Model (Network Only)")
        print("3. Human vs MCTS (True AlphaZero) ⭐")
        print("4. Random vs Random")
        print("5. Model vs Model")
        print("6. MCTS vs MCTS")
        print("7. Random vs MCTS")
        print("8. Human vs Human")
        print("9. Quit")
        
        try:
            choice = input("\nSelect option (1-9): ").strip()
            
            # Default model path
            default_model = "data/checkpoints_model_20250716-202800_20250717-000043_iter61.pkl"
            
            if choice == "1":
                interface.play_game("human", "random")
            elif choice == "2":
                interface.play_game("human", "model", default_model)
            elif choice == "3":
                interface.play_game("human", "mcts", default_model)
            elif choice == "4":
                interface.play_game("random", "random")
            elif choice == "5":
                interface.play_game("model", "model", default_model)
            elif choice == "6":
                interface.play_game("mcts", "mcts", default_model)
            elif choice == "7":
                interface.play_game("random", "mcts", default_model)
            elif choice == "8":
                interface.play_game("human", "human")
            elif choice == "9":
                print("Thanks for playing!")
                break
            else:
                print("Invalid choice. Please select 1-9.")
                
        except KeyboardInterrupt:
            print("\nThanks for playing!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()