import React, { useState, useEffect } from 'react';
import Board from './Board';
import { coordToString, stringToCoord } from '../utils/boardUtils';
import './Game.css';

const Game = () => {
  const [gameState, setGameState] = useState(null);
  const [selectedMarbles, setSelectedMarbles] = useState([]);
  const [showDirections, setShowDirections] = useState(false);
  const [gameStatus, setGameStatus] = useState('waiting'); // waiting, playing, finished
  const [currentPlayer, setCurrentPlayer] = useState(1); // 1 = black, -1 = white
  const [errorMessage, setErrorMessage] = useState('');

  // Initialize game on component mount
  useEffect(() => {
    initializeGame();
  }, []);

  const initializeGame = async () => {
    try {
      // Call backend API to start new game
      const response = await fetch('http://localhost:8000/game/new', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        throw new Error(`Failed to start game: ${response.statusText}`);
      }
      
      const gameState = await response.json();
      setGameState(gameState);
      setCurrentPlayer(gameState.actual_player); // Use actual_player for UI
      setGameStatus('playing');
      setSelectedMarbles([]);
      setShowDirections(false);
      setErrorMessage('');
      
    } catch (error) {
      console.error('Error initializing game:', error);
      setErrorMessage('Failed to connect to game server: ' + error.message);
    }
  };


  const handleMarbleClick = (x, y, z) => {
    if (gameStatus !== 'playing') return;
    
    const coordStr = coordToString(x, y, z);
    const marbleValue = getMarbleValue(x, y, z);
    
    // In canonical representation, current player's marbles are always 1
    // But we need to check against the actual visual player
    const actualCurrentPlayer = gameState.actual_player;
    
    // Check if this marble belongs to the current player
    // For visual consistency: 1 = black marbles, -1 = white marbles
    let isCurrentPlayerMarble = false;
    
    if (actualCurrentPlayer === 1) {
      // Black's turn - can select marbles that appear as 1 (black) 
      isCurrentPlayerMarble = (marbleValue === 1);
    } else {
      // White's turn - can select marbles that appear as -1 (white)
      isCurrentPlayerMarble = (marbleValue === -1);
    }
    
    if (!isCurrentPlayerMarble) {
      setErrorMessage(`It's ${actualCurrentPlayer === 1 ? 'Black' : 'White'}'s turn`);
      return;
    }

    setErrorMessage('');

    // Toggle marble selection
    const newSelectedMarbles = [...selectedMarbles];
    const marbleIndex = newSelectedMarbles.indexOf(coordStr);
    
    if (marbleIndex >= 0) {
      // Deselect marble
      newSelectedMarbles.splice(marbleIndex, 1);
    } else {
      // Select marble (max 3)
      if (newSelectedMarbles.length < 3) {
        newSelectedMarbles.push(coordStr);
      } else {
        setErrorMessage('Maximum 3 marbles can be selected');
        return;
      }
    }
    
    setSelectedMarbles(newSelectedMarbles);
    setShowDirections(newSelectedMarbles.length > 0);
  };

  const handleDirectionClick = async (direction) => {
    if (selectedMarbles.length === 0) {
      setErrorMessage('Select marbles first');
      return;
    }

    try {
      // Convert selected marbles to the format expected by backend
      const selectedCoords = selectedMarbles.map(coordStr => stringToCoord(coordStr));
      
      console.log('Making move:', {
        marbles: selectedCoords,
        direction: direction,
        player: currentPlayer
      });

      // TODO: Call backend API to validate and execute move
      await executeMove(selectedCoords, direction);
      
      // Clear selection after move
      setSelectedMarbles([]);
      setShowDirections(false);
      setErrorMessage('');
      
    } catch (error) {
      setErrorMessage('Invalid move: ' + error.message);
    }
  };

  const executeMove = async (marbles, direction) => {
    console.log('Executing move:', { marbles, direction });
    
    try {
      // Call backend API to execute move
      const response = await fetch('http://localhost:8000/game/move', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          marbles: marbles,
          direction: direction
        })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Move failed');
      }
      
      const newGameState = await response.json();
      
      // Update game state
      setGameState(newGameState);
      setCurrentPlayer(newGameState.actual_player); // Use actual_player for UI
      
      // Check if game is finished
      if (newGameState.is_terminal) {
        setGameStatus('finished');
        if (newGameState.winner === 1) {
          setErrorMessage('🎉 Black wins!');
        } else if (newGameState.winner === -1) {
          setErrorMessage('🎉 White wins!');
        } else {
          setErrorMessage('🤝 Game ended in a draw!');
        }
      }
      
      console.log('Move executed successfully');
      
    } catch (error) {
      console.error('Error executing move:', error);
      setErrorMessage(error.message);
    }
  };

  const getMarbleValue = (x, y, z) => {
    if (!gameState || !gameState.board) return 0;
    
    const arrayX = x + 4;
    const arrayY = y + 4;
    const arrayZ = z + 4;
    
    if (arrayX < 0 || arrayX >= 9 || arrayY < 0 || arrayY >= 9 || arrayZ < 0 || arrayZ >= 9) {
      return null; // Out of bounds
    }
    
    const value = gameState.board[arrayX][arrayY][arrayZ];
    // Handle null values (which were NaN in the original board - invalid positions)
    if (value === null || value === undefined) {
      return null; // Invalid position
    }
    
    return value; // 0 for empty, 1 for black, -1 for white
  };

  const resetGame = () => {
    initializeGame();
  };

  const loadCustomBoard = async () => {
    try {
      // Call backend API to load custom test board
      const response = await fetch('http://localhost:8000/game/custom', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        throw new Error(`Failed to load custom board: ${response.statusText}`);
      }
      
      const gameState = await response.json();
      setGameState(gameState);
      setCurrentPlayer(gameState.actual_player);
      setGameStatus('playing');
      setSelectedMarbles([]);
      setShowDirections(false);
      setErrorMessage('Custom test board loaded!');
      
    } catch (error) {
      console.error('Error loading custom board:', error);
      setErrorMessage('Failed to load custom board: ' + error.message);
    }
  };

  return (
    <div className="game-container">
      <div className="game-header">
        <h1>Abalone AlphaZero</h1>
        <div className="game-info">
          <div className="player-turn">
            <span className={`player-indicator ${currentPlayer === 1 ? 'active' : ''}`}>
              ● Black
            </span>
            <span className="vs">vs</span>
            <span className={`player-indicator ${currentPlayer === -1 ? 'active' : ''}`}>
              ○ White
            </span>
          </div>
          
          {gameState && (
            <div className="score">
              <span>Black out: {gameState.black_out}</span>
              <span>White out: {gameState.white_out}</span>
              <span>Moves: {gameState.moves_count}</span>
            </div>
          )}
        </div>
        
        <div className="game-buttons">
          <button onClick={resetGame} className="reset-button">
            New Game
          </button>
          <button onClick={loadCustomBoard} className="custom-button">
            Load Test Board
          </button>
        </div>
      </div>

      {errorMessage && (
        <div className="error-message">
          {errorMessage}
        </div>
      )}

      <div className="selection-info">
        {selectedMarbles.length > 0 && (
          <div>
            <strong>Selected marbles:</strong> {selectedMarbles.length}/3
            <div className="selected-coords">
              {selectedMarbles.map(coordStr => {
                const [x, y, z] = stringToCoord(coordStr);
                return (
                  <span key={coordStr} className="coord-chip">
                    ({x},{y},{z})
                  </span>
                );
              })}
            </div>
          </div>
        )}
        
        {showDirections && (
          <div className="direction-instruction">
            <strong>Click a direction arrow to move</strong>
          </div>
        )}
      </div>

      <Board
        gameState={gameState}
        onMarbleClick={handleMarbleClick}
        onDirectionClick={handleDirectionClick}
        selectedMarbles={selectedMarbles}
        showDirections={showDirections}
      />
    </div>
  );
};

export default Game;