import React, { useState, useEffect } from 'react';
import { 
  getValidPositions, 
  cubeToScreen, 
  coordToString, 
  stringToCoord,
  getDirectionArrowPositions,
  DIRECTION_NAMES 
} from '../utils/boardUtils';
import './Board.css';

const Board = ({ 
  gameState, 
  onMarbleClick, 
  onDirectionClick, 
  selectedMarbles = [], 
  showDirections = false 
}) => {
  const [screenPositions, setScreenPositions] = useState({});
  const [directionArrows, setDirectionArrows] = useState([]);
  
  const hexSize = 25;
  const centerX = 300;
  const centerY = 300;
  const svgWidth = 600;
  const svgHeight = 600;

  // Calculate screen positions for all valid board positions
  useEffect(() => {
    const positions = {};
    getValidPositions().forEach(([x, y, z]) => {
      const screen = cubeToScreen(x, y, z, hexSize, centerX, centerY);
      positions[coordToString(x, y, z)] = { cube: [x, y, z], screen };
    });
    setScreenPositions(positions);
  }, []);

  // Update direction arrows when marbles are selected
  useEffect(() => {
    if (showDirections && selectedMarbles.length > 0) {
      const selectedPositions = selectedMarbles.map(coordStr => {
        const [x, y, z] = stringToCoord(coordStr);
        const screen = cubeToScreen(x, y, z, hexSize, centerX, centerY);
        return { cube: [x, y, z], screen };
      });
      
      const arrows = getDirectionArrowPositions(selectedPositions, hexSize);
      setDirectionArrows(arrows);
    } else {
      setDirectionArrows([]);
    }
  }, [selectedMarbles, showDirections]);

  // Get marble value at position from game state
  const getMarbleValue = (x, y, z) => {
    if (!gameState || !gameState.board) return 0;
    
    // Convert from cube coordinates to array indices (add radius=4)
    const arrayX = x + 4;
    const arrayY = y + 4;  
    const arrayZ = z + 4;
    
    // Check bounds
    if (arrayX < 0 || arrayX >= 9 || arrayY < 0 || arrayY >= 9 || arrayZ < 0 || arrayZ >= 9) {
      return 0;
    }
    
    const value = gameState.board[arrayX][arrayY][arrayZ];
    return value; // 0 for empty, 1 for black, -1 for white
  };

  // Handle marble click
  const handleMarbleClick = (x, y, z) => {
    if (onMarbleClick) {
      onMarbleClick(x, y, z);
    }
  };

  // Handle direction arrow click
  const handleDirectionClick = (direction) => {
    if (onDirectionClick) {
      onDirectionClick(direction);
    }
  };

  // Render a single hexagon
  const renderHex = (x, y, z) => {
    const coordStr = coordToString(x, y, z);
    const position = screenPositions[coordStr];
    if (!position) return null;

    const marbleValue = getMarbleValue(x, y, z);
    if (marbleValue === null) return null; // Invalid position

    const isSelected = selectedMarbles.includes(coordStr);
    // Visual marble types - always consistent regardless of canonical representation
    const isBlackMarble = marbleValue === 1;   // Black marbles (always visually black)
    const isWhiteMarble = marbleValue === -1;  // White marbles (always visually white)
    const isEmpty = marbleValue === 0;
    
    // Hexagon path (flat-top hexagon)
    const hexPath = [];
    for (let i = 0; i < 6; i++) {
      const angle = (i * 60 - 30) * (Math.PI / 180);
      const hexX = position.screen.x + hexSize * 0.9 * Math.cos(angle);
      const hexY = position.screen.y + hexSize * 0.9 * Math.sin(angle);
      hexPath.push(`${i === 0 ? 'M' : 'L'} ${hexX} ${hexY}`);
    }
    hexPath.push('Z');

    return (
      <g key={coordStr}>
        {/* Hexagon background */}
        <path
          d={hexPath.join(' ')}
          fill={isEmpty ? "#f0f0f0" : "transparent"}
          stroke="#333"
          strokeWidth="1"
          className="hex-cell"
        />
        
        {/* Marble */}
        {!isEmpty && (
          <circle
            cx={position.screen.x}
            cy={position.screen.y}
            r={hexSize * 0.7}
            fill={isBlackMarble ? "#2c3e50" : "#ecf0f1"}
            stroke={isSelected ? "#e74c3c" : "#34495e"}
            strokeWidth={isSelected ? 4 : 2}
            className={`marble ${isBlackMarble ? 'black-marble' : 'white-marble'} ${isSelected ? 'selected' : ''}`}
            style={{ cursor: 'pointer' }}
            onClick={() => handleMarbleClick(x, y, z)}
          />
        )}
        
      </g>
    );
  };

  // Render direction arrows
  const renderDirectionArrows = () => {
    return directionArrows.map(({ direction, x, y, angle }) => (
      <g key={direction}>
        {/* Arrow background circle */}
        <circle
          cx={x}
          cy={y}
          r={hexSize * 0.8}
          fill="#3498db"
          stroke="#2980b9"
          strokeWidth="2"
          className="direction-arrow"
          style={{ cursor: 'pointer' }}
          onClick={() => handleDirectionClick(direction)}
        />
        
        {/* Arrow icon (simple triangle) */}
        <polygon
          points={`${x-8},${y+4} ${x+8},${y} ${x-8},${y-4}`}
          fill="white"
          transform={`rotate(${angle} ${x} ${y})`}
          pointerEvents="none"
        />
        
        {/* Direction label */}
        <text
          x={x}
          y={y + hexSize + 15}
          textAnchor="middle"
          fontSize="12"
          fill="#2980b9"
          fontWeight="bold"
          pointerEvents="none"
          style={{ userSelect: 'none' }}
        >
          {direction}
        </text>
      </g>
    ));
  };

  return (
    <div className="board-container">
      <svg
        width={svgWidth}
        height={svgHeight}
        viewBox={`0 0 ${svgWidth} ${svgHeight}`}
        className="board-svg"
      >
        {/* Rotated board group */}
        <g transform={`rotate(-30 ${centerX} ${centerY})`}>
          {/* Board hexagons */}
          {getValidPositions().map(([x, y, z]) => renderHex(x, y, z))}
          
          {/* Direction arrows */}
          {showDirections && renderDirectionArrows()}
        </g>
      </svg>
    </div>
  );
};

export default Board;