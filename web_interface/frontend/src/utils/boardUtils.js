// Board utility functions for converting between cubic coordinates and screen positions

/**
 * Convert cubic coordinates to screen coordinates for SVG rendering
 * Based on the hexagonal grid layout from your display_board function
 */
export function cubeToScreen(x, y, z, hexSize = 30, centerX = 300, centerY = 300) {
  // For hexagonal grid, we use the axial coordinate system
  // Convert cubic (x,y,z) to axial (q,r) where q=x, r=z
  const q = x;
  const r = z;
  
  // Standard hex-to-pixel conversion (flat-top hexagons)
  // z=-4 should be at the top, z=4 should be at the bottom
  const screenX = centerX + hexSize * (3/2 * q);
  const screenY = centerY + hexSize * (Math.sqrt(3)/2 * q + Math.sqrt(3) * r);
  
  return { x: screenX, y: screenY };
}

/**
 * Generate all valid board positions based on your board layout
 * This matches the valid_coords from your create_board_mask function
 */
export function getValidPositions() {
  return [
    // Row 1 (z = -4)
    [0,4,-4], [1,3,-4], [2,2,-4], [3,1,-4], [4,0,-4],
    // Row 2 (z = -3)
    [-1,4,-3], [0,3,-3], [1,2,-3], [2,1,-3], [3,0,-3], [4,-1,-3],
    // Row 3 (z = -2)
    [-2,4,-2], [-1,3,-2], [0,2,-2], [1,1,-2], [2,0,-2], [3,-1,-2], [4,-2,-2],
    // Row 4 (z = -1)
    [-3,4,-1], [-2,3,-1], [-1,2,-1], [0,1,-1], [1,0,-1], [2,-1,-1], [3,-2,-1], [4,-3,-1],
    // Row 5 (z = 0)
    [-4,4,0], [-3,3,0], [-2,2,0], [-1,1,0], [0,0,0], [1,-1,0], [2,-2,0], [3,-3,0], [4,-4,0],
    // Row 6 (z = 1)
    [-4,3,1], [-3,2,1], [-2,1,1], [-1,0,1], [0,-1,1], [1,-2,1], [2,-3,1], [3,-4,1],
    // Row 7 (z = 2)
    [-4,2,2], [-3,1,2], [-2,0,2], [-1,-1,2], [0,-2,2], [1,-3,2], [2,-4,2],
    // Row 8 (z = 3)
    [-4,1,3], [-3,0,3], [-2,-1,3], [-1,-2,3], [0,-3,3], [1,-4,3],
    // Row 9 (z = 4)
    [-4,0,4], [-3,-1,4], [-2,-2,4], [-1,-3,4], [0,-4,4]
  ];
}

/**
 * Create coordinate string for position lookup
 */
export function coordToString(x, y, z) {
  return `${x},${y},${z}`;
}

/**
 * Parse coordinate string back to array
 */
export function stringToCoord(coordString) {
  return coordString.split(',').map(Number);
}

/**
 * Direction vectors matching your DIRECTIONS from core.py
 */
export const DIRECTIONS = {
  NE: [1, 0, -1],   // Northeast
  E:  [1, -1, 0],   // East  
  SE: [0, -1, 1],   // Southeast
  SW: [-1, 0, 1],   // Southwest
  W:  [-1, 1, 0],   // West
  NW: [0, 1, -1]    // Northwest
};

export const DIRECTION_NAMES = ['NE', 'E', 'SE', 'SW', 'W', 'NW'];

/**
 * Get direction arrow positions around selected marbles
 */
export function getDirectionArrowPositions(selectedMarbles, hexSize = 30) {
  if (selectedMarbles.length === 0) return [];
  
  // Calculate center of selected marbles
  const centerX = selectedMarbles.reduce((sum, pos) => sum + pos.screen.x, 0) / selectedMarbles.length;
  const centerY = selectedMarbles.reduce((sum, pos) => sum + pos.screen.y, 0) / selectedMarbles.length;
  
  // Position arrows in a circle around the center
  const arrowDistance = hexSize * 2.5;
  const arrows = [];
  
  DIRECTION_NAMES.forEach((direction, index) => {
    const angle = (index * 60) * (Math.PI / 180); // 60 degrees apart
    const arrowX = centerX + Math.cos(angle) * arrowDistance;
    const arrowY = centerY + Math.sin(angle) * arrowDistance;
    
    arrows.push({
      direction,
      x: arrowX,
      y: arrowY,
      angle: (index * 60) // For rotating arrow icons
    });
  });
  
  return arrows;
}