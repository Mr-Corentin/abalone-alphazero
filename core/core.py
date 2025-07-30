import jax
import jax.numpy as jnp
import chex
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple

@dataclass(frozen=True)
class CubeCoord:
    """
    Representation of a cubic coordinate with constraint x + y + z = 0
    """
    x: int
    y: int
    z: int
    
    def __post_init__(self):
        if self.x + self.y + self.z != 0:
            raise ValueError(f"Invalid cube coordinates: {self.x}+{self.y}+{self.z}≠0")
    
    def to_array(self) -> chex.Array:
        """Convert to JAX array"""
        return jnp.array([self.x, self.y, self.z])
    
    @staticmethod
    def from_array(arr: chex.Array) -> 'CubeCoord':
        """Create coordinate from JAX array"""
        return CubeCoord(int(arr[0]), int(arr[1]), int(arr[2]))

class Direction(Enum):
    """
    Possible directions on hexagonal grid in cubic coordinates
    """
    NE = (1, 0, -1)   # North-East
    E  = (1, -1, 0)   # East
    SE = (0, -1, 1)   # South-East
    SW = (-1, 0, 1)   # South-West
    W  = (-1, 1, 0)   # West
    NW = (0, 1, -1)   # North-West
    
    def to_array(self) -> chex.Array:
        """Convert direction to JAX array"""
        return jnp.array(self.value)
    
    @staticmethod
    def all_directions() -> chex.Array:
        """Return all directions as matrix"""
        return jnp.array([d.value for d in Direction])

# Useful constants
DIRECTIONS = Direction.all_directions()
DIR_TO_IDX = {d: i for i, d in enumerate(Direction)}