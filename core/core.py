import jax
import jax.numpy as jnp
import chex
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple

@dataclass(frozen=True)
class CubeCoord:
    """
    Représentation d'une coordonnée cubique avec contrainte x + y + z = 0
    """
    x: int
    y: int
    z: int
    
    def __post_init__(self):
        if self.x + self.y + self.z != 0:
            raise ValueError(f"Invalid cube coordinates: {self.x}+{self.y}+{self.z}≠0")
    
    def to_array(self) -> chex.Array:
        """Convertit en tableau JAX"""
        return jnp.array([self.x, self.y, self.z])
    
    @staticmethod
    def from_array(arr: chex.Array) -> 'CubeCoord':
        """Crée une coordonnée à partir d'un tableau JAX"""
        return CubeCoord(int(arr[0]), int(arr[1]), int(arr[2]))

class Direction(Enum):
    """
    Directions possibles sur la grille hexagonale en coordonnées cubiques
    """
    NE = (1, 0, -1)   # Nord-Est
    E  = (1, -1, 0)   # Est
    SE = (0, -1, 1)   # Sud-Est
    SW = (-1, 0, 1)   # Sud-Ouest
    W  = (-1, 1, 0)   # Ouest
    NW = (0, 1, -1)   # Nord-Ouest
    
    def to_array(self) -> chex.Array:
        """Convertit la direction en tableau JAX"""
        return jnp.array(self.value)
    
    @staticmethod
    def all_directions() -> chex.Array:
        """Retourne toutes les directions sous forme de matrice"""
        return jnp.array([d.value for d in Direction])

# Constantes utiles
DIRECTIONS = Direction.all_directions()
DIR_TO_IDX = {d: i for i, d in enumerate(Direction)}