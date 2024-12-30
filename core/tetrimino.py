# File: core/tetrimino.py


import os
import sys
from typing import List, Dict, Tuple, Optional
from utils.logging_setup import setup_logging
loggers = setup_logging()
resource_logger = loggers['resource']

Shape = List[List[int]]
Grid = 'Grid'  # Forward declaration for type hinting

class TetriminoData:
    """Static data for Tetrimino shapes and wall kick data."""
    
    SHAPES: Dict[str, List[Shape]] = {
        'I': [
            [
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [0, 0, 0, 0],
            ],
            [
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
            ],
        ],
        'O': [
            [
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        ],
        'T': [
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 0, 0],
            ],
            [
                [0, 1, 0],
                [0, 1, 1],
                [0, 1, 0],
            ],
            [
                [0, 0, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            [
                [0, 1, 0],
                [1, 1, 0],
                [0, 1, 0],
            ],
        ],
        'S': [
            [
                [0, 1, 1],
                [1, 1, 0],
                [0, 0, 0],
            ],
            [
                [0, 1, 0],
                [0, 1, 1],
                [0, 0, 1],
            ],
            [
                [0, 0, 0],
                [0, 1, 1],
                [1, 1, 0],
            ],
            [
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
            ],
        ],
        'Z': [
            [
                [1, 1, 0],
                [0, 1, 1],
                [0, 0, 0],
            ],
            [
                [0, 0, 1],
                [0, 1, 1],
                [0, 1, 0],
            ],
            [
                [0, 0, 0],
                [1, 1, 0],
                [0, 1, 1],
            ],
            [
                [0, 1, 0],
                [1, 1, 0],
                [1, 0, 0],
            ],
        ],
        'J': [
            [
                [1, 0, 0],
                [1, 1, 1],
                [0, 0, 0],
            ],
            [
                [0, 1, 1],
                [0, 1, 0],
                [0, 1, 0],
            ],
            [
                [0, 0, 0],
                [1, 1, 1],
                [0, 0, 1],
            ],
            [
                [0, 1, 0],
                [0, 1, 0],
                [1, 1, 0],
            ],
        ],
        'L': [
            [
                [0, 0, 1],
                [1, 1, 1],
                [0, 0, 0],
            ],
            [
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 1],
            ],
            [
                [0, 0, 0],
                [1, 1, 1],
                [1, 0, 0],
            ],
            [
                [1, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
            ],
        ],
    }

    WALL_KICK_DATA: Dict[str, Dict[Tuple[int, int], List[Tuple[int, int]]]] = {
        'I': {
            (0, 1): [(0, 0), (-2, 0), (+1, 0), (-2, -1), (+1, +2)],
            (1, 0): [(0, 0), (+2, 0), (-1, 0), (+2, +1), (-1, -2)],
            (1, 2): [(0, 0), (-1, 0), (+2, 0), (-1, +2), (+2, -1)],
            (2, 1): [(0, 0), (+1, 0), (-2, 0), (+1, -2), (-2, +1)],
            (2, 3): [(0, 0), (+2, 0), (-1, 0), (+2, +1), (-1, -2)],
            (3, 2): [(0, 0), (-2, 0), (+1, 0), (-2, -1), (+1, +2)],
            (3, 0): [(0, 0), (+1, 0), (-2, 0), (+1, -2), (-2, +1)],
            (0, 3): [(0, 0), (-1, 0), (+2, 0), (-1, +2), (+2, -1)],
        },
        'J': {
            (0, 1): [(0, 0), (-1, 0), (-1, +1), (0, -2), (-1, -2)],
            (1, 0): [(0, 0), (+1, 0), (+1, -1), (0, +2), (+1, +2)],
            (1, 2): [(0, 0), (+1, 0), (+1, -1), (0, +2), (+1, +2)],
            (2, 1): [(0, 0), (-1, 0), (-1, +1), (0, -2), (-1, -2)],
            (2, 3): [(0, 0), (+1, 0), (+1, +1), (0, -2), (+1, -2)],
            (3, 2): [(0, 0), (-1, 0), (-1, -1), (0, +2), (-1, +2)],
            (3, 0): [(0, 0), (-1, 0), (-1, -1), (0, +2), (-1, +2)],
            (0, 3): [(0, 0), (+1, 0), (+1, +1), (0, -2), (+1, -2)],
        },
        'L': {
            (0, 1): [(0, 0), (-1, 0), (-1, +1), (0, -2), (-1, -2)],
            (1, 0): [(0, 0), (+1, 0), (+1, -1), (0, +2), (+1, +2)],
            (1, 2): [(0, 0), (+1, 0), (+1, -1), (0, +2), (+1, +2)],
            (2, 1): [(0, 0), (-1, 0), (-1, +1), (0, -2), (-1, -2)],
            (2, 3): [(0, 0), (+1, 0), (+1, +1), (0, -2), (+1, -2)],
            (3, 2): [(0, 0), (-1, 0), (-1, -1), (0, +2), (-1, +2)],
            (3, 0): [(0, 0), (-1, 0), (-1, -1), (0, +2), (-1, +2)],
            (0, 3): [(0, 0), (+1, 0), (+1, +1), (0, -2), (+1, -2)],
        },
        'S': {
            (0, 1): [(0, 0), (-1, 0), (-1, +1), (0, -2), (-1, -2)],
            (1, 0): [(0, 0), (+1, 0), (+1, -1), (0, +2), (+1, +2)],
            (1, 2): [(0, 0), (+1, 0), (+1, -1), (0, +2), (+1, +2)],
            (2, 1): [(0, 0), (-1, 0), (-1, +1), (0, -2), (-1, -2)],
            (2, 3): [(0, 0), (+1, 0), (+1, +1), (0, -2), (+1, -2)],
            (3, 2): [(0, 0), (-1, 0), (-1, -1), (0, +2), (-1, +2)],
            (3, 0): [(0, 0), (-1, 0), (-1, -1), (0, +2), (-1, +2)],
            (0, 3): [(0, 0), (+1, 0), (+1, +1), (0, -2), (+1, -2)],
        },
        'T': {
            (0, 1): [(0, 0), (-1, 0), (-1, +1), (0, -2), (-1, -2)],
            (1, 0): [(0, 0), (+1, 0), (+1, -1), (0, +2), (+1, +2)],
            (1, 2): [(0, 0), (+1, 0), (+1, -1), (0, +2), (+1, +2)],
            (2, 1): [(0, 0), (-1, 0), (-1, +1), (0, -2), (-1, -2)],
            (2, 3): [(0, 0), (+1, 0), (+1, +1), (0, -2), (+1, -2)],
            (3, 2): [(0, 0), (-1, 0), (-1, -1), (0, +2), (-1, +2)],
            (3, 0): [(0, 0), (-1, 0), (-1, -1), (0, +2), (-1, +2)],
            (0, 3): [(0, 0), (+1, 0), (+1, +1), (0, -2), (+1, -2)],
        },
    }

    @classmethod
    def validate_shape(cls, shape: List[List[int]]) -> bool:
        if not shape:
            return False
        size = len(shape)
        for row in shape:
            if len(row) != size:
                return False
        return True

    @classmethod
    def get_initial_shape(cls, piece_type: str, rotation: int = 0) -> Shape:
        try:
            shapes = cls.SHAPES[piece_type]
            return shapes[rotation % len(shapes)]
        except KeyError:
            resource_logger.error("Invalid piece type '%s'", piece_type)
            raise
        except IndexError:
            resource_logger.error("Invalid rotation %d for piece type '%s'", rotation, piece_type)
            raise

class Tetrimino:
    def __init__(self, piece_type: str, grid_size: int, columns: int):
        self.piece_type = piece_type
        self.grid_size = grid_size
        self.columns = columns
        self.rotation = 0
        self.shape = TetriminoData.get_initial_shape(piece_type, self.rotation)
        self.height = len(self.shape)
        self.width = len(self.shape[0])
        if piece_type == 'I':
            self.x = (self.columns - 4) // 2  # Center the piece
        elif piece_type == 'O':
            self.x = (self.columns - 2) // 2
        else:
            self.x = (self.columns - 3) // 2
        self.y = 0
        resource_logger.debug("Initialized Tetrimino '%s' at position (%d, %d) with rotation %d",
                              self.piece_type, self.x, self.y, self.rotation)

    def get_rotated_shape(self, new_rotation: int) -> Shape:
        return TetriminoData.get_initial_shape(self.piece_type, new_rotation)

    def try_rotation(self, grid: Grid, clockwise: bool = True) -> bool:
        old_rotation = self.rotation
        if clockwise:
            self.rotation = (self.rotation + 1) % 4
        else:
            self.rotation = (self.rotation - 1) % 4
        new_shape = self.get_rotated_shape(self.rotation)
        wall_kicks = TetriminoData.WALL_KICK_DATA.get(self.piece_type, {})
        rotation_key = (old_rotation, self.rotation)
        kicks = wall_kicks.get(rotation_key, [(0, 0)])
        resource_logger.debug("Attempting rotation for '%s' from %d to %d with %d kicks",
                              self.piece_type, old_rotation, self.rotation, len(kicks))
        for dx, dy in kicks:
            new_x = self.x + dx
            new_y = self.y + dy
            if self.is_valid_position(grid, new_x, new_y, new_shape):
                self.x = new_x
                self.y = new_y
                self.shape = new_shape
                resource_logger.info("Rotation successful for '%s' to rotation %d with kick (%d, %d)",
                                     self.piece_type, self.rotation, dx, dy)
                return True
        self.rotation = old_rotation
        resource_logger.info("Rotation failed for '%s'; reverted to rotation %d",
                             self.piece_type, self.rotation)
        return False

    def is_valid_position(self, grid: Grid, x: int, y: int,
                          shape: Optional[Shape] = None) -> bool:
        if shape is None:
            shape = self.shape
        for row_idx, row in enumerate(shape):
            for col_idx, cell in enumerate(row):
                if cell:
                    grid_x = x + col_idx
                    grid_y = y + row_idx
                    if grid_x < 0 or grid_x >= self.columns:
                        resource_logger.debug("Invalid position: x=%d out of bounds", grid_x)
                        return False
                    if grid_y >= len(grid):
                        resource_logger.debug("Invalid position: y=%d exceeds grid height", grid_y)
                        return False
                    if grid_y >= 0 and grid[grid_y][grid_x] is not None:
                        resource_logger.debug("Invalid position: cell (%d, %d) is occupied", grid_x, grid_y)
                        return False
        return True

    def get_ghost_position(self, grid: Grid) -> int:
        ghost_y = self.y
        while self.is_valid_position(grid, self.x, ghost_y + 1):
            ghost_y += 1
            if ghost_y >= len(grid):
                break
        resource_logger.debug("Ghost position for '%s' is y=%d", self.piece_type, ghost_y)
        return ghost_y

    def save_position(self) -> None:
        pass