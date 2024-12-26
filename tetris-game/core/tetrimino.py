# File: core/tetrimino.py

import os
import sys
from typing import List, Dict, Tuple, Optional
from utils.logging_setup import setup_logging

# Initialize loggers
loggers = setup_logging()
resource_logger = loggers['resource']

# Type aliases for clarity
Shape = List[List[int]]
Grid = List[List[Optional[str]]]

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
        """
        Validate that the shape matrix is square and non-empty.
        :param shape: The shape matrix to validate
        :return: True if the shape matrix is square (NxN) and not empty
        """
        if not shape:
            return False
        size = len(shape)
        for row in shape:
            if len(row) != size:
                return False
        return True

    @classmethod
    def get_initial_shape(cls, piece_type: str, rotation: int = 0) -> Shape:
        """
        Retrieve the shape matrix for a given piece type and rotation index.
        :param piece_type: The Tetrimino type, e.g. 'I', 'O', 'T', ...
        :param rotation: 0-based rotation index (0 to 3)
        :return: The shape matrix corresponding to that piece type and rotation
        :raises KeyError: If an invalid piece_type is provided
        :raises IndexError: If an invalid rotation is provided
        """
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
    """
    Represents a single Tetris piece, including its type, rotation, position,
    and collision checks. Also handles rotation logic with SRS wall kicks.
    """
    
    def __init__(self, piece_type: str, grid_size: int, columns: int):
        """
        Initialize a Tetrimino object.
        
        :param piece_type: The type of the piece, e.g. 'I', 'O', 'T', ...
        :param grid_size: The size of each cell in the grid (pixel dimension)
        :param columns: Number of columns in the Tetris grid
        """
        self.piece_type = piece_type
        self.grid_size = grid_size
        self.columns = columns
        self.rotation = 0
        self.shape = TetriminoData.get_initial_shape(piece_type, self.rotation)
        self.height = len(self.shape)
        self.width = len(self.shape[0])
        
        # Center the piece horizontally based on its width
        if piece_type == 'I':
            self.x = (self.columns - 4) // 2  # 'I' piece is 4 blocks wide
        elif piece_type == 'O':
            self.x = (self.columns - 2) // 2  # 'O' piece is 2 blocks wide
        else:
            self.x = (self.columns - 3) // 2  # Other pieces are 3 blocks wide
        self.y = 0  # Start at the top of the grid

    def get_rotated_shape(self, new_rotation: int) -> Shape:
        """
        Get the shape matrix for a new rotation without modifying the current state.
        
        :param new_rotation: The target rotation index
        :return: The shape matrix for that rotation
        """
        return TetriminoData.get_initial_shape(self.piece_type, new_rotation)

    def try_rotation(self, grid: Grid, clockwise: bool = True) -> bool:
        """
        Attempt to rotate the piece with SRS wall kicks.
        If a valid position is found (i.e., the rotation can occur without collision),
        apply it. Otherwise revert to the old rotation.
        
        :param grid: The game grid (2D array of piece types or None)
        :param clockwise: Whether the rotation is clockwise (True) or counterclockwise (False)
        :return: True if the rotation succeeds, False if it fails
        """
        old_rotation = self.rotation
        if clockwise:
            self.rotation = (self.rotation + 1) % 4
        else:
            self.rotation = (self.rotation - 1) % 4
        new_shape = self.get_rotated_shape(self.rotation)
        wall_kicks = TetriminoData.WALL_KICK_DATA.get(self.piece_type, {})
        rotation_key = (old_rotation, self.rotation)
        kicks = wall_kicks.get(rotation_key, [(0, 0)])
        
        for dx, dy in kicks:
            new_x = self.x + dx
            new_y = self.y + dy
            if self.is_valid_position(grid, new_x, new_y, new_shape):
                self.x = new_x
                self.y = new_y
                self.shape = new_shape
                resource_logger.debug("Rotation successful with kick (%d, %d)", dx, dy)
                return True
        
        # If all kicks fail, revert to the old rotation
        self.rotation = old_rotation
        resource_logger.debug("Rotation blocked; reverting to rotation %d", self.rotation)
        return False

    def is_valid_position(self, grid: Grid, x: int, y: int,
                          shape: Optional[Shape] = None) -> bool:
        """
        Check if the piece at position (x, y) with the given shape (defaults to self.shape)
        is in a valid position within the grid. A position is valid if no blocks fall
        outside the grid boundaries or into occupied cells.
        
        :param grid: The game grid
        :param x: Left coordinate in grid cells
        :param y: Top coordinate in grid cells
        :param shape: The shape matrix to check. If None, use self.shape
        :return: True if the position is valid, False otherwise
        """
        if shape is None:
            shape = self.shape
        for row_idx, row in enumerate(shape):
            for col_idx, cell in enumerate(row):
                if cell:
                    grid_x = x + col_idx
                    grid_y = y + row_idx
                    # Check horizontal boundaries
                    if grid_x < 0 or grid_x >= self.columns:
                        return False
                    # Check vertical boundaries
                    if grid_y >= len(grid):
                        return False
                    # Check for collision with existing locked pieces
                    if grid_y >= 0 and grid[grid_y][grid_x] is not None:
                        return False
        return True

    def get_ghost_position(self, grid: Grid) -> int:
        """
        Calculate the y position where the piece would land if it were hard dropped now.
        
        :param grid: The game grid
        :return: The y coordinate of the final position
        """
        ghost_y = self.y
        max_iterations = len(grid)
        iterations = 0
        while self.is_valid_position(grid, self.x, ghost_y + 1) and iterations < max_iterations:
            ghost_y += 1
            iterations += 1
        return ghost_y

    def save_position(self) -> None:
        """
        Placeholder method to save the current (x, y, rotation) if you want to keep
        a history or an undo/redo stack. Currently unused.
        """
        pass
