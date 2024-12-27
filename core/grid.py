# File: core/grid.py
from typing import List, Optional
import logging
from utils.logging_setup import setup_logging
from render.colors import Colors, Color

loggers = setup_logging()
logger = logging.getLogger('tetris.grid')

class Grid:
    """
    Represents the game grid for Tetris with proper indexing support.
    Includes methods for detecting and clearing completed lines, as well
    as collision checks for Tetrimino placement.
    """

    def __init__(self, config):
        """
        Initialize the grid with game configuration.
        
        Args:
            config: The game configuration object, which should have
                    .columns, .rows, and .grid_size at minimum.
        """
        self.columns = config.columns
        self.rows = config.rows
        self.grid_size = config.grid_size

        # The main 2D array for storing piece type or None
        self.grid: List[List[Optional[str]]] = [
            [None for _ in range(self.columns)] for _ in range(self.rows)
        ]
        self.cleared_lines: List[int] = []
        self.colors = Colors()
        self.flash_animation = False

        logger.debug(f"Grid initialized with {self.rows} rows and {self.columns} columns")

    def __getitem__(self, index: int) -> List[Optional[str]]:
        """
        Enable grid[y] indexing. Returns a single row of the grid.
        
        Args:
            index: Row index
        Returns:
            The specified row of the grid
        """
        return self.grid[index]

    def __len__(self) -> int:
        """Return the number of rows in the grid."""
        return len(self.grid)

    def reset(self) -> None:
        """
        Reset the grid to empty state, clearing all stored piece data.
        """
        self.grid = [
            [None for _ in range(self.columns)] for _ in range(self.rows)
        ]
        self.cleared_lines = []
        self.flash_animation = False
        logger.debug("Grid reset to empty state")

    def detect_cleared_lines(self) -> List[int]:
        """
        Detect completed lines that should be cleared.
        
        Returns:
            List of row indices that are fully occupied.
        """
        cleared = []
        # Check rows from bottom (rows-1) to top (0)
        for y in range(self.rows - 1, -1, -1):
            if all(cell is not None for cell in self.grid[y]):
                cleared.append(y)
                logger.debug(f"Line {y} is complete and marked for clearing")

        # If we found cleared lines, set them to flash_color
        if cleared:
            self.flash_animation = True
            self.cleared_lines = cleared
            for y in cleared:
                for x in range(self.columns):
                    self.grid[y][x] = 'flash_color'
            logger.info(f"Total lines detected for clearing: {len(cleared)}")

        return cleared

    def clear_cleared_lines(self) -> int:
        """
        Perform the actual line removal and apply gravity to shift
        the upper lines down.
        
        Returns:
            Number of lines cleared.
        """
        if not self.cleared_lines:
            return 0

        new_grid = []
        num_cleared = len(self.cleared_lines)

        # Keep rows that are not cleared
        for y in range(self.rows):
            if y not in self.cleared_lines:
                new_grid.append(self.grid[y])

        # Insert empty rows at the top
        for _ in range(num_cleared):
            new_grid.insert(0, [None for _ in range(self.columns)])

        self.grid = new_grid
        logger.info(f"Cleared {num_cleared} lines")
        self.flash_animation = False
        self.cleared_lines = []

        return num_cleared

    def can_move(self, piece, new_x: int, new_y: int) -> bool:
        """
        Check if the given piece can be moved to (new_x, new_y).
        
        Args:
            piece: The Tetrimino or piece object, which has 'shape'
            new_x: Left coordinate
            new_y: Top coordinate
        Returns:
            True if the move is valid, False otherwise.
        """
        for y, row in enumerate(piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    grid_x = new_x + x
                    grid_y = new_y + y

                    # Check horizontal bounds
                    if grid_x < 0 or grid_x >= self.columns:
                        logger.debug(f"Movement blocked: x={grid_x} out of bounds")
                        return False

                    # Check vertical bounds
                    if grid_y >= len(self):
                        logger.debug(f"Movement blocked: y={grid_y} exceeds grid height")
                        return False

                    # Check collision with existing piece
                    if grid_y >= 0 and self.grid[grid_y][grid_x] is not None:
                        if self.grid[grid_y][grid_x] != 'flash_color':
                            logger.debug(f"Movement blocked: cell ({grid_x}, {grid_y}) occupied")
                            return False

        return True

    def can_place(self, piece, x: int, y: int) -> bool:
        """
        Check if the piece can be placed at (x, y).
        
        Args:
            piece: Tetrimino object
            x: Left coordinate
            y: Top coordinate
        Returns:
            True if piece placement is valid, False otherwise.
        """
        return self.can_move(piece, x, y)

    def lock_piece(self, piece) -> None:
        """
        Lock current piece into the grid, writing its piece_type into
        the grid array at the piece's coordinates.
        """
        if piece is None:
            return

        for y, row in enumerate(piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    grid_x = piece.x + x
                    grid_y = piece.y + y
                    if 0 <= grid_y < self.rows and 0 <= grid_x < self.columns:
                        self.grid[grid_y][grid_x] = piece.piece_type
                        logger.debug(f"Locked piece type '{piece.piece_type}' at ({grid_x}, {grid_y})")
                    else:
                        logger.warning(f"Attempted to lock piece out of bounds at ({grid_x}, {grid_y})")

    def get_cell_color(self, x: int, y: int) -> Color:
        """
        Get the color for the cell at (x, y).
        
        Args:
            x: Column index
            y: Row index
        Returns:
            A tuple representing the RGB color or RGBA color.
        """
        try:
            cell = self.grid[y][x]
            if cell == 'flash_color':
                return self.colors.FLASH_COLOR
            elif cell == 'normal_color':
                return self.colors.NORMAL_COLOR
            elif cell is not None:
                return self.colors.get_piece_color(cell)
            else:
                return self.colors.UI_COLORS['background']
        except IndexError:
            logger.error(f"Attempted to access cell ({x}, {y}) out of bounds")
            return self.colors.UI_COLORS['background']

    def get_cleared_lines(self) -> List[int]:
        """
        Get a copy of the currently recorded cleared line indices.
        """
        return self.cleared_lines.copy()
