from typing import List, Optional, Dict, Tuple
import logging
from utils.logging_setup import setup_logging
from render.colors import Colors, Color

loggers = setup_logging()
logger = logging.getLogger('tetris.grid')

class Grid:
    """Represents the game grid for Tetris."""
    
    def __init__(self, config):
        """Initialize the grid with game configuration."""
        self.columns = config.columns
        self.rows = config.rows
        self.grid_size = config.grid_size
        self.grid: List[List[Optional[str]]] = [
            [None for _ in range(self.columns)] for _ in range(self.rows)
        ]
        self.cleared_lines: List[int] = []
        self.colors = Colors()
        self.flash_animation = False
        logger.debug(f"Grid initialized with {self.rows} rows and {self.columns} columns")

    def __len__(self) -> int:
        return len(self.grid)

    def reset(self) -> None:
        """Reset the grid to empty state."""
        self.grid = [
            [None for _ in range(self.columns)] for _ in range(self.rows)
        ]
        self.cleared_lines = []
        self.flash_animation = False
        logger.debug("Grid reset to empty state")

    def detect_cleared_lines(self) -> List[int]:
        """Detect completed lines that should be cleared."""
        cleared = []
        for y in range(self.rows - 1, -1, -1):  # Start from bottom
            if all(cell is not None for cell in self.grid[y]):
                cleared.append(y)
                logger.debug(f"Line {y} is complete and marked for clearing")
        
        if cleared:
            self.flash_animation = True
            self.cleared_lines = cleared
            # Mark lines for flash animation
            for y in cleared:
                for x in range(self.columns):
                    self.grid[y][x] = 'flash_color'
                    
        logger.info(f"Total lines detected for clearing: {len(cleared)}")
        return cleared

    def apply_gravity(self) -> None:
        """Apply gravity to make pieces fall after line clear."""
        if not self.cleared_lines:
            return

        # Remove completed lines and add new empty lines at top
        new_grid = []
        for y in range(self.rows):
            if y not in self.cleared_lines:
                new_grid.append(self.grid[y])
        
        # Add new empty lines at top
        for _ in range(len(self.cleared_lines)):
            new_grid.insert(0, [None for _ in range(self.columns)])
        
        self.grid = new_grid
        logger.debug(f"Applied gravity after clearing {len(self.cleared_lines)} lines")

    def clear_lines(self) -> int:
        """Clear completed lines and apply gravity."""
        if not self.cleared_lines:
            return 0

        # Apply gravity to make remaining pieces fall
        self.apply_gravity()

        num_cleared = len(self.cleared_lines)
        logger.info(f"Cleared {num_cleared} lines")
        
        # Reset state
        self.flash_animation = False
        self.cleared_lines = []
        return num_cleared

    def can_move(self, piece, new_x: int, new_y: int) -> bool:
        """Check if piece can move to specified position."""
        for y, row in enumerate(piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    grid_x = new_x + x
                    grid_y = new_y + y

                    # Check bounds
                    if grid_x < 0 or grid_x >= self.columns:
                        logger.debug(f"Movement blocked: x={grid_x} out of bounds")
                        return False

                    if grid_y >= len(self):
                        logger.debug(f"Movement blocked: y={grid_y} exceeds grid height")
                        return False

                    # Check collision with existing pieces
                    if grid_y >= 0 and self.grid[grid_y][grid_x] is not None:
                        if self.grid[grid_y][grid_x] != 'flash_color':  # Allow movement during flash
                            logger.debug(f"Movement blocked: cell ({grid_x}, {grid_y}) occupied")
                            return False
        return True

    def can_place(self, piece, x: int, y: int) -> bool:
        """Check if piece can be placed at position."""
        return self.can_move(piece, x, y)

    def lock_piece(self, piece) -> None:
        """Lock current piece into grid."""
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
        """Get color for grid cell."""
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
        """Get list of currently marked cleared lines."""
        return self.cleared_lines.copy()