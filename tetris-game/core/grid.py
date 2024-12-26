# File: core/grid.py

from typing import List, Optional, Any, Iterator
from utils.logging_setup import setup_logging

loggers = setup_logging()
game_logger = loggers['game']

class Grid:
    """
    Represents the Tetris game grid with complete sequence protocol implementation.
    
    Key Features:
    - Full sequence protocol support (iteration, indexing, length)
    - Bounds checking and validation
    - Safe cell access and modification
    - Efficient line clearing
    
    Implementation Details:
    The grid is implemented as a 2D list where:
    - Outer list represents rows (y-coordinate)
    - Inner lists represent columns (x-coordinate)
    - None represents empty cells
    - String values represent piece types
    """

    def __init__(self, config: Any):
        """
        Initialize the grid with dimensions from config.
        
        Args:
            config: GameConfig instance with grid dimensions
        """
        self.columns = config.columns
        self.rows = config.rows
        self._grid: List[List[Optional[str]]] = self._create_empty_grid()
        game_logger.debug(f"Initialized grid: {self.rows}x{self.columns}")

    def _create_empty_grid(self) -> List[List[Optional[str]]]:
        """Create a new empty grid with configured dimensions."""
        return [[None for _ in range(self.columns)] for _ in range(self.rows)]

    def __getitem__(self, index: int) -> List[Optional[str]]:
        """
        Enable grid[y] access with bounds checking.
        
        Args:
            index: Row index to access
            
        Returns:
            List[Optional[str]]: The requested row
            
        Raises:
            IndexError: If index is out of bounds
        """
        if not isinstance(index, int):
            raise TypeError(f"Grid indices must be integers, not {type(index)}")
        if not 0 <= index < self.rows:
            raise IndexError(f"Row index {index} out of range [0, {self.rows})")
        return self._grid[index]

    def __setitem__(self, index: int, value: List[Optional[str]]) -> None:
        """
        Enable grid[y] = row assignment with validation.
        
        Args:
            index: Row index to modify
            value: New row contents
            
        Raises:
            IndexError: If index is out of bounds
            ValueError: If row has incorrect length
        """
        if not isinstance(index, int):
            raise TypeError(f"Grid indices must be integers, not {type(index)}")
        if not 0 <= index < self.rows:
            raise IndexError(f"Row index {index} out of range [0, {self.rows})")
        if len(value) != self.columns:
            raise ValueError(f"Row must have length {self.columns}, got {len(value)}")
        self._grid[index] = value

    def __len__(self) -> int:
        """Return number of rows in grid."""
        return self.rows

    def __iter__(self) -> Iterator[List[Optional[str]]]:
        """Enable iteration over grid rows."""
        return iter(self._grid)

    def reset(self) -> None:
        """Reset grid to initial empty state."""
        self._grid = self._create_empty_grid()
        game_logger.debug("Grid reset")

    def get_cell(self, x: int, y: int) -> Optional[str]:
        """
        Get cell content with bounds checking.
        
        Args:
            x: Column index
            y: Row index
            
        Returns:
            Optional[str]: Cell contents or None if empty/invalid
        """
        if 0 <= x < self.columns and 0 <= y < self.rows:
            return self._grid[y][x]
        return None

    def set_cell(self, x: int, y: int, value: Optional[str]) -> bool:
        """
        Set cell content with bounds checking.
        
        Args:
            x: Column index
            y: Row index
            value: New cell content
            
        Returns:
            bool: True if cell was set, False if out of bounds
        """
        if 0 <= x < self.columns and 0 <= y < self.rows:
            self._grid[y][x] = value
            return True
        return False

    def can_move(self, piece: Any, x: int, y: int) -> bool:
        """Test if piece can move to specified position."""
        return piece.is_valid_position(self._grid, x, y)

    def can_place(self, piece: Any, x: int, y: int) -> bool:
        """Test if piece can be placed at specified position."""
        return self.can_move(piece, x, y)

    def lock_piece(self, piece: Any) -> None:
        """
        Lock piece into grid, marking its occupied cells.
        
        Args:
            piece: Tetrimino to lock in place
        """
        try:
            shape = piece.shape
            for row_idx, row in enumerate(shape):
                for col_idx, cell in enumerate(row):
                    if cell:
                        grid_x = piece.x + col_idx
                        grid_y = piece.y + row_idx
                        if 0 <= grid_x < self.columns and 0 <= grid_y < self.rows:
                            self._grid[grid_y][grid_x] = piece.piece_type
            game_logger.debug(f"Locked piece {piece.piece_type} at ({piece.x}, {piece.y})")
        except Exception as e:
            game_logger.error(f"Error locking piece: {e}")
            raise

    def clear_lines(self) -> int:
        """
        Remove completed lines and return count.
        
        Returns:
            int: Number of lines cleared
        """
        try:
            # Find completed lines
            full_rows = [
                y for y in range(self.rows)
                if all(cell is not None for cell in self._grid[y])
            ]
            
            if not full_rows:
                return 0
                
            # Remove completed lines
            new_grid = [
                row for i, row in enumerate(self._grid)
                if i not in full_rows
            ]
            
            # Add new empty lines at top
            lines_cleared = len(full_rows)
            for _ in range(lines_cleared):
                new_grid.insert(0, [None] * self.columns)
                
            self._grid = new_grid
            game_logger.debug(f"Cleared {lines_cleared} lines")
            return lines_cleared
            
        except Exception as e:
            game_logger.error(f"Error clearing lines: {e}")
            return 0

    @property
    def grid(self) -> List[List[Optional[str]]]:
        """Provide read-only access to underlying grid data."""
        return self._grid