# File: core/grid.py
from typing import List, Optional
import logging
from utils.logging_setup import setup_logging
from render.colors import Colors, Color

loggers = setup_logging()
logger = logging.getLogger('tetris.grid')

class Grid:
    def __init__(self, config):
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

    def __getitem__(self, index: int) -> List[Optional[str]]:
        return self.grid[index]

    def __len__(self) -> int:
        return len(self.grid)

    def reset(self) -> None:
        self.grid = [
            [None for _ in range(self.columns)] for _ in range(self.rows)
        ]
        self.cleared_lines = []
        self.flash_animation = False
        logger.debug("Grid reset to empty state")

    def detect_cleared_lines(self) -> List[int]:
        cleared = []
        for y in range(self.rows - 1, -1, -1):
            if all(cell is not None for cell in self.grid[y]):
                cleared.append(y)
                logger.debug(f"Line {y} is complete and marked for clearing")

        if cleared:
            self.flash_animation = True
            self.cleared_lines = cleared
            for y in cleared:
                for x in range(self.columns):
                    self.grid[y][x] = 'flash_color'
            logger.info(f"Total lines detected for clearing: {len(cleared)}")

        return cleared

    def clear_cleared_lines(self) -> int:
        if not self.cleared_lines:
            return 0

        new_grid = []
        num_cleared = len(self.cleared_lines)

        for y in range(self.rows):
            if y not in self.cleared_lines:
                new_grid.append(self.grid[y])

        for _ in range(num_cleared):
            new_grid.insert(0, [None for _ in range(self.columns)])

        self.grid = new_grid
        logger.info(f"Cleared {num_cleared} lines")
        self.flash_animation = False
        self.cleared_lines = []

        return num_cleared

    def can_move(self, piece, new_x: int, new_y: int) -> bool:
        for y, row in enumerate(piece.shape):
            for x, cell in enumerate(row):
                if cell:
                    grid_x = new_x + x
                    grid_y = new_y + y

                    if grid_x < 0 or grid_x >= self.columns:
                        logger.debug(f"Movement blocked: x={grid_x} out of bounds")
                        return False

                    if grid_y >= len(self):
                        logger.debug(f"Movement blocked: y={grid_y} exceeds grid height")
                        return False

                    if grid_y >= 0 and self.grid[grid_y][grid_x] is not None:
                        if self.grid[grid_y][grid_x] != 'flash_color':
                            logger.debug(f"Movement blocked: cell ({grid_x}, {grid_y}) occupied")
                            return False

        return True

    def can_place(self, piece, x: int, y: int) -> bool:
        return self.can_move(piece, x, y)

    def lock_piece(self, piece) -> None:
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
        return self.cleared_lines.copy()