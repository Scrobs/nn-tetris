#!/usr/bin/env python3
"""
Unit tests for the Grid class.
"""

import pytest
from unittest.mock import Mock, patch
from core.grid import Grid
from core.tetrimino import Tetrimino
from config.game_config import GameConfig
from render.colors import Colors


@pytest.fixture
def config():
    """Fixture providing a game configuration."""
    config = GameConfig()
    config.rows = 20
    config.columns = 10
    config.grid_size = 30
    return config


@pytest.fixture
def grid(config):
    """Fixture providing a Grid instance."""
    return Grid(config)


@pytest.fixture
def filled_grid(grid):
    """Fixture providing a grid with some filled cells."""
    # Fill bottom row
    for x in range(grid.columns):
        grid.grid[grid.rows - 1][x] = "I"
    
    # Fill second to bottom row except last cell
    for x in range(grid.columns - 1):
        grid.grid[grid.rows - 2][x] = "T"
    
    return grid


@pytest.fixture
def mock_tetrimino():
    """Fixture providing a mock Tetrimino."""
    piece = Mock(spec=Tetrimino)
    piece.shape = [
        [1, 1],
        [1, 1]
    ]
    piece.piece_type = "O"
    piece.x = 0
    piece.y = 0
    return piece


class TestGrid:
    """Test suite for the Grid class."""

    def test_initialization(self, grid):
        """Test grid initialization."""
        assert len(grid.grid) == 20  # rows
        assert len(grid.grid[0]) == 10  # columns
        assert all(cell is None for row in grid.grid for cell in row)
        assert isinstance(grid.colors, Colors)
        assert not grid.flash_animation
        assert grid.cleared_lines == []

    def test_reset(self, filled_grid):
        """Test grid reset functionality."""
        filled_grid.reset()
        assert all(cell is None for row in filled_grid.grid for cell in row)
        assert not filled_grid.flash_animation
        assert filled_grid.cleared_lines == []

    def test_detect_cleared_lines_none(self, grid):
        """Test line detection with no complete lines."""
        cleared = grid.detect_cleared_lines()
        assert len(cleared) == 0
        assert not grid.flash_animation

    def test_detect_cleared_lines_single(self, filled_grid):
        """Test detection of a single complete line."""
        cleared = filled_grid.detect_cleared_lines()
        assert len(cleared) == 1
        assert cleared[0] == filled_grid.rows - 1
        assert filled_grid.flash_animation

    def test_detect_cleared_lines_multiple(self, grid):
        """Test detection of multiple complete lines."""
        # Fill two complete rows
        for x in range(grid.columns):
            grid.grid[grid.rows - 1][x] = "I"
            grid.grid[grid.rows - 2][x] = "T"

        cleared = grid.detect_cleared_lines()
        assert len(cleared) == 2
        assert grid.flash_animation
        assert grid.rows - 1 in cleared
        assert grid.rows - 2 in cleared

    def test_clear_cleared_lines(self, filled_grid):
        """Test clearing of detected lines."""
        filled_grid.detect_cleared_lines()
        num_cleared = filled_grid.clear_cleared_lines()
        
        assert num_cleared == 1
        assert not filled_grid.flash_animation
        assert filled_grid.cleared_lines == []
        
        # Check that top row is now empty
        assert all(cell is None for cell in filled_grid.grid[0])

    def test_can_move_valid(self, grid, mock_tetrimino):
        """Test valid movement detection."""
        assert grid.can_move(mock_tetrimino, 0, 0)
        assert grid.can_move(mock_tetrimino, grid.columns - 2, 0)
        assert grid.can_move(mock_tetrimino, 0, grid.rows - 2)

    def test_can_move_invalid_bounds(self, grid, mock_tetrimino):
        """Test invalid movement detection due to bounds."""
        # Test left boundary
        assert not grid.can_move(mock_tetrimino, -1, 0)
        
        # Test right boundary
        assert not grid.can_move(mock_tetrimino, grid.columns - 1, 0)
        
        # Test bottom boundary
        assert not grid.can_move(mock_tetrimino, 0, grid.rows)
        
        # Test top boundary
        assert not grid.can_move(mock_tetrimino, 0, -1)

    def test_can_move_collision(self, filled_grid, mock_tetrimino):
        """Test movement collision detection."""
        # Try to move to occupied space
        mock_tetrimino.x = 0
        mock_tetrimino.y = filled_grid.rows - 2
        
        assert not filled_grid.can_move(mock_tetrimino, 0, filled_grid.rows - 1)

    def test_lock_piece(self, grid, mock_tetrimino):
        """Test piece locking functionality."""
        mock_tetrimino.x = 0
        mock_tetrimino.y = 0
        
        grid.lock_piece(mock_tetrimino)
        
        # Check that cells are filled with piece type
        assert grid.grid[0][0] == "O"
        assert grid.grid[0][1] == "O"
        assert grid.grid[1][0] == "O"
        assert grid.grid[1][1] == "O"

    def test_lock_piece_out_of_bounds(self, grid, mock_tetrimino):
        """Test piece locking with out of bounds coordinates."""
        mock_tetrimino.x = -1
        mock_tetrimino.y = -1
        
        # Should not raise exception
        grid.lock_piece(mock_tetrimino)

    def test_lock_piece_none(self, grid):
        """Test locking with None piece."""
        grid.lock_piece(None)  # Should not raise exception

    def test_get_cell_color(self, grid):
        """Test cell color retrieval."""
        # Test empty cell
        assert grid.get_cell_color(0, 0) == grid.colors.UI_COLORS['background']
        
        # Test piece cell
        grid.grid[0][0] = "I"
        assert grid.get_cell_color(0, 0) == grid.colors.get_piece_color("I")
        
        # Test flash color
        grid.grid[0][1] = "flash_color"
        assert grid.get_cell_color(0, 1) == grid.colors.FLASH_COLOR
        
        # Test normal color
        grid.grid[0][2] = "normal_color"
        assert grid.get_cell_color(0, 2) == grid.colors.NORMAL_COLOR

    def test_get_cell_color_out_of_bounds(self, grid):
        """Test color retrieval for out of bounds coordinates."""
        assert grid.get_cell_color(-1, 0) == grid.colors.UI_COLORS['background']
        assert grid.get_cell_color(0, -1) == grid.colors.UI_COLORS['background']
        assert grid.get_cell_color(grid.columns, 0) == grid.colors.UI_COLORS['background']
        assert grid.get_cell_color(0, grid.rows) == grid.colors.UI_COLORS['background']

    def test_get_cleared_lines(self, filled_grid):
        """Test cleared lines getter."""
        filled_grid.detect_cleared_lines()
        cleared = filled_grid.get_cleared_lines()
        
        assert isinstance(cleared, list)
        assert len(cleared) == 1
        assert cleared != filled_grid.cleared_lines  # Should be a copy

    def test_can_place(self, grid, mock_tetrimino):
        """Test piece placement validation."""
        assert grid.can_place(mock_tetrimino, 0, 0)
        assert not grid.can_place(mock_tetrimino, -1, 0)
        assert not grid.can_place(mock_tetrimino, 0, grid.rows)

    @pytest.mark.parametrize("rows,cols", [
        (15, 8),
        (25, 12),
        (10, 6),
    ])
    def test_different_grid_sizes(self, config, rows, cols):
        """Test grid initialization with different dimensions."""
        config.rows = rows
        config.columns = cols
        grid = Grid(config)
        
        assert len(grid.grid) == rows
        assert len(grid.grid[0]) == cols
        assert all(cell is None for row in grid.grid for cell in row)


if __name__ == "__main__":
    pytest.main(["-v", __file__])