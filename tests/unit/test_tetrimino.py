#!/usr/bin/env python3
"""
Unit tests for the Tetrimino class.
"""

import pytest
from unittest.mock import Mock, patch
from core.tetrimino import Tetrimino, TetriminoData
from core.grid import Grid
from config.game_config import GameConfig

@pytest.fixture
def config():
    """Fixture providing a game configuration."""
    return GameConfig()

@pytest.fixture
def mock_grid():
    """Fixture providing a mock grid."""
    grid = Mock(spec=Grid)
    grid.columns = 10
    grid.rows = 20
    return grid

@pytest.fixture
def grid(config):
    """Fixture providing a real grid instance."""
    return Grid(config)

class TestTetrimino:
    """Test suite for the Tetrimino class."""

    @pytest.mark.parametrize("piece_type,expected_width", [
        ("I", 4),
        ("O", 2),
        ("T", 3),
        ("L", 3),
        ("J", 3),
        ("S", 3),
        ("Z", 3),
    ])
    def test_piece_dimensions(self, piece_type, expected_width):
        """Test that each piece type has the correct dimensions."""
        piece = Tetrimino(piece_type, grid_size=30, columns=10)
        assert len(piece.shape[0]) == expected_width

    @pytest.mark.parametrize("piece_type,expected_start_x", [
        ("I", 6),  # 10 - 4 (width)
        ("O", 8),  # 10 - 2 (width)
        ("T", 7),  # 10 - 3 (width)
        ("L", 7),  # 10 - 3 (width)
    ])
    def test_initial_position(self, piece_type, expected_start_x):
        """Test that pieces spawn at the correct initial position."""
        piece = Tetrimino(piece_type, grid_size=30, columns=10)
        assert piece.x == expected_start_x
        assert piece.y == 0

    def test_get_positions(self):
        """Test that get_positions returns correct block coordinates."""
        piece = Tetrimino("T", grid_size=30, columns=10)
        positions = piece.get_positions()
        
        # T piece in initial rotation should have blocks at:
        # [0 1 0]
        # [1 1 1]
        # [0 0 0]
        expected_positions = {
            (piece.x + 1, 0),  # Top middle
            (piece.x, 1),      # Middle left
            (piece.x + 1, 1),  # Middle middle
            (piece.x + 2, 1),  # Middle right
        }
        
        assert len(positions) == 4
        assert set(positions) == expected_positions

    @pytest.mark.parametrize("piece_type,rotations", [
        ("I", 4),
        ("O", 1),
        ("T", 4),
        ("L", 4),
        ("J", 4),
        ("S", 4),
        ("Z", 4),
    ])
    def test_rotation_count(self, piece_type, rotations):
        """Test that each piece has the correct number of rotation states."""
        assert len(TetriminoData.SHAPES[piece_type]) == rotations

    def test_try_rotation_valid(self, grid):
        """Test that rotation succeeds when valid."""
        piece = Tetrimino("T", grid_size=30, columns=10)
        piece.x = 4  # Place in middle of grid
        piece.y = 4
        
        initial_shape = [row[:] for row in piece.shape]
        assert piece.try_rotation(grid, clockwise=True)
        assert piece.shape != initial_shape
        assert piece.rotation == 1

    def test_try_rotation_invalid_wall(self, grid):
        """Test that rotation fails when blocked by wall."""
        piece = Tetrimino("I", grid_size=30, columns=10)
        piece.x = 0  # Place at left wall
        piece.y = 0
        
        initial_shape = [row[:] for row in piece.shape]
        initial_rotation = piece.rotation
        
        assert not piece.try_rotation(grid, clockwise=True)
        assert piece.shape == initial_shape
        assert piece.rotation == initial_rotation

    def test_get_ghost_position(self, grid):
        """Test ghost piece positioning."""
        piece = Tetrimino("T", grid_size=30, columns=10)
        piece.x = 4
        piece.y = 0
        
        ghost_y = piece.get_ghost_position(grid)
        assert ghost_y == grid.rows - 2  # T piece is 2 blocks tall

    def test_is_valid_position(self, grid):
        """Test position validation."""
        piece = Tetrimino("O", grid_size=30, columns=10)
        
        # Valid position
        assert piece.is_valid_position(grid, x=4, y=4)
        
        # Invalid - out of bounds
        assert not piece.is_valid_position(grid, x=-1, y=0)
        assert not piece.is_valid_position(grid, x=9, y=0)
        assert not piece.is_valid_position(grid, x=0, y=-1)
        assert not piece.is_valid_position(grid, x=0, y=20)

    @pytest.mark.parametrize("piece_type", [
        "I", "O", "T", "L", "J", "S", "Z"
    ])
    def test_wall_kick_data_exists(self, piece_type):
        """Test that wall kick data exists for all relevant rotations."""
        if piece_type == "O":
            # O piece doesn't need wall kicks
            return
            
        for initial_rotation in range(4):
            next_rotation = (initial_rotation + 1) % 4
            kick_data = TetriminoData.WALL_KICK_DATA.get(piece_type, {})
            assert (initial_rotation, next_rotation) in kick_data

    def test_shape_validation(self):
        """Test shape validation logic."""
        # Valid shape
        valid_shape = [
            [0, 0, 0],
            [1, 1, 1],
            [0, 1, 0]
        ]
        assert TetriminoData.validate_shape(valid_shape)

        # Invalid - not square
        invalid_shape = [
            [0, 0],
            [1, 1, 1],
            [0, 1, 0]
        ]
        assert not TetriminoData.validate_shape(invalid_shape)

        # Invalid - empty
        assert not TetriminoData.validate_shape([])

    def test_rotation_with_wall_kicks(self, grid):
        """Test wall kick behavior during rotation."""
        # Place I piece near wall
        piece = Tetrimino("I", grid_size=30, columns=10)
        piece.x = 0
        piece.y = 5
        
        # Initial rotation should succeed with wall kick
        assert piece.try_rotation(grid, clockwise=True)
        # Piece should have moved right to accommodate rotation
        assert piece.x > 0

    def test_invalid_piece_type(self):
        """Test error handling for invalid piece type."""
        with pytest.raises(KeyError):
            Tetrimino("X", grid_size=30, columns=10)

if __name__ == "__main__":
    pytest.main(["-v", __file__])