#!/usr/bin/env python3

import unittest
from core.game_state import GameState
from core.tetrimino import Tetrimino, TetriminoData
from core.grid import Grid
from config.game_config import GameConfig


class TestGameState(unittest.TestCase):
    def test_enum_values(self):
        """
        Test that GameState enum contains all expected values.
        """
        expected_states = ["MENU", "PLAYING", "PAUSED", "GAME_OVER", "ANIMATING"]
        actual_states = [state.name for state in GameState]
        self.assertEqual(expected_states, actual_states)

    def test_enum_str(self):
        """
        Test that the string representation of GameState matches its name.
        """
        for state in GameState:
            self.assertEqual(str(state), state.name)


class TestTetriminoData(unittest.TestCase):
    def test_validate_shape(self):
        """
        Test validation of Tetrimino shapes.
        """
        valid_shape = [[1, 0, 0], [1, 1, 1], [0, 0, 0]]
        invalid_shape = [[1, 0], [1, 1, 1], [0, 0, 0]]  # Uneven rows
        self.assertTrue(TetriminoData.validate_shape(valid_shape))
        self.assertFalse(TetriminoData.validate_shape(invalid_shape))

    def test_get_initial_shape(self):
        """
        Test retrieval of initial shapes.
        """
        shape = TetriminoData.get_initial_shape("T")
        self.assertEqual(len(shape), 3)
        self.assertEqual(len(shape[0]), 3)

    def test_get_invalid_piece(self):
        """
        Test that accessing an invalid piece raises an error.
        """
        with self.assertRaises(KeyError):
            TetriminoData.get_initial_shape("InvalidPiece")


class TestTetrimino(unittest.TestCase):
    def setUp(self):
        """
        Initialize a basic Tetrimino and config for testing.
        """
        self.config = GameConfig()
        self.grid = Grid(self.config)
        self.tetrimino = Tetrimino(piece_type="T", grid_size=self.config.grid_size, columns=self.config.columns)

    def test_initial_position(self):
        """
        Test that a Tetrimino is initialized at the correct position.
        """
        piece_type = 'T'
        self.tetrimino = Tetrimino(piece_type, self.config.grid_size, self.config.columns)
        expected_x = (self.config.columns - len(TetriminoData.get_initial_shape(piece_type)[0])) // 2
        self.assertEqual(self.tetrimino.x, expected_x)

    def test_get_positions(self):
        """
        Test that Tetrimino block positions are calculated correctly.
        """
        positions = self.tetrimino.get_positions()
        expected_positions = [
            (self.tetrimino.x + col, self.tetrimino.y + row)
            for row, line in enumerate(self.tetrimino.shape)
            for col, cell in enumerate(line)
            if cell
        ]
        self.assertEqual(positions, expected_positions)

    def test_rotation_valid(self):
        """
        Test that a valid rotation is applied correctly.
        """
        initial_shape = self.tetrimino.shape
        self.tetrimino.try_rotation(self.grid, clockwise=True)
        rotated_shape = self.tetrimino.shape
        self.assertNotEqual(initial_shape, rotated_shape)

    '''
    def test_rotation_invalid(self):
        """
        Test that an invalid rotation is rejected.
        """
        self.grid = [[None] * 10 for _ in range(20)]
        self.grid[0][4] = 'OCCUPIED'  # Place a block in the rotation path
        self.tetrimino.x = 3
        print("Grid State Before Rotation:")
        for row in self.grid:
            print(row)
        print(f"Tetrimino Position: x={self.tetrimino.x}, y={self.tetrimino.y}")
        print(f"Tetrimino Shape Before Rotation: {self.tetrimino.shape}")
        self.assertFalse(self.tetrimino.try_rotation(self.grid, clockwise=True))
        print(f"Tetrimino Rotation After Attempt: {self.tetrimino.rotation}")
        print(f"Tetrimino Position After Attempt: x={self.tetrimino.x}, y={self.tetrimino.y}")
    '''

    def test_ghost_position(self):
        """
        Test that ghost position is calculated correctly.
        """
        ghost_y = self.tetrimino.get_ghost_position(self.grid)
        self.assertGreaterEqual(ghost_y, self.tetrimino.y)

    def test_is_valid_position(self):
        """
        Test that is_valid_position correctly identifies invalid positions.
        """
        self.grid = [[None] * 10 for _ in range(20)]
        self.grid[0][4] = 'OCCUPIED'
        shape = self.tetrimino.shape
        self.assertFalse(self.tetrimino.is_valid_position(self.grid, 3, 0, shape))
        self.assertTrue(self.tetrimino.is_valid_position(self.grid, 0, 0, shape))


class TestGame(unittest.TestCase):
    def test_game_initialization(self):
        """
        Test initialization of the game with valid configuration.
        """
        config = GameConfig()
        from core.game import TetrisGame
        game = TetrisGame(config)
        self.assertEqual(game.config.screen_width, config.screen_width)

    def test_game_reset(self):
        """
        Test that the game resets correctly.
        """
        from core.game import TetrisGame
        config = GameConfig()
        game = TetrisGame(config)
        game.reset()
        self.assertEqual(game.lines_cleared, 0)
        self.assertEqual(game.combo_counter, 0)
        self.assertEqual(len(game.next_pieces), config.preview_pieces)

    def test_game_over_state(self):
        """
        Test that the game correctly handles the game-over state.
        """
        from core.game import TetrisGame
        config = GameConfig()
        game = TetrisGame(config)
        game.state = GameState.GAME_OVER
        self.assertEqual(game.state, GameState.GAME_OVER)


if __name__ == "__main__":
    unittest.main()
