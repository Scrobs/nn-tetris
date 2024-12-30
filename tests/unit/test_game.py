#!/usr/bin/env python3
"""
Unit tests for the TetrisGame class.
"""

import pytest
import pygame
from unittest.mock import Mock, patch, MagicMock
from core.game import TetrisGame
from core.game_state import GameState
from core.tetrimino import TetriminoData
from config.game_config import GameConfig


@pytest.fixture(autouse=True)
def mock_pygame():
    """Fixture to mock pygame initialization and display."""
    with patch('pygame.init'), \
         patch('pygame.display.init'), \
         patch('pygame.display.set_mode', return_value=Mock()), \
         patch('pygame.time.Clock', return_value=Mock()):
        yield


@pytest.fixture
def config():
    """Fixture providing game configuration."""
    config = GameConfig()
    config.particle_effects = False  # Disable for testing
    config.debug_mode = False
    return config


@pytest.fixture
def mock_renderer():
    """Fixture providing a mock renderer."""
    renderer = Mock()
    renderer.render_config = Mock()
    renderer.render_config.sounds = {}
    return renderer


@pytest.fixture
def game(config, mock_renderer):
    """Fixture providing a game instance with mocked components."""
    with patch('core.game.RenderSystem', return_value=mock_renderer):
        game = TetrisGame(config)
        game.renderer = mock_renderer
        return game


class TestTetrisGame:
    """Test suite for the TetrisGame class."""

    def test_initialization(self, game):
        """Test game initialization."""
        assert game.state == GameState.MENU
        assert game.frame_counter == 0
        assert game.level == game.config.initial_level
        assert game.lines_cleared == 0
        assert game.combo_counter == 0
        assert isinstance(game.next_pieces, list)
        assert len(game.next_pieces) == game.config.preview_pieces
        assert game.held_piece is None

    def test_reset(self, game):
        """Test game reset functionality."""
        # Modify game state
        game.level = 5
        game.lines_cleared = 10
        game.combo_counter = 3
        game.state = GameState.GAME_OVER

        game.reset()

        assert game.level == game.config.initial_level
        assert game.lines_cleared == 0
        assert game.combo_counter == 0
        assert game.state == GameState.PLAYING
        assert len(game.next_pieces) == game.config.preview_pieces

    @patch('core.game.time.perf_counter', side_effect=[0.0, 0.016667])
    def test_game_loop_timing(self, mock_time, game):
        """Test game loop timing mechanics."""
        game.state = GameState.PLAYING
        game.running = True

        # Simulate one frame
        with patch.object(game, '_handle_events', return_value=None) as mock_handle_events:
            game._update_physics(game.fixed_timestep)
            
            assert mock_handle_events.called
            assert game.physics_updates > 0

    def test_piece_movement(self, game):
        """Test piece movement mechanics."""
        game.state = GameState.PLAYING
        initial_x = game.current_piece.x
        initial_y = game.current_piece.y

        # Test left movement
        game.move_left()
        assert game.current_piece.x == initial_x - 1

        # Test right movement
        game.move_right()
        assert game.current_piece.x == initial_x

        # Test soft drop
        game.soft_drop()
        assert game.current_piece.y == initial_y + 1

    def test_piece_rotation(self, game):
        """Test piece rotation mechanics."""
        game.state = GameState.PLAYING
        initial_rotation = game.current_piece.rotation

        # Test clockwise rotation
        game.rotate_cw()
        assert game.current_piece.rotation == (initial_rotation + 1) % 4

        # Test counter-clockwise rotation
        game.rotate_ccw()
        assert game.current_piece.rotation == initial_rotation

    def test_hard_drop(self, game):
        """Test hard drop functionality."""
        game.state = GameState.PLAYING
        initial_y = game.current_piece.y

        game.hard_drop()

        # Piece should be at bottom or locked
        assert (game.current_piece is None or 
                game.current_piece.y > initial_y)

    def test_hold_piece(self, game):
        """Test piece hold mechanics."""
        game.state = GameState.PLAYING
        original_piece = game.current_piece
        
        # First hold
        game.hold()
        assert game.held_piece == original_piece
        assert game.current_piece != original_piece

        # Second hold (shouldn't change held piece)
        current_piece = game.current_piece
        game.hold()
        assert game.held_piece == original_piece
        assert game.current_piece == original_piece

    def test_line_clearing(self, game):
        """Test line clearing mechanics."""
        game.state = GameState.PLAYING
        initial_lines = game.lines_cleared
        
        # Fill a line
        for x in range(game.config.columns):
            game.grid.grid[game.grid.rows - 1][x] = "I"

        game._clear_lines()
        
        assert game.lines_cleared == initial_lines + 1

    def test_game_over_condition(self, game):
        """Test game over detection."""
        # Fill top rows to force game over
        for x in range(game.config.columns):
            game.grid.grid[0][x] = "I"
            game.grid.grid[1][x] = "I"

        game._spawn_new_piece()
        assert game.state == GameState.GAME_OVER

    def test_pause_functionality(self, game):
        """Test pause mechanics."""
        game.state = GameState.PLAYING
        game.toggle_pause()
        assert game.state == GameState.PAUSED

        game.toggle_pause()
        assert game.state == GameState.PLAYING

    def test_scoring_system(self, game):
        """Test scoring mechanics."""
        initial_score = game.score_handler.value
        
        # Clear one line
        game._calculate_reward(1)
        assert game.score_handler.value > initial_score

        # Clear multiple lines (should give bonus)
        game._calculate_reward(4)
        assert game.score_handler.value > initial_score + game.config.scoring_values['tetris']

    def test_level_progression(self, game):
        """Test level progression mechanics."""
        initial_level = game.level
        
        # Clear enough lines to trigger level up
        for _ in range(game.config.lines_per_level):
            game._clear_lines()
            game.lines_cleared += 1

        assert game.level > initial_level

    @pytest.mark.parametrize("piece_type", list(TetriminoData.SHAPES.keys()))
    def test_piece_spawning(self, game, piece_type):
        """Test spawning of different piece types."""
        game.bag = [piece_type]
        piece = game._get_next_piece()
        
        assert piece.piece_type == piece_type
        assert piece.x >= 0
        assert piece.x + len(piece.shape[0]) <= game.config.columns

    def test_gravity_mechanics(self, game):
        """Test gravity mechanics."""
        game.state = GameState.PLAYING
        initial_y = game.current_piece.y
        
        # Simulate gravity
        game._apply_gravity()
        
        assert game.current_piece.y > initial_y

    def test_lock_delay(self, game):
        """Test piece locking delay."""
        game.state = GameState.PLAYING
        game.current_piece.y = game.grid.rows - 2  # Place near bottom
        
        # Simulate lock delay
        game._apply_gravity()
        assert game.lock_delay_start is not None
        
        # Simulate lock delay expiration
        game.lock_delay_accumulated = game.config.lock_delay
        game._apply_gravity()
        
        # Piece should be locked
        assert game.current_piece is not None

    def test_debug_mode(self, game):
        """Test debug mode functionality."""
        game.config.debug_mode = True
        game.toggle_debug()
        assert not game.config.debug_mode

        game.toggle_debug()
        assert game.config.debug_mode

    def test_cleanup(self, game):
        """Test cleanup functionality."""
        with patch('pygame.quit') as mock_quit:
            game.cleanup()
            assert mock_quit.called

    @patch('pygame.event.get')
    def test_menu_navigation(self, mock_event, game):
        """Test menu navigation."""
        game.state = GameState.MENU
        
        # Simulate down key press
        mock_event.return_value = [Mock(type=pygame.KEYDOWN, key=pygame.K_DOWN)]
        initial_index = game.selected_menu_index
        game._handle_events()
        assert game.selected_menu_index == (initial_index + 1) % 4

        # Simulate menu selection
        mock_event.return_value = [Mock(type=pygame.KEYDOWN, key=pygame.K_RETURN)]
        game._handle_events()


if __name__ == "__main__":
    pytest.main(["-v", __file__])