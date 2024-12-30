#!/usr/bin/env python3
"""
Unit tests for the UIManager class.
"""

import pytest
import pygame
from unittest.mock import Mock, patch, call
from config.game_config import GameConfig
from render.render_config import RenderConfig
from render.ui_manager import UIManager
from core.tetrimino import Tetrimino


@pytest.fixture
def config():
    """Fixture providing game configuration."""
    config = GameConfig()
    config.screen_width = 300
    config.screen_height = 600
    config.grid_size = 30
    config.ui_width = 200
    config.preview_pieces = 3
    return config


@pytest.fixture
def mock_surface():
    """Fixture providing a mock pygame surface."""
    surface = Mock(spec=pygame.Surface)
    surface.get_rect.return_value = Mock(
        center=(150, 300),
        topleft=(0, 0),
        width=200,
        height=600
    )
    return surface


@pytest.fixture
def mock_font():
    """Fixture providing a mock font."""
    font = Mock(spec=pygame.font.Font)
    font.render.return_value = Mock(spec=pygame.Surface)
    font.render.return_value.get_rect.return_value = Mock(
        width=100,
        height=20,
        center=(100, 300)
    )
    return font


@pytest.fixture
def render_config(config):
    """Fixture providing render configuration with mocked fonts."""
    with patch('pygame.font.Font'):
        render_config = RenderConfig(config)
        render_config.fonts = {
            'small': Mock(spec=pygame.font.Font),
            'medium': Mock(spec=pygame.font.Font),
            'large': Mock(spec=pygame.font.Font)
        }
        for font in render_config.fonts.values():
            font.render.return_value = Mock(spec=pygame.Surface)
            font.render.return_value.get_rect.return_value = Mock(
                width=100,
                height=20,
                center=(100, 300)
            )
        return render_config


@pytest.fixture
def ui_manager(config, render_config):
    """Fixture providing a UIManager instance."""
    return UIManager(config, render_config)


@pytest.fixture
def mock_tetrimino():
    """Fixture providing a mock Tetrimino."""
    tetrimino = Mock(spec=Tetrimino)
    tetrimino.shape = [[1, 1], [1, 1]]
    tetrimino.piece_type = 'O'
    return tetrimino


class TestUIManager:
    """Test suite for the UIManager class."""

    def test_initialization(self, ui_manager, config, render_config):
        """Test UIManager initialization."""
        assert ui_manager.config == config
        assert ui_manager.render_config == render_config
        assert ui_manager._last_next_piece_y is None
        assert ui_manager.preview_scale == 0.8

    def test_draw_game_ui(self, ui_manager, mock_surface, mock_tetrimino):
        """Test game UI drawing."""
        score = 1000
        high_score = 2000
        level = 1
        lines = 10
        next_pieces = [mock_tetrimino]
        held_piece = mock_tetrimino
        combo = 2

        ui_manager.draw_game_ui(
            mock_surface,
            score,
            high_score,
            level,
            lines,
            next_pieces,
            held_piece,
            combo
        )

        # Verify surface was used
        assert mock_surface.blit.called

    def test_draw_menu(self, ui_manager, mock_surface):
        """Test menu drawing."""
        menu_options = ["Start", "Options", "Quit"]
        selected_index = 1

        ui_manager.draw_menu(mock_surface, menu_options, selected_index)

        # Verify text rendering for each option
        assert ui_manager.render_config.fonts['large'].render.call_count == len(menu_options)
        assert mock_surface.blit.called

    def test_draw_score_section(self, ui_manager, mock_surface):
        """Test score section drawing."""
        score = 1000
        high_score = 2000
        level = 1
        lines = 10
        combo = 2

        ui_manager._draw_score_section(
            mock_surface,
            score,
            high_score,
            level,
            lines,
            combo
        )

        # Verify all score components were rendered
        font_calls = ui_manager.render_config.fonts['small'].render.call_args_list
        assert any('Score: 1000' in str(call) for call in font_calls)
        assert any('Level: 1' in str(call) for call in font_calls)
        assert any('Lines: 10' in str(call) for call in font_calls)
        assert any('Combo: x2' in str(call) for call in font_calls)

    def test_draw_next_pieces(self, ui_manager, mock_surface, mock_tetrimino):
        """Test next pieces preview drawing."""
        next_pieces = [mock_tetrimino, mock_tetrimino]

        ui_manager._draw_next_pieces(mock_surface, next_pieces)

        # Verify "Next" label was rendered
        assert ui_manager.render_config.fonts['small'].render.called
        assert mock_surface.blit.called
        
        # Verify _last_next_piece_y was updated
        assert ui_manager._last_next_piece_y is not None

    def test_draw_held_piece(self, ui_manager, mock_surface, mock_tetrimino):
        """Test held piece drawing."""
        ui_manager._last_next_piece_y = 400
        ui_manager._draw_held_piece(mock_surface, mock_tetrimino)

        # Verify "Hold" label was rendered
        assert ui_manager.render_config.fonts['small'].render.called
        assert mock_surface.blit.called

    def test_draw_held_piece_no_last_y(self, ui_manager, mock_surface, mock_tetrimino):
        """Test held piece drawing without last_next_piece_y."""
        ui_manager._last_next_piece_y = None
        ui_manager._draw_held_piece(mock_surface, mock_tetrimino)

        # Should still render with default position
        assert ui_manager.render_config.fonts['small'].render.called
        assert mock_surface.blit.called

    @pytest.mark.parametrize("score,expected_width", [
        (1000, 100),
        (10000, 100),
        (1000000, 100)
    ])
    def test_score_text_scaling(self, ui_manager, mock_surface, score, expected_width):
        """Test score text rendering with different lengths."""
        ui_manager._draw_score_section(
            mock_surface,
            score,
            high_score=0,
            level=1,
            lines=0,
            combo=0
        )
        assert mock_surface.blit.called

    def test_menu_selection_highlight(self, ui_manager, mock_surface):
        """Test menu selection highlighting."""
        menu_options = ["Start", "Options", "Quit"]
        selected_index = 1

        ui_manager.draw_menu(mock_surface, menu_options, selected_index)

        # Verify selected option uses highlight color
        highlight_color = ui_manager.render_config.colors.UI_COLORS['highlight']
        font_calls = ui_manager.render_config.fonts['large'].render.call_args_list
        assert any(call[0][2] == highlight_color for call in font_calls)

    def test_combo_display_threshold(self, ui_manager, mock_surface):
        """Test combo display threshold."""
        # Test with no combo
        ui_manager._draw_score_section(
            mock_surface,
            score=0,
            high_score=0,
            level=1,
            lines=0,
            combo=0
        )
        font_calls = ui_manager.render_config.fonts['small'].render.call_args_list
        assert not any('Combo' in str(call) for call in font_calls)

        # Test with active combo
        ui_manager._draw_score_section(
            mock_surface,
            score=0,
            high_score=0,
            level=1,
            lines=0,
            combo=2
        )
        font_calls = ui_manager.render_config.fonts['small'].render.call_args_list
        assert any('Combo' in str(call) for call in font_calls)

    def test_preview_piece_scaling(self, ui_manager, mock_surface, mock_tetrimino):
        """Test preview piece scaling."""
        next_pieces = [mock_tetrimino]
        
        with patch('pygame.transform.smoothscale') as mock_scale:
            ui_manager._draw_next_pieces(mock_surface, next_pieces)
            assert mock_scale.called

    def test_ui_element_spacing(self, ui_manager, mock_surface):
        """Test UI element vertical spacing."""
        score = 1000
        high_score = 2000
        level = 1
        lines = 10
        combo = 2

        ui_manager._draw_score_section(
            mock_surface,
            score,
            high_score,
            level,
            lines,
            combo
        )

        # Verify vertical positioning of elements
        blit_calls = mock_surface.blit.call_args_list
        y_positions = [call[0][1][1] for call in blit_calls]
        assert all(y2 > y1 for y1, y2 in zip(y_positions, y_positions[1:]))

    def test_error_handling(self, ui_manager, mock_surface):
        """Test error handling for UI drawing."""
        # Test with invalid surface
        with pytest.raises(AttributeError):
            ui_manager.draw_game_ui(
                None,
                score=0,
                high_score=0,
                level=1,
                lines=0,
                next_pieces=[],
                held_piece=None,
                combo=0
            )


if __name__ == "__main__":
    pytest.main(["-v", __file__])