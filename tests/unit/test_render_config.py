#!/usr/bin/env python3
"""
Unit tests for the RenderConfig class.
"""

import pytest
import pygame
from unittest.mock import Mock, patch, mock_open
from config.game_config import GameConfig
from render.render_config import RenderConfig
from render.colors import Colors


@pytest.fixture
def game_config():
    """Fixture providing a basic game configuration."""
    config = GameConfig()
    config.grid_size = 30
    config.cell_padding = 2
    config.grid_line_width = 1
    config.preview_margin = 10
    config.preview_pieces = 3
    config.ui_width = 200
    config.screen_width = 300
    config.screen_height = 600
    config.debug_mode = False
    config.particle_effects = True
    return config


@pytest.fixture
def mock_pygame_font():
    """Fixture providing mocked pygame font functionality."""
    with patch('pygame.font.Font') as mock_font:
        mock_font.return_value = Mock(spec=pygame.font.Font)
        yield mock_font


@pytest.fixture
def mock_pygame_mixer():
    """Fixture providing mocked pygame mixer functionality."""
    with patch('pygame.mixer.Sound') as mock_sound:
        mock_sound.return_value = Mock(spec=pygame.mixer.Sound)
        yield mock_sound


@pytest.fixture
def render_config(game_config, mock_pygame_font, mock_pygame_mixer):
    """Fixture providing a RenderConfig instance with mocked dependencies."""
    return RenderConfig(game_config)


class TestRenderConfig:
    """Test suite for the RenderConfig class."""

    def test_initialization(self, render_config, game_config):
        """Test basic initialization of RenderConfig."""
        assert isinstance(render_config.colors, Colors)
        assert render_config.grid_size == game_config.grid_size
        assert render_config.cell_padding == game_config.cell_padding
        assert render_config.grid_line_width == game_config.grid_line_width
        assert render_config.preview_margin == game_config.preview_margin
        assert render_config.preview_pieces == game_config.preview_pieces
        assert render_config.ui_width == game_config.ui_width
        assert render_config.screen_width == game_config.screen_width
        assert render_config.screen_height == game_config.screen_height
        assert render_config.debug_mode == game_config.debug_mode
        assert render_config.particle_effects == game_config.particle_effects

    def test_font_initialization(self, render_config):
        """Test font initialization and sizes."""
        assert 'small' in render_config.fonts
        assert 'medium' in render_config.fonts
        assert 'large' in render_config.fonts
        
        assert render_config.font_small_size == 12
        assert render_config.font_medium_size == 18
        assert render_config.font_large_size == 26

    @patch('pygame.font.Font')
    def test_font_fallback(self, mock_font):
        """Test font fallback to system font when custom font not found."""
        mock_font.side_effect = FileNotFoundError
        
        with patch('pygame.font.SysFont') as mock_sysfont:
            config = RenderConfig(GameConfig())
            assert mock_sysfont.called
            assert 'small' in config.fonts
            assert 'medium' in config.fonts
            assert 'large' in config.fonts

    def test_sound_initialization(self, render_config):
        """Test sound effect initialization."""
        expected_sounds = {
            'move',
            'rotate',
            'clear',
            'hold',
            'drop',
            'game_over',
            'lock'
        }
        assert set(render_config.sounds.keys()) == expected_sounds

    @patch('pygame.mixer.Sound', side_effect=pygame.error)
    def test_sound_error_handling(self, mock_sound, game_config):
        """Test graceful handling of sound loading errors."""
        config = RenderConfig(game_config)
        for sound in config.sounds.values():
            assert sound is None

    def test_sound_file_paths(self, render_config):
        """Test that sound file paths are correctly constructed."""
        expected_paths = [
            'resources/sounds/move.wav',
            'resources/sounds/rotate.wav',
            'resources/sounds/clear.wav',
            'resources/sounds/hold.wav',
            'resources/sounds/drop.wav',
            'resources/sounds/game_over.wav',
            'resources/sounds/lock.wav'
        ]
        
        with patch('pygame.mixer.Sound') as mock_sound:
            config = RenderConfig(GameConfig())
            calls = mock_sound.call_args_list
            actual_paths = [call[0][0] for call in calls]
            assert sorted(actual_paths) == sorted(expected_paths)

    @patch('pygame.font.Font')
    def test_font_file_paths(self, mock_font):
        """Test that font file paths are correctly constructed."""
        config = RenderConfig(GameConfig())
        font_path = 'resources/fonts/PressStart2P-Regular.ttf'
        
        expected_calls = [
            ((font_path, config.font_small_size),),
            ((font_path, config.font_medium_size),),
            ((font_path, config.font_large_size),),
        ]
        
        actual_calls = [call[0] for call in mock_font.call_args_list]
        assert actual_calls == expected_calls

    def test_partial_font_loading(self, game_config):
        """Test handling of partial font loading failures."""
        with patch('pygame.font.Font') as mock_font:
            # Simulate failure for medium font only
            def mock_font_load(*args):
                if args[1] == 18:  # medium font size
                    raise FileNotFoundError
                return Mock(spec=pygame.font.Font)
            
            mock_font.side_effect = mock_font_load
            config = RenderConfig(game_config)
            
            assert config.fonts['small'] is not None
            assert config.fonts['large'] is not None

    def test_resource_path_construction(self, render_config):
        """Test resource path construction logic."""
        expected_sound_path = 'resources/sounds/move.wav'
        expected_font_path = 'resources/fonts/PressStart2P-Regular.ttf'
        
        with patch('pygame.mixer.Sound') as mock_sound, \
             patch('pygame.font.Font') as mock_font:
            
            config = RenderConfig(GameConfig())
            
            # Verify sound path
            mock_sound.assert_any_call(expected_sound_path)
            
            # Verify font path
            mock_font.assert_any_call(expected_font_path, config.font_small_size)

    def test_debug_mode_inheritance(self, game_config):
        """Test that debug mode is properly inherited from game config."""
        game_config.debug_mode = True
        config = RenderConfig(game_config)
        assert config.debug_mode is True
        
        game_config.debug_mode = False
        config = RenderConfig(game_config)
        assert config.debug_mode is False

    def test_particle_effects_inheritance(self, game_config):
        """Test that particle effects setting is properly inherited."""
        game_config.particle_effects = True
        config = RenderConfig(game_config)
        assert config.particle_effects is True
        
        game_config.particle_effects = False
        config = RenderConfig(game_config)
        assert config.particle_effects is False


if __name__ == "__main__":
    pytest.main(["-v", __file__])