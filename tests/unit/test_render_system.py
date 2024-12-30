#!/usr/bin/env python3
"""
Unit tests for the RenderSystem class.
"""

import pytest
import pygame
from unittest.mock import Mock, patch, call
from config.game_config import GameConfig
from render.render_config import RenderConfig
from render.render_system import RenderSystem, Layer, ScorePopup
from render.colors import Colors
from core.tetrimino import Tetrimino
from core.grid import Grid


@pytest.fixture
def config():
    """Fixture providing game configuration."""
    config = GameConfig()
    config.screen_width = 300
    config.screen_height = 600
    config.grid_size = 30
    config.particle_effects = False
    return config


@pytest.fixture
def render_config(config):
    """Fixture providing render configuration."""
    with patch('pygame.font.Font'):
        return RenderConfig(config)


@pytest.fixture
def mock_surface():
    """Fixture providing a mock pygame surface."""
    surface = Mock(spec=pygame.Surface)
    surface.get_rect.return_value = Mock(
        center=(150, 300),
        topleft=(0, 0)
    )
    return surface


@pytest.fixture(autouse=True)
def mock_pygame_setup():
    """Fixture setting up pygame mocks."""
    with patch('pygame.Surface') as mock_surface, \
         patch('pygame.draw') as mock_draw, \
         patch('pygame.transform') as mock_transform:
        mock_surface.return_value = Mock(spec=pygame.Surface)
        yield mock_surface, mock_draw, mock_transform


@pytest.fixture
def render_system(config, render_config):
    """Fixture providing a RenderSystem instance."""
    return RenderSystem(config, render_config)


class TestRenderSystem:
    """Test suite for the RenderSystem class."""

    def test_initialization(self, render_system):
        """Test render system initialization."""
        assert render_system.config is not None
        assert render_system.render_config is not None
        assert all(layer in render_system.surfaces for layer in Layer)
        assert render_system.frame_count == 0
        assert isinstance(render_system.particles, list)
        assert isinstance(render_system.score_popups, list)

    def test_clear_surfaces(self, render_system, mock_pygame_setup):
        """Test surface clearing functionality."""
        render_system.clear_surfaces()
        for surface in render_system.surfaces.values():
            assert surface.fill.called

    @patch('pygame.draw.rect')
    def test_draw_block(self, mock_draw_rect, render_system, mock_surface):
        """Test block drawing functionality."""
        color = (255, 0, 0)
        render_system.draw_block(mock_surface, 1, 1, color)
        assert mock_draw_rect.called

    @patch('pygame.draw.rect')
    def test_draw_block_with_alpha(self, mock_draw_rect, render_system, mock_surface):
        """Test block drawing with alpha."""
        color = (255, 0, 0)
        render_system.draw_block(mock_surface, 1, 1, color, alpha=128)
        assert mock_surface.blit.called

    def test_draw_piece(self, render_system):
        """Test piece drawing."""
        piece = Mock(spec=Tetrimino)
        piece.shape = [[1, 1], [1, 1]]
        piece.piece_type = 'O'
        piece.x = 0
        piece.y = 0

        with patch.object(render_system, 'draw_block') as mock_draw_block:
            render_system.draw_piece(piece)
            assert mock_draw_block.called

    def test_draw_ghost_piece(self, render_system):
        """Test ghost piece drawing."""
        piece = Mock(spec=Tetrimino)
        piece.shape = [[1, 1], [1, 1]]
        piece.piece_type = 'O'
        piece.x = 0
        piece.y = 0

        with patch.object(render_system, 'draw_block') as mock_draw_block:
            render_system.draw_piece(piece, ghost=True)
            for call_args in mock_draw_block.call_args_list:
                assert call_args[1]['alpha'] == 64

    @patch('time.time')
    def test_score_popup_lifecycle(self, mock_time, render_system):
        """Test score popup creation and expiration."""
        mock_time.return_value = 100.0
        position = (100, 100)
        
        render_system.add_score_popup(100, position)
        assert len(render_system.score_popups) == 1
        
        # Simulate time passing
        mock_time.return_value = 101.5
        render_system.draw_score_popups()
        assert len(render_system.score_popups) == 0

    def test_particle_effects(self, render_system):
        """Test particle effect system."""
        position = (5, 5)
        initial_particle_count = len(render_system.particles)
        
        render_system.trigger_particle_effect(position)
        if render_system.particle_effects_enabled:
            assert len(render_system.particles) > initial_particle_count
        else:
            assert len(render_system.particles) == initial_particle_count

    def test_particle_update(self, render_system):
        """Test particle physics updates."""
        render_system.particles = [{
            'position': [100, 100],
            'velocity': [1, 1],
            'lifetime': 0.5
        }]
        
        render_system.update_particles(0.1)
        assert render_system.particles[0]['position'] != [100, 100]
        assert render_system.particles[0]['lifetime'] < 0.5

    @patch('pygame.draw.circle')
    def test_particle_drawing(self, mock_draw_circle, render_system):
        """Test particle rendering."""
        render_system.particles = [{
            'position': [100, 100],
            'velocity': [1, 1],
            'lifetime': 0.5
        }]
        
        render_system.draw_particles()
        assert mock_draw_circle.called

    def test_ui_drawing(self, render_system):
        """Test UI element drawing."""
        score = 1000
        high_score = 2000
        level = 1
        lines = 10
        next_pieces = []
        held_piece = None
        combo = 0

        with patch.object(render_system.ui_manager, 'draw_game_ui') as mock_draw_ui:
            render_system.draw_ui(
                score, high_score, level, lines, 
                next_pieces, held_piece, combo
            )
            assert mock_draw_ui.called

    def test_menu_drawing(self, render_system):
        """Test menu rendering."""
        menu_options = ["Start", "Options", "Quit"]
        selected_index = 0

        with patch.object(render_system.ui_manager, 'draw_menu') as mock_draw_menu:
            render_system.draw_menu(menu_options, selected_index)
            assert mock_draw_menu.called

    def test_pause_screen(self, render_system):
        """Test pause screen rendering."""
        with patch.object(render_system.render_config, 'fonts') as mock_fonts:
            mock_font = Mock()
            mock_font.render.return_value = Mock(spec=pygame.Surface)
            mock_fonts.__getitem__.return_value = mock_font
            
            render_system.draw_pause_screen()
            assert mock_font.render.called

    def test_game_over_screen(self, render_system):
        """Test game over screen rendering."""
        score = 1000
        with patch.object(render_system.render_config, 'fonts') as mock_fonts:
            mock_font = Mock()
            mock_font.render.return_value = Mock(spec=pygame.Surface)
            mock_fonts.__getitem__.return_value = mock_font
            
            render_system.draw_game_over(score)
            assert mock_font.render.called

    def test_grid_drawing(self, render_system, config):
        """Test grid rendering."""
        grid = Mock(spec=Grid)
        grid.grid = [[None for _ in range(config.columns)] 
                    for _ in range(config.rows)]
        
        with patch('pygame.draw.line') as mock_draw_line:
            render_system.draw_grid(grid)
            assert mock_draw_line.called

    def test_score_popup_cleanup(self, render_system):
        """Test cleanup of expired score popups."""
        render_system.score_popups = [
            ScorePopup("100", (100, 100), time.time() - 2.0),
            ScorePopup("200", (200, 200), time.time())
        ]
        
        render_system.draw_score_popups()
        assert len(render_system.score_popups) == 1

    def test_compose_frame(self, render_system, mock_surface):
        """Test frame composition."""
        screen = mock_surface
        frame_count = 0
        
        with patch.object(render_system, 'surfaces') as mock_surfaces:
            mock_surfaces.__getitem__.return_value = Mock(spec=pygame.Surface)
            render_system.compose_frame(screen, frame_count)
            assert screen.blit.called

    @pytest.mark.parametrize("layer", Layer)
    def test_layer_initialization(self, render_system, layer):
        """Test initialization of each render layer."""
        assert layer in render_system.surfaces
        assert isinstance(render_system.surfaces[layer], Mock)


if __name__ == "__main__":
    pytest.main(["-v", __file__])