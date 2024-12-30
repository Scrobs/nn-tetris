# File: conftest.py
"""
Pytest configuration and fixtures for Tetris game testing.
"""
import sys
from pathlib import Path
import pytest
import pygame
from unittest.mock import Mock, create_autospec, patch

# Adjust the system path to include the project root directory
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from config.game_config import GameConfig
from render.render_config import RenderConfig
from render.colors import Colors
from core.grid import Grid
from core.tetrimino import Tetrimino
from core.game_state import GameState

class MockGrid:
    """Mock implementation of a Grid for testing."""
    def __init__(self, columns, rows):
        self.columns = columns
        self.rows = rows
        self.grid = [[None for _ in range(columns)] for _ in range(rows)]
        self.cleared_lines = []
        self.flash_animation = False

    def __getitem__(self, idx):
        return self.grid[idx]

    def __len__(self):
        """Return the number of rows (height of the grid)."""
        return self.rows

    def can_move(self, piece, x, y):
        """Mock piece movement validation."""
        if not piece:
            return False
        return (0 <= x < self.columns and 0 <= y < self.rows)

    def lock_piece(self, piece):
        """Mock piece locking."""
        pass

    def detect_cleared_lines(self):
        """Mock line clearing detection."""
        return []

    def clear_cleared_lines(self):
        """Mock line clearing."""
        return 0

@pytest.fixture
def mock_pygame_surface():
    """Create a properly mocked Pygame surface."""
    surface = create_autospec(pygame.Surface)
    surface.get_rect.return_value = Mock(
        center=(150, 300),
        topleft=(0, 0),
        width=300,
        height=600
    )
    return surface

@pytest.fixture
def mock_pygame_font():
    """Create a properly mocked Pygame font."""
    font = create_autospec(pygame.font.Font)
    text_surface = create_autospec(pygame.Surface)
    text_surface.get_rect.return_value = Mock(
        width=100,
        height=20,
        center=(100, 20)
    )
    font.render.return_value = text_surface
    return font

@pytest.fixture(autouse=True)
def mock_pygame_setup():
    """Fixture to mock pygame initialization and basic functionality."""
    with patch('pygame.init'), \
         patch('pygame.display.init'), \
         patch('pygame.display.set_mode', return_value=create_autospec(pygame.Surface)), \
         patch('pygame.time.Clock', return_value=Mock()):
        yield

@pytest.fixture
def config():
    """Fixture providing game configuration."""
    config = GameConfig()
    config.rows = 20
    config.columns = 10
    config.grid_size = 30
    config.cell_padding = 1
    config.screen_width = 300
    config.screen_height = 600
    config.ui_width = 200
    config.particle_effects = False
    config.debug_mode = False
    config.initial_state = GameState.MENU
    return config

@pytest.fixture
def render_config(config):
    """Fixture providing render configuration with properly mocked fonts and sounds."""
    with patch('pygame.font.Font', return_value=create_autospec(pygame.font.Font)) as mock_font, \
         patch('pygame.mixer.Sound', return_value=create_autospec(pygame.mixer.Sound)) as mock_sound:
        render_config = RenderConfig(config)
        render_config.fonts = {
            'small': mock_font(),
            'medium': mock_font(),
            'large': mock_font()
        }
        render_config.sounds = {
            'move': mock_sound(),
            'rotate': mock_sound(),
            'clear': mock_sound(),
            'hold': mock_sound(),
            'drop': mock_sound(),
            'game_over': mock_sound(),
            'lock': mock_sound()
        }
        return render_config

@pytest.fixture
def grid(config):
    """Fixture providing a MockGrid instance."""
    return MockGrid(columns=config.columns, rows=config.rows)

@pytest.fixture
def game_grid(config):
    """Fixture providing a real Grid instance."""
    return Grid(config)

@pytest.fixture
def tetrimino(grid):
    """Fixture providing a default Tetrimino instance."""
    return Tetrimino(piece_type="I", grid_size=30, columns=grid.columns)

@pytest.fixture
def mock_tetrimino():
    """Fixture providing a mock Tetrimino."""
    piece = Mock(spec=Tetrimino)
    piece.shape = [[1, 1], [1, 1]]
    piece.piece_type = "O"
    piece.x = 0
    piece.y = 0
    piece.rotation = 0
    return piece

@pytest.fixture
def filled_grid(game_grid):
    """Fixture providing a grid with some filled cells."""
    for x in range(game_grid.columns):
        game_grid.grid[game_grid.rows - 1][x] = "I"
    for x in range(game_grid.columns - 1):
        game_grid.grid[game_grid.rows - 2][x] = "T"
    return game_grid

def create_mock_surface(width=300, height=600):
    """Create a properly mocked surface with specified dimensions."""
    surface = create_autospec(pygame.Surface)
    surface.get_rect.return_value = Mock(
        center=(width // 2, height // 2),
        topleft=(0, 0),
        width=width,
        height=height
    )
    return surface

def create_mock_font(text_width=100, text_height=20):
    """Create a properly mocked font with specified text dimensions."""
    font = create_autospec(pygame.font.Font)
    text_surface = create_autospec(pygame.Surface)
    text_surface.get_rect.return_value = Mock(
        width=text_width,
        height=text_height,
        center=(text_width // 2, text_height // 2)
    )
    font.render.return_value = text_surface
    return font
