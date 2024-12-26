#!/usr/bin/env python3
"""
tetris_game.py: Enhanced Tetris game environment with proper error handling,
type hints, improved game mechanics, and modern Tetris features.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, NewType, SupportsInt, Sequence
from threading import Lock
import random
import logging
import numpy as np
import pygame
import time
from enum import Enum, auto

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Type definitions
Grid = NewType('Grid', List[List[int]])
Position = Tuple[int, int]
Shape = List[List[int]]

class GameState(Enum):
    """Game state enumeration."""
    PLAYING = auto()
    PAUSED = auto()
    GAME_OVER = auto()
    ANIMATING = auto()  # For animations

@dataclass
class GameConfig:
    """Game configuration parameters with validation."""
    screen_width: int = 300
    screen_height: int = 600
    grid_size: int = 30
    fps: int = 60
    das_delay: int = 16  # Delayed Auto Shift initial delay (frames)
    das_repeat: int = 6  # Delayed Auto Shift repeat rate (frames)
    soft_drop_speed: int = 2  # Soft drop speed multiplier
    input_buffer_frames: int = 10  # Number of frames to buffer inputs
    max_score: int = 999_999_999  # Maximum possible score
    vsync: bool = True  # Enable vertical sync
    preview_pieces: int = 3  # Number of preview pieces to show
    grid_line_width: int = 1  # Width of grid lines
    cell_padding: int = 1  # Padding between cells
    preview_margin: int = 30  # Margin for preview area
    ui_width: int = 200  # Width of UI area

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.screen_width <= 0 or self.screen_height <= 0:
            raise ValueError("Screen dimensions must be positive")
        if self.grid_size <= 0:
            raise ValueError("Grid size must be positive")
        if self.fps <= 0:
            raise ValueError("FPS must be positive")
        if self.preview_pieces <= 0:
            raise ValueError("Preview pieces must be positive")

    @property
    def columns(self) -> int:
        """Calculate number of columns."""
        return self.screen_width // self.grid_size

    @property
    def rows(self) -> int:
        """Calculate number of rows."""
        return self.screen_height // self.grid_size

    @property
    def total_width(self) -> int:
        """Calculate total screen width including UI."""
        return self.screen_width + self.ui_width

class RenderConfig:
    """Configuration for rendering-specific parameters."""
    def __init__(self, game_config: GameConfig):
        self.cell_size = game_config.grid_size - (2 * game_config.cell_padding)
        self.grid_line_width = game_config.grid_line_width
        self.cell_padding = game_config.cell_padding

        # Surface configurations
        self.main_surface_flags = pygame.SRCALPHA
        if game_config.vsync:
            self.main_surface_flags |= pygame.DOUBLEBUF | pygame.HWSURFACE

        # Color configurations with alpha support
        self.colors = Colors()

        # Font configurations
        self.fonts = self._initialize_fonts()

    def _initialize_fonts(self) -> Dict[str, Optional[pygame.font.Font]]:
        """Initialize fonts with error handling."""
        fonts = {}
        try:
            fonts['small'] = pygame.font.Font(None, 36)
            fonts['medium'] = pygame.font.Font(None, 48)
            fonts['large'] = pygame.font.Font(None, 72)
        except pygame.error as e:
            logger.error(f"Failed to initialize fonts: {e}")
            # Fallback to system font if custom font fails
            try:
                default_font = pygame.font.get_default_font()
                fonts = {
                    'small': pygame.font.Font(default_font, 36),
                    'medium': pygame.font.Font(default_font, 48),
                    'large': pygame.font.Font(default_font, 72)
                }
            except pygame.error as e:
                logger.error(f"Failed to initialize default fonts: {e}")
                return None
        return fonts

class Colors:
    """Color definitions with alpha support and validation."""
    def __init__(self):
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GRAY = (128, 128, 128)
        self.RED = (255, 0, 0)
        self.DEFAULT_COLOR = (200, 200, 200)
        self.GHOST_ALPHA = 128

        self.PIECE_COLORS = {
            'I': (0, 240, 240),   # Cyan
            'O': (240, 240, 0),   # Yellow
            'T': (160, 0, 240),   # Purple
            'S': (0, 240, 0),     # Green
            'Z': (240, 0, 0),     # Red
            'J': (0, 0, 240),     # Blue
            'L': (240, 160, 0),   # Orange
        }

    def get_piece_color(self, piece_type: str, alpha: int = 255) -> Tuple[int, int, int, int]:
        """Get color for piece type with alpha support."""
        base_color = self.PIECE_COLORS.get(piece_type, self.DEFAULT_COLOR)
        return (*base_color, alpha)

    def validate_color(self, color: Tuple[int, ...]) -> bool:
        """Validate color values."""
        if not isinstance(color, tuple):
            return False
        if len(color) not in (3, 4):  # RGB or RGBA
            return False
        return all(isinstance(c, int) and 0 <= c <= 255 for c in color)

class Score:
    """Score handling with overflow protection and validation."""
    def __init__(self, max_score: int):
        self._score: int = 0
        self._max_score = max_score
        self._validate_max_score(max_score)

    def _validate_max_score(self, max_score: int) -> None:
        """Validate maximum score value."""
        if not isinstance(max_score, int) or max_score <= 0:
            raise ValueError("Maximum score must be a positive integer")

    def add(self, points: SupportsInt) -> None:
        """Add points with overflow protection and validation."""
        try:
            points_int = int(points)
            if points_int < 0:
                raise ValueError("Points cannot be negative")
            new_score = self._score + points_int
            self._score = min(new_score, self._max_score)
        except (OverflowError, ValueError) as e:
            logger.warning(f"Score addition failed: {e}")
            self._score = self._max_score

    @property
    def value(self) -> int:
        """Get current score."""
        return self._score

class InputBuffer:
    """Handle input buffering for moves and rotations with validation."""
    def __init__(self, buffer_frames: int = 10):
        if buffer_frames <= 0:
            raise ValueError("Buffer frames must be positive")
        self.buffer_frames = buffer_frames
        self.buffer: List[Tuple[int, int]] = []  # (action, frames_left)
        self.last_action_time: Dict[int, float] = {}  # Track last time each action was performed

    def add_input(self, action: int) -> bool:
        """Add input to buffer with rate limiting."""
        current_time = time.time()

        # Check rate limiting
        if action in self.last_action_time:
            if current_time - self.last_action_time[action] < 0.016:  # ~60fps
                return False

        self.last_action_time[action] = current_time
        self.buffer.append((action, self.buffer_frames))
        return True

    def update(self) -> Optional[int]:
        """Update buffer and return oldest valid input."""
        if not self.buffer:
            return None

        # Decrease frames_left for all inputs
        self.buffer = [(action, frames - 1) for action, frames in self.buffer]

        # Remove expired inputs
        self.buffer = [(action, frames) for action, frames in self.buffer if frames > 0]

        # Return oldest valid input if any
        return self.buffer.pop(0)[0] if self.buffer else None

    def clear(self) -> None:
        """Clear input buffer."""
        self.buffer.clear()
        self.last_action_time.clear()

class GameTimer:
    """Handle game timing with delta time and frame synchronization."""
    def __init__(self, target_fps: int):
        if target_fps <= 0:
            raise ValueError("Target FPS must be positive")

        self.target_fps = target_fps
        self.target_frame_time = 1.0 / target_fps
        self.last_time = time.perf_counter()
        self.accumulated_time = 0.0
        self._frame_count = 0
        self._fps_update_time = self.last_time
        self._current_fps = 0.0

    def tick(self) -> int:
        """Calculate delta time and return the number of physics updates needed."""
        current_time = time.perf_counter()
        delta_time = current_time - self.last_time
        self.last_time = current_time

        # Prevent spiral of death
        if delta_time > 0.25:  # More than 250ms
            delta_time = 0.25

        self.accumulated_time += delta_time
        self._frame_count += 1

        # Update FPS calculation every second
        if current_time - self._fps_update_time >= 1.0:
            self._current_fps = self._frame_count / (current_time - self._fps_update_time)
            self._frame_count = 0
            self._fps_update_time = current_time

        # Calculate required physics updates
        updates = 0
        while self.accumulated_time >= self.target_frame_time:
            updates += 1
            self.accumulated_time -= self.target_frame_time

        return updates

    @property
    def fps(self) -> float:
        """Get current FPS."""
        return self._current_fps

    def reset(self) -> None:
        """Reset timer state."""
        self.last_time = time.perf_counter()
        self.accumulated_time = 0.0
        self._frame_count = 0
        self._fps_update_time = self.last_time

class TetriminoData:
    """Tetrimino shapes and spawn positions with validation."""
    SHAPES: Dict[str, List[List[List[int]]]] = {
        'I': [[[0, 0, 0, 0],
              [1, 1, 1, 1],
              [0, 0, 0, 0],
              [0, 0, 0, 0]]],
        'O': [[[1, 1],
              [1, 1]]],
        'T': [[[0, 1, 0],
              [1, 1, 1],
              [0, 0, 0]]],
        'S': [[[0, 1, 1],
              [1, 1, 0],
              [0, 0, 0]]],
        'Z': [[[1, 1, 0],
              [0, 1, 1],
              [0, 0, 0]]],
        'J': [[[1, 0, 0],
              [1, 1, 1],
              [0, 0, 0]]],
        'L': [[[0, 0, 1],
              [1, 1, 1],
              [0, 0, 0]]]
    }

    # Complete wall kick data including 180-degree rotations
    WALL_KICK_DATA = {
        'JLSTZ': [
            # 0->R
            [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
            # R->0
            [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
            # R->2
            [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
            # 2->R
            [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
            # 2->L
            [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
            # L->2
            [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
            # L->0
            [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
            # 0->L
            [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
            # 0->2 (180-degree)
            [(0, 0), (0, 1), (1, 1), (-1, 1), (1, 0), (-1, 0)],
            # 2->0 (180-degree)
            [(0, 0), (0, -1), (-1, -1), (1, -1), (-1, 0), (1, 0)]
        ],
        'I': [
            # Complete set for I piece including 180-degree rotations
            [(0, 0), (-2, 0), (1, 0), (-2, 1), (1, -2)],
            [(0, 0), (2, 0), (-1, 0), (2, -1), (-1, 2)],
            [(0, 0), (-1, 0), (2, 0), (-1, -2), (2, 1)],
            [(0, 0), (1, 0), (-2, 0), (1, 2), (-2, -1)],
            [(0, 0), (-1, 0), (2, 0), (-1, -2), (2, 1)],
            [(0, 0), (1, 0), (-2, 0), (1, 2), (-2, -1)],
            [(0, 0), (2, 0), (-1, 0), (2, -1), (-1, 2)],
            [(0, 0), (-2, 0), (1, 0), (-2, 1), (1, -2)],
            [(0, 0), (0, 1), (0, -1), (1, 1), (-1, -1)],
            [(0, 0), (0, -1), (0, 1), (-1, -1), (1, 1)]
        ]
    }

    @classmethod
    def validate_shape(cls, shape: List[List[int]]) -> bool:
        """Validate tetrimino shape."""
        if not shape or not shape[0]:
            return False
        width = len(shape[0])
        return all(len(row) == width for row in shape)

    @classmethod
    def get_initial_shape(cls, piece_type: str) -> Shape:
        """Get initial shape for piece type with validation."""
        if piece_type not in cls.SHAPES:
            raise ValueError(f"Invalid piece type: {piece_type}")
        return cls.SHAPES[piece_type][0]

class Tetrimino:
    """Represents a Tetris piece with enhanced rotation and movement mechanics."""

    def __init__(self, piece_type: str, grid_size: int, columns: int):
        """Initialize tetrimino with validation."""
        self.piece_type = piece_type
        self.shape = TetriminoData.get_initial_shape(piece_type)
        self.can_rotate = piece_type != 'O'  # O piece cannot rotate
        self.rotation_state = 0
        self.grid_size = grid_size

        # Calculate spawn position
        shape_width = len(self.shape[0])
        self.x = (columns - shape_width) // 2
        self.y = 0 if piece_type != 'I' else -1

    def get_rotated_shape(self, rotation_state: int) -> Shape:
        """Get piece shape for a specific rotation state with validation."""
        if not self.can_rotate:
            return self.shape

        shape = TetriminoData.get_initial_shape(self.piece_type)
        # Normalize rotation state
        normalized_rotation = ((rotation_state % 4) + 4) % 4

        # Rotate shape
        for _ in range(normalized_rotation):
            shape = list(zip(*shape[::-1]))  # Rotate 90 degrees clockwise

        # Convert back to list of lists and validate
        result = [list(row) for row in shape]
        if not TetriminoData.validate_shape(result):
            raise ValueError("Invalid shape after rotation")

        return result

    def try_rotation(self, grid: Grid, clockwise: bool = True) -> bool:
        """Attempt to rotate the piece with wall kicks and validation."""
        if not self.can_rotate:
            return False

        old_rotation = self.rotation_state
        new_rotation = (old_rotation + (1 if clockwise else -1)) % 4

        # Get wall kick data
        kick_data = TetriminoData.WALL_KICK_DATA['I' if self.piece_type == 'I' else 'JLSTZ']
        kicks = kick_data[old_rotation * 2 + (1 if clockwise else 0)]

        # Try each wall kick
        old_shape = self.shape
        try:
            self.shape = self.get_rotated_shape(new_rotation)
        except ValueError:
            return False

        for dx, dy in kicks:
            if self.is_valid_position(grid, self.x + dx, self.y + dy):
                self.x += dx
                self.y += dy
                self.rotation_state = new_rotation
                return True

        # If no wall kick worked, revert rotation
        self.shape = old_shape
        return False

    def is_valid_position(self, grid: Grid, x: int, y: int) -> bool:
        """Check if piece can be placed at given position with complete boundary checking."""
        shape_height = len(self.shape)
        shape_width = len(self.shape[0])

        # Check piece bounds
        if x < 0 or x + shape_width > len(grid[0]):
            return False

        # Check if piece extends below grid
        if y + shape_height > len(grid):
            return False

        # Check collision with existing blocks and grid boundaries
        for row_idx, row in enumerate(self.shape):
            for col_idx, cell in enumerate(row):
                if cell:
                    # Allow blocks above grid (y < 0)
                    if y + row_idx >= 0:
                        if y + row_idx >= len(grid):
                            return False
                        if grid[y + row_idx][x + col_idx]:
                            return False

        return True

    def get_ghost_position(self, grid: Grid) -> int:
        """Get the lowest valid position for ghost piece with validation."""
        if not grid or not grid[0]:
            raise ValueError("Invalid grid")

        ghost_y = self.y
        while self.is_valid_position(grid, self.x, ghost_y + 1):
            ghost_y += 1
        return ghost_y

class RenderSystem:
    """Handles all rendering operations with proper layering and buffering."""
    def __init__(self, config: GameConfig, render_config: RenderConfig):
        """Initialize the rendering system with proper error handling."""
        self.config = config
        self.render_config = render_config
        self.piece_cache = {}
        self._initialize_surfaces()

    def _initialize_surfaces(self) -> None:
        """Initialize all rendering surfaces with proper layering."""
        try:
            # Main game surface (playfield)
            self.game_surface = pygame.Surface(
                (self.config.screen_width, self.config.screen_height)
            )

            # Separate UI surface
            self.ui_surface = pygame.Surface(
                (self.config.ui_width, self.config.screen_height)
            )

            # Ghost piece surface with alpha support
            self.ghost_surface = pygame.Surface(
                (self.config.screen_width, self.config.screen_height),
                pygame.SRCALPHA
            )

            # Debug surface at the top
            self.debug_surface = pygame.Surface(
                (self.config.total_width, 30),
                pygame.SRCALPHA
            )

            # Create the static grid background
            self._initialize_static_grid()

            # Clear all surfaces initially
            self.clear_surfaces()

        except pygame.error as e:
            logger.error(f"Error initializing surfaces: {e}")
            raise

    def _initialize_static_grid(self) -> None:
        """Create the static grid background with proper grid lines."""
        try:
            self.grid_surface = pygame.Surface(
                (self.config.screen_width, self.config.screen_height)
            )
            self.grid_surface.fill((0, 0, 0))  # Black background

            # Draw subtle grid lines
            for x in range(0, self.config.screen_width + 1, self.config.grid_size):
                pygame.draw.line(
                    self.grid_surface,
                    (40, 40, 40),  # Dark gray lines
                    (x, 0),
                    (x, self.config.screen_height),
                    self.config.grid_line_width
                )
            for y in range(0, self.config.screen_height + 1, self.config.grid_size):
                pygame.draw.line(
                    self.grid_surface,
                    (40, 40, 40),
                    (0, y),
                    (self.config.screen_width, y),
                    self.config.grid_line_width
                )
        except pygame.error as e:
            logger.error(f"Error initializing grid surface: {e}")
            raise

    def clear_surfaces(self) -> None:
        """Clear all rendering surfaces to their base state."""
        try:
            if self.game_surface:
                self.game_surface.fill((0, 0, 0))
            if self.ui_surface:
                self.ui_surface.fill((0, 0, 0))
            if self.ghost_surface:
                self.ghost_surface.fill((0, 0, 0, 0))
            if self.debug_surface:
                self.debug_surface.fill((0, 0, 0, 0))
        except pygame.error as e:
            logger.error(f"Error clearing surfaces: {e}")

    def draw_grid(self, grid: Grid) -> None:
        """Draw the game grid with filled cells."""
        try:
            # Draw pre-rendered grid background
            if self.game_surface and self.grid_surface:
                self.game_surface.blit(self.grid_surface, (0, 0))

                # Draw filled cells
                for y, row in enumerate(grid):
                    for x, cell in enumerate(row):
                        if cell:
                            rect = pygame.Rect(
                                x * self.config.grid_size + self.config.cell_padding,
                                y * self.config.grid_size + self.config.cell_padding,
                                self.config.grid_size - (2 * self.config.cell_padding),
                                self.config.grid_size - (2 * self.config.cell_padding)
                            )
                            pygame.draw.rect(self.game_surface, (200, 200, 200), rect)
        except (pygame.error, TypeError, IndexError) as e:
            logger.error(f"Error drawing grid: {e}")

    def draw_piece(self, piece: Tetrimino, ghost: bool = False,
                  override_y: Optional[int] = None,
                  override_x: Optional[int] = None) -> None:
        """Draw a tetrimino with proper shading."""
        try:
            if not piece:
                return

            target_surface = self.ghost_surface if ghost else self.game_surface
            if not target_surface:
                return

            y_pos = override_y if override_y is not None else piece.y
            x_pos = override_x if override_x is not None else piece.x

            color = self.render_config.colors.get_piece_color(piece.piece_type)
            alpha = 64 if ghost else 255

            for y, row in enumerate(piece.shape):
                for x, cell in enumerate(row):
                    if cell:
                        rect = pygame.Rect(
                            (x_pos + x) * self.config.grid_size + self.config.cell_padding,
                            (y_pos + y) * self.config.grid_size + self.config.cell_padding,
                            self.config.grid_size - (2 * self.config.cell_padding),
                            self.config.grid_size - (2 * self.config.cell_padding)
                        )

                        if ghost:
                            s = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
                            pygame.draw.rect(s, (*color[:3], alpha), s.get_rect())
                            target_surface.blit(s, rect)
                        else:
                            # Main block
                            pygame.draw.rect(target_surface, color[:3], rect)
                            # Add shading effects
                            self._draw_piece_shading(target_surface, rect, color)

        except (pygame.error, AttributeError) as e:
            logger.error(f"Error drawing piece: {e}")

    def _draw_piece_shading(self, surface: pygame.Surface, rect: pygame.Rect,
                          color: Tuple[int, ...]) -> None:
        """Add shading effects to a piece."""
        try:
            # Top highlight
            pygame.draw.line(
                surface,
                tuple(min(c + 50, 255) for c in color[:3]),
                rect.topleft,
                rect.topright
            )
            # Left highlight
            pygame.draw.line(
                surface,
                tuple(min(c + 30, 255) for c in color[:3]),
                rect.topleft,
                rect.bottomleft
            )
            # Bottom shadow
            pygame.draw.line(
                surface,
                tuple(max(c - 50, 0) for c in color[:3]),
                rect.bottomleft,
                rect.bottomright
            )
            # Right shadow
            pygame.draw.line(
                surface,
                tuple(max(c - 30, 0) for c in color[:3]),
                rect.topright,
                rect.bottomright
            )
        except pygame.error as e:
            logger.error(f"Error drawing piece shading: {e}")

    def draw_next_pieces(self, next_pieces: List[Tetrimino]) -> None:
        """Draw preview of next pieces."""
        try:
            if not self.render_config.fonts or not self.ui_surface:
                return

            # Draw "Next:" label
            label = self.render_config.fonts['small'].render(
                "Next", True, (200, 200, 200))
            self.ui_surface.blit(label, (20, 20))

            # Draw preview pieces
            preview_y = 60
            preview_size = 4 * self.config.grid_size
            for piece in next_pieces[:self.config.preview_pieces]:
                preview_surface = pygame.Surface((preview_size, preview_size))
                preview_surface.fill((0, 0, 0))

                # Center the piece
                x_offset = (preview_size -
                          len(piece.shape[0]) * self.config.grid_size) // 2
                y_offset = (preview_size -
                          len(piece.shape) * self.config.grid_size) // 2

                self.draw_piece(
                    piece,
                    override_x=x_offset // self.config.grid_size,
                    override_y=y_offset // self.config.grid_size
                )

                self.ui_surface.blit(preview_surface, (10, preview_y))
                preview_y += 80

        except (pygame.error, AttributeError) as e:
            logger.error(f"Error drawing next pieces: {e}")

    def draw_held_piece(self, held_piece: Optional[str],
                       held_tetrimino: Optional[Tetrimino]) -> None:
        """Draw the held piece."""
        try:
            if not held_piece or not held_tetrimino or not self.render_config.fonts:
                return

            # Draw "Hold:" label
            label = self.render_config.fonts['small'].render(
                "Hold", True, (200, 200, 200))
            self.ui_surface.blit(label, (20, 250))

            # Draw held piece
            held_surface = pygame.Surface((
                4 * self.config.grid_size,
                4 * self.config.grid_size
            ))
            held_surface.fill((0, 0, 0))

            # Center the piece
            x_offset = (held_surface.get_width() -
                       len(held_tetrimino.shape[0]) * self.config.grid_size) // 2

            self.draw_piece(
                held_tetrimino,
                override_x=x_offset // self.config.grid_size,
                override_y=290 // self.config.grid_size
            )

            self.ui_surface.blit(held_surface, (10, 290))

        except (pygame.error, AttributeError) as e:
            logger.error(f"Error drawing held piece: {e}")

    def draw_info(self, score: int, level: int, lines: int, combo: int) -> None:
        """Draw game information."""
        try:
            if not self.render_config.fonts or not self.ui_surface:
                return

            y_pos = 400
            spacing = 40

            # Score
            score_text = self.render_config.fonts['small'].render(
                f"Score: {score:,}", True, (200, 200, 200))
            self.ui_surface.blit(score_text, (20, y_pos))

            # Level
            level_text = self.render_config.fonts['small'].render(
                f"Level: {level}", True, (200, 200, 200))
            self.ui_surface.blit(level_text, (20, y_pos + spacing))

            # Lines
            lines_text = self.render_config.fonts['small'].render(
                f"Lines: {lines}", True, (200, 200, 200))
            self.ui_surface.blit(lines_text, (20, y_pos + spacing * 2))

            # Combo
            if combo > 0:
                combo_text = self.render_config.fonts['small'].render(
                    f"Combo: {combo}", True, (255, 200, 0))
                self.ui_surface.blit(combo_text, (20, y_pos + spacing * 3))

        except (pygame.error, AttributeError) as e:
            logger.error(f"Error drawing info: {e}")

    def draw_debug_info(self, frame_count: int, fps: float) -> None:
        """Draw debug information."""
        try:
            if not self.render_config.fonts or not self.debug_surface:
                return

            debug_text = self.render_config.fonts['small'].render(
                f"Frame: {frame_count} | FPS: {fps:.1f}",
                True,
                (150, 150, 150)
            )

            # Position in top-right corner
            text_rect = debug_text.get_rect(
                topright=(self.config.total_width - 10, 5)
            )
            self.debug_surface.blit(debug_text, text_rect)

        except (pygame.error, AttributeError) as e:
            logger.error(f"Error drawing debug info: {e}")

    def draw_pause_screen(self) -> None:
        """Draw pause overlay."""
        try:
            if not self.render_config.fonts or not self.game_surface:
                return

            overlay = pygame.Surface((
                self.config.total_width,
                self.config.screen_height
            ), pygame.SRCALPHA)

            overlay.fill((0, 0, 0, 128))

            pause_text = self.render_config.fonts['large'].render(
                "PAUSED", True, (200, 200, 200))
            text_rect = pause_text.get_rect(center=(
                self.config.total_width // 2,
                self.config.screen_height // 2
            ))

            overlay.blit(pause_text, text_rect)
            self.game_surface.blit(overlay, (0, 0))

        except (pygame.error, AttributeError) as e:
            logger.error(f"Error drawing pause screen: {e}")

    def draw_game_over(self, final_score: int) -> None:
        """Draw game over overlay."""
        try:
            if not self.render_config.fonts or not self.game_surface:
                return

            overlay = pygame.Surface((
                self.config.total_width,
                self.config.screen_height
            ), pygame.SRCALPHA)

            overlay.fill((0, 0, 0, 192))

            # Game Over text
            game_over = self.render_config.fonts['large'].render(
                "GAME OVER", True, (255, 50, 50))
            game_over_rect = game_over.get_rect(center=(
                self.config.total_width // 2,
                self.config.screen_height // 2 - 60
            ))

            # Final score
            score_text = self.render_config.fonts['medium'].render(
                f"Final Score: {final_score:,}", True, (200, 200, 200))
            score_rect = score_text.get_rect(center=(
                self.config.total_width // 2,
                self.config.screen_height // 2
            ))

            # Instructions
            restart_text = self.render_config.fonts['small'].render(
                "Press R to Restart", True, (200, 200, 200))
            restart_rect = restart_text.get_rect(center=(
                self.config.total_width // 2,
                self.config.screen_height // 2 + 50
            ))

            quit_text = self.render_config.fonts['small'].render(
                "Press Q to Quit", True, (200, 200, 200))
            quit_rect = quit_text.get_rect(center=(
                self.config.total_width // 2,
                self.config.screen_height // 2 + 90
            ))

            # Draw all elements
            overlay.blit(game_over, game_over_rect)
            overlay.blit(score_text, score_rect)
            overlay.blit(restart_text, restart_rect)
            overlay.blit(quit_text, quit_rect)

            self.game_surface.blit(overlay, (0, 0))

        except (pygame.error, AttributeError) as e:
            logger.error(f"Error drawing game over screen: {e}")

    def compose_frame(self, screen: pygame.Surface) -> None:
        """Compose final frame with proper positioning."""
        try:
            if not screen:
                return

            # Fill main screen with background color
            screen.fill((0, 0, 0))

            # Draw main game area
            if self.game_surface:
                screen.blit(self.game_surface, (0, 0))

            # Draw ghost piece
            if self.ghost_surface:
                screen.blit(self.ghost_surface, (0, 0))

            # Draw UI area (to the right of the game area)
            if self.ui_surface:
                screen.blit(self.ui_surface, (self.config.screen_width, 0))

            # Draw debug overlay
            if self.debug_surface:
                screen.blit(self.debug_surface, (0, 0))

            # Update display
            pygame.display.flip()

        except pygame.error as e:
            logger.error(f"Error composing frame: {e}")

    def cleanup(self) -> None:
        """Clean up rendering resources."""
        try:
            # Clear the piece cache
            self.piece_cache.clear()

            # Clear all surfaces
            self.clear_surfaces()

            # Clean up surface references
            self.game_surface = None
            self.ui_surface = None
            self.ghost_surface = None
            self.debug_surface = None
            self.grid_surface = None

        except Exception as e:
            logger.error(f"Error during renderer cleanup: {e}")
        finally:
            # Ensure all references are cleared even if an error occurs
            self.piece_cache = {}


class TetrisGame:
    """Main Tetris game class with improved state management and error handling."""

    def __init__(self, config: Optional[GameConfig] = None):
        self.config = config or GameConfig()
        logger.info(f"Initializing game with dimensions: {self.config.total_width}x{self.config.screen_height}")

        # Initialize subsystems
        self._initialize_subsystems()

        # Game state
        self._state_lock = Lock()
        self.state = GameState.PLAYING
        self.score_handler = Score(self.config.max_score)
        self.lines_cleared = 0
        self.level = 1
        self.combo_counter = -1
        self.frame_counter = 0

        # Initialize game grid and pieces
        self.reset()

    def _apply_gravity(self) -> bool:
        """Apply gravity to current piece. Returns False if piece can't move down."""
        if not self.current_piece:
            return False
        if self.current_piece.is_valid_position(self.grid,
            self.current_piece.x, self.current_piece.y + 1):
            self.current_piece.y += 1
            return True
        return False

    def _get_fall_speed(self) -> int:
        """Calculate piece fall speed based on level."""
        return max(1, int(60 - (self.level * 5)))  # Frames per drop

    def _handle_keydown(self, key: int) -> None:
        """Handle keyboard input."""
        if self.state == GameState.PLAYING:
            if key == pygame.K_LEFT:
                self._try_move(-1)
            elif key == pygame.K_RIGHT:
                self._try_move(1)
            elif key == pygame.K_DOWN:
                self._soft_drop()
            elif key == pygame.K_UP:
                self._try_rotate()
            elif key == pygame.K_SPACE:
                self._hard_drop()
            elif key == pygame.K_c:
                self._try_hold()
        elif key == pygame.K_p:
            self.toggle_pause()
        elif self.state == GameState.GAME_OVER:
            if key == pygame.K_r:
                self.reset()
            elif key == pygame.K_q:
                pygame.event.post(pygame.event.Event(pygame.QUIT))

    def _try_move(self, dx: int) -> bool:
        """Try to move piece horizontally."""
        if not self.current_piece:
            return False
        if self.current_piece.is_valid_position(self.grid,
            self.current_piece.x + dx, self.current_piece.y):
            self.current_piece.x += dx
            return True
        return False

    def _try_rotate(self) -> bool:
        """Try to rotate current piece."""
        if not self.current_piece:
            return False
        return self.current_piece.try_rotation(self.grid)

    def _soft_drop(self) -> None:
        """Perform soft drop."""
        if not self.current_piece:
            return
        if self.current_piece.is_valid_position(self.grid,
            self.current_piece.x, self.current_piece.y + 1):
            self.current_piece.y += 1
            self.score_handler.add(1)

    def _hard_drop(self) -> None:
        """Perform hard drop."""
        if not self.current_piece:
            return
        drop_distance = 0
        while self.current_piece.is_valid_position(self.grid,
            self.current_piece.x, self.current_piece.y + 1):
            self.current_piece.y += 1
            drop_distance += 1
        self.score_handler.add(drop_distance * 2)
        self._lock_piece()
        cleared_lines = self._clear_lines()
        reward = self._calculate_reward(cleared_lines)
        self.score_handler.add(reward)
        self._spawn_new_piece()

    def _try_hold(self) -> None:
        """Try to hold current piece."""
        if not self.can_hold:
            return

        current_type = self.current_piece.piece_type
        if self.held_piece is None:
            self.held_piece = current_type
            self.current_piece = self._get_next_piece()
        else:
            self.current_piece = Tetrimino(self.held_piece, self.config.grid_size, self.config.columns)
            self.held_piece = current_type
        self.can_hold = False

    def _lock_piece(self) -> None:
        """Lock current piece in place."""
        if not self.current_piece:
            return
        for y, row in enumerate(self.current_piece.shape):
            for x, cell in enumerate(row):
                if cell and self.current_piece.y + y >= 0:
                    self.grid[self.current_piece.y + y][self.current_piece.x + x] = cell

    def _clear_lines(self) -> int:
        """Clear complete lines and return number cleared."""
        lines_cleared = 0
        y = self.config.rows - 1
        while y >= 0:
            if all(self.grid[y]):
                lines_cleared += 1
                for ny in range(y, 0, -1):
                    self.grid[ny] = self.grid[ny - 1][:]
                self.grid[0] = [0] * self.config.columns
            else:
                y -= 1
        self.lines_cleared += lines_cleared
        # Update level
        self.level = (self.lines_cleared // 10) + 1
        return lines_cleared

    def _calculate_reward(self, lines_cleared: int) -> int:
        """Calculate score reward for cleared lines."""
        if lines_cleared == 0:
            self.combo_counter = -1
            return 0

        # Base score from lines
        base_scores = {1: 100, 2: 300, 3: 500, 4: 800}
        base_reward = base_scores.get(lines_cleared, 0)

        # Combo bonus
        self.combo_counter += 1
        combo_bonus = 50 * self.combo_counter if self.combo_counter > 0 else 0

        # Level multiplier
        return (base_reward + combo_bonus) * self.level

    def _spawn_new_piece(self) -> bool:
        """Spawn a new piece and check if it's valid."""
        self.current_piece = self._get_next_piece()
        self.can_hold = True
        return self.current_piece.is_valid_position(self.grid,
            self.current_piece.x, self.current_piece.y)

    def toggle_pause(self) -> None:
        """Toggle pause state."""
        if self.state == GameState.PLAYING:
            self.state = GameState.PAUSED
        elif self.state == GameState.PAUSED:
            self.state = GameState.PLAYING

    def _initialize_subsystems(self) -> None:
        """Initialize all game subsystems with error handling."""
        try:
            # Initialize Pygame
            pygame.init()
            # Set display mode with hardware acceleration if available
            display_flags = pygame.DOUBLEBUF
            if hasattr(pygame, 'HWSURFACE'):
                display_flags |= pygame.HWSURFACE

            screen = pygame.display.set_mode(
                (self.config.total_width, self.config.screen_height),
                display_flags
            )

            # Try to enable vsync if requested
            if self.config.vsync:
                try:
                    # On some systems, vsync can be enabled via SDL
                    import os
                    os.environ['SDL_VIDEO_VSYNC'] = '1'
                except Exception as e:
                    logger.warning(f"Failed to enable vsync: {e}")

            pygame.display.set_caption("Tetris")

            # Initialize subsystems
            self.render_config = RenderConfig(self.config)
            self.renderer = RenderSystem(self.config, self.render_config)
            self.input_buffer = InputBuffer(self.config.input_buffer_frames)
            self.timer = GameTimer(self.config.fps)

            # Initialize sound if available
            try:
                pygame.mixer.init()
            except pygame.error:
                logger.warning("Sound initialization failed, continuing without sound")

        except Exception as e:
            logger.error(f"Failed to initialize game: {e}")
            raise

    def _fill_bag(self) -> List[str]:
        """Implement bag-7 randomization."""
        if len(self.bag) < 7:
            new_bag = list(TetriminoData.SHAPES.keys())
            random.shuffle(new_bag)
            self.bag.extend(new_bag)
        return self.bag

    def _get_next_piece(self) -> Tetrimino:
        """Get the next piece from the bag."""
        self._fill_bag()
        next_piece = self.next_pieces.pop(0)
        piece_type = self.bag.pop(0)
        self.next_pieces.append(Tetrimino(piece_type, self.config.grid_size, self.config.columns))
        return next_piece

    def reset(self, randomize_start: bool = False) -> None:
        """Reset the game state with proper cleanup."""
        with self._state_lock:
            # Initialize empty grid
            self.grid = [[0] * self.config.columns for _ in range(self.config.rows)]
            # Clear existing state
            self.grid = [[0] * self.config.columns for _ in range(self.config.rows)]
            self.score_handler = Score(self.config.max_score)
            self.lines_cleared = 0
            self.level = 1
            self.state = GameState.PLAYING
            self.combo_counter = -1
            self.frame_counter = 0

            # Reset piece generation
            self.bag = []
            self.held_piece = None
            self.can_hold = True

            # Generate initial pieces
            piece_types = self._fill_bag()[:3]
            self.next_pieces = [
                Tetrimino(piece_type, self.config.grid_size, self.config.columns)
                for piece_type in piece_types
            ]
            self.current_piece = self._get_next_piece()

            # Randomize if requested
            if randomize_start:
                self._randomize_starting_grid()

            # Reset subsystems
            self.input_buffer.clear()
            self.timer.reset()

    def _randomize_starting_grid(self, max_height: int = 10) -> None:
        """Create a random starting grid state with validation."""
        attempts = 0
        while attempts < 100:  # Prevent infinite loop
            # Create temporary grid
            temp_grid = [[0] * self.config.columns for _ in range(self.config.rows)]
            height = random.randint(0, max_height)

            for y in range(self.config.rows - height, self.config.rows):
                row = [1] * self.config.columns
                empty_cells = random.randint(1, 3)  # 1-3 empty cells per row
                empty_positions = random.sample(range(self.config.columns), empty_cells)
                for pos in empty_positions:
                    row[pos] = 0
                temp_grid[y] = row

            if self._is_valid_board_state(temp_grid):
                self.grid = temp_grid
                return

            attempts += 1

        # If no valid state found, start with empty grid
        self.grid = [[0] * self.config.columns for _ in range(self.config.rows)]

    def _is_valid_board_state(self, grid: Grid) -> bool:
        """Validate board state for impossible configurations."""
        # Check for floating blocks
        for x in range(self.config.columns):
            gap_found = False
            for y in range(self.config.rows - 1, -1, -1):
                if not grid[y][x]:
                    gap_found = True
                elif gap_found:
                    return False
        return True

    def run(self) -> None:
        """Main game loop with proper error handling and cleanup."""
        try:
            running = True
            clock = pygame.time.Clock()

            while running:
                try:
                    # Event handling
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                        elif event.type == pygame.KEYDOWN:
                            self._handle_keydown(event.key)

                    # Game state update
                    if self.state == GameState.PLAYING:
                        self._update_game_state()

                    # Rendering
                    self._render_frame()

                    # Frame timing
                    clock.tick(self.config.fps)
                    self.frame_counter += 1

                except Exception as e:
                    logger.error(f"Error in game loop: {e}")
                    if self.state != GameState.GAME_OVER:
                        self.state = GameState.GAME_OVER

            self.cleanup()

        except Exception as e:
            logger.error(f"Fatal error in game loop: {e}")
            raise
        finally:
            self.cleanup()

    def _update_game_state(self) -> None:
        """Update game state with proper synchronization."""
        with self._state_lock:
            # Process inputs
            action = self.input_buffer.update()
            if action is not None:
                self._process_action(action)

            # Apply gravity
            if self.frame_counter % self._get_fall_speed() == 0:
                if not self._apply_gravity():
                    self._lock_piece()
                    cleared_lines = self._clear_lines()
                    reward = self._calculate_reward(cleared_lines)
                    self.score_handler.add(reward)

                    if not self._spawn_new_piece():
                        self.state = GameState.GAME_OVER

    def _render_frame(self) -> None:
        """Render the current frame."""
        self.renderer.clear_surfaces()

        if self.state == GameState.PLAYING:
            # Draw active game elements
            self.renderer.draw_grid()
            if self.current_piece:
                ghost_y = self.current_piece.get_ghost_position(self.grid)
                self.renderer.draw_piece(self.current_piece, ghost=True, override_y=ghost_y)
                self.renderer.draw_piece(self.current_piece)
            self.renderer.draw_next_pieces(self.next_pieces)
            if self.held_piece:
                held_tetrimino = Tetrimino(self.held_piece, self.config.grid_size, self.config.columns)
                self.renderer.draw_held_piece(self.held_piece, held_tetrimino)
            self.renderer.draw_info(
                self.score_handler.value,
                self.level,
                self.lines_cleared,
                self.combo_counter
            )
        elif self.state == GameState.PAUSED:
            self.renderer.draw_pause_screen()
        elif self.state == GameState.GAME_OVER:
            self.renderer.draw_game_over(self.score_handler.value)

        # Draw debug information
        self.renderer.draw_debug_info(self.frame_counter, self.timer.fps)

        # Compose the final frame
        screen = pygame.display.get_surface()
        if screen:
            self.renderer.compose_frame(screen)


    def cleanup(self) -> None:
        """Clean up game resources."""
        try:
            # Clean up subsystems
            if hasattr(self, 'renderer'):
                self.renderer.cleanup()

            # Clean up Pygame
            pygame.mixer.quit()
            pygame.quit()

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            logger.info("Game cleaned up")

def main():
    """Main entry point with proper error handling."""
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Create and run game
        config = GameConfig()
        game = TetrisGame(config)
        game.run()

    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
