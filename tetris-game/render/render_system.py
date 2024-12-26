# File: render/render_system.py

import pygame
import time
from typing import List, Optional, Tuple
from config.game_config import GameConfig
from render.render_config import RenderConfig
from core.tetrimino import Tetrimino
from render.colors import Color, ColorAlpha
from core.grid import Grid
from utils.logging_setup import setup_logging

# Initialize loggers
loggers = setup_logging()
render_logger = loggers['render']
game_logger = loggers['game']


class RenderSystem:
    """Handles all rendering operations with proper layering and buffering."""

    def __init__(self, config: GameConfig, render_config: RenderConfig):
        """Initialize the RenderSystem with configurations."""
        self.config = config
        self.render_config = render_config
        self.game_surface = pygame.Surface((self.config.screen_width, self.config.screen_height))
        self.ui_surface = pygame.Surface((self.config.ui_width, self.config.screen_height))
        self.ghost_surface = pygame.Surface((self.config.screen_width, self.config.screen_height), pygame.SRCALPHA)
        self.debug_surface = pygame.Surface((self.config.screen_width, self.config.screen_height), pygame.SRCALPHA)
        self.grid_surface = pygame.Surface((self.config.screen_width, self.config.screen_height))
        self.particles: List[Tuple[pygame.Rect, Color]] = []
        self.frame_count = 0
        self.frame_times: List[float] = []
        self.last_frame_time = time.perf_counter()
        self._initialize_surfaces()
        self._initialize_static_grid()

    def _initialize_surfaces(self) -> None:
        """Initialize all rendering surfaces with proper layering."""
        try:
            self.game_surface.fill(self.render_config.colors.UI_COLORS['background'])
            self.ui_surface.fill(self.render_config.colors.UI_COLORS['background'])
            self.ghost_surface.fill((0, 0, 0, 0))
            self.debug_surface.fill((0, 0, 0, 0))
            self.grid_surface.fill(self.render_config.colors.UI_COLORS['background'])
            render_logger.debug("All rendering surfaces initialized")
        except pygame.error as e:
            render_logger.error("Error initializing surfaces: %s", str(e))

    def _initialize_static_grid(self) -> None:
        """Create the static grid background with proper grid lines."""
        try:
            render_logger.debug("Initializing static grid background")
            self.grid_surface.fill(self.render_config.colors.UI_COLORS['background'])
            for x in range(0, self.config.screen_width + 1, self.config.grid_size):
                pygame.draw.line(
                    self.grid_surface,
                    self.render_config.colors.UI_COLORS['grid_lines'],
                    (x, 0),
                    (x, self.config.screen_height),
                    self.config.grid_line_width
                )
            for y in range(0, self.config.screen_height + 1, self.config.grid_size):
                pygame.draw.line(
                    self.grid_surface,
                    self.render_config.colors.UI_COLORS['grid_lines'],
                    (0, y),
                    (self.config.screen_width, y),
                    self.config.grid_line_width
                )
            render_logger.debug("Static grid background initialized successfully")
        except pygame.error as e:
            render_logger.error("Error initializing grid surface: %s", str(e))

    def clear_surfaces(self) -> None:
        """Clear all rendering surfaces to their base state."""
        try:
            render_logger.debug("Clearing all surfaces")
            self.game_surface.blit(self.grid_surface, (0, 0))
            self.ui_surface.fill(self.render_config.colors.UI_COLORS['background'])
            self.ghost_surface.fill((0, 0, 0, 0))
            self.debug_surface.fill((0, 0, 0, 0))
            render_logger.debug("All surfaces cleared successfully")
        except pygame.error as e:
            render_logger.error("Error clearing surfaces: %s", str(e))

    def draw_block(self, surface: pygame.Surface, x: int, y: int, color: Color, alpha: int = 255) -> None:
        """Draw a single block on the given surface."""
        try:
            rect = pygame.Rect(
                x * self.config.grid_size + self.config.cell_padding,
                y * self.config.grid_size + self.config.cell_padding,
                self.config.grid_size - (2 * self.config.cell_padding),
                self.config.grid_size - (2 * self.config.cell_padding)
            )
            if alpha < 255:
                s = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
                s.fill((*color, alpha))
                surface.blit(s, rect)
            else:
                pygame.draw.rect(surface, color, rect)
            self._draw_piece_shading(surface, rect, color)
        except pygame.error as e:
            render_logger.error("Error drawing block: %s", str(e))

    def draw_piece(self, piece: Tetrimino, ghost: bool = False,
                  override_y: Optional[int] = None,
                  override_x: Optional[int] = None) -> None:
        """Draw a tetrimino with proper shading and logging."""
        try:
            if not piece:
                render_logger.warning("Attempted to draw None piece")
                return
            render_logger.debug(
                "Drawing piece: type=%s, ghost=%s, pos=(%s, %s)",
                piece.piece_type, ghost,
                override_x if override_x is not None else piece.x,
                override_y if override_y is not None else piece.y
            )
            target_surface = self.ghost_surface if ghost else self.game_surface
            if not target_surface:
                render_logger.error("Target surface not available for piece drawing")
                return
            y_pos = override_y if override_y is not None else piece.y
            x_pos = override_x if override_x is not None else piece.x
            color = self.render_config.colors.get_piece_color(piece.piece_type)
            alpha = 64 if ghost else 255
            blocks_drawn = 0
            for y, row in enumerate(piece.shape):
                for x, cell in enumerate(row):
                    if cell:
                        blocks_drawn += 1
                        self.draw_block(target_surface, x_pos + x, y_pos + y, color, alpha)
            render_logger.debug("Successfully drew piece with %d blocks", blocks_drawn)
        except (pygame.error, AttributeError) as e:
            render_logger.error("Error drawing piece: %s", str(e))

    def _draw_piece_shading(self, surface: pygame.Surface, rect: pygame.Rect,
                            color: Color) -> None:
        """Add shading effects to a piece."""
        try:
            pygame.draw.line(
                surface,
                tuple(min(c + 50, 255) for c in color[:3]),
                rect.topleft,
                rect.topright
            )
            pygame.draw.line(
                surface,
                tuple(min(c + 30, 255) for c in color[:3]),
                rect.topleft,
                rect.bottomleft
            )
            pygame.draw.line(
                surface,
                tuple(max(c - 50, 0) for c in color[:3]),
                rect.bottomleft,
                rect.bottomright
            )
            pygame.draw.line(
                surface,
                tuple(max(c - 30, 0) for c in color[:3]),
                rect.topright,
                rect.bottomright
            )
        except pygame.error as e:
            render_logger.error("Error drawing piece shading: %s", str(e))

    def draw_grid(self, grid: Grid) -> None:
        """Draw the game grid with filled cells."""
        try:
            render_logger.debug("Drawing game grid")
            if self.game_surface and self.grid_surface:
                self.game_surface.blit(self.grid_surface, (0, 0))
            cells_drawn = 0
            for y, row in enumerate(grid.grid):
                for x, cell in enumerate(row):
                    if cell:
                        cells_drawn += 1
                        color = self.render_config.colors.get_piece_color(cell)
                        rect = pygame.Rect(
                            x * self.config.grid_size + self.config.cell_padding,
                            y * self.config.grid_size + self.config.cell_padding,
                            self.config.grid_size - (2 * self.config.cell_padding),
                            self.config.grid_size - (2 * self.config.cell_padding)
                        )
                        pygame.draw.rect(self.game_surface, color, rect)
            render_logger.debug("Drew %d filled cells", cells_drawn)
        except (pygame.error, TypeError, IndexError) as e:
            render_logger.error("Error drawing grid: %s", str(e))

    def draw_ui(self, score: int, high_score: int, level: int, lines: int,
               next_pieces: List[Tetrimino], held_piece: Optional[Tetrimino] = None,
               combo: int = 0) -> None:
        """Draw all UI elements."""
        try:
            render_logger.debug("Drawing UI elements")
            if not self.render_config.fonts or not self.ui_surface:
                render_logger.error("Fonts or UI surface not available")
                return
            self._draw_score_section(score, high_score, level, lines, combo)
            self._draw_next_pieces(next_pieces)
            if held_piece is not None:
                self._draw_held_piece(held_piece)
            else:
                render_logger.debug("No held piece to draw")
            render_logger.debug("UI elements drawn successfully")
        except Exception as e:
            render_logger.error("Error drawing UI: %s", str(e))

    def _draw_score_section(self, score: int, high_score: int, level: int, lines: int, combo: int) -> None:
        """Draw score and statistics section, including high score."""
        try:
            y_pos = 20
            spacing = 40
            score_text = self.render_config.fonts['small'].render(
                f"Score: {score:,}", True, self.render_config.colors.UI_COLORS['text']
            )
            self.ui_surface.blit(score_text, (20, y_pos))

            high_score_text = self.render_config.fonts['small'].render(
                f"High Score: {high_score:,}", True, self.render_config.colors.UI_COLORS['text']
            )
            self.ui_surface.blit(high_score_text, (20, y_pos + spacing))

            level_text = self.render_config.fonts['small'].render(
                f"Level: {level}", True, self.render_config.colors.UI_COLORS['text']
            )
            self.ui_surface.blit(level_text, (20, y_pos + spacing * 2))

            lines_text = self.render_config.fonts['small'].render(
                f"Lines: {lines}", True, self.render_config.colors.UI_COLORS['text']
            )
            self.ui_surface.blit(lines_text, (20, y_pos + spacing * 3))

            if combo > 0:
                combo_text = self.render_config.fonts['small'].render(
                    f"Combo: x{combo}", True, self.render_config.colors.UI_COLORS['highlight']
                )
                self.ui_surface.blit(combo_text, (20, y_pos + spacing * 4))
        except Exception as e:
            render_logger.error("Error drawing score section: %s", str(e))

    def _draw_held_piece(self, held_piece: Tetrimino) -> None:
        """Draw the held piece with logging."""
        try:
            if not self.render_config.fonts or not self.ui_surface:
                render_logger.error("Fonts or UI surface not available for held piece")
                return
            label = self.render_config.fonts['small'].render(
                "Hold", True, self.render_config.colors.UI_COLORS['text']
            )
            self.ui_surface.blit(label, (20, 150))

            held_surface = pygame.Surface((
                4 * self.config.grid_size,
                4 * self.config.grid_size
            ), pygame.SRCALPHA)
            held_surface.fill(self.render_config.colors.UI_COLORS['background'])

            # Center the held piece within the held_surface
            x_offset = (held_surface.get_width() - len(held_piece.shape[0]) * self.config.grid_size) // 2
            y_offset = (held_surface.get_height() - len(held_piece.shape) * self.config.grid_size) // 2

            for y, row in enumerate(held_piece.shape):
                for x, cell in enumerate(row):
                    if cell:
                        rect = pygame.Rect(
                            x_offset + x * self.config.grid_size + self.config.cell_padding,
                            y_offset + y * self.config.grid_size + self.config.cell_padding,
                            self.config.grid_size - (2 * self.config.cell_padding),
                            self.config.grid_size - (2 * self.config.cell_padding)
                        )
                        pygame.draw.rect(held_surface, self.render_config.colors.get_piece_color(held_piece.piece_type), rect)

            self.ui_surface.blit(held_surface, (20, 180))
            render_logger.debug("Drew held piece: %s", held_piece.piece_type)
        except Exception as e:
            render_logger.error("Error drawing held piece: %s", str(e))

    def _draw_next_pieces(self, next_pieces: List[Tetrimino]) -> None:
        """Draw preview of next pieces."""
        try:
            if not self.render_config.fonts or not self.ui_surface:
                return
            label = self.render_config.fonts['small'].render(
                "Next", True, self.render_config.colors.UI_COLORS['text']
            )
            self.ui_surface.blit(label, (20, 20))

            preview_y = 60
            for piece in next_pieces[:self.config.preview_pieces]:
                preview_surface = pygame.Surface((
                    4 * self.config.grid_size,
                    4 * self.config.grid_size
                ), pygame.SRCALPHA)
                preview_surface.fill(self.render_config.colors.UI_COLORS['background'])

                # Center the next piece within the preview_surface
                x_offset = (preview_surface.get_width() - len(piece.shape[0]) * self.config.grid_size) // 2
                y_offset = (preview_surface.get_height() - len(piece.shape) * self.config.grid_size) // 2

                for y, row in enumerate(piece.shape):
                    for x, cell in enumerate(row):
                        if cell:
                            rect = pygame.Rect(
                                x_offset + x * self.config.grid_size + self.config.cell_padding,
                                y_offset + y * self.config.grid_size + self.config.cell_padding,
                                self.config.grid_size - (2 * self.config.cell_padding),
                                self.config.grid_size - (2 * self.config.cell_padding)
                            )
                            pygame.draw.rect(preview_surface, self.render_config.colors.get_piece_color(piece.piece_type), rect)

                self.ui_surface.blit(preview_surface, (20, preview_y))
                preview_y += 80
        except Exception as e:
            render_logger.error("Error drawing next pieces: %s", str(e))

    def draw_pause_screen(self) -> None:
        """Draw the pause screen overlay."""
        try:
            if not self.render_config.fonts or not self.ui_surface:
                render_logger.error("Fonts or UI surface not available for pause screen")
                return
            pause_text = self.render_config.fonts['large'].render(
                "PAUSED", True, self.render_config.colors.UI_COLORS['highlight']
            )
            # Complete the center tuple with both x and y coordinates
            pause_rect = pause_text.get_rect(center=(self.config.screen_width // 2, self.config.screen_height // 2))
            self.game_surface.blit(pause_text, pause_rect)
            render_logger.debug("Pause screen drawn")
        except Exception as e:
            render_logger.error("Error drawing pause screen: %s", str(e))

    def draw_game_over(self, score: int) -> None:
        """Draw the game over screen."""
        try:
            if not self.render_config.fonts or not self.ui_surface:
                render_logger.error("Fonts or UI surface not available for game over screen")
                return
            game_over_text = self.render_config.fonts['large'].render(
                "GAME OVER", True, self.render_config.colors.UI_COLORS['highlight']
            )
            score_text = self.render_config.fonts['medium'].render(
                f"Score: {score:,}", True, self.render_config.colors.UI_COLORS['text']
            )
            restart_text = self.render_config.fonts['small'].render(
                "Press R to Restart or Q to Quit", True, self.render_config.colors.UI_COLORS['text']
            )
            # Complete the center tuples with both x and y coordinates
            game_over_rect = game_over_text.get_rect(center=(self.config.screen_width // 2, self.config.screen_height // 2 - 50))
            score_rect = score_text.get_rect(center=(self.config.screen_width // 2, self.config.screen_height // 2))
            restart_rect = restart_text.get_rect(center=(self.config.screen_width // 2, self.config.screen_height // 2 + 50))
            self.game_surface.blit(game_over_text, game_over_rect)
            self.game_surface.blit(score_text, score_rect)
            self.game_surface.blit(restart_text, restart_rect)
            if self.render_config.sounds.get('game_over'):
                self.render_config.sounds['game_over'].play()
            render_logger.debug("Game over screen drawn")
        except Exception as e:
            render_logger.error("Error drawing game over screen: %s", str(e))

    def draw_debug_info(self, frame_counter: int, fps: float) -> None:
        """Draw debug information on the debug surface."""
        try:
            if not self.render_config.fonts or not self.debug_surface:
                render_logger.error("Fonts or debug surface not available for debug info")
                return
            debug_text = self.render_config.fonts['small'].render(
                f"Frame: {frame_counter} | FPS: {fps:.1f}",
                True,
                self.render_config.colors.UI_COLORS['text']
            )
            self.debug_surface.blit(debug_text, (10, 5))
            render_logger.debug("Debug information drawn")
        except Exception as e:
            render_logger.error("Error drawing debug info: %s", str(e))

    def draw_menu(self, options: List[str], selected_index: int) -> None:
        """Draw a menu with given options and highlight the selected one."""
        try:
            menu_font = self.render_config.fonts['large']
            for idx, option in enumerate(options):
                color = self.render_config.colors.UI_COLORS['highlight'] if idx == selected_index else self.render_config.colors.UI_COLORS['text']
                text_surface = menu_font.render(option, True, color)
                # Complete the center tuple with both x and y coordinates
                text_rect = text_surface.get_rect(center=(self.config.screen_width // 2, 200 + idx * 50))
                self.ui_surface.blit(text_surface, text_rect)
        except Exception as e:
            render_logger.error("Error drawing menu: %s", str(e))

    def compose_frame(self, screen: pygame.Surface, frame_count: int) -> None:
        """Compose final frame with proper positioning and timing logging."""
        frame_start = time.perf_counter()
        try:
            if not screen:
                render_logger.error("No screen surface available for composition")
                return
            render_logger.debug("Starting frame composition")
            screen.fill(self.render_config.colors.UI_COLORS['background'])
            if self.game_surface:
                screen.blit(self.game_surface, (0, 0))
                render_logger.debug("Blitted game surface")
            if self.ghost_surface:
                screen.blit(self.ghost_surface, (0, 0))
                render_logger.debug("Blitted ghost surface")
            if self.ui_surface:
                screen.blit(self.ui_surface, (self.config.screen_width, 0))
                render_logger.debug("Blitted UI surface")
            if self.debug_surface and self.render_config.debug_mode:
                screen.blit(self.debug_surface, (0, 0))
                render_logger.debug("Blitted debug surface")
            frame_end = time.perf_counter()
            frame_time = (frame_end - frame_start) * 1000
            frame_delta = frame_end - self.last_frame_time
            fps = 1.0 / frame_delta if frame_delta > 0 else 0
            self.frame_times.append(frame_time)
            if len(self.frame_times) > 120:
                self.frame_times.pop(0)
            if frame_count % 60 == 0:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                loggers['performance'].info(
                    "Frame stats - Avg: %.2fms, FPS: %.1f",
                    avg_frame_time, fps
                )
            self.frame_count += 1
            self.last_frame_time = frame_end
            pygame.display.flip()
        except pygame.error as e:
            render_logger.error("Error composing frame: %s", str(e))

    def add_particle(self, rect: pygame.Rect, color: Color) -> None:
        """Add a particle to the particle system."""
        if self.config.particle_effects:
            self.particles.append((rect, color))
            render_logger.debug("Added particle at %s with color %s", rect.topleft, color)

    def update_particles(self, delta_time: float) -> None:
        """Update particle positions and lifespans."""
        updated_particles = []
        for rect, color in self.particles:
            rect.y += int(100 * delta_time)  # Adjusted to integer for pixel movement
            if rect.y < self.config.screen_height:
                updated_particles.append((rect, color))
        self.particles = updated_particles

    def draw_particles(self) -> None:
        """Draw all particles."""
        for rect, color in self.particles:
            pygame.draw.rect(self.game_surface, color, rect)
