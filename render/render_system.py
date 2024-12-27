# File: render/render_system.py

import pygame
import time
import random
from typing import List, Optional, Tuple, Dict
from config.game_config import GameConfig
from render.render_config import RenderConfig
from core.tetrimino import Tetrimino
from render.colors import Color, ColorAlpha
from core.grid import Grid
from utils.logging_setup import setup_logging
from dataclasses import dataclass

loggers = setup_logging()
render_logger = loggers['render']
game_logger = loggers['game']

@dataclass
class ScorePopup:
    text: str
    position: Tuple[int, int]
    start_time: float
    duration: float = 1.0

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
        self.particles: List[Dict] = []
        self.score_popups: List[ScorePopup] = []
        self.frame_count = 0
        self.frame_times: List[float] = []
        self.last_frame_time = time.perf_counter()
        self.last_render_time: float = 0.0
        self._initialize_surfaces()
        self._initialize_static_grid()
        self._initialize_particle_effects()

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

    def _initialize_particle_effects(self) -> None:
        """Initialize particle effects settings."""
        self.particle_effects_enabled = self.config.particle_effects
        self.particle_color = self.render_config.colors.UI_COLORS['highlight']
        self.particle_size = 5

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
                  override_x: Optional[int] = None,
                  highlight: bool = False) -> None:
        """Draw a tetrimino with proper shading, glow, and logging."""
        try:
            if not piece:
                render_logger.warning("Attempted to draw None piece")
                return
            render_logger.debug(
                "Drawing piece: type=%s, ghost=%s, pos=(%s, %s), highlight=%s",
                piece.piece_type, ghost,
                override_x if override_x is not None else piece.x,
                override_y if override_y is not None else piece.y,
                highlight
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
                        if not ghost:
                            shadow_rect = pygame.Rect(
                                (x_pos + x) * self.config.grid_size + self.config.cell_padding + 2,
                                (y_pos + y) * self.config.grid_size + self.config.cell_padding + 2,
                                self.config.grid_size - (2 * self.config.cell_padding),
                                self.config.grid_size - (2 * self.config.cell_padding)
                            )
                            pygame.draw.rect(target_surface, self.render_config.colors.UI_COLORS['shadow'], shadow_rect)
                        self.draw_block(target_surface, x_pos + x, y_pos + y, color, alpha)
                        if highlight and not ghost:
                            self._draw_glow(target_surface, x_pos + x, y_pos + y, color)
            render_logger.debug("Successfully drew piece with %d blocks", blocks_drawn)
        except (pygame.error, AttributeError) as e:
            render_logger.error("Error drawing piece: %s", str(e))

    def _draw_piece_shading(self, surface: pygame.Surface, rect: pygame.Rect,
                            color: Color) -> None:
        """Add shading effects to a piece."""
        try:
            # Light shading on top and left
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
            # Dark shading on bottom and right
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

    def _draw_glow(self, surface: pygame.Surface, x: int, y: int, color: Color) -> None:
        """Draw a glow effect around a block."""
        try:
            glow_color = tuple(min(c + 100, 255) for c in color)
            rect = pygame.Rect(
                x * self.config.grid_size + self.config.cell_padding - 2,
                y * self.config.grid_size + self.config.cell_padding - 2,
                self.config.grid_size - (2 * self.config.cell_padding) + 4,
                self.config.grid_size - (2 * self.config.cell_padding) + 4
            )
            pygame.draw.rect(surface, glow_color, rect, 2)
        except pygame.error as e:
            render_logger.error("Error drawing glow effect: %s", str(e))

    def draw_grid(self, grid: 'Grid') -> None:
        """
        Draw the game grid with filled cells.
        Args:
            grid: The game grid to render
        """
        try:
            if self.game_surface and self.grid_surface:
                self.game_surface.blit(self.grid_surface, (0, 0))
                
                for y in range(len(grid)):
                    for x in range(grid.columns):
                        cell = grid[y][x]  # Use Grid's __getitem__ method
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

    def add_score_popup(self, points: int, position: Tuple[int, int]) -> None:
        """Add a score popup at the specified position."""
        popup = ScorePopup(text=f"+{points}", position=position, start_time=time.time())
        self.score_popups.append(popup)
        render_logger.debug("Added score popup: %s at %s", popup.text, popup.position)

    def draw_score_popups(self) -> None:
        """Draw all active score popups with fade-out effect."""
        current_time = time.time()
        for popup in self.score_popups[:]:
            elapsed = current_time - popup.start_time
            if elapsed > popup.duration:
                self.score_popups.remove(popup)
                continue
            alpha = max(255 - int((elapsed / popup.duration) * 255), 0)
            text_surface = self.render_config.fonts['small'].render(
                popup.text, True, self.render_config.colors.UI_COLORS['highlight']
            )
            text_surface.set_alpha(alpha)
            self.ui_surface.blit(text_surface, popup.position)
            render_logger.debug("Drawing score popup: %s with alpha: %d", popup.text, alpha)

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
            self._draw_background_image()
            self._draw_border()
            self.draw_score_popups()
            render_logger.debug("UI elements drawn successfully")
        except Exception as e:
            render_logger.error("Error drawing UI: %s", str(e))

    def _draw_score_section(self, score: int, high_score: int, level: int, lines: int, combo: int) -> None:
        """Draw score and statistics section, including high score."""
        try:
            margin = 20
            y_start = margin
            spacing = 40
            elements = [
                f"Score: {score:,}",
                f"High Score: {high_score:,}",
                f"Level: {level}",
                f"Lines: {lines}"
            ]
            for idx, text in enumerate(elements):
                text_surface = self.render_config.fonts['small'].render(
                    text, True, self.render_config.colors.UI_COLORS['text']
                )
                self.ui_surface.blit(text_surface, (margin, y_start + idx * spacing))
            if combo > 0:
                combo_text = self.render_config.fonts['small'].render(
                    f"Combo: x{combo}", True, self.render_config.colors.UI_COLORS['highlight']
                )
                self.ui_surface.blit(combo_text, (margin, y_start + len(elements) * spacing))
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
                        self._draw_glow(held_surface, x, y, self.render_config.colors.get_piece_color(held_piece.piece_type))
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

    def _draw_background_image(self) -> None:
        """Draw a background image or pattern for the UI panel."""
        try:
            overlay = pygame.Surface((self.config.ui_width, self.config.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 50))
            self.ui_surface.blit(overlay, (0, 0))
            render_logger.debug("Background image or overlay drawn on UI")
        except pygame.error as e:
            render_logger.error("Error drawing background image: %s", str(e))

    def _draw_border(self) -> None:
        """Draw borders around the UI panel for better separation."""
        try:
            border_color = self.render_config.colors.UI_COLORS['highlight']
            pygame.draw.rect(
                self.ui_surface,
                border_color,
                self.ui_surface.get_rect(),
                2
            )
            render_logger.debug("UI border drawn")
        except pygame.error as e:
            render_logger.error("Error drawing UI border: %s", str(e))

    def draw_pause_screen(self) -> None:
        """Draw the pause screen overlay."""
        try:
            if not self.render_config.fonts or not self.ui_surface:
                render_logger.error("Fonts or UI surface not available for pause screen")
                return
            pause_text = self.render_config.fonts['large'].render(
                "PAUSED", True, self.render_config.colors.UI_COLORS['highlight']
            )
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
                if idx == selected_index:
                    text_surface = menu_font.render(option, True, self.render_config.colors.UI_COLORS['highlight'])
                    arrow_surface = self.render_config.fonts['small'].render('➤', True, self.render_config.colors.UI_COLORS['highlight'])
                    self.ui_surface.blit(arrow_surface, (20, 20 + idx * 60))
                    text_rect = text_surface.get_rect(topleft=(50, 20 + idx * 60))
                    self.ui_surface.blit(text_surface, text_rect)
                else:
                    text_surface = menu_font.render(option, True, self.render_config.colors.UI_COLORS['text'])
                    text_rect = text_surface.get_rect(topleft=(50, 20 + idx * 60))
                    self.ui_surface.blit(text_surface, text_rect)
        except Exception as e:
            render_logger.error("Error drawing menu: %s", str(e))

    def trigger_particle_effect(self, position: Tuple[int, int]) -> None:
        """Trigger a particle effect at the specified grid position."""
        if not self.particle_effects_enabled:
            return
        x, y = position
        pixel_x = x * self.config.grid_size + self.config.cell_padding + self.config.grid_size // 2
        pixel_y = y * self.config.grid_size + self.config.cell_padding + self.config.grid_size // 2
        for _ in range(20):
            velocity = [random.randint(-3, 3), random.randint(-3, -1)]
            lifetime = random.uniform(0.5, 1.0)
            self.particles.append({
                'position': [pixel_x, pixel_y],
                'velocity': velocity,
                'lifetime': lifetime
            })
        render_logger.debug("Triggered particle effect at (%d, %d)", x, y)

    def update_particles(self, delta_time: float) -> None:
        """Update particle positions and lifespans."""
        updated_particles = []
        for particle in self.particles:
            particle['position'][0] += particle['velocity'][0]
            particle['position'][1] += particle['velocity'][1]
            particle['velocity'][1] += 9.81 * delta_time  # Gravity effect
            particle['lifetime'] -= delta_time
            if particle['lifetime'] > 0:
                updated_particles.append(particle)
        self.particles = updated_particles

    def draw_particles(self) -> None:
        """Draw all particles with fade-out effect."""
        for particle in self.particles:
            alpha = int(255 * (particle['lifetime'] / 1.0))
            particle_color = (*self.particle_color, alpha)
            pygame.draw.circle(self.game_surface, particle_color, (int(particle['position'][0]), int(particle['position'][1])), self.particle_size)

    def compose_frame(self, screen: pygame.Surface, frame_count: int) -> None:
        """
        Compose final frame with proper positioning and timing logging.
        Includes summary logs at controlled intervals to prevent log spam.
        """
        frame_start = time.perf_counter()
        try:
            if not screen:
                render_logger.error("No screen surface available for composition")
                return
            render_logger.debug("Starting frame composition")
            screen.fill(self.render_config.colors.UI_COLORS['background'])
            if self.game_surface:
                screen.blit(self.game_surface, (0, 0))
            if self.ghost_surface:
                screen.blit(self.ghost_surface, (0, 0))
            if self.ui_surface:
                screen.blit(self.ui_surface, (self.config.screen_width, 0))
            if self.debug_surface and self.render_config.debug_mode:
                screen.blit(self.debug_surface, (0, 0))
            frame_end = time.perf_counter()
            render_time = (frame_end - frame_start) * 1000  # ms
            self.last_render_time = render_time
            self.frame_times.append(render_time)
            if frame_count % 60 == 0 and self.frame_times:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                fps = 1000.0 / avg_frame_time if avg_frame_time > 0 else 0.0
                render_logger.info(
                    f"Frame {frame_count}: Avg Frame Time: {avg_frame_time:.2f}ms, FPS: {fps:.1f}"
                )
                self.frame_times.clear()
            pygame.display.flip()
        except pygame.error as e:
            render_logger.error("Error composing frame: %s", str(e))
        finally:
            self.last_frame_time = frame_end if 'frame_end' in locals() else self.last_frame_time
