import pygame
import time
import random
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from enum import Enum

from config.game_config import GameConfig
from render.render_config import RenderConfig
from core.tetrimino import Tetrimino
from core.grid import Grid
from render.colors import Color, ColorAlpha
from utils.logging_setup import setup_logging
from render.ui_manager import UIManager

loggers = setup_logging()
render_logger = loggers['render']
game_logger = loggers['game']

@dataclass
class ScorePopup:
    text: str
    position: Tuple[int, int]
    start_time: float
    duration: float = 1.0

class Layer(Enum):
    """
    Enumerates different rendering layers:
      - BACKGROUND: The static grid or background
      - GAME: Active Tetriminos, ghost pieces, etc.
      - UI: Score, Next/Hold previews, etc.
      - DEBUG: Optional debug overlay
    """
    BACKGROUND = 0
    GAME = 1
    UI = 2
    DEBUG = 3

class RenderSystem:
    """
    Handles all rendering operations with layered surfaces and a UI Manager for
    drawing score, next pieces, hold pieces, etc. Also manages particle effects and
    score popups.
    """

    def __init__(self, config: GameConfig, render_config: RenderConfig):
        """
        :param config: GameConfig with screen/UI dimensions and Tetris logic settings
        :param render_config: RenderConfig for fonts, colors, and resource references
        """
        self.config = config
        self.render_config = render_config

        # Dictionary of layered pygame.Surface objects
        self.surfaces: Dict[Layer, pygame.Surface] = {}

        # Manager that draws UI elements (score, next/held previews, menu, etc.)
        self.ui_manager = UIManager(config, render_config)

        # Particle effect + score popup tracking
        self.particles: List[Dict] = []
        self.score_popups: List[ScorePopup] = []

        # Frame/time metrics
        self.frame_count = 0
        self.frame_times: List[float] = []
        self.last_frame_time = time.perf_counter()
        self.last_render_time: float = 0.0

        self._initialize_surfaces()
        self._initialize_particle_effects()

    def _initialize_surfaces(self) -> None:
        """
        Create the layered surfaces: BACKGROUND, GAME, UI, DEBUG.
        Fill the background layer with a default color.
        """
        try:
            width = self.config.screen_width
            height = self.config.screen_height
            ui_width = self.config.ui_width

            self.surfaces[Layer.BACKGROUND] = pygame.Surface((width, height))
            self.surfaces[Layer.GAME] = pygame.Surface((width, height), pygame.SRCALPHA)
            self.surfaces[Layer.UI] = pygame.Surface((ui_width, height), pygame.SRCALPHA)
            self.surfaces[Layer.DEBUG] = pygame.Surface((width, height), pygame.SRCALPHA)

            bg_color = self.render_config.colors.UI_COLORS['background']
            self.surfaces[Layer.BACKGROUND].fill(bg_color)

            render_logger.debug("Layered surfaces initialized successfully.")
        except pygame.error as e:
            render_logger.error(f"Error initializing surfaces: {e}")

    def _initialize_particle_effects(self) -> None:
        """
        Initialize default particle effect parameters (color, size).
        """
        self.particle_effects_enabled = self.config.particle_effects
        self.particle_color = self.render_config.colors.UI_COLORS['highlight']
        self.particle_size = 5

    def clear_surfaces(self) -> None:
        """
        Clear each layered surface to a base state. Typically called once per frame,
        before drawing any new objects or UI.
        """
        try:
            bg_surface = self.surfaces[Layer.BACKGROUND]
            bg_surface.fill(self.render_config.colors.UI_COLORS['background'])

            self.surfaces[Layer.GAME].fill((0, 0, 0, 0))
            self.surfaces[Layer.UI].fill((0, 0, 0, 0))
            self.surfaces[Layer.DEBUG].fill((0, 0, 0, 0))

            render_logger.debug("All surfaces cleared successfully.")
        except pygame.error as e:
            render_logger.error(f"Error clearing surfaces: {e}")

    def draw_block(self,
                   surface: pygame.Surface,
                   x: int,
                   y: int,
                   color: Color,
                   alpha: int = 255) -> None:
        """
        Draw a single Tetris block at grid cell (x, y) with optional alpha,
        plus simple shading.
        """
        try:
            rect = pygame.Rect(
                x * self.config.grid_size + self.config.cell_padding,
                y * self.config.grid_size + self.config.cell_padding,
                self.config.grid_size - (2 * self.config.cell_padding),
                self.config.grid_size - (2 * self.config.cell_padding)
            )

            if alpha < 255:
                block_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
                block_surf.fill((*color, alpha))
                surface.blit(block_surf, rect)
            else:
                pygame.draw.rect(surface, color, rect)

            self._draw_piece_shading(surface, rect, color)
        except pygame.error as e:
            render_logger.error(f"Error drawing block at ({x},{y}): {e}")

    def _draw_piece_shading(self,
                            surface: pygame.Surface,
                            rect: pygame.Rect,
                            color: Color) -> None:
        """
        Draw highlight and shadow lines on the edges of a block for a 3D look.
        """
        try:
            # Lighter top edge
            pygame.draw.line(
                surface,
                tuple(min(c + 50, 255) for c in color[:3]),
                rect.topleft,
                rect.topright
            )
            # Lighter left edge
            pygame.draw.line(
                surface,
                tuple(min(c + 30, 255) for c in color[:3]),
                rect.topleft,
                rect.bottomleft
            )
            # Darker bottom edge
            pygame.draw.line(
                surface,
                tuple(max(c - 50, 0) for c in color[:3]),
                rect.bottomleft,
                rect.bottomright
            )
            # Darker right edge
            pygame.draw.line(
                surface,
                tuple(max(c - 30, 0) for c in color[:3]),
                rect.topright,
                rect.bottomright
            )
        except pygame.error as e:
            render_logger.error(f"Error drawing piece shading: {e}")

    def _draw_glow(self,
                   surface: pygame.Surface,
                   x: int,
                   y: int,
                   color: Color) -> None:
        """
        Draw a glow outline around a single block at grid coords (x, y).
        """
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
            render_logger.error(f"Error drawing glow: {e}")

    def draw_piece(self,
                   piece: Tetrimino,
                   ghost: bool = False,
                   override_y: Optional[int] = None,
                   override_x: Optional[int] = None,
                   highlight: bool = False) -> None:
        """
        Draw an entire Tetrimino piece onto the GAME layer.

        :param piece: The Tetrimino object
        :param ghost: If True, draw at reduced alpha
        :param override_y, override_x: If given, override piece's default x,y
        :param highlight: If True, add an extra glow around each block
        """
        if not piece:
            render_logger.warning("draw_piece called with None piece.")
            return

        game_surface = self.surfaces.get(Layer.GAME)
        if not game_surface:
            render_logger.error("No GAME surface available to draw the Tetrimino.")
            return

        alpha = 64 if ghost else 255
        base_x = override_x if override_x is not None else piece.x
        base_y = override_y if override_y is not None else piece.y

        try:
            for row_idx, row in enumerate(piece.shape):
                for col_idx, cell in enumerate(row):
                    if cell:
                        # Draw a small shadow offset if not ghost
                        if not ghost:
                            shadow_rect = pygame.Rect(
                                (base_x + col_idx) * self.config.grid_size + self.config.cell_padding + 2,
                                (base_y + row_idx) * self.config.grid_size + self.config.cell_padding + 2,
                                self.config.grid_size - (2 * self.config.cell_padding),
                                self.config.grid_size - (2 * self.config.cell_padding)
                            )
                            pygame.draw.rect(
                                game_surface,
                                self.render_config.colors.UI_COLORS['shadow'],
                                shadow_rect
                            )
                        # Draw the block
                        block_color = self.render_config.colors.get_piece_color(piece.piece_type)
                        self.draw_block(game_surface, base_x + col_idx, base_y + row_idx, block_color, alpha)

                        # Optional glow
                        if highlight and not ghost:
                            self._draw_glow(game_surface, base_x + col_idx, base_y + row_idx, block_color)
        except pygame.error as e:
            render_logger.error(f"Error drawing piece '{piece.piece_type}': {e}")

    def draw_grid(self, grid: Grid) -> None:
        """
        Draw the Tetris grid lines and locked cells onto the BACKGROUND layer.
        """
        bg_surface = self.surfaces.get(Layer.BACKGROUND)
        if not bg_surface:
            render_logger.error("No BACKGROUND surface found to draw_grid.")
            return

        try:
            # Clear background
            bg_surface.fill(self.render_config.colors.UI_COLORS['background'])

            # Draw grid lines
            for x in range(0, self.config.screen_width + 1, self.config.grid_size):
                pygame.draw.line(
                    bg_surface,
                    self.render_config.colors.UI_COLORS['grid_lines'],
                    (x, 0),
                    (x, self.config.screen_height),
                    self.config.grid_line_width
                )
            for y in range(0, self.config.screen_height + 1, self.config.grid_size):
                pygame.draw.line(
                    bg_surface,
                    self.render_config.colors.UI_COLORS['grid_lines'],
                    (0, y),
                    (self.config.screen_width, y),
                    self.config.grid_line_width
                )

            # Draw locked cells
            for row_index, row in enumerate(grid.grid):
                for col_index, cell in enumerate(row):
                    if cell:
                        rect = pygame.Rect(
                            col_index * self.config.grid_size + self.config.cell_padding,
                            row_index * self.config.grid_size + self.config.cell_padding,
                            self.config.grid_size - (2 * self.config.cell_padding),
                            self.config.grid_size - (2 * self.config.cell_padding)
                        )
                        pygame.draw.rect(bg_surface, (200, 200, 200), rect)

        except (pygame.error, TypeError, IndexError) as e:
            game_logger.error(f"Error drawing grid: {e}")

    def add_score_popup(self, points: int, position: Tuple[int, int]) -> None:
        """
        Enqueue a small floating text popup at pixel (position).
        """
        popup = ScorePopup(
            text=f"+{points}",
            position=position,
            start_time=time.time(),
            duration=1.0
        )
        self.score_popups.append(popup)
        render_logger.debug(f"Added score popup '{popup.text}' at {popup.position}")

    def draw_score_popups(self) -> None:
        """
        Draw all active score popups onto the UI layer. 
        NO external argument is needed; we fetch the UI surface ourselves.
        """
        ui_surface = self.surfaces.get(Layer.UI)
        if not ui_surface:
            render_logger.error("No UI surface available for score popups.")
            return

        current_time = time.time()
        for popup in self.score_popups[:]:
            elapsed = current_time - popup.start_time
            if elapsed > popup.duration:
                self.score_popups.remove(popup)
                continue
            alpha = max(255 - int((elapsed / popup.duration) * 255), 0)
            text_surface = self.render_config.fonts['small'].render(
                popup.text,
                True,
                self.render_config.colors.UI_COLORS['highlight']
            )
            text_surface.set_alpha(alpha)
            ui_surface.blit(text_surface, popup.position)

    def trigger_particle_effect(self, position: Tuple[int, int]) -> None:
        """
        Spawn basic particle effects around grid cell (x,y).
        position is grid coords, so convert to approximate pixel coords.
        """
        if not self.particle_effects_enabled:
            return

        x_grid, y_grid = position
        px = x_grid * self.config.grid_size + self.config.grid_size
        py = y_grid * self.config.grid_size + self.config.grid_size
        for _ in range(20):
            vx = random.randint(-3, 3)
            vy = random.randint(-3, -1)
            lifetime = random.uniform(0.5, 1.0)
            self.particles.append({
                "position": [px, py],
                "velocity": [vx, vy],
                "lifetime": lifetime
            })
        render_logger.debug(f"Triggered particle effect at ({x_grid},{y_grid}).")

    def update_particles(self, delta_time: float) -> None:
        """
        Update active particles (movement + lifetime). Remove expired.
        """
        updated_list = []
        for p in self.particles:
            p["position"][0] += p["velocity"][0]
            p["position"][1] += p["velocity"][1]
            p["velocity"][1] += 9.81 * delta_time  # gravity
            p["lifetime"] -= delta_time
            if p["lifetime"] > 0:
                updated_list.append(p)
        self.particles = updated_list

    def draw_particles(self) -> None:
        """
        Draw all active particles onto the GAME layer.
        """
        game_surface = self.surfaces.get(Layer.GAME)
        if not game_surface:
            return

        for p in self.particles:
            alpha = int(255 * (p["lifetime"] / 1.0))
            color = (*self.particle_color, alpha)
            px = int(p["position"][0])
            py = int(p["position"][1])
            pygame.draw.circle(game_surface, color, (px, py), self.particle_size)

    def compose_frame(self,
                      screen: pygame.Surface,
                      frame_count: int) -> None:
        """
        Final step: Blit the layered surfaces in order (BACKGROUND->GAME->UI->DEBUG)
        onto the main display, then flip.
        """
        frame_start = time.perf_counter()
        try:
            if not screen:
                render_logger.error("No main screen surface provided to compose_frame.")
                return

            screen.fill(self.render_config.colors.UI_COLORS["background"])

            # Blit in order
            bg_surf = self.surfaces[Layer.BACKGROUND]
            if bg_surf:
                screen.blit(bg_surf, (0, 0))

            game_surf = self.surfaces[Layer.GAME]
            if game_surf:
                screen.blit(game_surf, (0, 0))

            ui_surf = self.surfaces[Layer.UI]
            if ui_surf:
                screen.blit(ui_surf, (self.config.screen_width, 0))

            debug_surf = self.surfaces[Layer.DEBUG]
            if debug_surf and self.render_config.debug_mode:
                screen.blit(debug_surf, (0, 0))

            frame_end = time.perf_counter()
            render_time = (frame_end - frame_start) * 1000.0
            self.last_render_time = render_time
            self.frame_times.append(render_time)

            # Log average FPS every 60 frames
            if frame_count % 60 == 0 and self.frame_times:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                fps = 1000.0 / avg_frame_time if avg_frame_time > 0 else 0.0
                render_logger.info(
                    f"Frame {frame_count}: Avg Frame Time: {avg_frame_time:.2f}ms, FPS: {fps:.1f}"
                )
                self.frame_times.clear()

            pygame.display.flip()
        except pygame.error as e:
            render_logger.error(f"Error composing frame: {e}")
        finally:
            self.last_frame_time = frame_end if "frame_end" in locals() else self.last_frame_time

    #
    # These methods forward UI tasks to UIManager, then handle popups internally
    #

    def draw_ui(self,
                score: int,
                high_score: int,
                level: int,
                lines: int,
                next_pieces: List[Tetrimino],
                held_piece: Optional[Tetrimino] = None,
                combo: int = 0) -> None:
        """
        Draw the entire UI (score, next/held, combos) onto the UI layer,
        then internally draw score popups.
        """
        ui_surface = self.surfaces.get(Layer.UI)
        if not ui_surface:
            render_logger.error("UI surface not available for draw_ui.")
            return

        # Clear the UI layer
        ui_surface.fill(self.render_config.colors.UI_COLORS['background'])

        # Let UIManager handle the scoreboard, next/held logic
        self.ui_manager.draw_game_ui(
            ui_surface,
            score,
            high_score,
            level,
            lines,
            next_pieces,
            held_piece,
            combo
        )

        # Now draw any score popups over that
        self.draw_score_popups()

        render_logger.debug("UI elements drawn successfully.")

    def draw_menu(self,
                  menu_options: List[str],
                  selected_index: int) -> None:
        """
        Draw a menu on the UI layer, highlight the selected option.
        """
        ui_surface = self.surfaces.get(Layer.UI)
        if not ui_surface:
            render_logger.error("UI surface not available for menu rendering.")
            return

        ui_surface.fill(self.render_config.colors.UI_COLORS['background'])
        self.ui_manager.draw_menu(ui_surface, menu_options, selected_index)

    def draw_pause_screen(self) -> None:
        """
        Draw a "PAUSED" message on the GAME layer.
        """
        game_surface = self.surfaces.get(Layer.GAME)
        if not game_surface or not self.render_config.fonts:
            render_logger.error("Cannot draw pause screen (missing surface or fonts).")
            return

        pause_text = self.render_config.fonts['large'].render(
            "PAUSED", True, self.render_config.colors.UI_COLORS['highlight']
        )
        pause_rect = pause_text.get_rect(
            center=(self.config.screen_width // 2, self.config.screen_height // 2)
        )
        game_surface.blit(pause_text, pause_rect)
        render_logger.debug("Pause screen drawn successfully.")

    def draw_game_over(self, score: int) -> None:
        """
        Draw a "Game Over" overlay on the GAME layer, with final score.
        """
        game_surface = self.surfaces.get(Layer.GAME)
        if not game_surface or not self.render_config.fonts:
            render_logger.error("Cannot draw game over screen (missing surface or fonts).")
            return

        overlay = pygame.Surface((self.config.screen_width, self.config.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))  # semi-transparent black

        game_over_text = self.render_config.fonts['large'].render(
            "GAME OVER", True, self.render_config.colors.UI_COLORS['highlight']
        )
        game_over_rect = game_over_text.get_rect(
            center=(self.config.screen_width // 2, self.config.screen_height // 3)
        )

        score_text = self.render_config.fonts['medium'].render(
            f"Score: {score:,}",
            True,
            self.render_config.colors.UI_COLORS['text']
        )
        score_rect = score_text.get_rect(
            center=(self.config.screen_width // 2, self.config.screen_height // 3 + 80)
        )

        restart_text = self.render_config.fonts['small'].render(
            "Press R to Restart or Q to Quit",
            True,
            self.render_config.colors.UI_COLORS['text']
        )
        restart_rect = restart_text.get_rect(
            center=(self.config.screen_width // 2, self.config.screen_height // 3 + 130)
        )

        overlay.blit(game_over_text, game_over_rect)
        overlay.blit(score_text, score_rect)
        overlay.blit(restart_text, restart_rect)

        game_surface.blit(overlay, (0, 0))

        # Optional "game_over" sound
        if self.render_config.sounds.get("game_over"):
            self.render_config.sounds["game_over"].play()

        render_logger.debug("Game Over screen drawn successfully.")
