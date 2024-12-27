"""
ui_manager.py
Centralized UI logic for Tetris:
- Score, level, lines, combos
- Next/held pieces (optionally scaled)
- Menu drawing
"""
import pygame
from typing import List, Optional
from config.game_config import GameConfig
from render.render_config import RenderConfig
from core.tetrimino import Tetrimino
from utils.logging_setup import setup_logging

loggers = setup_logging()
ui_logger = loggers['render']

class UIManager:
    """
    Handles the layout and drawing of all UI elements:
    - Score, high score, level, lines, combo
    - Next piece previews
    - Held piece
    - Menu
    """

    def __init__(self, config: GameConfig, render_config: RenderConfig):
        self.config = config
        self.render_config = render_config

        # Storing the final Y of the last next-piece to place "Hold" below it
        self._last_next_piece_y: Optional[int] = None

        # Adjust if your UI is narrow and you want to scale down the Next/Hold pieces
        self.preview_scale = 0.8

    def draw_game_ui(self,
                     ui_surface: pygame.Surface,
                     score: int,
                     high_score: int,
                     level: int,
                     lines: int,
                     next_pieces: List[Tetrimino],
                     held_piece: Optional[Tetrimino],
                     combo: int) -> None:
        """
        Draw all in-game UI elements:
          - Score, level, lines, combo
          - Next piece previews
          - Held piece
        """
        if not self.render_config.fonts:
            ui_logger.error("No fonts available for UI rendering.")
            return

        # 1) Score/Level/Lines
        self._draw_score_section(ui_surface, score, high_score, level, lines, combo)

        # 2) Next pieces
        self._draw_next_pieces(ui_surface, next_pieces)

        # 3) Held piece (if any)
        if held_piece is not None:
            self._draw_held_piece(ui_surface, held_piece)
        else:
            ui_logger.debug("No held piece to draw.")

    def draw_menu(self,
                  ui_surface: pygame.Surface,
                  menu_options: List[str],
                  selected_index: int) -> None:
        """
        Draw a vertical menu on the UI. The selected option is highlighted.
        """
        if not self.render_config.fonts or not ui_surface:
            ui_logger.error("No fonts or UI surface for menu.")
            return

        menu_font = self.render_config.fonts['large']

        start_y = 50
        spacing = 60

        for idx, option in enumerate(menu_options):
            color = self.render_config.colors.UI_COLORS['text']
            if idx == selected_index:
                color = self.render_config.colors.UI_COLORS['highlight']

            text_surface = menu_font.render(option, True, color)
            text_rect = text_surface.get_rect(topleft=(20, start_y + idx * spacing))

            # Draw arrow for selected
            if idx == selected_index:
                arrow_surface = self.render_config.fonts['small'].render(
                    "➤", True, self.render_config.colors.UI_COLORS['highlight']
                )
                ui_surface.blit(arrow_surface, (0, start_y + idx * spacing + 5))

            ui_surface.blit(text_surface, text_rect)

    #
    # Internal / Helper Methods
    #

    def _draw_score_section(self,
                            ui_surface: pygame.Surface,
                            score: int,
                            high_score: int,
                            level: int,
                            lines: int,
                            combo: int) -> None:
        """
        Show score, hi-score, level, lines, combo in a vertical column.
        """
        try:
            margin = 20
            spacing = 14
            y_start = margin

            lines_to_draw = [
                f"Score: {score:,}",
                f"HiScore: {high_score:,}",
                f"Level: {level}",
                f"Lines: {lines}"
            ]
            if combo > 0:
                lines_to_draw.append(f"Combo: x{combo}")

            for text_label in lines_to_draw:
                surf = self.render_config.fonts['small'].render(
                    text_label, True, self.render_config.colors.UI_COLORS['text']
                )
                ui_surface.blit(surf, (margin, y_start))
                y_start += surf.get_height() + spacing

        except Exception as e:
            ui_logger.error(f"Error drawing score section: {e}")

    def _draw_next_pieces(self,
                          ui_surface: pygame.Surface,
                          next_pieces: List[Tetrimino]) -> None:
        """
        Draw upcoming "Next" pieces, each scaled to fit inside a 4x4 region,
        then scaled by preview_scale if needed.
        """
        try:
            if not self.render_config.fonts:
                return

            label_font = self.render_config.fonts['small']
            label_surf = label_font.render("Next", True, self.render_config.colors.UI_COLORS['text'])

            next_label_y = 150
            ui_surface.blit(label_surf, (20, next_label_y))

            preview_y = next_label_y + 30
            base_size = 4 * self.config.grid_size

            for piece in next_pieces[:self.config.preview_pieces]:
                # Create a base surface for the 4x4 bounding box
                preview_surface = pygame.Surface((base_size, base_size), pygame.SRCALPHA)
                preview_surface.fill((0, 0, 0, 0))

                shape_width = len(piece.shape[0])
                shape_height = len(piece.shape)
                x_offset = (base_size - shape_width * self.config.grid_size) // 2
                y_offset = (base_size - shape_height * self.config.grid_size) // 2

                # Draw the piece in normal grid size
                for row_idx, row in enumerate(piece.shape):
                    for col_idx, cell in enumerate(row):
                        if cell:
                            rect = pygame.Rect(
                                x_offset + col_idx * self.config.grid_size + self.config.cell_padding,
                                y_offset + row_idx * self.config.grid_size + self.config.cell_padding,
                                self.config.grid_size - (2 * self.config.cell_padding),
                                self.config.grid_size - (2 * self.config.cell_padding)
                            )
                            color = self.render_config.colors.get_piece_color(piece.piece_type)
                            pygame.draw.rect(preview_surface, color, rect)

                # Scale if needed
                scaled_width = int(base_size * self.preview_scale)
                scaled_height = int(base_size * self.preview_scale)
                scaled_surface = pygame.transform.smoothscale(
                    preview_surface, (scaled_width, scaled_height)
                )

                # Blit the scaled surface
                ui_surface.blit(scaled_surface, (20, preview_y))
                preview_y += scaled_height + 10

            # Store final y position for "Hold"
            self._last_next_piece_y = preview_y

        except Exception as e:
            ui_logger.error(f"Error drawing next pieces: {e}")

    def _draw_held_piece(self,
                         ui_surface: pygame.Surface,
                         held_piece: Tetrimino) -> None:
        """
        Draw the held piece similarly, scaled if needed, below the last Next piece.
        """
        try:
            if not self.render_config.fonts:
                ui_logger.error("No fonts available for held piece.")
                return

            if self._last_next_piece_y is None:
                self._last_next_piece_y = 350

            label_surf = self.render_config.fonts['small'].render(
                "Hold", True, self.render_config.colors.UI_COLORS['text']
            )
            hold_label_y = self._last_next_piece_y + 10
            ui_surface.blit(label_surf, (20, hold_label_y))

            base_size = 4 * self.config.grid_size
            held_surface = pygame.Surface((base_size, base_size), pygame.SRCALPHA)
            held_surface.fill((0, 0, 0, 0))

            shape_width = len(held_piece.shape[0])
            shape_height = len(held_piece.shape)
            x_offset = (base_size - shape_width * self.config.grid_size) // 2
            y_offset = (base_size - shape_height * self.config.grid_size) // 2

            for row_idx, row in enumerate(held_piece.shape):
                for col_idx, cell in enumerate(row):
                    if cell:
                        rect = pygame.Rect(
                            x_offset + col_idx * self.config.grid_size + self.config.cell_padding,
                            y_offset + row_idx * self.config.grid_size + self.config.cell_padding,
                            self.config.grid_size - (2 * self.config.cell_padding),
                            self.config.grid_size - (2 * self.config.cell_padding)
                        )
                        color = self.render_config.colors.get_piece_color(held_piece.piece_type)
                        pygame.draw.rect(held_surface, color, rect)

            scaled_width = int(base_size * self.preview_scale)
            scaled_height = int(base_size * self.preview_scale)
            scaled_surface = pygame.transform.smoothscale(
                held_surface, (scaled_width, scaled_height)
            )

            hold_piece_y = hold_label_y + 30
            ui_surface.blit(scaled_surface, (20, hold_piece_y))

            ui_logger.debug(f"Drew held piece: {held_piece.piece_type}")
        except Exception as e:
            ui_logger.error(f"Error drawing held piece: {e}")
