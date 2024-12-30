#!/usr/bin/env python3
"""
Unit tests for the Colors class and color constants.
"""

import pytest
from typing import Tuple
from render.colors import Colors, Color, ColorAlpha
from render.colors import (
    BLACK, WHITE, GRAY, DARK_GRAY, RED, GREEN, BLUE,
    YELLOW, PURPLE, ORANGE, CYAN, MAGENTA,
    FLASH_COLOR, NORMAL_COLOR
)


def is_valid_color(color: Color) -> bool:
    """Helper function to validate color tuples."""
    return (
        isinstance(color, tuple) and
        len(color) == 3 and
        all(isinstance(c, int) and 0 <= c <= 255 for c in color)
    )


def is_valid_color_alpha(color: ColorAlpha) -> bool:
    """Helper function to validate color tuples with alpha."""
    return (
        isinstance(color, tuple) and
        len(color) == 4 and
        all(isinstance(c, int) and 0 <= c <= 255 for c in color)
    )


class TestColorConstants:
    """Test suite for color constants."""

    @pytest.mark.parametrize("color,name", [
        (BLACK, "BLACK"),
        (WHITE, "WHITE"),
        (GRAY, "GRAY"),
        (DARK_GRAY, "DARK_GRAY"),
        (RED, "RED"),
        (GREEN, "GREEN"),
        (BLUE, "BLUE"),
        (YELLOW, "YELLOW"),
        (PURPLE, "PURPLE"),
        (ORANGE, "ORANGE"),
        (CYAN, "CYAN"),
        (MAGENTA, "MAGENTA"),
        (FLASH_COLOR, "FLASH_COLOR"),
        (NORMAL_COLOR, "NORMAL_COLOR")
    ])
    def test_color_constants(self, color: Color, name: str):
        """Test that all color constants are valid RGB tuples."""
        assert is_valid_color(color), f"Invalid color format for {name}: {color}"

    def test_color_uniqueness(self):
        """Test that all color constants are unique."""
        colors = [
            BLACK, WHITE, GRAY, DARK_GRAY, RED, GREEN, BLUE,
            YELLOW, PURPLE, ORANGE, CYAN, MAGENTA,
            FLASH_COLOR, NORMAL_COLOR
        ]
        assert len(colors) == len(set(colors)), "Duplicate colors found"

    @pytest.mark.parametrize("color", [
        BLACK, WHITE, GRAY, DARK_GRAY, RED, GREEN, BLUE,
        YELLOW, PURPLE, ORANGE, CYAN, MAGENTA
    ])
    def test_color_immutability(self, color: Color):
        """Test that color tuples are immutable."""
        with pytest.raises((AttributeError, TypeError)):
            color[0] = 0


class TestColors:
    """Test suite for the Colors class."""

    @pytest.fixture
    def colors(self):
        """Fixture providing a Colors instance."""
        return Colors()

    def test_piece_colors_mapping(self, colors):
        """Test piece color mappings."""
        expected_pieces = {'I', 'O', 'T', 'S', 'Z', 'J', 'L'}
        assert set(colors.PIECE_COLORS.keys()) == expected_pieces

    def test_ui_colors_mapping(self, colors):
        """Test UI color mappings."""
        expected_ui_elements = {
            'background',
            'text',
            'highlight',
            'grid_lines',
            'shadow'
        }
        assert set(colors.UI_COLORS.keys()) == expected_ui_elements

    @pytest.mark.parametrize("piece_type", ['I', 'O', 'T', 'S', 'Z', 'J', 'L'])
    def test_get_piece_color(self, colors, piece_type):
        """Test color retrieval for each piece type."""
        color = colors.get_piece_color(piece_type)
        assert is_valid_color(color), f"Invalid color for piece {piece_type}"
        assert color == colors.PIECE_COLORS[piece_type]

    def test_get_piece_color_invalid(self, colors):
        """Test color retrieval for invalid piece type."""
        color = colors.get_piece_color('INVALID')
        assert color == WHITE, "Invalid piece type should return WHITE"

    @pytest.mark.parametrize("piece_type,expected_color", [
        ('I', CYAN),
        ('O', YELLOW),
        ('T', PURPLE),
        ('S', GREEN),
        ('Z', RED),
        ('J', BLUE),
        ('L', ORANGE)
    ])
    def test_piece_specific_colors(self, colors, piece_type, expected_color):
        """Test specific color assignments for each piece type."""
        assert colors.get_piece_color(piece_type) == expected_color

    def test_ui_color_values(self, colors):
        """Test UI color values."""
        assert colors.UI_COLORS['background'] == BLACK
        assert colors.UI_COLORS['text'] == WHITE
        assert colors.UI_COLORS['highlight'] == YELLOW
        assert colors.UI_COLORS['grid_lines'] == GRAY
        assert colors.UI_COLORS['shadow'] == DARK_GRAY

    def test_flash_color_constant(self, colors):
        """Test flash color constant."""
        assert colors.FLASH_COLOR == FLASH_COLOR
        assert is_valid_color(colors.FLASH_COLOR)

    def test_normal_color_constant(self, colors):
        """Test normal color constant."""
        assert colors.NORMAL_COLOR == NORMAL_COLOR
        assert is_valid_color(colors.NORMAL_COLOR)

    @pytest.mark.parametrize("ui_element", [
        'background', 'text', 'highlight', 'grid_lines', 'shadow'
    ])
    def test_ui_color_validity(self, colors, ui_element):
        """Test that all UI colors are valid RGB tuples."""
        color = colors.UI_COLORS[ui_element]
        assert is_valid_color(color), f"Invalid color for UI element {ui_element}"

    def test_color_instance_independence(self):
        """Test that different Colors instances are independent."""
        colors1 = Colors()
        colors2 = Colors()
        assert colors1.PIECE_COLORS is not colors2.PIECE_COLORS
        assert colors1.UI_COLORS is not colors2.UI_COLORS

    def test_color_consistency(self, colors):
        """Test that colors remain consistent when accessed multiple times."""
        color1 = colors.get_piece_color('I')
        color2 = colors.get_piece_color('I')
        assert color1 == color2
        assert color1 is color2  # Should be the same tuple object


if __name__ == "__main__":
    pytest.main(["-v", __file__])