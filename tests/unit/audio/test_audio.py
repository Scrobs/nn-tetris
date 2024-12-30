#!/usr/bin/env python3

import unittest
from unittest.mock import MagicMock, patch
from audio.sound_manager import SoundManager
from render.render_config import RenderConfig


class TestSoundManager(unittest.TestCase):
    def setUp(self):
        # Mock RenderConfig and sounds dictionary
        self.mock_render_config = MagicMock(spec=RenderConfig)
        self.mock_sounds = {
            "test_sound": MagicMock(),
            "error_sound": MagicMock(),
        }
        self.mock_render_config.sounds = self.mock_sounds

        # Initialize SoundManager with mocked RenderConfig
        self.sound_manager = SoundManager(self.mock_render_config)

    @patch("audio.sound_manager.game_logger")
    @patch("audio.sound_manager.resource_logger")
    def test_play_existing_sound(self, mock_resource_logger, mock_game_logger):
        """Test playing an existing sound."""
        sound_name = "test_sound"
        self.sound_manager.play_sound(sound_name)

        # Ensure the sound was played
        self.mock_sounds[sound_name].play.assert_called_once()
        # Ensure the debug log was called
        mock_game_logger.debug.assert_called_with("Played sound: %s", sound_name)

    @patch("audio.sound_manager.game_logger")
    @patch("audio.sound_manager.resource_logger")
    def test_play_missing_sound(self, mock_resource_logger, mock_game_logger):
        """Test attempting to play a sound that does not exist."""
        sound_name = "non_existent_sound"
        self.sound_manager.play_sound(sound_name)

        # Ensure no sound was played because it does not exist
        mock_resource_logger.warning.assert_called_with(
            "Sound '%s' not found or failed to load", sound_name
        )

    @patch("audio.sound_manager.game_logger")
    @patch("audio.sound_manager.resource_logger")
    def test_play_sound_with_error(self, mock_resource_logger, mock_game_logger):
        """Test playing a sound that raises an exception."""
        sound_name = "error_sound"
        self.mock_sounds[sound_name].play.side_effect = Exception("Sound play error")
        
        self.sound_manager.play_sound(sound_name)

        # Ensure the error was logged
        mock_resource_logger.error.assert_called_with(
            "Error playing sound '%s': %s", sound_name, "Sound play error"
        )

    def tearDown(self):
        # Cleanup mock data
        self.mock_render_config = None
        self.mock_sounds = None
        self.sound_manager = None


if __name__ == "__main__":
    unittest.main()
