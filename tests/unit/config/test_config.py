#!/usr/bin/env python3

import unittest
from unittest.mock import patch, mock_open, MagicMock
import json
from config.game_config import GameConfig
from pathlib import Path


class TestGameConfig(unittest.TestCase):
    def setUp(self):
        """
        Setup the default configuration for testing.
        """
        self.default_config = {
            "screen_width": 300,
            "screen_height": 600,
            "grid_size": 30,
            "fps": 60,
            "vsync": True,
            "fixed_timestep": 0.016667,
            "max_frame_time": 0.25,
            "particle_pool_size": 1000,
            "fullscreen": False,
            "preview_pieces": 3,
            "gravity_delay": 1.0,
            "lines_per_level": 10,
            "debug_mode": False,
            "logging_level": "INFO",
            "key_bindings": {
                "MOVE_LEFT": "LEFT",
                "MOVE_RIGHT": "RIGHT",
                "SOFT_DROP": "DOWN",
            },
        }

    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps({"screen_width": 500, "screen_height": 800}))
    def test_load_config(self, mock_file):
        """
        Test loading a configuration from a JSON file.
        """
        config = GameConfig.load_config("mock_config.json")
        self.assertEqual(config.screen_width, 500)
        self.assertEqual(config.screen_height, 800)
        mock_file.assert_called_once_with("mock_config.json", "r")

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.rename")
    def test_save_config(self, mock_rename, mock_file):
        """
        Test saving a configuration to a JSON file.
        """
        config = GameConfig()
        config.save_config("mock_save_config.json")
        mock_file.assert_called_once_with("mock_save_config.json.tmp", "w")
        mock_rename.assert_called_once_with("mock_save_config.json")

    def test_apply_config(self):
        """
        Test applying a custom configuration.
        """
        # Ensure the screen_width and screen_height are multiples of grid_size
        custom_config = {"screen_width": 600, "screen_height": 900, "fps": 120}
        config = GameConfig(custom_config)
        self.assertEqual(config.screen_width, 600)
        self.assertEqual(config.screen_height, 900)
        self.assertEqual(config.fps, 120)



    def test_validate_config_success(self):
        """
        Test validation of a valid configuration.
        """
        config = GameConfig(self.default_config)
        try:
            config._validate_config()
        except ValueError:
            self.fail("Validation raised ValueError unexpectedly!")

    def test_validate_config_failure(self):
        """
        Test validation of an invalid configuration.
        """
        invalid_config = {"screen_width": -300, "screen_height": 600}
        with self.assertRaises(ValueError):
            GameConfig(invalid_config)

    def test_unknown_config_keys(self):
        """
        Test applying unknown configuration keys.
        """
        unknown_config = {"unknown_key": "value", "screen_width": 400}
        with self.assertLogs("game", level="WARNING") as log:
            config = GameConfig(unknown_config)
            self.assertIn("Unknown configuration key: unknown_key", log.output[0])
        self.assertEqual(config.screen_width, 400)

    def test_key_bindings(self):
        """
        Test applying custom key bindings.
        """
        custom_config = {"key_bindings": {"MOVE_LEFT": "A", "MOVE_RIGHT": "D"}}
        config = GameConfig(custom_config)
        self.assertEqual(config.key_bindings["MOVE_LEFT"], "A")
        self.assertEqual(config.key_bindings["MOVE_RIGHT"], "D")

    @patch("builtins.open", new_callable=mock_open, read_data="invalid_json")
    def test_load_invalid_json(self, mock_file):
        """
        Test loading an invalid JSON configuration.
        """
        with self.assertRaises(json.JSONDecodeError):
            GameConfig.load_config("invalid_config.json")

    @patch("builtins.open", new_callable=mock_open, side_effect=FileNotFoundError)
    def test_load_config_file_not_found(self, mock_file):
        """
        Test loading a configuration from a missing file.
        """
        with self.assertRaises(FileNotFoundError):
            GameConfig.load_config("non_existent_config.json")

    def test_validate_display_settings_failure(self):
        """
        Test validation of invalid display settings.
        """
        invalid_config = {"screen_width": 300, "grid_size": 40}
        config = GameConfig(invalid_config)
        with self.assertRaises(ValueError):
            config._validate_display_settings()

    def tearDown(self):
        """
        Clean up after tests.
        """
        pass


if __name__ == "__main__":
    unittest.main()
