#!/usr/bin/env python3

import pygame
from typing import Dict, Optional
from render.render_config import RenderConfig
from utils.logging_setup import setup_logging

# Setup logging
loggers = setup_logging()
game_logger = loggers['game']
resource_logger = loggers['resource']


class SoundManager:
    """Handles loading and playing sound effects."""

    def __init__(self, render_config: RenderConfig):
        """
        Initialize the SoundManager with a RenderConfig instance.

        Args:
            render_config (RenderConfig): Configuration containing sound data.
        """
        self.sounds = render_config.sounds

    def play_sound(self, sound_name: str) -> None:
        """
        Play a sound effect by name.

        Args:
            sound_name (str): The name of the sound to play.
        """
        sound = self.sounds.get(sound_name)
        if not sound:
            resource_logger.warning(
                "Sound '%s' not found or failed to load", sound_name
            )
            return

        try:
            sound.play()
            game_logger.debug("Played sound: %s", sound_name)
        except Exception as e:
            resource_logger.error(
                "Error playing sound '%s': %s", sound_name, str(e)
            )
