# File: audio/sound_manager.py
import pygame
from typing import Dict, Optional
from render.render_config import RenderConfig
from utils.logging_setup import setup_logging

loggers = setup_logging()
game_logger = loggers['game']
resource_logger = loggers['resource']

class SoundManager:
    """Handles loading and playing sound effects."""
    def __init__(self, render_config: RenderConfig):
        self.sounds = render_config.sounds

    def play_sound(self, sound_name: str) -> None:
        """Play a sound effect by name."""
        sound = self.sounds.get(sound_name)
        if sound:
            sound.play()
            game_logger.debug("Played sound: %s", sound_name)
        else:
            resource_logger.warning("Sound '%s' not found or failed to load", sound_name)
