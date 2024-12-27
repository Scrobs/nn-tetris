# File: input/input_processor.py
import pygame
from typing import Set
from input.key_mapper import KeyMapper
from input.action_registry import ActionRegistry
from utils.logging_setup import setup_logging

loggers = setup_logging()
game_logger = loggers['game']

class InputProcessor:
    def __init__(self, key_mapper: KeyMapper, action_registry: ActionRegistry):
        """
        Initialize the InputProcessor.
        :param key_mapper: The KeyMapper instance for resolving actions.
        :param action_registry: The ActionRegistry instance for triggering callbacks.
        """
        self.key_mapper = key_mapper
        self.action_registry = action_registry
        self.pressed_keys: Set[int] = set()

    def process_event(self, event: pygame.event.Event) -> None:
        """
        Process a single input event.
        :param event: A Pygame event.
        """
        if event.type == pygame.KEYDOWN:
            self.pressed_keys.add(event.key)
            action = self.key_mapper.map_key_to_action(event.key)
            if action:
                callback = self.action_registry.get_callback(action)
                if callback:
                    try:
                        callback()
                        game_logger.info(f"Action triggered: {action}")
                    except Exception as e:
                        game_logger.error(f"Error executing callback for action '{action}': {e}")
        elif event.type == pygame.KEYUP:
            self.pressed_keys.discard(event.key)

    def update(self, delta_time: float) -> None:
        """
        Handle continuous input like long presses or key combinations.
        This should be called every frame with the time since the last frame.
        :param delta_time: Time in seconds since the last frame.
        """
        # Future implementation for advanced input features
        pass
