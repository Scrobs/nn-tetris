import pygame

from input.key_mapper import KeyMapper
from input.action_registry import ActionRegistry
from input.input_processor import InputProcessor
from typing import Callable, Dict
from utils.logging_setup import setup_logging

loggers = setup_logging()
game_logger = loggers['game']

class InputHandler:
    """Handles user input using the modular Input Handling System."""
    def __init__(self, config_file: str, callbacks: Dict[str, Callable[[], None]]):
        """
        Initialize the InputHandler with configuration and callback functions.
        :param config_file: Path to the JSON configuration file for key mappings.
        :param callbacks: A dictionary mapping action names to callback functions.
        """
        self.key_mapper = KeyMapper(config_file)
        self.action_registry = ActionRegistry()
        for action, callback in callbacks.items():
            self.action_registry.register_action(action, callback)
        # Validate all required actions have callbacks
        required_actions = list(callbacks.keys())
        self.action_registry.validate_actions(required_actions)
        self.input_processor = InputProcessor(self.key_mapper, self.action_registry)

    def handle_event(self, event: pygame.event.Event) -> None:
        """
        Handle a single Pygame event.
        :param event: A Pygame event.
        """
        self.input_processor.process_event(event)

    def update(self, delta_time: float) -> None:
        """
        Update the input processor, handling continuous inputs.
        :param delta_time: Time in seconds since the last frame.
        """
        self.input_processor.update(delta_time)
