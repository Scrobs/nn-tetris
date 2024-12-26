# File: input/action_registry.py
from typing import Callable, Dict, Optional

class ActionRegistry:
    def __init__(self):
        """
        Initialize the ActionRegistry.
        """
        self.action_to_callback: Dict[str, Callable[[], None]] = {}

    def register_action(self, action: str, callback: Callable[[], None]) -> None:
        """
        Register an action with its corresponding callback.
        :param action: The name of the action.
        :param callback: The function to call when the action is triggered.
        """
        self.action_to_callback[action] = callback

    def get_callback(self, action: str) -> Optional[Callable[[], None]]:
        """
        Retrieve the callback for a given action.
        :param action: The name of the action.
        :return: The callback function, or None if not registered.
        """
        return self.action_to_callback.get(action)

    def validate_actions(self, required_actions: list) -> None:
        """
        Ensure all required actions have callbacks registered.
        :param required_actions: List of action names that must have callbacks.
        """
        for action in required_actions:
            if action not in self.action_to_callback:
                print(f"Warning: No callback registered for action '{action}'.")
