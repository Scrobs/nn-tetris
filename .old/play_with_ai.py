#!/usr/bin/env python3
"""
play_with_ai.py: Enhanced interface for playing Tetris with trained AI model.
"""

import os
import sys
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import argparse

import pygame
import torch
import torch.nn as nn
import numpy as np

from tetris_game import TetrisEnv, GameConfig, GameState
from tetris_dqn import DQNTetris

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AIPlayer:
    """AI player interface with visualization and controls."""

    def __init__(self, model_path: str, config: Optional[GameConfig] = None):
        """Initialize AI player with model and game environment."""
        self.config = config or GameConfig()
        self.config.fps = 30  # Adjust FPS for better visualization

        # Initialize game environment
        self.env = TetrisEnv(self.config)

        # Load AI model
        self.model = self._load_model(model_path)
        self.device = next(self.model.parameters()).device

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((
            self.config.screen_width + 400,  # Extra space for AI visualization
            self.config.screen_height
        ))
        pygame.display.set_caption("Tetris AI Player")
        self.clock = pygame.time.Clock()

        # Font for rendering
        self.font = pygame.font.Font(None, 24)

        # Statistics
        self.stats = {
            'games_played': 0,
            'total_score': 0,
            'max_score': 0,
            'total_lines': 0,
            'max_lines': 0
        }

        # Control flags
        self.paused = False
        self.show_predictions = True
        self.ai_active = True
        self.speed_multiplier = 1

    def _load_model(self, model_path: str) -> nn.Module:
        """Load the trained model with error handling."""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Determine device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {device}")

            # Initialize model
            input_size = self.env.config.rows * self.env.config.columns + 4
            model = DQNTetris(input_size=input_size, output_size=4).to(device)

            # Load state dict
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()

            logger.info(f"Successfully loaded model from {model_path}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def get_ai_action(self, state: np.ndarray) -> int:
        """Get action from AI model with prediction visualization."""
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            action_probs = torch.softmax(q_values, dim=1)
            action = torch.argmax(q_values).item()

            if self.show_predictions:
                self._draw_predictions(action_probs[0].cpu().numpy())

            return action

    def _draw_predictions(self, action_probs: np.ndarray) -> None:
        """Visualize AI's action predictions."""
        actions = ['Move Left', 'Move Right', 'Rotate', 'Hard Drop']
        start_x = self.config.screen_width + 20
        start_y = 400

        # Draw prediction bars
        bar_width = 150
        bar_height = 20
        spacing = 30

        for i, (action, prob) in enumerate(zip(actions, action_probs)):
            # Draw action name
            text = self.font.render(f"{action}: {prob:.3f}", True, (255, 255, 255))
            self.screen.blit(text, (start_x, start_y + i * spacing))

            # Draw probability bar
            bar_rect = pygame.Rect(
                start_x + 200,
                start_y + i * spacing,
                int(bar_width * prob),
                bar_height
            )
            pygame.draw.rect(self.screen, (0, 255, 0), bar_rect)
            pygame.draw.rect(self.screen, (255, 255, 255),
                           (start_x + 200, start_y + i * spacing, bar_width, bar_height), 1)

    def _draw_stats(self) -> None:
        """Draw game statistics."""
        stats_x = self.config.screen_width + 20
        stats_y = 50
        spacing = 30

        stats_text = [
            f"Games Played: {self.stats['games_played']}",
            f"Current Score: {self.env.score}",
            f"Max Score: {self.stats['max_score']}",
            f"Current Lines: {self.env.lines_cleared}",
            f"Max Lines: {self.stats['max_lines']}",
            f"Average Score: {self.stats['total_score'] / max(1, self.stats['games_played']):.1f}"
        ]

        for i, text in enumerate(stats_text):
            surface = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(surface, (stats_x, stats_y + i * spacing))

    def _draw_controls(self) -> None:
        """Draw control information."""
        controls_x = self.config.screen_width + 20
        controls_y = 250
        spacing = 25

        controls = [
            "Controls:",
            "P - Pause/Unpause",
            "V - Toggle AI Predictions",
            "A - Toggle AI Control",
            "Up/Down - Adjust Speed",
            "R - Reset Game",
            "Esc - Quit"
        ]

        for i, text in enumerate(controls):
            surface = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(surface, (controls_x, controls_y + i * spacing))

    def _update_stats(self) -> None:
        """Update game statistics."""
        self.stats['games_played'] += 1
        self.stats['total_score'] += self.env.score
        self.stats['max_score'] = max(self.stats['max_score'], self.env.score)
        self.stats['total_lines'] += self.env.lines_cleared
        self.stats['max_lines'] = max(self.stats['max_lines'], self.env.lines_cleared)

    def run(self) -> None:
        """Main game loop."""
        try:
            running = True
            while running:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_p:
                            self.paused = not self.paused
                        elif event.key == pygame.K_v:
                            self.show_predictions = not self.show_predictions
                        elif event.key == pygame.K_a:
                            self.ai_active = not self.ai_active
                        elif event.key == pygame.K_r:
                            self._update_stats()
                            self.env.reset()
                        elif event.key == pygame.K_UP:
                            self.speed_multiplier = min(4, self.speed_multiplier + 0.5)
                        elif event.key == pygame.K_DOWN:
                            self.speed_multiplier = max(0.5, self.speed_multiplier - 0.5)

                if not self.paused:
                    # Get game state
                    state = self.env.get_state()

                    # Get action (either from AI or human)
                    action = None
                    if self.ai_active:
                        action = self.get_ai_action(state)
                    else:
                        action = self.env.handle_input()

                    # Execute action if available
                    if action is not None:
                        _, reward, done = self.env.step(action)

                        if done:
                            self._update_stats()
                            self.env.reset()

                # Draw game state
                self.env.draw(self.screen)
                self._draw_stats()
                self._draw_controls()
                pygame.display.flip()

                # Control game speed
                self.clock.tick(self.config.fps * self.speed_multiplier)

        except Exception as e:
            logger.error(f"Game crashed with error: {e}")
            raise
        finally:
            pygame.quit()

def parse_args() -> Dict[str, Any]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Play Tetris with trained AI")
    parser.add_argument('--model', type=str, default='models/best_tetris_dqn_model.pth',
                      help='Path to trained model file')
    parser.add_argument('--fps', type=int, default=30,
                      help='Game FPS')
    parser.add_argument('--ai-start', action='store_true',
                      help='Start with AI control enabled')
    return vars(parser.parse_args())

def main():
    """Main entry point."""
    try:
        args = parse_args()

        # Create game configuration
        config = GameConfig()
        config.fps = args['fps']

        # Initialize and run AI player
        player = AIPlayer(args['model'], config)
        player.ai_active = args['ai_start']
        player.run()

    except Exception as e:
        logger.error(f"Application failed with error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
