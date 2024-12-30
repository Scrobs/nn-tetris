#!/usr/bin/env python3
"""
File: main_ai.py
AI-driven Tetris game implementation using DQN (Deep Q-Network).
"""

import os
import sys
import time
import numpy as np
import pygame
import tensorflow as tf
from typing import Tuple, Optional
from pathlib import Path

from core.game import TetrisGame
from core.game_state import GameState
from config.game_config import GameConfig
from utils.logging_setup import setup_logging
from utils.resource_checker import verify_resources
from ai.dqn_agent import DQNAgent

class AITetrisGame(TetrisGame):
    """AI-controlled version of Tetris using Deep Q-Learning."""
    
    def __init__(self, config: GameConfig, loggers):
        """
        Initialize the AI Tetris game.
        
        Args:
            config (GameConfig): Game configuration settings
            loggers: Dictionary of logging instances
        """
        # Force TensorFlow to use CPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        # Disable particles and sounds during training
        config.particle_effects = False
        
        # Initialize Pygame without sound
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
        
        super().__init__(config)
        
        # Explicitly empty the sounds dictionary in render config
        if hasattr(self, 'render_config'):
            self.render_config.sounds = {}
        
        # Set up logging
        self.game_logger = loggers['game']
        self.game_logger.info("Initializing AI Tetris game")
        
        # Initialize DQN agent with optimized parameters
        self.agent = DQNAgent(
            state_shape=(config.rows, config.columns, 1),
            action_size=6,  # left, right, rotate_cw, rotate_ccw, soft_drop, hard_drop
            learning_rate=0.001,
            gamma=0.95,       # Slightly reduced to focus on immediate rewards
            epsilon=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.997,  # Slower decay
            batch_size=32,        # Smaller batch for faster training
            memory_size=10000     # Reduced memory size
        )
        
        # Training metrics
        self.episode = 0
        self.best_score = 0
        self.total_rewards = 0
        self.frame_counter = 0
        
        # Set up logging directory
        self.log_dir = Path('logs/tetris_ai_training')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up TensorBoard logging
        self.tensorboard_writer = tf.summary.create_file_writer(str(self.log_dir))
        
        # Performance monitoring
        self.clock = pygame.time.Clock()
        
        # Save initial screen reference
        self.screen = pygame.display.get_surface()
        
        self.game_logger.info("AI Tetris game initialized successfully with particles and sounds disabled")

    def get_state(self) -> np.ndarray:
        """
        Generate the current game state for the AI agent.
        
        Returns:
            np.ndarray: The current grid state as a NumPy array (rows x columns x 1)
        """
        try:
            state = np.zeros((self.config.rows, self.config.columns, 1), dtype=np.float32)
            
            # Set occupied cells to 1
            for y in range(self.config.rows):
                for x in range(self.config.columns):
                    if self.grid.grid[y][x] is not None:
                        state[y, x, 0] = 1
                        
            return state
            
        except Exception as e:
            self.game_logger.error(f"Error generating game state: {e}")
            raise

    def calculate_reward(self, lines_cleared: int, is_game_over: bool) -> float:
        """
        Calculate the reward for the current action.
        
        Args:
            lines_cleared (int): Number of lines cleared in this step
            is_game_over (bool): Whether the game is over
            
        Returns:
            float: The calculated reward
        """
        try:
            reward = 0.0
            
            # Large penalty for game over
            if is_game_over:
                reward -= 50.0
                return reward
            
            # Rewards for clearing lines (exponential scaling)
            if lines_cleared > 0:
                reward += (2 ** lines_cleared) * 10  # Much higher rewards for lines
            
            # Height penalty
            max_height = 0
            holes = 0
            heights = [0] * self.config.columns
            
            # Calculate column heights and holes
            for col in range(self.config.columns):
                for row in range(self.config.rows):
                    if self.grid.grid[row][col] is not None:
                        heights[col] = self.config.rows - row
                        break
                
                max_height = max(max_height, heights[col])
                
                # Count holes in this column
                for row in range(self.config.rows - heights[col], self.config.rows):
                    if self.grid.grid[row][col] is None:
                        holes += 1
            
            # Penalty for high blocks and holes
            reward -= (max_height * 0.5)  # Height penalty
            reward -= (holes * 2.0)       # Holes penalty
            
            # Small positive reward for surviving
            reward += 0.1
            
            return reward
            
        except Exception as e:
            self.game_logger.error(f"Error calculating reward: {e}")
            return 0.0

    def execute_action(self, action: int) -> None:
        """
        Execute the chosen action in the game.
        
        Args:
            action (int): The action index to execute
        """
        try:
            if self.current_piece is None:
                return
                
            # Map action index to game actions
            if action == 0:  # Move left
                self.move_left()
            elif action == 1:  # Move right
                self.move_right()
            elif action == 2:  # Rotate clockwise
                self.rotate_cw()
            elif action == 3:  # Rotate counter-clockwise
                self.rotate_ccw()
            elif action == 4:  # Soft drop
                self.soft_drop()
            elif action == 5:  # Hard drop
                self.hard_drop()
                
        except Exception as e:
            self.game_logger.error(f"Error executing action {action}: {e}")

    def update_display(self) -> None:
        """Update the game display."""
        if hasattr(self, 'renderer'):
            self.renderer.clear_surfaces()
            self.renderer.draw_grid(self.grid)
            
            if self.current_piece:
                ghost_y = self.current_piece.get_ghost_position(self.grid)
                self.renderer.draw_piece(self.current_piece, ghost=True, override_y=ghost_y)
                self.renderer.draw_piece(self.current_piece)
            
            self.renderer.update_particles(self.fixed_timestep)
            self.renderer.draw_particles()
            self.renderer.draw_ui(
                score=self.score_handler.value,
                high_score=self.score_handler.high_score,
                level=self.level,
                lines=self.lines_cleared,
                next_pieces=self.next_pieces,
                held_piece=self.held_piece,
                combo=self.combo_counter
            )
            
            if self.state == GameState.PAUSED:
                self.renderer.draw_pause_screen()
            elif self.state == GameState.GAME_OVER:
                self.renderer.draw_game_over(self.score_handler.value)
            
            if self.config.debug_mode:
                metrics = self.performance_monitor.get_metrics()
                self.renderer.draw_debug_info(
                    frame_counter=self.frame_counter,
                    fps=metrics.fps
                )
            
            self.renderer.compose_frame(
                self.screen,
                self.frame_counter
            )

    def run_episode(self) -> Tuple[int, float]:
        """
        Run a single episode of the game.
        
        Returns:
            Tuple[int, float]: (score achieved, total reward)
        """
        try:
            self.reset()
            episode_reward = 0.0
            state = self.get_state()
            
            # Time management
            last_frame_time = time.perf_counter()
            game_timer = 0.0
            training_timer = 0.0
            TRAINING_INTERVAL = 0.1  # Train every 100ms instead of every frame
            
            while True:
                # Calculate delta time
                current_time = time.perf_counter()
                dt = min(current_time - last_frame_time, 0.1)  # Cap at 100ms
                last_frame_time = current_time
                
                game_timer += dt
                training_timer += dt
                
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.cleanup()
                        return self.score_handler.value, episode_reward
                
                # Get agent's action
                action = self.agent.act(state)
                
                # Execute action and get new state
                prev_lines = self.lines_cleared
                self.execute_action(action)
                next_state = self.get_state()
                
                # Calculate reward
                lines_cleared = self.lines_cleared - prev_lines
                is_game_over = self.state == GameState.GAME_OVER
                reward = self.calculate_reward(lines_cleared, is_game_over)
                episode_reward += reward
                
                # Store experience
                self.agent.remember(state, action, reward, next_state, is_game_over)
                
                # Train periodically rather than every frame
                if training_timer >= TRAINING_INTERVAL and len(self.agent.memory) >= self.agent.batch_size:
                    self.agent.replay()
                    training_timer = 0
                
                if is_game_over:
                    self.update_display()
                    return self.score_handler.value, episode_reward
                
                # Natural game speed control
                if game_timer >= self.config.gravity_delay:
                    if not self._apply_gravity():
                        self._lock_piece()
                        self._clear_lines()
                        if not self._spawn_new_piece():
                            self.update_display()
                            return self.score_handler.value, episode_reward
                    game_timer = 0.0
                
                # Update state
                state = next_state
                
                # Update display at capped frame rate
                self.update_display()
                self.clock.tick(30)  # Cap at 30 FPS during training
            
        except Exception as e:
            self.game_logger.error(f"Error in episode: {e}", exc_info=True)
            return 0, 0.0

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if hasattr(self, 'score_handler'):
                self.score_handler.save_high_score()
            # Don't quit Pygame here, as it's handled by parent class
        except Exception as e:
            self.game_logger.error(f"Error during cleanup: {e}")

    def _apply_gravity(self) -> bool:
        """
        Apply gravity to the current piece.
        
        Returns:
            bool: True if the piece moved down, False otherwise.
        """
        try:
            if not self.current_piece:
                return False
            
            if self.grid.can_move(
                self.current_piece,
                self.current_piece.x,
                self.current_piece.y + 1
            ):
                self.current_piece.y += 1
                return True
            return False
            
        except Exception as e:
            self.game_logger.error(f"Error applying gravity: {e}", exc_info=True)
            return False

    def run_training(self, episodes: int = 1000) -> None:
        """
        Run the training loop for specified number of episodes.
        
        Args:
            episodes (int): Number of episodes to train for
        """
        try:
            self.game_logger.info(f"Starting training for {episodes} episodes")
            
            checkpoint_dir = Path('models')
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            for episode in range(1, episodes + 1):
                self.episode = episode
                episode_start = time.time()
                
                score, reward = self.run_episode()
                episode_duration = time.time() - episode_start
                
                # Update best score
                self.best_score = max(self.best_score, score)
                self.total_rewards += reward
                
                # Log metrics
                with self.tensorboard_writer.as_default():
                    tf.summary.scalar('episode_score', score, step=episode)
                    tf.summary.scalar('episode_reward', reward, step=episode)
                    tf.summary.scalar('epsilon', self.agent.epsilon, step=episode)
                
                # Periodic progress logging
                if episode % 10 == 0:
                    self.game_logger.info(
                        f"Episode {episode}/{episodes} - "
                        f"Score: {score:,d} | "
                        f"Best: {self.best_score:,d} | "
                        f"Avg Reward: {self.total_rewards/episode:.2f} | "
                        f"Epsilon: {self.agent.epsilon:.3f} | "
                        f"Duration: {episode_duration:.2f}s"
                    )
                
                # Save model checkpoints
                if episode % 100 == 0:
                    model_path = checkpoint_dir / f'tetris_model_{episode}.keras'
                    self.agent.save_model(str(model_path))
                    self.game_logger.info(f"Model checkpoint saved: {model_path}")
                
                # Update target network periodically
                if episode % 10 == 0:
                    self.agent.update_target_model()
                
        except KeyboardInterrupt:
            self.game_logger.warning("Training interrupted by user")
        except Exception as e:
            self.game_logger.critical(f"Training failed: {e}", exc_info=True)
        finally:
            # Save final model
            try:
                final_model_path = checkpoint_dir / 'final_model.keras'
                self.agent.save_model(str(final_model_path))
                self.game_logger.info(f"Final model saved: {final_model_path}")
            except Exception as e:
                self.game_logger.error(f"Failed to save final model: {e}")
            
            self.cleanup()

def main():
    """Main entry point for the AI Tetris game."""
    try:
        # Load configuration
        try:
            config = GameConfig.load_config('config/default_config.json')
        except FileNotFoundError as e:
            print(f"Configuration file not found: {e}")
            config = GameConfig()
        except Exception as e:
            print(f"Error loading configuration: {e}")
            config = GameConfig()

        # Setup logging
        loggers = setup_logging(debug_mode=config.debug_mode)
        main_logger = loggers['main']
        
        # Verify required resources
        required_fonts = ['fonts/PressStart2P-Regular.ttf']
        if not verify_resources(required_fonts, 'resources'):
            main_logger.error("Required resources are missing")
            sys.exit(1)
            
        # Initialize Pygame
        main_logger.info("Starting AI-driven Tetris game")
        pygame.init()
        
        try:
            pygame.mixer.init()
        except pygame.error as e:
            main_logger.warning(f"Pygame mixer initialization failed: {e}")
            
        # Create and run game
        try:
            ai_game = AITetrisGame(config, loggers)
            ai_game.run_training(episodes=1000)
        except Exception as e:
            main_logger.critical(f"Fatal error in game execution: {e}", exc_info=True)
            raise
            
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()