#!/usr/bin/env python3
"""
tetris_dqn.py: Improved DQN implementation for Tetris with proper error handling,
memory management, and training visualization.
"""

import os
import logging
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tetris_game import TetrisEnv, GameConfig, GameState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for DQN training parameters."""
    episodes: int = 1000
    batch_size: int = 64
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.995
    learning_rate: float = 0.001
    target_update_frequency: int = 10
    evaluation_frequency: int = 50
    replay_buffer_size: int = 10000
    save_dir: str = "models"
    checkpoint_frequency: int = 100

class PrioritizedReplayBuffer:
    """Improved Prioritized Experience Replay Buffer with proper memory management."""
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: List[Tuple] = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self._max_priority = 1.0

    def add(self, experience: Tuple, td_error: Optional[float] = None) -> None:
        """Add experience to buffer with thread-safe operations."""
        priority = self._max_priority if td_error is None else abs(td_error)

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities[len(self.buffer) - 1] = priority
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority

        self.position = (self.position + 1) % self.capacity
        self._max_priority = max(self._max_priority, priority)

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List, np.ndarray, torch.Tensor]:
        """Sample batch with importance sampling."""
        buffer_len = len(self.buffer)
        if buffer_len == 0:
            raise ValueError("Cannot sample from an empty buffer")

        priorities = self.priorities[:buffer_len]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(buffer_len, batch_size, p=probabilities)
        experiences = [self.buffer[idx] for idx in indices]

        weights = (buffer_len * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return experiences, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities for sampled experiences."""
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error)
            self.priorities[idx] = priority
            self._max_priority = max(self._max_priority, priority)

    def __len__(self) -> int:
        return len(self.buffer)

class DQNTetris(nn.Module):
    """Improved DQN architecture with proper initialization."""
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )

        # Initialize weights using Xavier initialization
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class TetrisTrainer:
    """Main trainer class with proper error handling and visualization."""
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create save directory
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)

        # Initialize environment and networks
        self.env = TetrisEnv()
        self.input_size = self.env.config.rows * self.env.config.columns + 4
        self.output_size = 4

        self.policy_net = DQNTetris(self.input_size, self.output_size).to(self.device)
        self.target_net = DQNTetris(self.input_size, self.output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
        self.replay_buffer = PrioritizedReplayBuffer(config.replay_buffer_size)

        # Initialize visualization
        pygame.init()
        self.screen = pygame.display.set_mode((
            self.env.config.screen_width + 200,
            self.env.config.screen_height
        ))
        pygame.display.set_caption("Tetris AI Training")
        self.clock = pygame.time.Clock()

        # Initialize tensorboard
        self.writer = SummaryWriter(log_dir="runs/tetris_training")

        # Training metrics
        self.best_eval_reward = float('-inf')
        self.episode_rewards = []

    def save_checkpoint(self, episode: int, is_best: bool = False) -> None:
        try:
            checkpoint = {
                'episode': episode,
                'policy_state_dict': self.policy_net.state_dict(),
                'target_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_eval_reward': self.best_eval_reward
            }

            checkpoint_path = os.path.join(self.config.save_dir, f'checkpoint_ep{episode}.pth')
            torch.save(checkpoint, checkpoint_path)

            if is_best:
                best_path = os.path.join(self.config.save_dir, 'best_model.pth')
                torch.save(checkpoint, best_path)

            logger.info(f"Saved checkpoint at episode {episode}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_eval_reward = checkpoint['best_eval_reward']
            return checkpoint['episode']
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    def choose_action(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.output_size - 1)

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            return self.policy_net(state_tensor).argmax().item()

    def train_step(self, beta: float = 0.4) -> Optional[float]:
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        try:
            batch, indices, weights = self.replay_buffer.sample(self.config.batch_size, beta)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
            actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
            weights = weights.to(self.device)

            current_q_values = self.policy_net(states).gather(1, actions)

            with torch.no_grad():
                next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
                target_q_values = rewards + (1 - dones) * self.config.gamma * next_q_values

            td_errors = (current_q_values - target_q_values).detach().cpu().numpy()
            loss = (nn.MSELoss(reduction='none')(current_q_values, target_q_values) * weights).mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.optimizer.step()

            self.replay_buffer.update_priorities(indices, td_errors)

            return loss.item()
        except Exception as e:
            logger.error(f"Error during training step: {e}")
            return None

    def evaluate(self, num_episodes: int = 10) -> float:
        total_rewards = []
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                    action = self.policy_net(state_tensor).argmax().item()
                state, reward, done = self.env.step(action)
                episode_reward += reward

            total_rewards.append(episode_reward)

        return sum(total_rewards) / len(total_rewards)

    def train(self) -> None:
        try:
            epsilon = self.config.epsilon_start

            for episode in range(self.config.episodes):
                state = self.env.reset()
                total_reward = 0
                losses = []

                while True:
                    action = self.choose_action(state, epsilon)
                    next_state, reward, done = self.env.step(action)
                    total_reward += reward

                    self.replay_buffer.add((state, action, reward, next_state, done))
                    state = next_state

                    if loss := self.train_step():
                        losses.append(loss)

                    self.env.draw(self.screen)
                    pygame.display.flip()
                    self.clock.tick(30)

                    if done:
                        break

                epsilon = max(self.config.epsilon_end, epsilon * self.config.epsilon_decay)

                self.episode_rewards.append(total_reward)
                avg_reward = sum(self.episode_rewards[-100:]) / len(self.episode_rewards[-100:])
                avg_loss = sum(losses) / len(losses) if losses else 0

                self.writer.add_scalar('Training/Average_Reward', avg_reward, episode)
                self.writer.add_scalar('Training/Epsilon', epsilon, episode)
                self.writer.add_scalar('Training/Loss', avg_loss, episode)

                if episode % self.config.evaluation_frequency == 0:
                    eval_reward = self.evaluate()
                    self.writer.add_scalar('Evaluation/Reward', eval_reward, episode)

                    if eval_reward > self.best_eval_reward:
                        self.best_eval_reward = eval_reward
                        self.save_checkpoint(episode, is_best=True)
                        logger.info(f"New best model at episode {episode} with reward: {eval_reward:.2f}")

                if episode % self.config.target_update_frequency == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                if episode % self.config.checkpoint_frequency == 0:
                    self.save_checkpoint(episode)

                if episode % 10 == 0:
                    logger.info(
                        f"Episode {episode}/{self.config.episodes} - "
                        f"Avg Reward: {avg_reward:.2f}, "
                        f"Epsilon: {epsilon:.3f}, "
                        f"Loss: {avg_loss:.6f}"
                    )

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self.save_checkpoint(episode, is_best=False)
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        try:
            pygame.quit()
            self.writer.close()
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Cleanup failed with error: {e}")

def parse_arguments() -> Dict[str, Any]:
    import argparse
    parser = argparse.ArgumentParser(description='Train DQN agent for Tetris')

    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of episodes to train')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--load-checkpoint', type=str, default=None,
                      help='Path to checkpoint to load')
    parser.add_argument('--no-visualization', action='store_true',
                      help='Disable visualization during training')

    return vars(parser.parse_args())

def main() -> None:
    """Main entry point."""
    try:
        # Parse arguments and create config
        args = parse_arguments()
        config = TrainingConfig(
            episodes=args['episodes'],
            batch_size=args['batch_size'],
            learning_rate=args['learning_rate'],
            save_dir=args['save_dir']
        )

        # Set up logging configuration
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"training_{int(time.time())}.log")
        )
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)

        # Initialize trainer
        trainer = TetrisTrainer(config)

        # Load checkpoint if specified
        if args['load_checkpoint']:
            start_episode = trainer.load_checkpoint(args['load_checkpoint'])
            logger.info(f"Resumed training from episode {start_episode}")

        # Start training
        logger.info("Starting training with configuration:")
        for key, value in vars(config).items():
            logger.info(f"{key}: {value}")

        trainer.train()

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    finally:
        logger.info("Training completed")

if __name__ == "__main__":
    main()
