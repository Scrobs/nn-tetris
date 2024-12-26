# main.py

import pygame
import sys
from core.game import TetrisGame
from config.game_config import GameConfig
from utils.logging_setup import setup_logging

def main():
    loggers = setup_logging()
    main_logger = loggers['main']
    game_logger = loggers['game']
    main_logger.info("Starting Tetris game")
    pygame.init()
    try:
        pygame.mixer.init()
        game_logger.debug("Pygame mixer initialized")
    except pygame.error as e:
        game_logger.error("Failed to initialize Pygame mixer: %s", str(e))
    try:
        config = GameConfig.load_config('config/default_config.json')
        game_logger.info("Game configuration loaded")
    except FileNotFoundError as e:
        game_logger.error("Failed to load configuration from config/default_config.json: %s", str(e))
        game_logger.info("Using default configuration values.")
        config = GameConfig()
    except Exception as e:
        game_logger.error("Unexpected error loading configuration: %s", str(e))
        game_logger.info("Using default configuration values.")
        config = GameConfig()
    try:
        game = TetrisGame(config)
        game_logger.info("Game instance created successfully")
    except Exception as e:
        game_logger.critical("Fatal error in main: %s", str(e), exc_info=True)
        sys.exit(1)
    try:
        game.run()
    except Exception as e:
        game_logger.critical("Fatal error during game execution: %s", str(e), exc_info=True)
    finally:
        game.cleanup()
        main_logger.info("Tetris game exited successfully")

if __name__ == "__main__":
    main()
