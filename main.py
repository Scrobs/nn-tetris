# File: main.py

import pygame
import sys
from core.game import TetrisGame
from config.game_config import GameConfig
from utils.logging_setup import setup_logging
from utils.resource_checker import verify_resources

def main():
    try:
        config = GameConfig.load_config('config/default_config.json')
    except FileNotFoundError as e:
        print(f"Configuration file not found: {e}")
        config = GameConfig()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        config = GameConfig()

    loggers = setup_logging(debug_mode=config.debug_mode)
    main_logger = loggers['main']
    game_logger = loggers['game']
    resource_logger = loggers['resource']

    required_fonts = [
        'fonts/PressStart2P-Regular.ttf',
    ]
    required_sounds = [
        'sounds/move.wav',
        'sounds/rotate.wav',
        'sounds/clear.wav',
        'sounds/hold.wav',
        'sounds/drop.wav',
        'sounds/game_over.wav',
    ]
    required_resources = required_fonts + required_sounds
    base_resource_dir = 'resources'
    resources_ok = verify_resources(required_resources, base_resource_dir)
    if not resources_ok:
        print("One or more resource files are missing. Please check the logs for details.")
        sys.exit(1)

    main_logger.info("Starting Tetris game")
    pygame.init()
    try:
        pygame.mixer.init()
        game_logger.debug("Pygame mixer initialized")
    except pygame.error as e:
        game_logger.error("Failed to initialize Pygame mixer: %s", str(e))
    
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
