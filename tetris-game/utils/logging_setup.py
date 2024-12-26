# File: utils/logging_setup.py
import logging
import sys

def setup_logging() -> dict:
    """Set up logging for different modules and return a dictionary of loggers."""
    loggers = {}
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler('tetris.log')
    file_handler.setFormatter(formatter)
    logger_names = ['main', 'game', 'render', 'resource', 'performance']
    for name in logger_names:
        logger = logging.getLogger(f'tetris.{name}')
        logger.setLevel(logging.DEBUG if name == 'performance' else logging.INFO)
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
        logger.propagate = False
        loggers[name] = logger
    return loggers
