# File: utils/logging_setup.py
import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Dict

class NoSpamFilter(logging.Filter):
    """
    A filter that prevents logging of duplicate messages.
    Useful for suppressing repetitive warnings or errors.
    """
    def __init__(self):
        super().__init__()
        self.logged_messages = set()

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if msg in self.logged_messages:
            return False
        self.logged_messages.add(msg)
        return True

def setup_logging(debug_mode: bool = False) -> Dict[str, logging.Logger]:
    """
    Set up logging for different modules and return a dictionary of loggers.
    By default, we reduce the log level on the 'resource' and 'performance' modules
    so they only show warnings and errors unless debug mode is active.
    """
    loggers = {}
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    # In debug mode everything is shown, otherwise default is INFO
    stream_handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)

    file_handler = RotatingFileHandler('tetris.log', maxBytes=5*1024*1024, backupCount=5)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)

    no_spam_filter = NoSpamFilter()
    stream_handler.addFilter(no_spam_filter)
    file_handler.addFilter(no_spam_filter)

    # Lower the logging threshold for certain modules
    logger_names = ['main', 'game', 'render', 'resource', 'performance']

    for name in logger_names:
        logger = logging.getLogger(f'tetris.{name}')
        
        # For resource and performance, we set WARNING if not in debug mode
        if name in ['resource', 'performance'] and not debug_mode:
            logger.setLevel(logging.WARNING)
        elif debug_mode:
            logger.setLevel(logging.DEBUG)
        else:
            # Everything else at INFO if not debug
            logger.setLevel(logging.INFO)

        # clear any existing handlers to avoid duplication
        logger.handlers = []
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
        logger.propagate = False

        loggers[name] = logger

    return loggers
