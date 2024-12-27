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
    """Set up logging for different modules and return a dictionary of loggers."""
    loggers = {}
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    
    file_handler = RotatingFileHandler('tetris.log', maxBytes=5*1024*1024, backupCount=5)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    
    no_spam_filter = NoSpamFilter()
    stream_handler.addFilter(no_spam_filter)
    file_handler.addFilter(no_spam_filter)
    
    logger_names = ['main', 'game', 'render', 'resource', 'performance']
    for name in logger_names:
        logger = logging.getLogger(f'tetris.{name}')
        if name == 'performance':
            logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
        else:
            logger.setLevel(logging.INFO)
        logger.handlers = []
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
        logger.propagate = False
        loggers[name] = logger
    return loggers
