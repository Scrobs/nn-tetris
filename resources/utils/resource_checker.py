# File: utils/resource_checker.py

import os
from typing import List
from utils.logging_setup import setup_logging

loggers = setup_logging()
resource_logger = loggers['resource']

def verify_resources(required_files: List[str], base_dir: str) -> bool:
    """
    Verify that all required resource files exist.
    
    Args:
        required_files: List of relative file paths to verify.
        base_dir: Base directory to prepend to each file path.
    
    Returns:
        bool: True if all files exist, False otherwise.
    """
    all_exist = True
    for file in required_files:
        full_path = os.path.join(base_dir, file)
        if not os.path.isfile(full_path):
            resource_logger.error("Missing resource file: %s", full_path)
            all_exist = False
    return all_exist
