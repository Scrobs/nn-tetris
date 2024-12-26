# File: utils/performance_monitor.py
import time
import logging
from typing import List, Optional
from utils.logging_setup import setup_logging  # Imported from logging_setup.py

loggers = setup_logging()
performance_logger = loggers['performance']

class PerformanceMonitor:
    """
    Tracks performance metrics such as frame times and calculates average FPS.
    """
    def __init__(self):
        self.frame_times: List[float] = []
        self.frame_count: int = 0
        self.start_time: float = time.perf_counter()
        self.last_log_time: float = self.start_time
        self.frame_start: Optional[float] = None

    def start_frame(self) -> None:
        """Mark the start of a frame."""
        self.frame_start = time.perf_counter()

    def end_frame(self) -> None:
        """Mark the end of a frame, calculate frame time, and update statistics."""
        if self.frame_start is None:
            performance_logger.warning("end_frame called without start_frame")
            self.frame_start = time.perf_counter()
        frame_end = time.perf_counter()
        frame_time = (frame_end - self.frame_start) * 1000
        self.frame_times.append(frame_time)
        self.frame_count += 1
        if len(self.frame_times) > 120:
            self.frame_times.pop(0)

    def get_fps(self) -> float:
        """Calculate and return the average FPS based on recent frame times."""
        if not self.frame_times:
            return 0.0
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        fps = 1000.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        return fps

    def log_performance(self) -> None:
        """Log performance metrics at regular intervals."""
        current_time = time.perf_counter()
        if current_time - self.last_log_time >= 1.0:
            if self.frame_times:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                fps = self.get_fps()
                performance_logger.info("Frame stats - Avg: %.2fms, FPS: %.2f", avg_frame_time, fps)
            else:
                performance_logger.info("No frame times recorded yet.")
            self.last_log_time = current_time
