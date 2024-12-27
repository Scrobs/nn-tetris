# File: utils/performance_monitor.py
import time
import logging
import statistics
from dataclasses import dataclass
from typing import List, Optional, Dict, Deque
from collections import deque
from utils.logging_setup import setup_logging

loggers = setup_logging()
performance_logger = loggers['performance']

@dataclass
class PerformanceMetrics:
    """Container for performance metrics calculations."""
    avg_frame_time: float
    min_frame_time: float
    max_frame_time: float
    fps: float
    frame_time_variance: float
    frame_time_std_dev: float

class PerformanceMonitor:
    """
    Tracks performance metrics with fixed timestep support and detailed analytics.
    """

    def __init__(self, 
                 fixed_timestep: float = 1/60.0, 
                 vsync_enabled: bool = True,
                 metrics_window_size: int = 120):
        self.frame_times: Deque[float] = deque(maxlen=metrics_window_size)
        self.frame_count: int = 0
        self.start_time: float = time.perf_counter()
        self.last_log_time: float = self.start_time
        self.frame_start: Optional[float] = None
        self.fixed_timestep: float = fixed_timestep
        self.accumulated_time: float = 0.0
        self.vsync_enabled: bool = vsync_enabled
        self.metrics_window_size: int = metrics_window_size
        self.min_frame_time: float = float('inf')
        self.max_frame_time: float = 0.0

        # Use a warning or debug statement instead of always info
        performance_logger.debug(
            "Performance monitor initialized - "
            f"Fixed timestep: {fixed_timestep:.3f}s, "
            f"VSync: {vsync_enabled}, "
            f"Metrics window: {metrics_window_size} frames"
        )

    def start_frame(self) -> None:
        if self.frame_start is not None:
            performance_logger.debug("start_frame called before previous frame ended")
            self.end_frame()
        self.frame_start = time.perf_counter()

    def end_frame(self) -> None:
        if self.frame_start is None:
            performance_logger.warning("end_frame called without start_frame")
            return
        frame_end = time.perf_counter()
        frame_time = (frame_end - self.frame_start) * 1000
        self.frame_times.append(frame_time)
        self.frame_count += 1
        self.min_frame_time = min(self.min_frame_time, frame_time)
        self.max_frame_time = max(self.max_frame_time, frame_time)
        self.frame_start = None

    def get_metrics(self) -> PerformanceMetrics:
        if not self.frame_times:
            return PerformanceMetrics(
                avg_frame_time=0.0,
                min_frame_time=0.0,
                max_frame_time=0.0,
                fps=0.0,
                frame_time_variance=0.0,
                frame_time_std_dev=0.0
            )

        avg_frame_time = statistics.mean(self.frame_times)
        fps = 1000.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        variance = statistics.variance(self.frame_times) if len(self.frame_times) > 1 else 0.0
        std_dev = statistics.stdev(self.frame_times) if len(self.frame_times) > 1 else 0.0

        return PerformanceMetrics(
            avg_frame_time=avg_frame_time,
            min_frame_time=self.min_frame_time,
            max_frame_time=self.max_frame_time,
            fps=fps,
            frame_time_variance=variance,
            frame_time_std_dev=std_dev
        )

    def log_performance(self, log_interval_sec: float = 5.0) -> None:
        """
        Log performance metrics at longer intervals to reduce spam (default: 5 seconds).
        """
        current_time = time.perf_counter()
        if current_time - self.last_log_time >= log_interval_sec:
            metrics = self.get_metrics()
            performance_logger.warning(
                # Now using `warning` or even `info` if you want some console output
                # but less frequent than debug in normal mode
                f"Performance Stats - FPS: {metrics.fps:.2f}, "
                f"Frame Time - Avg: {metrics.avg_frame_time:.2f}ms, "
                f"Min: {metrics.min_frame_time:.2f}ms, "
                f"Max: {metrics.max_frame_time:.2f}ms, "
                f"StdDev: {metrics.frame_time_std_dev:.2f}"
            )
            # Log a warning if frame_time_std_dev is significantly high
            if metrics.frame_time_std_dev > 16.67:
                performance_logger.warning("High frame time variance detected")

            self.last_log_time = current_time

    def reset_metrics(self) -> None:
        self.frame_times.clear()
        self.frame_count = 0
        self.min_frame_time = float('inf')
        self.max_frame_time = 0.0
        self.accumulated_time = 0.0
        performance_logger.debug("Performance metrics reset")
